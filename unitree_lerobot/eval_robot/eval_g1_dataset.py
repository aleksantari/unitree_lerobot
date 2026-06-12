"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import json
import torch
import tqdm
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pathlib import Path
from pprint import pformat
from typing import Any
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)

from unitree_lerobot.eval_robot.utils.utils import (
    extract_observation,
    log_inference_times,
    predict_chunk,
    OfflineEvalConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data


import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def _resolve_output_dir(cfg: OfflineEvalConfig, variant_tag: str | None = None) -> Path:
    # Honors cfg.output_dir if set (verbatim, no variant tag — explicit path = full manual control).
    # Otherwise lands under the checkpoint's run dir at <run>/eval/<dataset_safe>/<step>/<variant>, so
    # evals on different checkpoints don't clobber each other AND different inference variants of the same
    # checkpoint (e.g. denoise steps 4 vs 8) land in sibling dirs. The step subdir is omitted if the path
    # doesn't follow the canonical lerobot checkpoint layout (<run>/checkpoints/<step>/pretrained_model) —
    # detected by isdigit() on the parent dir name. The variant subdir is omitted when variant_tag is None.
    # Fallback for random-weight runs (no pretrained_path): write to ./eval_outputs/<dataset_safe>/<variant>.
    if cfg.output_dir is not None:
        return Path(cfg.output_dir)
    if cfg.policy is not None and cfg.policy.pretrained_path is not None:
        ckpt_path = Path(cfg.policy.pretrained_path)
        run_dir = ckpt_path.parent.parent.parent
        dataset_safe = cfg.repo_id.replace("/", "__")
        eval_dir = run_dir / "eval" / dataset_safe
        step_dir_name = ckpt_path.parent.name
        if step_dir_name.isdigit():
            eval_dir = eval_dir / step_dir_name
    else:
        eval_dir = Path("eval_outputs") / cfg.repo_id.replace("/", "__")
    if variant_tag:
        eval_dir = eval_dir / variant_tag
    return eval_dir


def _compute_episode_metrics(
    predicted_chunks: np.ndarray,
    ground_truth: np.ndarray,
    deployed: np.ndarray,
) -> dict:
    # mean_l2 / per-dim errors are over the fresh-prediction stream (chunk[0] per frame, no deployment
    # staleness — the upper bound). mean_l2_deployed_full_chunk is the same metric on the n_action_steps=K
    # deployment stream, so its ratio to mean_l2 quantifies the staleness penalty of full-chunk deployment.
    first_action_stream = predicted_chunks[:, 0, :]
    error = first_action_stream - ground_truth
    deployed_error = deployed - ground_truth
    return {
        "mean_l2": float(np.linalg.norm(error, axis=1).mean()),
        "mse_per_dim": np.mean(error**2, axis=0).tolist(),
        "mae_per_dim": np.mean(np.abs(error), axis=0).tolist(),
        "mean_l2_deployed_full_chunk": float(np.linalg.norm(deployed_error, axis=1).mean()),
    }


def _aggregate_metrics(per_episode: dict[int, dict]) -> dict:
    if not per_episode:
        return {}
    mean_l2s = np.array([m["mean_l2"] for m in per_episode.values()])
    deployed_l2s = np.array([m["mean_l2_deployed_full_chunk"] for m in per_episode.values()])
    mse_per_dims = np.stack([np.array(m["mse_per_dim"]) for m in per_episode.values()])
    agg = {
        "mean_l2_mean": float(mean_l2s.mean()),
        "mean_l2_std": float(mean_l2s.std()),
        "mean_l2_deployed_full_chunk_mean": float(deployed_l2s.mean()),
        "mse_per_dim_mean": np.mean(mse_per_dims, axis=0).tolist(),
        "n_episodes": len(per_episode),
    }
    # Cross-episode boundary-discontinuity aggregate (per arm joint): mean of per-episode means, max of maxes.
    if all("boundary_discontinuity" in m for m in per_episode.values()):
        joints = list(next(iter(per_episode.values()))["boundary_discontinuity"]["per_joint"].keys())
        pj = lambda m, nm: m["boundary_discontinuity"]["per_joint"][nm]  # noqa: E731
        agg["boundary_discontinuity"] = {
            "per_joint_mean_deg": {
                nm: float(np.nanmean([pj(m, nm)["mean_deg"] for m in per_episode.values()])) for nm in joints
            },
            "per_joint_max_deg": {
                nm: float(np.nanmax([pj(m, nm)["max_deg"] for m in per_episode.values()])) for nm in joints
            },
        }
    # Cross-episode horizon-decay curve: nanmean per chunk position across episodes (short episodes leave
    # trailing NaN at high k, which nanmean ignores).
    if all("horizon_decay_l2" in m for m in per_episode.values()):
        hds = np.array([m["horizon_decay_l2"] for m in per_episode.values()], dtype=float)
        agg["horizon_decay_l2_mean"] = np.nanmean(hds, axis=0).tolist()
    return agg


def _resolve_action_dim_names(dataset: LeRobotDataset, action_dim: int) -> list[str]:
    # Reads dataset.meta.features["action"]["names"] (a list-of-lists per the lerobot per-axis convention).
    # The converter at convert_unitree_json_to_lerobot.py populates this from ROBOT_CONFIGS at dataset build time,
    # so the dataset is the authoritative record. Falls back to "Dim N" if missing/malformed.
    try:
        names = dataset.meta.features["action"]["names"][0]
        if isinstance(names, (list, tuple)) and len(names) == action_dim:
            return list(names)
    except (KeyError, TypeError, IndexError):
        pass
    logger_mp.warning(
        f"Could not read action dim names from dataset.meta.features['action']['names']; "
        f"falling back to generic 'Dim N' labels (expected {action_dim} names)."
    )
    return [f"Dim {i + 1}" for i in range(action_dim)]


def _denoising_heads(policy: nn.Module) -> list[nn.Module]:
    # GR00T's flow-matching action head exposes num_inference_timesteps (read live in its sampling loop,
    # flow_matching_action_head.py). Found by attribute rather than a hardcoded path so it survives the
    # _groot_model/action_head nesting. Empty for ACT (no denoising loop).
    return [m for m in policy.modules() if hasattr(m, "num_inference_timesteps")]


def _set_denoising_steps(policy: nn.Module, n: int) -> None:
    # Overriding the attribute on the built module changes the denoising-step count with no rebuild. The
    # value isn't a GrootConfig field (it comes from the base GR00T-N1.5 action_head_cfg, default 4), so a
    # --policy.* CLI override can't reach it — we set it here.
    heads = _denoising_heads(policy)
    if not heads:
        logger_mp.warning(
            f"--num_inference_timesteps={n} requested but no flow-matching action head was found on this "
            f"policy. This knob only applies to GR00T-style diffusion policies; ACT has no denoising loop. "
            f"Leaving the policy unchanged."
        )
        return
    for h in heads:
        old = h.num_inference_timesteps
        h.num_inference_timesteps = n
        logger_mp.info(f"Denoising steps on {type(h).__name__}.num_inference_timesteps: {old} -> {n}")


def _get_denoising_steps(policy: nn.Module) -> int | None:
    # Effective denoising-step count after any override (the base GR00T-N1.5 default is 4). None for ACT.
    # Used to tag the output dir so denoise-step variants of the same checkpoint don't collide.
    heads = _denoising_heads(policy)
    return heads[0].num_inference_timesteps if heads else None


def _variant_tag(cfg: OfflineEvalConfig, denoising_steps: int | None) -> str | None:
    # Encodes the inference-time configuration that distinguishes runs of the SAME checkpoint+episode, so
    # e.g. 4-step and 8-step GR00T evals land in sibling dirs instead of clobbering each other. Pieces are
    # only added when relevant: `steps{N}` for diffusion policies (omitted for ACT), `seed{N}` when a seed
    # is set, plus any free-form cfg.tag. Returns None when there's nothing to distinguish (e.g. plain ACT,
    # no seed, no tag) so that case keeps the flat <step>/ layout.
    pieces = []
    if denoising_steps is not None:
        pieces.append(f"steps{denoising_steps}")
    if cfg.seed is not None:
        pieces.append(f"seed{cfg.seed}")
    if cfg.tag:
        pieces.append(cfg.tag)
    return "_".join(pieces) if pieces else None


def _build_deployed_stream(predicted_chunks: np.ndarray, ground_truth: np.ndarray, k: int) -> np.ndarray:
    # Reconstructs what a real-robot deployment at n_action_steps=k would have executed, purely from the
    # saved chunks: re-query inference at frames 0, k, 2k, ... and pop the pre-computed chunk actions in
    # between. deployed[t] = chunks[(t // k) * k, t % k]. No extra inference. For k = chunk_size this is
    # the most-stale cadence (one fresh observation per full chunk).
    T = ground_truth.shape[0]
    deployed = np.empty_like(ground_truth)
    for t in range(T):
        start = (t // k) * k
        deployed[t] = predicted_chunks[start, t - start]
    return deployed


def _plot_fresh_stream(
    ground_truth: np.ndarray,
    first_action_stream: np.ndarray,
    action_dim_names: list[str],
    out_path: Path,
    ep_idx: int,
) -> None:
    # Plot 1 — the fresh-prediction stream: chunk[0] at every frame vs GT. The n_action_steps=1 upper
    # bound (a brand-new inference per frame, zero deployment staleness).
    _, n_dims = ground_truth.shape
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2.5 * n_dims), sharex=True, constrained_layout=True)
    fig.suptitle(f"Plot 1 — Fresh-prediction stream (chunk[0] per frame) vs GT — Episode {ep_idx}")
    for i in range(n_dims):
        ax = axes[i] if n_dims > 1 else axes
        ax.plot(ground_truth[:, i], color="blue", label="Ground Truth")
        ax.plot(first_action_stream[:, i], color="red", linestyle="--", label="Predicted (chunk[0])")
        # Arm joint values are in radians (G1 DDS convention); gripper units are policy-specific so leave unlabeled.
        unit_suffix = "" if "Gripper" in action_dim_names[i] else " (rad)"
        ax.set_ylabel(f"{action_dim_names[i]}{unit_suffix}")
        if i == 0:
            ax.legend(loc="upper right")
    (axes[-1] if n_dims > 1 else axes).set_xlabel("Timestep")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_chunk_fan(
    ground_truth: np.ndarray,
    predicted_chunks: np.ndarray,
    action_dim_names: list[str],
    out_path: Path,
    ep_idx: int,
    stride: int,
) -> None:
    # Plot 2 — the full predicted chunk drawn at each obs step (every `stride` frames), so a chunk that
    # peels away from GT exposes which observation the policy started drifting from. The color evolves
    # ALONG each chunk by its within-chunk position k: blue at k=0 (the immediate prediction) -> red at
    # k=chunk_size-1 (the far-horizon extrapolation), so on each fan you can see which part is early vs
    # late. GT is the solid black reference. Chunks are clipped at the episode end so x matches GT.
    T, chunk_size, n_dims = predicted_chunks.shape
    norm = Normalize(vmin=0, vmax=max(chunk_size - 1, 1))
    cmap = LinearSegmentedColormap.from_list("chunk_pos", ["blue", "red"])
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2.5 * n_dims), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"Plot 2 — Full predicted chunks per obs step (stride={stride}), colored early(blue)→late(red) "
        f"within each chunk — Episode {ep_idx}"
    )
    for i in range(n_dims):
        ax = axes[i] if n_dims > 1 else axes
        # Break each drawn chunk into per-step segments so the color can vary along it (scalar = position k).
        seg_list, pos_list = [], []
        for t in range(0, T, stride):
            k_max = min(chunk_size, T - t)
            if k_max <= 1:
                continue
            pts = np.column_stack([np.arange(t, t + k_max), predicted_chunks[t, :k_max, i]])  # (k_max, 2)
            seg_list.append(np.stack([pts[:-1], pts[1:]], axis=1))  # (k_max-1, 2, 2)
            pos_list.append(np.arange(k_max - 1))  # color each segment by its start position k
        if seg_list:
            lc = LineCollection(np.concatenate(seg_list), cmap=cmap, norm=norm, linewidths=0.8, alpha=0.5)
            lc.set_array(np.concatenate(pos_list))
            ax.add_collection(lc)
        ax.plot(ground_truth[:, i], color="black", linewidth=1.6, label="Ground Truth", zorder=10)
        ax.autoscale()
        ax.set_xlim(0, T)
        unit_suffix = "" if "Gripper" in action_dim_names[i] else " (rad)"
        ax.set_ylabel(f"{action_dim_names[i]}{unit_suffix}")
        if i == 0:
            ax.legend(loc="upper right")
    (axes[-1] if n_dims > 1 else axes).set_xlabel("Timestep")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="within-chunk position k (early=blue → late=red)", fraction=0.015, pad=0.01)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_deployed_full_chunk(
    ground_truth: np.ndarray,
    deployed: np.ndarray,
    action_dim_names: list[str],
    out_path: Path,
    ep_idx: int,
    k: int,
) -> None:
    # Plot 3 — the realtime-deployment view at n_action_steps=k (= chunk_size here): re-query inference
    # only at frames 0, k, 2k, ... (gray verticals) and replay the pre-computed chunk actions in between.
    # The action drifts on stale observations between re-queries, then snaps at the next query — the
    # staleness pattern the fresh-stream plot (Plot 1) hides.
    T, n_dims = ground_truth.shape
    requery_starts = range(0, T, k)
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2.5 * n_dims), sharex=True, constrained_layout=True)
    fig.suptitle(f"Plot 3 — Realtime deployment (n_action_steps={k}, full chunk) vs GT — Episode {ep_idx}")
    for i in range(n_dims):
        ax = axes[i] if n_dims > 1 else axes
        ax.plot(ground_truth[:, i], color="blue", label="Ground Truth")
        ax.plot(deployed[:, i], color="red", linestyle="--", label=f"Deployed (re-query every {k})")
        for rs in requery_starts:
            ax.axvline(rs, color="gray", linewidth=0.6, alpha=0.5)
        unit_suffix = "" if "Gripper" in action_dim_names[i] else " (rad)"
        ax.set_ylabel(f"{action_dim_names[i]}{unit_suffix}")
        if i == 0:
            ax.legend(loc="upper right")
    (axes[-1] if n_dims > 1 else axes).set_xlabel("Timestep")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _boundary_jump_stats(deployed: np.ndarray, action_dim_names: list[str], k: int) -> dict:
    # Per-arm-joint chunk-boundary discontinuity on the deployed stream. At each re-query boundary
    # b = k, 2k, ... the action jumps from the stale last action of the old chunk (deployed[b-1]) to the
    # fresh first action of the new chunk (deployed[b]); |deployed[b,j] - deployed[b-1,j]| is that jump.
    # Arm joints only (grippers excluded — different units), converted to degrees (arm actions are radians).
    # Returns per-joint mean/max (json-safe lists) plus the raw per-boundary jumps for plotting.
    T, _ = deployed.shape
    boundaries = np.arange(k, T, k)  # index of the first deployed sample of each new chunk
    arm_idx = [i for i, n in enumerate(action_dim_names) if "Gripper" not in n]
    arm_names = [action_dim_names[i] for i in arm_idx]
    if boundaries.size == 0:
        nan = [float("nan")] * len(arm_idx)
        return {"arm_names": arm_names, "mean_deg": nan, "max_deg": nan,
                "n_boundaries": 0, "boundaries": boundaries, "jumps_deg": np.empty((0, len(arm_idx)))}
    jumps_deg = np.degrees(np.abs(deployed[boundaries] - deployed[boundaries - 1]))[:, arm_idx]  # (n_b, n_arm)
    return {
        "arm_names": arm_names,
        "mean_deg": jumps_deg.mean(axis=0).tolist(),
        "max_deg": jumps_deg.max(axis=0).tolist(),
        "n_boundaries": int(boundaries.size),
        "boundaries": boundaries,
        "jumps_deg": jumps_deg,
    }


def _plot_boundary_discontinuity(disc: dict, out_path: Path, ep_idx: int, k: int) -> None:
    # Plot 4 — one bar chart per ARM joint (grippers excluded). Each bar is one chunk-boundary
    # discontinuity event: x = boundary timestep, height = |Δaction| at that boundary (degrees). The
    # dashed line marks the per-joint mean; mean/max are in each subplot title. Shows the magnitude of
    # every discontinuity across the episode, per joint.
    arm_names, boundaries, jumps_deg = disc["arm_names"], disc["boundaries"], disc["jumps_deg"]
    n = len(arm_names)
    if disc["n_boundaries"] == 0 or n == 0:
        logger_mp.warning(
            "No chunk boundaries (episode shorter than one chunk) or no arm joints; skipping plot 4."
        )
        return
    # sharey=True puts all joints on one scale so bar heights are visually comparable across joints
    # (the per-joint titles still carry absolute mean/max in case a quiet joint's bars look flat).
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.8 * n), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle(f"Plot 4 — chunk-boundary discontinuity per arm joint (n_action_steps={k}) — Episode {ep_idx}")
    bar_w = max(1.0, k * 0.6)
    for row in range(n):
        ax = axes[row] if n > 1 else axes
        vals = jumps_deg[:, row]
        mean_v, max_v = float(vals.mean()), float(vals.max())
        ax.bar(boundaries, vals, width=bar_w, color="indianred", align="center")
        ax.axhline(mean_v, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel("|Δ| (deg)")
        ax.set_title(f"{arm_names[row]}    mean={mean_v:.2f}°   max={max_v:.2f}°", fontsize=9, loc="left")
    (axes[-1] if n > 1 else axes).set_xlabel("chunk-boundary timestep")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _horizon_decay_l2(predicted_chunks: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    # For each chunk position k: mean over valid frames t of ‖chunk[t,k] − GT[t+k]‖₂ — how well the policy
    # predicts k steps ahead from a single observation, averaged over the episode. Phase-free and
    # position-resolved (the "global view" of chunk predictive quality). hd[0] == fresh mean_l2 by
    # construction; same per-frame L2 convention as mean_l2/deployed, so E[deployed@k] ≈ mean(hd[0:k]).
    # Positions past the episode end (T − k ≤ 0) stay NaN.
    T, chunk_size, _ = predicted_chunks.shape
    hd = np.full(chunk_size, np.nan)
    for k in range(chunk_size):
        if T - k <= 0:
            continue
        diff = predicted_chunks[: T - k, k] - ground_truth[k:]
        hd[k] = float(np.linalg.norm(diff, axis=1).mean())
    return hd


def _plot_horizon_decay(hd: np.ndarray, out_path: Path, ep_idx: int) -> None:
    # Plot 5 — horizon decay: per-position prediction error vs chunk position k (how far ahead a single
    # observation can predict). The dashed running mean of hd[0..k] tracks the expected deployment error
    # if you re-query every k+1 frames; its endpoint ≈ deployed@chunk_size, and the curve's steepness is
    # the principled basis for choosing the deployment cadence (re-query before it climbs too far).
    chunk_size = hd.shape[0]
    ks = np.arange(chunk_size)
    running_mean = np.array([np.nanmean(hd[: k + 1]) for k in range(chunk_size)])
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(ks, hd, marker="o", markersize=3, color="purple", label="horizon decay  hd[k] = ‖chunk[k] − GT[t+k]‖")
    ax.plot(ks, running_mean, color="green", linestyle="--", linewidth=1.2,
            label="running mean of hd[0..k]  (≈ deployed error using positions 0..k)")
    ax.set_xlabel("chunk position k (steps ahead of the observation)")
    ax.set_ylabel("mean L2 error")
    ax.set_title(f"Plot 5 — horizon decay (chunk[k] vs GT[t+k] across t) — Episode {ep_idx}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def eval_policy(
    cfg: OfflineEvalConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    variant_tag: str | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    # ----- Optional Rerun visualization sink -----
    if cfg.visualization:
        rerun_logger = RerunLogger()

    # ----- One-time reset of policy + processor pipelines -----
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    # ----- Resolve and prepare output directory (variant_tag separates inference variants) -----
    output_dir = _resolve_output_dir(cfg, variant_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger_mp.info(f"Eval outputs will be written to: {output_dir}")

    # ----- Resolve action-dim names (joint labels) from dataset metadata -----
    action_dim = dataset.meta.features["action"]["shape"][0]
    action_dim_names = _resolve_action_dim_names(dataset, action_dim)
    logger_mp.info(f"Action dim names ({action_dim}): {action_dim_names}")

    # ----- Sampling seed (matters for stochastic policies like GR00T's diffusion head) -----
    if cfg.seed is None:
        logger_mp.info(
            "Random seed: None (sampling will be stochastic for diffusion policies like GR00T; "
            "re-runs will produce different metrics). Pass --seed=N for reproducibility."
        )
    else:
        logger_mp.info(f"Random seed: {cfg.seed} (resolved per-episode as cfg.seed + ep_idx)")

    # ----- Cross-episode accumulators -----
    all_inference_times_ms: list[float] = []
    per_episode_metrics: dict[int, dict] = {}

    # ===== Per-episode loop =====
    for ep_idx in cfg.episodes:
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]

        # Clears the policy's internal chunk queue so each episode starts cold
        # (otherwise the first frame would pop a stale action from the previous episode).
        policy.reset()

        # Per-episode seeding: lock RNG so re-running a single episode subset reproduces the same samples.
        # The +ep_idx offset means different episodes still use different seeds within one run.
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + ep_idx)
            np.random.seed(cfg.seed + ep_idx)

        # ----- Per-episode output directory -----
        episode_dir = output_dir / "episodes" / f"episode_{ep_idx:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        ground_truth_actions = []
        predicted_chunks = []
        inference_times_ms: list[float] = []

        # ----- Per-frame inference loop -----
        for step_idx in tqdm.tqdm(range(from_idx, to_idx), desc=f"episode {ep_idx}"):
            step = dataset[step_idx]
            observation = extract_observation(step)

            # `.to("cpu")` at the tail of predict_chunk forces a CUDA sync, so this
            # captures the real obs-in→action-out latency (incl. GPU work).
            infer_start = time.perf_counter()
            chunk = predict_chunk(
                observation,
                policy,
                get_safe_torch_device(policy.config.device),
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                step["task"],
                use_dataset=True,
                robot_type=None,
            )
            inference_times_ms.append((time.perf_counter() - infer_start) * 1000.0)
            chunk_np = chunk.cpu().numpy()  # shape (chunk_size, action_dim)

            ground_truth_actions.append(step["action"].numpy())
            predicted_chunks.append(chunk_np)

            if cfg.visualization:
                # Visualize chunk[0] — the action that would deploy if we used n_action_steps=1.
                visualization_data(step_idx, observation, observation["observation.state"], chunk_np[0], rerun_logger)

        # ----- Per-episode latency summary -----
        log_inference_times(f"Episode {ep_idx}", inference_times_ms)
        all_inference_times_ms.extend(inference_times_ms)

        # ----- Stack predictions + GT for this episode -----
        ground_truth_actions = np.array(ground_truth_actions)  # (T, action_dim)
        predicted_chunks = np.stack(predicted_chunks)  # (T, chunk_size, action_dim)
        first_action_stream = predicted_chunks[:, 0, :]  # (T, action_dim) — fresh-prediction stream
        T_steps, chunk_size, _ = predicted_chunks.shape

        # Plot 3's deployed stream: realtime cadence at n_action_steps = chunk_size (the most-stale case).
        deployed_full_chunk = _build_deployed_stream(predicted_chunks, ground_truth_actions, chunk_size)

        # ----- Three diagnostic plots (chunks stay in memory; nothing persisted to disk) -----
        # The fan plot gets dense on long episodes; stride keeps it legible while staying per-step on short ones.
        # cfg.fan_stride overrides the auto rule (fan_stride=1 => draw a chunk from every frame).
        fan_stride = cfg.fan_stride if cfg.fan_stride is not None else max(1, T_steps // 120)
        _plot_fresh_stream(
            ground_truth_actions, first_action_stream, action_dim_names,
            episode_dir / "1_fresh_chunk0.png", ep_idx,
        )
        _plot_chunk_fan(
            ground_truth_actions, predicted_chunks, action_dim_names,
            episode_dir / "2_chunk_fan.png", ep_idx, fan_stride,
        )
        _plot_deployed_full_chunk(
            ground_truth_actions, deployed_full_chunk, action_dim_names,
            episode_dir / "3_deployed_full_chunk.png", ep_idx, chunk_size,
        )
        # Plot 4 + per-joint boundary-discontinuity stats (same deployed stream / cadence as plot 3).
        disc = _boundary_jump_stats(deployed_full_chunk, action_dim_names, chunk_size)
        _plot_boundary_discontinuity(
            disc, episode_dir / "4_boundary_discontinuity.png", ep_idx, chunk_size,
        )
        # Plot 5 — horizon decay: per-position predictive error (the global, phase-free view of how well
        # one observation predicts the future; hd[0] == fresh mean_l2, mean(hd[0:k]) ≈ deployed@k).
        horizon_decay = _horizon_decay_l2(predicted_chunks, ground_truth_actions)
        _plot_horizon_decay(horizon_decay, episode_dir / "5_horizon_decay.png", ep_idx)

        # ----- Per-episode core metrics -----
        ep_metrics = _compute_episode_metrics(predicted_chunks, ground_truth_actions, deployed_full_chunk)
        ep_metrics["boundary_discontinuity"] = {
            "n_boundaries": disc["n_boundaries"],
            "per_joint": {
                nm: {"mean_deg": mn, "max_deg": mx}
                for nm, mn, mx in zip(disc["arm_names"], disc["mean_deg"], disc["max_deg"])
            },
        }
        ep_metrics["horizon_decay_l2"] = horizon_decay.tolist()
        per_episode_metrics[ep_idx] = ep_metrics
        mse_arr = np.array(ep_metrics["mse_per_dim"])
        logger_mp.info(
            f"Episode {ep_idx} metrics: mean_l2={ep_metrics['mean_l2']:.4f} "
            f"(deployed@{chunk_size}={ep_metrics['mean_l2_deployed_full_chunk']:.4f}) | "
            f"per-dim MSE min={mse_arr.min():.5f} max={mse_arr.max():.5f} mean={mse_arr.mean():.5f}"
        )
        if disc["n_boundaries"]:
            wj = int(np.argmax(disc["max_deg"]))
            logger_mp.info(
                f"Episode {ep_idx} boundary discontinuity (n={disc['n_boundaries']}): "
                f"worst joint {disc['arm_names'][wj]} max={disc['max_deg'][wj]:.2f}° "
                f"mean={disc['mean_deg'][wj]:.2f}°"
            )

    # ===== Cross-episode aggregate latency summary =====
    log_inference_times("All episodes", all_inference_times_ms)

    # ===== Cross-episode metrics aggregate + metrics.json dump =====
    aggregate_metrics = _aggregate_metrics(per_episode_metrics)
    if aggregate_metrics:
        logger_mp.info(
            f"All episodes mean_l2: {aggregate_metrics['mean_l2_mean']:.4f} ± {aggregate_metrics['mean_l2_std']:.4f} "
            f"(n={aggregate_metrics['n_episodes']} episodes)"
        )

    metrics_payload = {
        "action_dim_names": action_dim_names,
        "episodes": {str(ep_idx): m for ep_idx, m in per_episode_metrics.items()},
        "aggregate": aggregate_metrics,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger_mp.info(f"Metrics written to: {metrics_path}")


@parser.wrap()
def eval_main(cfg: OfflineEvalConfig):
    logging.info(pformat(asdict(cfg)))

    # ----- Device + cuDNN tuning -----
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ----- Load held-out dataset (metadata + frames) -----
    logging.info("Making policy.")
    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    # ----- Load policy weights + the matching pre/postprocessor pipelines from the checkpoint -----
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    # GR00T-only: override flow-matching denoising steps at eval time (no-op for ACT).
    if cfg.num_inference_timesteps is not None:
        _set_denoising_steps(policy, cfg.num_inference_timesteps)

    # Tag the output dir with the inference variant (effective denoise steps + seed + cfg.tag) so runs of
    # the same checkpoint/episode under different settings sit in sibling dirs instead of overwriting.
    variant_tag = _variant_tag(cfg, _get_denoising_steps(policy))
    logger_mp.info(f"Output variant tag: {variant_tag or '(none — flat <step>/ layout)'}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    # ----- Run eval inside no_grad + (optional) autocast context -----
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor, variant_tag=variant_tag)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
