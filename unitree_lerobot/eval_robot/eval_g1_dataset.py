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
    predict_chunk,
    OfflineEvalConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data


import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

_REALTIME_BUDGET_HZ = 30.0
_REALTIME_BUDGET_MS = 1000.0 / _REALTIME_BUDGET_HZ


def _log_inference_times(label: str, times_ms: list[float]) -> None:
    if not times_ms:
        return
    arr = np.array(times_ms)
    budget_ok = arr.max() < _REALTIME_BUDGET_MS
    logger_mp.info(
        f"{label} inference (ms): "
        f"mean={arr.mean():.2f} std={arr.std():.2f} "
        f"min={arr.min():.2f} max={arr.max():.2f} "
        f"p50={np.percentile(arr, 50):.2f} p95={np.percentile(arr, 95):.2f} p99={np.percentile(arr, 99):.2f} "
        f"| n={len(arr)} | first={arr[0]:.2f} (incl. warm-up) "
        f"| {_REALTIME_BUDGET_HZ:.0f}Hz budget ({_REALTIME_BUDGET_MS:.2f}ms): "
        f"{'OK' if budget_ok else 'BUSTED'}"
    )


def _resolve_output_dir(cfg: OfflineEvalConfig) -> Path:
    # Honors cfg.output_dir if set. Otherwise lands under the checkpoint's run dir at <run>/eval/<dataset_safe>.
    # Fallback for random-weight runs (no pretrained_path): write to ./eval_outputs/<dataset_safe>.
    if cfg.output_dir is not None:
        return Path(cfg.output_dir)
    if cfg.policy is not None and cfg.policy.pretrained_path is not None:
        run_dir = Path(cfg.policy.pretrained_path).parent.parent.parent
        dataset_safe = cfg.repo_id.replace("/", "__")
        return run_dir / "eval" / dataset_safe
    return Path("eval_outputs") / cfg.repo_id.replace("/", "__")


def _compute_episode_metrics(
    predicted_chunks: np.ndarray,
    ground_truth: np.ndarray,
    horizon_decay_mse: np.ndarray,
) -> dict:
    # All metrics here are over the fresh-prediction stream — chunk[0] per frame, no deployment staleness.
    first_action_stream = predicted_chunks[:, 0, :]
    error = first_action_stream - ground_truth
    return {
        "mean_l2": float(np.linalg.norm(error, axis=1).mean()),
        "mse_per_dim": np.mean(error ** 2, axis=0).tolist(),
        "mae_per_dim": np.mean(np.abs(error), axis=0).tolist(),
        "horizon_decay_mse": horizon_decay_mse.tolist(),
    }


def _aggregate_metrics(per_episode: dict[int, dict]) -> dict:
    # nanmean across episodes so short episodes (which leave trailing NaN in horizon_decay_mse) don't poison the aggregate curve.
    if not per_episode:
        return {}
    mean_l2s = np.array([m["mean_l2"] for m in per_episode.values()])
    mse_per_dims = np.stack([np.array(m["mse_per_dim"]) for m in per_episode.values()])
    horizon_decays = np.stack([np.array(m["horizon_decay_mse"]) for m in per_episode.values()])
    return {
        "mean_l2_mean": float(mean_l2s.mean()),
        "mean_l2_std": float(mean_l2s.std()),
        "mse_per_dim_mean": np.nanmean(mse_per_dims, axis=0).tolist(),
        "horizon_decay_mse_mean": np.nanmean(horizon_decays, axis=0).tolist(),
        "n_episodes": len(per_episode),
    }


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


def eval_policy(
    cfg: OfflineEvalConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
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

    # ----- Manual start gate (legacy from the real-robot script; harmless offline) -----
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() != "s":
        return

    # ----- Resolve and prepare output directory -----
    output_dir = _resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger_mp.info(f"Eval outputs will be written to: {output_dir}")

    # ----- Resolve action-dim names (joint labels) from dataset metadata -----
    action_dim = dataset.meta.features["action"]["shape"][0]
    action_dim_names = _resolve_action_dim_names(dataset, action_dim)
    logger_mp.info(f"Action dim names ({action_dim}): {action_dim_names}")

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
        _log_inference_times(f"Episode {ep_idx}", inference_times_ms)
        all_inference_times_ms.extend(inference_times_ms)

        # ----- Stack and derive analysis arrays -----
        ground_truth_actions = np.array(ground_truth_actions)       # (T, action_dim)
        predicted_chunks = np.stack(predicted_chunks)               # (T, chunk_size, action_dim)
        first_action_stream = predicted_chunks[:, 0, :]             # (T, action_dim) — fresh-prediction stream

        T_steps, chunk_size, _ = predicted_chunks.shape

        # ----- Horizon-decay MSE per chunk position k -----
        # For each k: mean over valid t of MSE(chunk[t, k], GT[t+k]). Positions past episode end stay NaN.
        horizon_decay_mse = np.full(chunk_size, np.nan)
        for k in range(chunk_size):
            valid_frames = T_steps - k
            if valid_frames <= 0:
                continue
            diff = predicted_chunks[:valid_frames, k] - ground_truth_actions[k:]
            horizon_decay_mse[k] = float(np.mean(diff ** 2))

        # ----- Per-episode trajectory plot: GT vs fresh-prediction stream (chunk[0] per frame) -----
        n_timesteps, n_dims = ground_truth_actions.shape

        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims), sharex=True)
        fig.suptitle(f"Ground Truth vs Fresh-Prediction Stream (chunk[0]) — Episode {ep_idx}")

        for i in range(n_dims):
            ax = axes[i] if n_dims > 1 else axes

            ax.plot(ground_truth_actions[:, i], label="Ground Truth", color="blue")
            ax.plot(first_action_stream[:, i], label="Predicted (chunk[0])", color="red", linestyle="--")
            ax.set_ylabel(action_dim_names[i])
            ax.legend()

        axes[-1].set_xlabel("Timestep")

        plt.tight_layout()
        plt.savefig(episode_dir / "actions_trajectory.png")
        plt.close(fig)

        # ----- Per-episode horizon-decay plot -----
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(np.arange(chunk_size), horizon_decay_mse, marker="o", markersize=3, color="purple")
        ax.set_xlabel("Chunk position k")
        ax.set_ylabel("Mean squared error")
        ax.set_title(f"Horizon decay — Episode {ep_idx} (MSE of chunk[k] vs GT[t+k] across t)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(episode_dir / "horizon_decay.png")
        plt.close(fig)

        # ----- Persist raw arrays so any n_action_steps cadence can be derived offline -----
        np.savez_compressed(
            episode_dir / "predictions.npz",
            chunks=predicted_chunks,
            ground_truth=ground_truth_actions,
            horizon_decay_mse=horizon_decay_mse,
        )

        # ----- Per-episode core metrics -----
        ep_metrics = _compute_episode_metrics(predicted_chunks, ground_truth_actions, horizon_decay_mse)
        per_episode_metrics[ep_idx] = ep_metrics
        mse_arr = np.array(ep_metrics["mse_per_dim"])
        logger_mp.info(
            f"Episode {ep_idx} metrics: mean_l2={ep_metrics['mean_l2']:.4f} | "
            f"per-dim MSE min={mse_arr.min():.5f} max={mse_arr.max():.5f} mean={mse_arr.mean():.5f}"
        )

    # ===== Cross-episode aggregate latency summary =====
    _log_inference_times("All episodes", all_inference_times_ms)

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
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
