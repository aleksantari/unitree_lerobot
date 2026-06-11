import json
import numpy as np
import torch
from pathlib import Path
from typing import Any
from contextlib import nullcontext
from copy import copy
import logging
from dataclasses import dataclass, field
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def log_inference_times(label: str, times_ms: list[float]) -> None:
    """Aggregate inference-time stats. Used by the offline eval (eval_g1_dataset.py).
    On-robot eval (eval_g1.py) uses TimingLog instead, which carries its own runtime
    budget derived from cfg.frequency."""
    if not times_ms:
        return
    arr = np.array(times_ms)
    logger_mp.info(
        f"{label} inference (ms): "
        f"mean={arr.mean():.2f} std={arr.std():.2f} "
        f"min={arr.min():.2f} max={arr.max():.2f} "
        f"p50={np.percentile(arr, 50):.2f} p95={np.percentile(arr, 95):.2f} p99={np.percentile(arr, 99):.2f} "
        f"| n={len(arr)} | first={arr[0]:.2f} (incl. warm-up)"
    )


def _format_stage_stats(label: str, arr_ms: np.ndarray, budget_ms: float | None = None) -> str:
    """One-line per-stage stats. Used by TimingLog's summary."""
    if arr_ms.size == 0:
        return f"{label}: (no data)"
    line = (
        f"{label}: mean={arr_ms.mean():.2f} std={arr_ms.std():.2f} "
        f"min={arr_ms.min():.2f} max={arr_ms.max():.2f} "
        f"p50={np.percentile(arr_ms, 50):.2f} p95={np.percentile(arr_ms, 95):.2f} p99={np.percentile(arr_ms, 99):.2f} "
        f"| n={len(arr_ms)}"
    )
    if budget_ms is not None:
        line += f" | budget {budget_ms:.2f}ms: {'OK' if arr_ms.max() < budget_ms else 'BUSTED'}"
    return line


@dataclass
class TimingLog:
    """Per-step latency capture for eval_g1.py's policy loop.

    Writes one line to timing.csv per loop iteration (line-buffered, atomic on
    most filesystems → survives SIGKILL up to the last completed step). Keeps
    rows in memory so finalize() can produce a compressed npz + summary at the
    end of the run (also runs from `finally` so Ctrl+C is preserved cleanly).
    """

    out_dir: Path
    # Budget for the OK/BUSTED labels in summary.txt and the Rerun budget reference line.
    # Required -- caller must pass cfg.frequency (or whatever rate this run is paced at).
    # No default: forces the caller to think about which budget applies to this run, so the
    # summary label can't silently mismatch the actual loop rate.
    budget_hz: float
    fields: tuple[str, ...] = (
        "step",
        "t_obs_ms",
        "t_infer_ms",
        "t_tau_ms",
        "t_ctrl_ms",
        "t_loop_ms",
        "t_sleep_ms",
        "chunk_boundary",
        "queue_len_before",
        "missed_deadline",
        "arm_delta_max",
    )

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "timing.csv"
        self.budget_ms = 1000.0 / self.budget_hz
        # buffering=1 → line-buffered text mode: each newline triggers a flush, so
        # `tail -f timing.csv` mid-run shows live data and SIGKILL leaves a usable file.
        self._csv = open(self.csv_path, "w", buffering=1)
        self._csv.write(",".join(self.fields) + "\n")
        self.rows: list[dict] = []
        self.actions: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self._finalized = False

    def append(self, **kwargs) -> None:
        # Store raw values (float/int) in self.rows so npz inherits proper dtypes.
        # Format only on the CSV side so the human-readable file stays tidy.
        row = {f: kwargs.get(f, "") for f in self.fields}

        def _csv_fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        self._csv.write(",".join(_csv_fmt(row[f]) for f in self.fields) + "\n")
        self.rows.append(row)

    def add_step_data(self, action: np.ndarray, state: np.ndarray) -> None:
        self.actions.append(np.asarray(action).copy())
        self.states.append(np.asarray(state).copy())

    def finalize(self, abort_reason: str, cfg_snapshot: dict | None = None) -> None:
        """Idempotent: safe to call from finally even after an early abort."""
        if self._finalized:
            return
        self._finalized = True
        try:
            self._csv.flush()
            self._csv.close()
        except Exception:
            pass
        if not self.rows:
            return
        arr = {f: np.array([r[f] for r in self.rows]) for f in self.fields}
        np.savez_compressed(
            self.out_dir / "timing.npz",
            **arr,
            actions=np.stack(self.actions) if self.actions else np.empty(0),
            states=np.stack(self.states) if self.states else np.empty(0),
            abort_reason=np.asarray(abort_reason),
        )
        if cfg_snapshot is not None:
            (self.out_dir / "config.json").write_text(
                json.dumps(cfg_snapshot, indent=2, default=str)
            )
        self._write_summary(abort_reason)

    def _write_summary(self, abort_reason: str) -> None:
        rows = self.rows
        n = len(rows)
        get = lambda f: np.array([float(r[f]) for r in rows])  # noqa: E731
        t_obs = get("t_obs_ms")
        t_infer = get("t_infer_ms")
        t_tau = get("t_tau_ms")
        t_ctrl = get("t_ctrl_ms")
        t_loop = get("t_loop_ms")
        t_sleep = get("t_sleep_ms")
        chunk_boundary = np.array([int(r["chunk_boundary"]) for r in rows])
        missed_deadline = np.array([int(r["missed_deadline"]) for r in rows])

        lines = [
            f"Realtime eval summary  |  n_steps={n}  |  abort_reason={abort_reason}  |  budget={self.budget_hz:.0f}Hz ({self.budget_ms:.2f}ms)",
            "",
            _format_stage_stats("process_obs_ms   ", t_obs),
            _format_stage_stats("predict_action_ms", t_infer, budget_ms=self.budget_ms),
            _format_stage_stats("solve_tau_ms     ", t_tau),
            _format_stage_stats("ctrl_arm_ms      ", t_ctrl),
            _format_stage_stats("loop_total_ms    ", t_loop, budget_ms=self.budget_ms),
            _format_stage_stats("sleep_ms         ", t_sleep),
            "",
            f"chunk_boundaries: {int(chunk_boundary.sum())} / {n} "
            f"({100.0 * chunk_boundary.mean():.1f}%)",
            f"missed_deadlines: {int(missed_deadline.sum())} / {n} "
            f"({100.0 * missed_deadline.mean():.1f}%)",
        ]
        if chunk_boundary.any() and (~chunk_boundary.astype(bool)).any():
            cb = chunk_boundary.astype(bool)
            lines.append("")
            lines.append(
                f"predict_action_ms at chunk_boundary=True : "
                f"mean={t_infer[cb].mean():.2f} max={t_infer[cb].max():.2f} (n={cb.sum()})"
            )
            lines.append(
                f"predict_action_ms at chunk_boundary=False: "
                f"mean={t_infer[~cb].mean():.2f} max={t_infer[~cb].max():.2f} (n={(~cb).sum()})"
            )
        summary = "\n".join(lines)
        (self.out_dir / "summary.txt").write_text(summary + "\n")
        for line in lines:
            logger_mp.info(line)


def extract_observation(step: dict):
    observation = {}

    for key, value in step.items():
        if key.startswith("observation.images."):
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in [1, 3]:
                value = np.transpose(value, (2, 0, 1))
            observation[key] = value

        elif key == "observation.state":
            observation[key] = value

    return observation


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    use_dataset: bool | None = False,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if not use_dataset:
                # Skip non-tensor observations (like task strings)
                if not hasattr(observation[name], "unsqueeze"):
                    continue
                if "images" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()

            observation[name] = observation[name].unsqueeze(0).to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)
        action = postprocessor(action)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def predict_chunk(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    use_dataset: bool | None = False,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        for name in observation:
            if not use_dataset:
                if not hasattr(observation[name], "unsqueeze"):
                    continue
                if "images" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()

            observation[name] = observation[name].unsqueeze(0).to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        observation = preprocessor(observation)

        # Returns the full chunk (B, chunk_size, action_dim_padded); does not touch the policy's internal action queue.
        action = policy.predict_action_chunk(observation)

        # Reshape to (B*chunk_size, D_padded) before postprocessing so the postprocessor sees a 2D
        # batch instead of a 3D chunk. GR00T's postprocessor has a `if dim == 3: action = action[:, -1, :]`
        # branch (processor_groot.py:589-591) that collapses the chunk to its last timestep — correct for
        # select_action (one popped action at a time) but wrong for us, since we want every chunk position.
        # Flattening to 2D avoids that branch and lets the postprocessor apply un-padding + un-normalization
        # batch-element-wise on each chunk step. Reshape back to (B, T, action_dim_real) after.
        B, T, _ = action.shape
        action = action.reshape(B * T, -1)
        action = postprocessor(action)
        action = action.reshape(B, T, -1)

        # Squeeze only the batch dim — keep (chunk_size, action_dim).
        action = action.squeeze(0)

        action = action.to("cpu")

    return action


def reset_policy(policy: PreTrainedPolicy):
    policy.reset()


def cleanup_resources(image_info: dict[str, Any]):
    """Safely close and unlink shared memory resources."""
    logger_mp.info("Cleaning up shared memory resources.")
    for shm in image_info["shm_resources"]:
        if shm:
            shm.close()
            shm.unlink()


def to_list(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().ravel().tolist()
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def to_scalar(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return float(x.detach().cpu().ravel()[0].item())
    if isinstance(x, np.ndarray):
        return float(x.ravel()[0])
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


@dataclass
class EvalRealConfig:
    repo_id: str
    policy: PreTrainedConfig | None = None

    root: str = ""
    episodes: int = 0
    frequency: float = 30.0

    # Network / image client
    image_host: str = "192.168.123.164"  # IP of the robot's image_server (ZMQ host)

    # Basic control parameters
    arm: str = "G1_29"  # G1_29, G1_23
    ee: str = "dex1"  # dex3, dex1, inspire1, brainco

    # Mode flags
    motion: bool = False
    headless: bool = False
    visualization: bool = False
    cam_check_only: bool = False  # If True, pull one observation, save the four camera frames to ./cam_dryrun/, log shapes, and exit without sending any motion command.
    # Stage gates for eval_g1.py — both default off so an accidental launch never commands motion.
    soft_start: bool = False  # Stage 1: linearly interpolate the arms from the current pose to init_arm_pose before the policy loop.
    run_policy: bool = False  # Stage 2: run the inference loop (sends actions to arms + EE). Assumes the robot is already at init_arm_pose unless soft_start is also set.
    max_steps: int = 0  # Cap on policy loop iterations; 0 means unlimited (original while-True behavior).
    save_rrd: bool = False  # If True, also record the Rerun session to <realtime_eval/<ts>/session.rrd so the run can be replayed later (rerun <path>.rrd). Independent of `visualization` — image/scalar logging is enabled by either flag.
    send_real_robot: bool = False  # Legacy; eval_g1.py no longer references this. Kept for dataclass-import compatibility.
    use_dataset: bool = False

    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@dataclass
class OfflineEvalConfig:
    repo_id: str
    episodes: list[int]
    policy: PreTrainedConfig | None = None

    root: str = ""
    visualization: bool = False
    output_dir: str | None = None
    seed: int | None = None
    # Chunk-fan plot (2_chunk_fan.png) draws one predicted chunk every `fan_stride` frames.
    # None => auto (max(1, T // 120)): legible on long episodes, every-frame on short ones.
    # Set an explicit int to override; fan_stride=1 forces a fan from every frame.
    fan_stride: int | None = None
    # GR00T-only: number of flow-matching denoising steps at inference (base GR00T-N1.5 default = 4).
    # None leaves the model's built-in count untouched; an int (e.g. 8) overrides it at eval time by
    # setting num_inference_timesteps on the action head. Ignored for ACT (no diffusion head).
    num_inference_timesteps: int | None = None
    # Free-form experiment label appended to the auto-generated output-dir variant tag (see
    # eval_g1_dataset._variant_tag). Use it to distinguish runs the auto-tag can't capture (e.g. an
    # ablation). Only applied to the default checkpoint-adjacent path, not to an explicit --output_dir.
    tag: str = ""

    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
