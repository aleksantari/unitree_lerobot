"""Headless smoke test for the realtime-eval timing log + Rerun visualization.

Runs N synthetic policy-loop iterations with fake timings (some chunk-boundary,
some cached pops, some missed deadlines). Exercises the exact `TimingLog` +
`rr.set_time` + `rr.Scalars` + `rr.Image` call sites that eval_g1.py uses during
its policy loop -- but without the robot, the policy, or the image_server, so
you can validate visualizer/logging changes safely off-hardware.

Validates:
  - TimingLog opens timing.csv at startup, writes one line per append (line-buffered)
  - TimingLog.finalize writes timing.npz + summary.txt + config.json
  - summary.txt contains the chunk_boundary=True/False split (the diagnostic that
    confirms or refutes the GR00T chunk-boundary jitter hypothesis)
  - The Rerun API calls don't raise on the installed rerun version
By default also spawns the Rerun viewer so you can eyeball the layout (4 synthetic
camera tiles + all `timings/*` and `events/*` scalar streams).

Usage (from any cwd, no robot needed):
    bash -ic 'use_conda unitree-lerobot-groot && python test/test_eval_visualizer.py'

Tunables via env var:
    VIZ_TEST_STEPS=300       number of synthetic loop iterations  (default 100)
    VIZ_TEST_NO_VIEWER=1     skip rr.spawn() so it runs headless    (default off)
    VIZ_TEST_OUT=<path>      output dir; default = a fresh /tmp/viz_test_<rand>/
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import rerun as rr

# Make the repo root importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unitree_lerobot.eval_robot.utils.utils import TimingLog, _REALTIME_BUDGET_MS

N_STEPS = int(os.environ.get("VIZ_TEST_STEPS", 100))
SPAWN_VIEWER = os.environ.get("VIZ_TEST_NO_VIEWER", "") != "1"
CHUNK_SIZE = 16  # matches GR00T-N1.5's n_action_steps
ACTION_DIM = 16  # G1_29 + dex1 (14 arm + 2 grippers)
STATE_DIM = 16
IMAGE_SHAPE = (480, 640, 3)  # H, W, C -- matches the resized policy input


def main() -> None:
    out_dir_env = os.environ.get("VIZ_TEST_OUT")
    out_dir = Path(out_dir_env) if out_dir_env else Path(tempfile.mkdtemp(prefix="viz_test_"))
    print(f"Test artifacts -> {out_dir}")

    if SPAWN_VIEWER:
        rr.init("viz_test")
        rr.spawn(memory_limit="200MB")
        print("Rerun viewer spawned. Drag timings/* and events/* into the grid to inspect.")
    else:
        rr.init("viz_test", spawn=False)
        print("VIZ_TEST_NO_VIEWER=1: viewer skipped, only disk artifacts will be produced.")

    timing = TimingLog(out_dir=out_dir)
    rng = np.random.default_rng(42)
    cams = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist")

    for idx in range(N_STEPS):
        # Chunk-boundary frames: simulate ~120 ms inference (full forward pass).
        # Cached-pop frames: ~0.5 ms.
        chunk_boundary = (idx % CHUNK_SIZE) == 0
        queue_len_before = 0 if chunk_boundary else CHUNK_SIZE - (idx % CHUNK_SIZE)
        t_infer_ms = (120.0 + rng.normal(0, 5)) if chunk_boundary else (0.5 + rng.normal(0, 0.1))
        t_obs_ms = 2.0 + rng.normal(0, 0.3)
        t_tau_ms = 0.5 + rng.normal(0, 0.05)
        t_ctrl_ms = 1.0 + rng.normal(0, 0.1)
        t_loop_ms = t_obs_ms + t_infer_ms + t_tau_ms + t_ctrl_ms
        missed_deadline = t_loop_ms > _REALTIME_BUDGET_MS
        t_sleep_ms = max(0.0, _REALTIME_BUDGET_MS - t_loop_ms)
        arm_delta_max = float(rng.uniform(0.001, 0.02))

        timing.append(
            step=idx,
            t_obs_ms=t_obs_ms,
            t_infer_ms=t_infer_ms,
            t_tau_ms=t_tau_ms,
            t_ctrl_ms=t_ctrl_ms,
            t_loop_ms=t_loop_ms,
            t_sleep_ms=t_sleep_ms,
            chunk_boundary=int(chunk_boundary),
            queue_len_before=queue_len_before,
            missed_deadline=int(missed_deadline),
            arm_delta_max=arm_delta_max,
        )
        timing.add_step_data(
            action=rng.standard_normal(ACTION_DIM).astype(np.float32),
            state=rng.standard_normal(STATE_DIM).astype(np.float32),
        )

        # --- Same Rerun calls eval_g1.py makes inside the policy loop ---
        rr.set_time("frame", sequence=idx)
        rr.log("timings/process_obs_ms",    rr.Scalars(t_obs_ms))
        rr.log("timings/predict_action_ms", rr.Scalars(t_infer_ms))
        rr.log("timings/solve_tau_ms",      rr.Scalars(t_tau_ms))
        rr.log("timings/ctrl_arm_ms",       rr.Scalars(t_ctrl_ms))
        rr.log("timings/loop_total_ms",     rr.Scalars(t_loop_ms))
        rr.log("timings/budget_ms",         rr.Scalars(_REALTIME_BUDGET_MS))
        rr.log("events/chunk_boundary",     rr.Scalars(1 if chunk_boundary else 0))
        rr.log("events/missed_deadline",    rr.Scalars(1 if missed_deadline else 0))
        rr.log("events/queue_len_before",   rr.Scalars(queue_len_before))
        # Synthetic camera tiles -- pure noise, just confirms rr.Image works.
        for cam in cams:
            rr.log(
                f"images/{cam}",
                rr.Image(rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8)),
            )

    timing.finalize(
        abort_reason="completed",
        cfg_snapshot={"test": True, "n_steps": N_STEPS, "chunk_size": CHUNK_SIZE},
    )

    # --- Disk-artifact assertions ---
    expected = ["timing.csv", "timing.npz", "summary.txt", "config.json"]
    print("\nArtifacts:")
    for name in expected:
        path = out_dir / name
        assert path.exists(), f"FAIL: missing {path}"
        print(f"  {name}: {path.stat().st_size} bytes")

    # CSV: header + N_STEPS data lines.
    csv_lines = (out_dir / "timing.csv").read_text().splitlines()
    assert len(csv_lines) == N_STEPS + 1, (
        f"FAIL: timing.csv has {len(csv_lines)} lines, expected {N_STEPS + 1}"
    )

    # npz: required keys + shapes.
    data = np.load(out_dir / "timing.npz")
    required_keys = (
        "step", "t_obs_ms", "t_infer_ms", "t_tau_ms", "t_ctrl_ms",
        "t_loop_ms", "t_sleep_ms",
        "chunk_boundary", "queue_len_before", "missed_deadline", "arm_delta_max",
        "actions", "states", "abort_reason",
    )
    for k in required_keys:
        assert k in data.files, f"FAIL: timing.npz missing key '{k}'"
    assert data["actions"].shape == (N_STEPS, ACTION_DIM), (
        f"FAIL: actions shape {data['actions'].shape}, expected ({N_STEPS}, {ACTION_DIM})"
    )
    assert data["states"].shape == (N_STEPS, STATE_DIM), (
        f"FAIL: states shape {data['states'].shape}, expected ({N_STEPS}, {STATE_DIM})"
    )
    expected_chunk_count = (N_STEPS + CHUNK_SIZE - 1) // CHUNK_SIZE
    assert int(data["chunk_boundary"].sum()) == expected_chunk_count, (
        f"FAIL: chunk_boundary sum={int(data['chunk_boundary'].sum())}, expected {expected_chunk_count}"
    )
    assert str(data["abort_reason"]) == "completed", (
        f"FAIL: abort_reason='{data['abort_reason']}', expected 'completed'"
    )

    # summary.txt: must include the chunk-boundary split (the actual diagnostic).
    summary = (out_dir / "summary.txt").read_text()
    assert "chunk_boundary=True" in summary, "FAIL: summary missing chunk_boundary=True row"
    assert "chunk_boundary=False" in summary, "FAIL: summary missing chunk_boundary=False row"

    print("\n--- summary.txt ---")
    print(summary)
    print(f"PASS. Artifacts at: {out_dir}")
    if not out_dir_env:
        print(f"(Temp dir; remove with:  rm -rf {out_dir})")


if __name__ == "__main__":
    main()
