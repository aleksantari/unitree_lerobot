"""Analyze a realtime_eval/<ts>/ directory produced by eval_g1.py.

Loads timing.npz, prints a structured summary, and writes four PNG plots
into the same dir:

    analysis_latency_timeline.png    per-step t_infer / t_loop with chunk_boundary markers + budget line
    analysis_latency_histogram.png   t_infer_ms histograms split by chunk_boundary
    analysis_per_stage_breakdown.png stacked area of obs/infer/tau/ctrl/sleep over time
    analysis_action_trajectory.png   16 commanded action lanes over time, separating arm vs gripper

Usage (module form, from repo root):
    python -m unitree_lerobot.eval_robot.analyze_realtime_eval \\
        --run_dir outputs/train/<...>/realtime_eval/<TS>
    # or auto-pick the most-recently-modified realtime_eval/<ts>/
    python -m unitree_lerobot.eval_robot.analyze_realtime_eval

Tunables:
    --skip_warmup   exclude step 0 from the stats and plots (default true; the
                    first inference always includes CUDA graph compile + kernel
                    autotune and is not representative of steady-state behavior)
    --no_plots      skip figure rendering; print summary only
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; we save PNGs, don't show
import matplotlib.pyplot as plt
import numpy as np


_DEFAULT_BUDGET_HZ = 30.0


def find_latest_run_dir(root: Path) -> Path:
    """Find the most recently modified realtime_eval/<ts>/ under outputs/train/*."""
    candidates = list(root.glob("outputs/train/*/*/realtime_eval/*/"))
    if not candidates:
        raise FileNotFoundError(f"No realtime_eval/<ts>/ dirs found under {root}/outputs/train/")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_run(run_dir: Path) -> dict:
    """Load the per-step arrays from timing.npz plus the budget from config.json.

    `budget_hz` is read from <run_dir>/config.json's `frequency` field (written by
    eval_g1.py's TimingLog.finalize via asdict(cfg)). Falls back to 30 Hz if the
    config is missing (old runs predating this analyzer's budget plumbing).
    """
    npz_path = run_dir / "timing.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing timing.npz in {run_dir}")
    data = dict(np.load(npz_path))
    # abort_reason is a 0-d string array; convert to scalar.
    data["abort_reason"] = str(data["abort_reason"])
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            data["budget_hz"] = float(cfg.get("frequency", _DEFAULT_BUDGET_HZ))
        except Exception:
            data["budget_hz"] = _DEFAULT_BUDGET_HZ
    else:
        data["budget_hz"] = _DEFAULT_BUDGET_HZ
    data["budget_ms"] = 1000.0 / data["budget_hz"]
    return data


def compute_arm_deltas(data: dict, skip_warmup: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step max abs action delta across arm joints, aligned with chunk_boundary.

    delta[i] = max |actions[i+1, :14] - actions[i, :14]|  (max across the 14 arm dims).
    chunk_boundary at index i+1 aligns with delta[i] -- the action AT step i+1 was the
    first sample of a new chunk, so the delta from i to i+1 is the boundary jump.

    Returns (deltas_rad, chunk_boundary_aligned, step_x_axis).
    """
    actions = data["actions"]
    deltas = np.abs(np.diff(actions[:, :14], axis=0)).max(axis=1)  # (N-1,)
    cb_aligned = data["chunk_boundary"][1:].astype(bool)            # (N-1,) aligned with deltas
    # Step x-axis: delta[i] = action[i+1]-action[i], so the "delta at step k" plots at x=k+1.
    steps = data["step"][1:]
    if skip_warmup:
        # Also drop the first delta (the 0→1 transition includes the warm-up step at index 0).
        deltas = deltas[1:]
        cb_aligned = cb_aligned[1:]
        steps = steps[1:]
    return deltas, cb_aligned, steps


def compute_tracking_error(data: dict, skip_warmup: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step PD tracking error (residual after one frame of PD response).

    tracking_error[i] = max |actions[i-1, :14] - states[i, :14]|  (max across arm joints).
    Index alignment:
      - states[i] was sampled at the start of loop iteration i, AFTER actions[i-1] was sent
        the previous iteration and the PD had ~33 ms to drive toward it.
      - So small tracking_error[i] means the PD caught up to action[i-1] in time; large
        tracking_error[i] means the PD fell behind.
    chunk_boundary[i-1] (aligned with the action that produced this error) tells us whether
    the command the PD was just trying to track was a chunk-boundary jump.

    Returns (errors_rad, chunk_boundary_aligned, step_x_axis).
    """
    actions = data["actions"]
    states = data["states"]
    # |action[i-1] - state[i]| for i = 1..N-1; arm joints only.
    errors = np.abs(actions[:-1, :14] - states[1:, :14]).max(axis=1)   # (N-1,)
    # The error at step i was produced by command at step i-1. So filter by chunk_boundary[i-1].
    cb_aligned = data["chunk_boundary"][:-1].astype(bool)              # (N-1,)
    steps = data["step"][1:]                                           # x-axis: step we OBSERVED the residual
    if skip_warmup:
        # The warm-up step is index 0; its command lands at step 1's observation. Drop it.
        errors = errors[1:]
        cb_aligned = cb_aligned[1:]
        steps = steps[1:]
    return errors, cb_aligned, steps


def print_summary(data: dict, skip_warmup: bool) -> None:
    n = len(data["step"])
    cb = data["chunk_boundary"].astype(bool)
    md = data["missed_deadline"].astype(bool)
    t_infer = data["t_infer_ms"]
    t_loop = data["t_loop_ms"]

    mask = slice(1, None) if skip_warmup else slice(None)
    t_infer_clean = t_infer[mask]
    cb_clean = cb[mask]

    print(f"\n=== Realtime eval analysis ===")
    print(f"abort_reason : {data['abort_reason']}")
    print(f"n_steps      : {n}")
    print(f"warm-up step : t_infer={t_infer[0]:.1f}ms (excluded from stats below)" if skip_warmup else "")
    print()
    print(f"Loop timing (steady state, n={t_infer_clean.size}):")
    print(f"  t_loop_ms    mean={t_loop[mask].mean():.2f}  max={t_loop[mask].max():.2f}  p95={np.percentile(t_loop[mask], 95):.2f}")
    print(f"  t_infer_ms   mean={t_infer_clean.mean():.2f}  max={t_infer_clean.max():.2f}  p95={np.percentile(t_infer_clean, 95):.2f}")
    print()
    print(f"Chunk-boundary split (the hypothesis test):")
    print(f"  boundary=True  n={cb_clean.sum():>4}  mean t_infer={t_infer_clean[cb_clean].mean():.2f} ms  max={t_infer_clean[cb_clean].max():.2f} ms")
    print(f"  boundary=False n={(~cb_clean).sum():>4}  mean t_infer={t_infer_clean[~cb_clean].mean():.2f} ms  max={t_infer_clean[~cb_clean].max():.2f} ms")
    print(f"  ratio (slow/fast) = {t_infer_clean[cb_clean].mean() / max(t_infer_clean[~cb_clean].mean(), 1e-6):.1f}x")
    print()
    print(f"Deadline misses:")
    md_clean = md[mask]
    print(f"  total {md_clean.sum()} / {md_clean.size} ({100*md_clean.mean():.1f}%)")
    print(f"  overlap with chunk_boundary=True: {int((md_clean & cb_clean).sum())} / {int(cb_clean.sum())} = {100 * (md_clean & cb_clean).sum() / max(cb_clean.sum(), 1):.0f}%")
    print(f"  overlap with chunk_boundary=False: {int((md_clean & ~cb_clean).sum())} / {int((~cb_clean).sum())} = {100 * (md_clean & ~cb_clean).sum() / max((~cb_clean).sum(), 1):.0f}%")

    # Action-discontinuity stats: does the per-step jump magnitude differ at chunk boundaries?
    # If the boundary mean is materially larger, that's the chunk-boundary discontinuity
    # mechanism documented in ACT / Diffusion Policy. Larger ratio → bigger PD torque spikes.
    arm_deltas, cb_aligned, _ = compute_arm_deltas(data, skip_warmup=skip_warmup)
    cb_d = arm_deltas[cb_aligned]
    nocb_d = arm_deltas[~cb_aligned]
    print()
    print(f"Action discontinuity at chunk boundaries (arm joints, max |Δaction| per step):")
    print(f"  AT boundary  (n={cb_d.size:>4}): mean={np.degrees(cb_d.mean()):.3f}° "
          f"max={np.degrees(cb_d.max()):.3f}° p95={np.degrees(np.percentile(cb_d, 95)):.3f}°")
    print(f"  WITHIN chunk (n={nocb_d.size:>4}): mean={np.degrees(nocb_d.mean()):.3f}° "
          f"max={np.degrees(nocb_d.max()):.3f}° p95={np.degrees(np.percentile(nocb_d, 95)):.3f}°")
    print(f"  ratio of means: {cb_d.mean() / max(nocb_d.mean(), 1e-9):.2f}x")

    # PD tracking error: how much did the joint miss by, ONE FRAME after the command was sent?
    # tracking_error[i] = max |action[i-1] - state[i]|. Small → PD caught up; large at boundaries → PD struggling.
    # Important: this is the residual AFTER the PD had ~33ms to chase the command. If it's still large
    # at chunk boundaries, it means the PD couldn't move the joint fast enough during that window
    # (i.e., the boundary jump exceeded the PD's effective bandwidth-times-window product).
    track_err, cb_track, _ = compute_tracking_error(data, skip_warmup=skip_warmup)
    cb_e = track_err[cb_track]
    nocb_e = track_err[~cb_track]
    print()
    print(f"PD tracking error (residual after one frame of PD chase, arm joints):")
    print(f"  AT boundary cmd  (n={cb_e.size:>4}): mean={np.degrees(cb_e.mean()):.3f}° "
          f"max={np.degrees(cb_e.max()):.3f}° p95={np.degrees(np.percentile(cb_e, 95)):.3f}°")
    print(f"  WITHIN-chunk cmd (n={nocb_e.size:>4}): mean={np.degrees(nocb_e.mean()):.3f}° "
          f"max={np.degrees(nocb_e.max()):.3f}° p95={np.degrees(np.percentile(nocb_e, 95)):.3f}°")
    print(f"  ratio of means: {cb_e.mean() / max(nocb_e.mean(), 1e-9):.2f}x")
    print(f"  (compare against the chunk-boundary action delta of {np.degrees(cb_d.mean()):.2f}° -- if "
          f"tracking error is much smaller than the boundary delta, the PD caught up cleanly)")


def plot_latency_timeline(data: dict, out: Path, skip_warmup: bool) -> None:
    mask = slice(1, None) if skip_warmup else slice(None)
    step = data["step"][mask]
    t_infer = data["t_infer_ms"][mask]
    t_loop = data["t_loop_ms"][mask]
    cb = data["chunk_boundary"][mask].astype(bool)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(step, t_loop, color="lightgrey", lw=0.7, label="t_loop_ms")
    ax.plot(step, t_infer, color="steelblue", lw=0.8, label="t_infer_ms")
    ax.scatter(step[cb], t_infer[cb], color="crimson", s=20, zorder=5, label="chunk_boundary=True")
    budget_hz = data["budget_hz"]
    budget_ms = data["budget_ms"]
    ax.axhline(budget_ms, color="orange", ls="--", lw=1.0, label=f"{budget_hz:.0f}Hz budget ({budget_ms:.1f}ms)")
    ax.set_xlabel("step")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Per-step inference & loop latency (chunk boundaries highlighted)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_latency_histogram(data: dict, out: Path, skip_warmup: bool) -> None:
    mask = slice(1, None) if skip_warmup else slice(None)
    t_infer = data["t_infer_ms"][mask]
    cb = data["chunk_boundary"][mask].astype(bool)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5))
    budget_ms = data["budget_ms"]
    bins = np.linspace(0, max(t_infer.max(), budget_ms * 1.1), 80)

    for ax in (ax_lin, ax_log):
        ax.hist(t_infer[~cb], bins=bins, alpha=0.6, color="steelblue",
                label=f"cached pop (n={(~cb).sum()})")
        ax.hist(t_infer[cb], bins=bins, alpha=0.6, color="crimson",
                label=f"chunk boundary (n={cb.sum()})")
        ax.axvline(budget_ms, color="orange", ls="--", lw=1.0, label=f"budget {budget_ms:.1f}ms")
        ax.set_xlabel("t_infer_ms")
        ax.legend()
        ax.grid(alpha=0.3)

    ax_lin.set_ylabel("count (linear)")
    ax_lin.set_title("predict_action latency histogram")
    ax_log.set_yscale("log")
    ax_log.set_ylabel("count (log)")
    ax_log.set_title("(log scale to see the slow tail)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_per_stage_breakdown(data: dict, out: Path, skip_warmup: bool) -> None:
    mask = slice(1, None) if skip_warmup else slice(None)
    step = data["step"][mask]
    layers = {
        "obs":   data["t_obs_ms"][mask],
        "infer": data["t_infer_ms"][mask],
        "tau":   data["t_tau_ms"][mask],
        "ctrl":  data["t_ctrl_ms"][mask],
        "sleep": data["t_sleep_ms"][mask],
    }
    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros_like(step, dtype=float)
    colors = {"obs": "tab:gray", "infer": "tab:blue", "tau": "tab:olive", "ctrl": "tab:purple", "sleep": "tab:green"}
    for name, vals in layers.items():
        ax.fill_between(step, bottom, bottom + vals, label=name, color=colors[name], alpha=0.8, lw=0)
        bottom = bottom + vals
    budget_ms = data["budget_ms"]
    ax.axhline(budget_ms, color="orange", ls="--", lw=1.0, label=f"budget {budget_ms:.1f}ms")
    ax.set_xlabel("step")
    ax.set_ylabel("time per loop iter (ms)")
    ax.set_title("Per-stage latency stacked over time (obs + infer + tau + ctrl + sleep)")
    ax.legend(loc="upper right", ncol=6, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_action_delta(data: dict, out: Path, skip_warmup: bool) -> None:
    """Per-step max abs action delta over arm joints, with chunk boundaries highlighted.

    This is the action-discontinuity equivalent of the latency-timeline plot. The two
    distributions side-by-side in the histogram subplot make the ACT-paper / Diffusion-
    Policy chunk-boundary jump pattern visible directly in this run's commanded actions.
    """
    deltas, cb_aligned, steps = compute_arm_deltas(data, skip_warmup)
    deltas_deg = np.degrees(deltas)

    fig, (ax_ts, ax_hist) = plt.subplots(2, 1, figsize=(14, 8))

    within_mean = deltas_deg[~cb_aligned].mean()
    boundary_mean = deltas_deg[cb_aligned].mean()
    ax_ts.plot(steps, deltas_deg, color="steelblue", lw=0.7, label="max |Δaction| arm joints")
    ax_ts.scatter(steps[cb_aligned], deltas_deg[cb_aligned], color="crimson", s=20, zorder=5,
                  label=f"chunk boundary (n={cb_aligned.sum()})")
    ax_ts.axhline(within_mean, color="gray", ls=":", lw=1.0,
                  label=f"within-chunk mean ({within_mean:.2f}°)")
    ax_ts.axhline(boundary_mean, color="crimson", ls=":", lw=1.0,
                  label=f"boundary mean ({boundary_mean:.2f}°)")
    ax_ts.set_xlabel("step")
    ax_ts.set_ylabel("max |Δaction| across arm joints (deg)")
    ax_ts.set_title("Per-step action discontinuity — chunk boundaries highlighted")
    ax_ts.legend(loc="upper right")
    ax_ts.grid(alpha=0.3)

    bins = np.linspace(0, deltas_deg.max() * 1.05, 60)
    ax_hist.hist(deltas_deg[~cb_aligned], bins=bins, alpha=0.6, color="steelblue",
                 label=f"within chunk (n={(~cb_aligned).sum()})")
    ax_hist.hist(deltas_deg[cb_aligned], bins=bins, alpha=0.6, color="crimson",
                 label=f"AT chunk boundary (n={cb_aligned.sum()})")
    ratio = boundary_mean / max(within_mean, 1e-9)
    ax_hist.set_xlabel("max |Δaction| across arm joints (deg)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"Jump-magnitude distribution — boundary mean {ratio:.2f}× within-chunk mean")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_tracking_error(data: dict, out: Path, skip_warmup: bool) -> None:
    """PD tracking error per step — does the controller catch up to each commanded setpoint?

    Three subplots:
      1. Timeline: per-step max |action[t-1] - state[t]| over arm joints, chunk boundaries red.
      2. Histogram split by chunk_boundary -- the chunk-boundary distribution is where the
         "PD struggling at boundaries" signal lives. If the boundary distribution is shifted right
         compared to within-chunk, the PD failed to catch up at boundaries.
      3. Boundary jump magnitude vs tracking error at the SAME step (scatter). If the PD always
         caught up, the cloud sits near zero regardless of jump size. If the PD struggled, the
         cloud climbs linearly with jump size (residual proportional to commanded step).
    """
    errors, cb_aligned, steps = compute_tracking_error(data, skip_warmup)
    errors_deg = np.degrees(errors)
    # Action delta at the SAME step alignment so we can scatter them against each other.
    deltas, cb_delta, _ = compute_arm_deltas(data, skip_warmup)
    # Align: deltas[i] is action[i+1]-action[i] (cb at index i+1).
    # tracking_error[i] is residual after action[i-1] (cb at index i-1).
    # Both are over the same set of "actions that landed at step i", but offset by one index.
    # For a direct per-step comparison we use the action that was COMMANDED last frame -- so the
    # delta we want is the one that produced the boundary, i.e., shift deltas by one to align.
    # Simpler: just plot deltas[t] (jump magnitude) on x, errors[t+1] (next-frame residual) on y.
    # Skip the last delta and the first error to align them.
    if errors.size > 1 and deltas.size > 1:
        scatter_delta = np.degrees(deltas[:-1])
        scatter_err = errors_deg[1:]
        scatter_cb = cb_delta[:-1]
    else:
        scatter_delta = np.degrees(deltas)
        scatter_err = errors_deg
        scatter_cb = cb_delta

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
    ax_ts = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])
    ax_scat = fig.add_subplot(gs[2])

    within_mean = errors_deg[~cb_aligned].mean()
    boundary_mean = errors_deg[cb_aligned].mean()
    ax_ts.plot(steps, errors_deg, color="steelblue", lw=0.7, label="max |action[t-1] - state[t]| (arm)")
    ax_ts.scatter(steps[cb_aligned], errors_deg[cb_aligned], color="crimson", s=20, zorder=5,
                  label=f"boundary-cmd residual (n={cb_aligned.sum()})")
    ax_ts.axhline(within_mean, color="gray", ls=":", lw=1.0,
                  label=f"within-chunk mean ({within_mean:.2f}°)")
    ax_ts.axhline(boundary_mean, color="crimson", ls=":", lw=1.0,
                  label=f"boundary-cmd mean ({boundary_mean:.2f}°)")
    ax_ts.set_xlabel("step")
    ax_ts.set_ylabel("PD tracking error (deg)")
    ax_ts.set_title("PD tracking error — residual after one frame of PD chase")
    ax_ts.legend(loc="upper right", fontsize=9)
    ax_ts.grid(alpha=0.3)

    bins = np.linspace(0, errors_deg.max() * 1.05, 60)
    ax_hist.hist(errors_deg[~cb_aligned], bins=bins, alpha=0.6, color="steelblue",
                 label=f"within-chunk cmd (n={(~cb_aligned).sum()})")
    ax_hist.hist(errors_deg[cb_aligned], bins=bins, alpha=0.6, color="crimson",
                 label=f"boundary cmd (n={cb_aligned.sum()})")
    ax_hist.set_xlabel("PD tracking error (deg)")
    ax_hist.set_ylabel("count")
    ratio = boundary_mean / max(within_mean, 1e-9)
    ax_hist.set_title(f"Tracking-error distribution — boundary mean {ratio:.2f}× within-chunk mean")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    # Scatter: commanded jump size on x, resulting tracking error on y.
    # If PD always catches up cleanly, y stays near zero regardless of x.
    # If PD struggles, the cloud climbs with x (large jumps → larger residuals).
    # The y=x line is the worst case: PD didn't move at all that frame.
    ax_scat.scatter(scatter_delta[~scatter_cb], scatter_err[~scatter_cb], s=8,
                    alpha=0.4, color="steelblue", label="within-chunk cmd")
    ax_scat.scatter(scatter_delta[scatter_cb], scatter_err[scatter_cb], s=20,
                    alpha=0.8, color="crimson", label="boundary cmd")
    xlim_max = max(scatter_delta.max(), scatter_err.max()) * 1.1
    xs = np.linspace(0, xlim_max, 100)
    ax_scat.plot(xs, xs, color="black", ls="--", lw=0.7, alpha=0.5, label="y = x (PD didn't move at all)")
    ax_scat.set_xlabel("commanded jump magnitude (deg) — max |Δaction| arm joints")
    ax_scat.set_ylabel("tracking error next frame (deg)")
    ax_scat.set_title("Jump magnitude vs resulting tracking error  (cloud near 0 = PD caught up; cloud near y=x = PD failed)")
    ax_scat.legend(loc="upper left", fontsize=9)
    ax_scat.grid(alpha=0.3)
    ax_scat.set_xlim(0, xlim_max)
    ax_scat.set_ylim(0, xlim_max)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_action_trajectory(data: dict, out: Path, skip_warmup: bool) -> None:
    mask = slice(1, None) if skip_warmup else slice(None)
    step = data["step"][mask]
    actions = data["actions"][mask]  # (n, 16): 14 arm joints + 2 grippers
    states = data["states"][mask]    # (n, 16): same layout

    fig, (ax_arm, ax_gripper) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    # Left/right arm: 7 joints each (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
    arm_left_dims = list(range(0, 7))
    arm_right_dims = list(range(7, 14))
    gripper_dims = [14, 15]

    cmap = plt.cm.viridis
    for i, d in enumerate(arm_left_dims + arm_right_dims):
        ax_arm.plot(step, actions[:, d], color=cmap(i / 14), lw=0.8, label=f"action[{d}]")
    ax_arm.set_ylabel("commanded action (rad)")
    ax_arm.set_title("Arm joints — commanded actions over time (14 lanes; left=cool colors, right=warm)")
    ax_arm.grid(alpha=0.3)

    for d, color in zip(gripper_dims, ("tab:cyan", "tab:orange")):
        ax_gripper.plot(step, actions[:, d], color=color, lw=1.0, label=f"action[{d}] (gripper)")
        ax_gripper.plot(step, states[:, d], color=color, ls=":", lw=0.7, alpha=0.6, label=f"state[{d}] (gripper)")
    ax_gripper.set_xlabel("step")
    ax_gripper.set_ylabel("gripper position")
    ax_gripper.set_title("Gripper commanded vs observed")
    ax_gripper.legend(loc="upper right", fontsize=9)
    ax_gripper.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=None,
                    help="realtime_eval/<ts>/ dir; default = most recent under outputs/train/*")
    ap.add_argument("--skip_warmup", action="store_true", default=True,
                    help="exclude step 0 (CUDA warm-up) from stats and plots (default true)")
    ap.add_argument("--include_warmup", dest="skip_warmup", action="store_false",
                    help="include step 0 in everything (the 1000ms warm-up will dominate plots)")
    ap.add_argument("--no_plots", action="store_true", help="skip PNG generation, print summary only")
    args = ap.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(repo_root)
    run_dir = run_dir.resolve()
    print(f"Run dir: {run_dir}")

    data = load_run(run_dir)
    print_summary(data, skip_warmup=args.skip_warmup)

    if args.no_plots:
        return

    print("\nPlots:")
    plot_latency_timeline(data, run_dir / "analysis_latency_timeline.png", args.skip_warmup)
    plot_latency_histogram(data, run_dir / "analysis_latency_histogram.png", args.skip_warmup)
    plot_per_stage_breakdown(data, run_dir / "analysis_per_stage_breakdown.png", args.skip_warmup)
    plot_action_trajectory(data, run_dir / "analysis_action_trajectory.png", args.skip_warmup)
    plot_action_delta(data, run_dir / "analysis_action_delta.png", args.skip_warmup)
    plot_tracking_error(data, run_dir / "analysis_tracking_error.png", args.skip_warmup)
    print(f"\nAll plots in {run_dir}")


if __name__ == "__main__":
    main()
