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
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; we save PNGs, don't show
import matplotlib.pyplot as plt
import numpy as np


_BUDGET_HZ = 30.0
_BUDGET_MS = 1000.0 / _BUDGET_HZ


def find_latest_run_dir(root: Path) -> Path:
    """Find the most recently modified realtime_eval/<ts>/ under outputs/train/*."""
    candidates = list(root.glob("outputs/train/*/*/realtime_eval/*/"))
    if not candidates:
        raise FileNotFoundError(f"No realtime_eval/<ts>/ dirs found under {root}/outputs/train/")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_run(run_dir: Path) -> dict:
    """Load the per-step arrays from timing.npz."""
    npz_path = run_dir / "timing.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing timing.npz in {run_dir}")
    data = dict(np.load(npz_path))
    # abort_reason is a 0-d string array; convert to scalar.
    data["abort_reason"] = str(data["abort_reason"])
    return data


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
    ax.axhline(_BUDGET_MS, color="orange", ls="--", lw=1.0, label=f"{_BUDGET_HZ:.0f}Hz budget ({_BUDGET_MS:.1f}ms)")
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
    bins = np.linspace(0, max(t_infer.max(), _BUDGET_MS * 1.1), 80)

    for ax in (ax_lin, ax_log):
        ax.hist(t_infer[~cb], bins=bins, alpha=0.6, color="steelblue",
                label=f"cached pop (n={(~cb).sum()})")
        ax.hist(t_infer[cb], bins=bins, alpha=0.6, color="crimson",
                label=f"chunk boundary (n={cb.sum()})")
        ax.axvline(_BUDGET_MS, color="orange", ls="--", lw=1.0, label=f"budget {_BUDGET_MS:.1f}ms")
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
    ax.axhline(_BUDGET_MS, color="orange", ls="--", lw=1.0, label=f"budget {_BUDGET_MS:.1f}ms")
    ax.set_xlabel("step")
    ax.set_ylabel("time per loop iter (ms)")
    ax.set_title("Per-stage latency stacked over time (obs + infer + tau + ctrl + sleep)")
    ax.legend(loc="upper right", ncol=6, fontsize=9)
    ax.grid(alpha=0.3)
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
    print(f"\nAll plots in {run_dir}")


if __name__ == "__main__":
    main()
