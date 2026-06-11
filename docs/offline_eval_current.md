# Offline policy evaluation — current implementation

> **Authoritative summary of what [`eval_g1_dataset.py`](../unitree_lerobot/eval_robot/eval_g1_dataset.py) produces today (as of 2026-06-10).**
> The earlier design docs — [`offline_eval_plan.md`](offline_eval_plan.md) and [`offline_eval_analysis.md`](offline_eval_analysis.md) — are now **historical/deprecated**: their core inference design still holds, but they describe outputs (horizon-decay curve, `predictions.npz`, old plot filenames) that have since been removed. Read them only for design rationale and the deferred/future-analysis ideas. [`action_generation.md`](action_generation.md) is **not** deprecated — the chunk-generation research it documents is still load-bearing.

## What it does

Pure offline diagnostic — no robot, no IK, no DDS, no Rerun-required path. It loads a trained checkpoint and, for each episode index passed via `--episodes`, runs `predict_chunk` (a thin wrapper over `policy.predict_action_chunk`, which bypasses the policy's internal action queue) at **every frame** of the held-out episode, capturing the full `(chunk_size, action_dim)` chunk. Predictions are compared against the dataset's ground-truth actions. ACT chunk = 100; GR00T-N1.5 = 16 (hard-capped by the pretrained architecture).

This is the first line of defense before any robot or sim eval. `eval_g1.py` (real robot) and `eval_g1_sim.py` (sim) keep all the hardware/IK/EE machinery; this script is deliberately stripped to pure analysis.

```bash
bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.eval_robot.eval_g1_dataset \
    --policy.path=<ckpt>/checkpoints/<step>/pretrained_model \
    --repo_id=<hf_user>/<dataset> \
    --episodes 0 10 20 30'
```

(GR00T: run in the `lerobot-gr00t` env and pass `--seed=N` for reproducible diffusion sampling. `--num_inference_timesteps=N` overrides the flow-matching denoising-step count at eval time — base GR00T-N1.5 default is 4; it is not a `GrootConfig` field, so the script sets `num_inference_timesteps` directly on the action-head module after load. No-op for ACT. More steps = finer ODE integration at the cost of a proportionally slower action head.)

## Three per-episode plots

Written under `episodes/episode_NNN/`. One subplot per action dimension, labeled with joint names read from dataset metadata.

1. **`1_fresh_chunk0.png`** — GT (blue) vs `chunk[0]` at every frame (red dashed). The **fresh-prediction stream**: a brand-new inference per frame, zero deployment staleness. This is the `n_action_steps = 1` upper bound on achievable quality — no deployment cadence can beat it.

2. **`2_chunk_fan.png`** — the **full predicted chunk drawn at each obs step**, as a faint polyline spanning `[t, t+chunk_size)`, colored by its start (obs) timestep via a viridis colorbar, with GT solid black on top. Where a chunk fan peels away from GT shows *which observation* the policy starts drifting from. Chunks are clipped at the episode end so the x-range matches GT. To stay legible on long episodes the fan is sub-sampled: by default `stride = max(1, T // 120)` (draws ~120 fans; short episodes draw every frame). Override with `--fan_stride=N` (e.g. `--fan_stride=1` to draw a chunk from every frame).

3. **`3_deployed_full_chunk.png`** — the **realtime-deployment view at `n_action_steps = chunk_size`**: GT (blue) vs the deployed stream (red dashed). Re-query inference only at frames `0, K, 2K…` (gray verticals) and replay the pre-computed chunk in between — the most-stale cadence. Shows the drift-then-snap-back pattern that the fresh-stream plot hides. The deployed stream is built by `_build_deployed_stream` via the identity `deployed[t] = chunks[(t // k) * k, t % k]`, which is general for any cadence `k` (here `k = chunk_size`).

## `metrics.json` (current schema)

Written at the output-dir root.

```json
{
  "action_dim_names": ["kLeftShoulderPitch", ..., "kRightGripper"],
  "episodes": {
    "0": { "mean_l2": ..., "mse_per_dim": [...], "mae_per_dim": [...], "mean_l2_deployed_full_chunk": ... },
    "10": { ... }
  },
  "aggregate": {
    "mean_l2_mean": ..., "mean_l2_std": ...,
    "mean_l2_deployed_full_chunk_mean": ...,
    "mse_per_dim_mean": [...],
    "n_episodes": 4
  }
}
```

- `mean_l2` / `mse_per_dim` / `mae_per_dim` — over the fresh stream (`chunk[0]` per frame). `mean_l2` is the single-number "is this checkpoint any good?" score.
- `mean_l2_deployed_full_chunk` — the same L2 metric on the full-chunk deployment stream (plot 3). Its **ratio to `mean_l2` quantifies the staleness penalty** of deploying the whole chunk between re-queries (≥ 1.0 in healthy runs).

## Latency stats

Per-episode + aggregate inference-time summary (mean / std / min / max / p50 / p95 / p99, plus `first` reported separately as cold-start warm-up) via `log_inference_times`. The distribution is **unimodal** because every frame is a real forward pass — there are no fast queue-pop frames as there would be on the robot.

## Output layout

```
outputs/train/<run>/eval/<dataset_safe>/<step>/<variant>/
├── metrics.json
└── episodes/
    └── episode_NNN/
        ├── 1_fresh_chunk0.png
        ├── 2_chunk_fan.png
        └── 3_deployed_full_chunk.png
```

`<dataset_safe>` = `repo_id.replace("/", "__")`; `<step>` inferred from the canonical `checkpoints/<step>/pretrained_model` layout (omitted for non-canonical paths). Override the whole path with `--output_dir` (verbatim, no `<variant>` appended). Falls back to `./eval_outputs/<dataset_safe>/<variant>` for from-scratch (no-pretrained) runs.

**`<variant>` subdir** — auto-encodes the inference configuration that distinguishes runs of the *same* checkpoint+episode, so e.g. a 4-step and an 8-step GR00T eval land in **sibling dirs instead of clobbering each other**. Built by `_variant_tag`, pieces added only when relevant:

- `steps{N}` — effective flow-matching denoising steps (diffusion policies only; omitted for ACT).
- `seed{N}` — when `--seed` is set.
- free-form `--tag=<label>` appended last (for ablations the auto-tag can't capture).

When none apply (plain ACT, no seed, no tag) the `<variant>` subdir is omitted and outputs stay flat at `<step>/` (backward-compatible). Example: `…/017500/steps4_seed42/` next to `…/017500/steps8_seed42/`.

## Removed vs the old docs

- **Horizon-decay** curve + metric (`horizon_decay_mse`, `horizon_decay_mse_mean`) — removed.
- **`predictions.npz`** per-episode dump — removed. Inference is cheap; re-run rather than persist. (The cadence reconstruction the npz used to enable is now done in-memory by `_build_deployed_stream`.)
- Old filenames `actions_trajectory.png` / `horizon_decay.png` — replaced by the numbered three-plot set above.

## Deferred / future-analysis ideas

Still un-implemented; the deprecated docs describe them in more detail and they may be worth revisiting:

- **Horizon-decay curve** (MSE of `chunk[k]` vs `GT[t+k]` across t) — was the principled basis for picking deployment cadence; removed but conceptually sound.
- **Full cadence sweep** — `mean_l2` vs every `n_action_steps` k, not just `k = chunk_size`. `_build_deployed_stream` already generalizes to any k.
- **Per-task breakdown** for the combined-dataset GR00T runs (split episodes by `task` string / source offsets).
- **Language-prompt override** (`--eval-task-prompts`) to probe language-conditioning robustness.
- **Cross-checkpoint comparison** plots (`mean_l2` vs training step).
