# Offline policy evaluation — current implementation

> **Authoritative summary of what [`eval_g1_dataset.py`](../unitree_lerobot/eval_robot/eval_g1_dataset.py) produces today (as of 2026-06-11).**
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

## Five per-episode plots

Written under `episodes/episode_NNN/`. Plots 1–3 have one subplot per action dimension, labeled with joint names read from dataset metadata.

1. **`1_fresh_chunk0.png`** — GT (blue) vs `chunk[0]` at every frame (red dashed). The **fresh-prediction stream**: a brand-new inference per frame, zero deployment staleness. This is the `n_action_steps = 1` upper bound on achievable quality — no deployment cadence can beat it.

2. **`2_chunk_fan.png`** — the **full predicted chunk drawn at each obs step**, as a faint polyline spanning `[t, t+chunk_size)`. The color **evolves *along* each chunk by within-chunk position `k`: blue at `k=0` (the immediate prediction) → red at `k=chunk_size-1` (the far-horizon extrapolation)** (colorbar), with GT solid black on top. So on each fan you can see which part is early (blue, hugging GT) vs late (red, fanning out and drifting). Chunks are clipped at the episode end so the x-range matches GT. To stay legible on long episodes the fan is sub-sampled: by default `stride = max(1, T // 120)`; override with `--fan_stride=N` (`--fan_stride=1` draws a chunk from every frame).

3. **`3_deployed_full_chunk.png`** — the **realtime-deployment view at `n_action_steps = chunk_size`**: GT (blue) vs the deployed stream (red dashed). Re-query inference only at frames `0, K, 2K…` (gray verticals) and replay the pre-computed chunk in between — the most-stale cadence. Shows the drift-then-snap-back pattern that the fresh-stream plot hides. The deployed stream is built by `_build_deployed_stream` via the identity `deployed[t] = chunks[(t // k) * k, t % k]`, which is general for any cadence `k` (here `k = chunk_size`).

4. **`4_boundary_discontinuity.png`** — **one bar chart per arm joint** (14 for G1+Dex1; grippers excluded, since their units aren't radians), quantifying the **chunk-boundary discontinuity** of the deployed stream (plot 3's cadence). At each re-query boundary `b = k, 2k, …` the deployed action jumps from the stale last action of the old chunk (`deployed[b-1]`) to the fresh first action of the new chunk (`deployed[b]`). Each **bar is one boundary event** — x = boundary timestep, height = `|deployed[b,j] − deployed[b-1,j]|` for that joint in **degrees** (arm actions are radians). The dashed line marks the joint's mean; the subplot title shows `mean / max`. Definition **mirrors the on-robot `compute_arm_deltas`** in `analyze_realtime_eval.py`, so the offline boundary numbers are directly comparable to a real rollout — and across models/checkpoints (this is how you check whether training reduced the boundary jerk).

5. **`5_horizon_decay.png`** — **horizon decay**: a single curve of per-position prediction error `hd[k] = mean_t ‖chunk[t,k] − GT[t+k]‖₂` vs chunk position `k` — i.e. how well a *single observation* predicts `k` steps ahead, averaged over the episode. This is the global, **phase-free** view of chunk predictive quality (unlike `deployed@k`, which mixes all staleness levels and depends on the re-query offset). Key identities: `hd[0]` **equals** the fresh `mean_l2`, and `mean(hd[0:k])` ≈ `deployed@k` — so the curve predicts deployment error at *any* cadence and its steepness is the principled basis for choosing `n_action_steps` (re-query before it climbs too far). A flatter curve = chunks that drift less = more internally-coherent multi-step predictions. The plot overlays the running mean `mean(hd[0:k])` (dashed; endpoint ≈ `deployed@chunk_size`).

## `metrics.json` (current schema)

Written at the output-dir root.

```json
{
  "action_dim_names": ["kLeftShoulderPitch", ..., "kRightGripper"],
  "episodes": {
    "0": {
      "mean_l2": ..., "mse_per_dim": [...], "mae_per_dim": [...], "mean_l2_deployed_full_chunk": ...,
      "boundary_discontinuity": {
        "n_boundaries": ...,
        "per_joint": { "kLeftShoulderPitch": { "mean_deg": ..., "max_deg": ... }, ... }
      },
      "horizon_decay_l2": [hd[0], hd[1], ..., hd[chunk_size-1]]
    },
    "10": { ... }
  },
  "aggregate": {
    "mean_l2_mean": ..., "mean_l2_std": ...,
    "mean_l2_deployed_full_chunk_mean": ...,
    "mse_per_dim_mean": [...],
    "n_episodes": 4,
    "boundary_discontinuity": {
      "per_joint_mean_deg": { "kLeftShoulderPitch": ..., ... },
      "per_joint_max_deg":  { "kLeftShoulderPitch": ..., ... }
    },
    "horizon_decay_l2_mean": [...]
  }
}
```

- `mean_l2` / `mse_per_dim` / `mae_per_dim` — over the fresh stream (`chunk[0]` per frame). `mean_l2` is the single-number "is this checkpoint any good?" score.
- `mean_l2_deployed_full_chunk` — the same L2 metric on the full-chunk deployment stream (plot 3). Its **ratio to `mean_l2` quantifies the staleness penalty** of deploying the whole chunk between re-queries (≥ 1.0 in healthy runs).
- `boundary_discontinuity` — the numbers behind plot 4. Per arm joint (grippers excluded), the chunk-boundary jump in **degrees**: per-episode `mean_deg` / `max_deg` over all boundaries; the aggregate takes the **mean of per-episode means** and the **max of per-episode maxes** per joint. This is the metric to compare across models/checkpoints to see whether the boundary discontinuity (plot 4) actually shrank.
- `horizon_decay_l2` — the plot-5 curve as raw numbers: a length-`chunk_size` list, `hd[k] = mean_t ‖chunk[t,k] − GT[t+k]‖₂` (positions past the episode end are NaN). `hd[0]` equals `mean_l2`. The aggregate `horizon_decay_l2_mean` is the per-position `nanmean` across episodes — a flatter curve = chunks that drift less, and `mean(hd[0:k])` ≈ `deployed@k` for any cadence.

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
        ├── 3_deployed_full_chunk.png
        ├── 4_boundary_discontinuity.png
        └── 5_horizon_decay.png
```

`<dataset_safe>` = `repo_id.replace("/", "__")`; `<step>` inferred from the canonical `checkpoints/<step>/pretrained_model` layout (omitted for non-canonical paths). Override the whole path with `--output_dir` (verbatim, no `<variant>` appended). Falls back to `./eval_outputs/<dataset_safe>/<variant>` for from-scratch (no-pretrained) runs.

**`<variant>` subdir** — auto-encodes the inference configuration that distinguishes runs of the *same* checkpoint+episode, so e.g. a 4-step and an 8-step GR00T eval land in **sibling dirs instead of clobbering each other**. Built by `_variant_tag`, pieces added only when relevant:

- `steps{N}` — effective flow-matching denoising steps (diffusion policies only; omitted for ACT).
- `seed{N}` — when `--seed` is set.
- free-form `--tag=<label>` appended last (for ablations the auto-tag can't capture).

When none apply (plain ACT, no seed, no tag) the `<variant>` subdir is omitted and outputs stay flat at `<step>/` (backward-compatible). Example: `…/017500/steps4_seed42/` next to `…/017500/steps8_seed42/`.

## Removed vs the old docs

- **Horizon-decay** — was removed in the 3-plot redesign, then **re-added 2026-06-12 as plot 5** (`5_horizon_decay.png` + `horizon_decay_l2` metric), now in **L2** units (was MSE) and consistent with `mean_l2`/`deployed` (`hd[0] == mean_l2`).
- **`predictions.npz`** per-episode dump — removed. Inference is cheap; re-run rather than persist. (The cadence reconstruction the npz used to enable is now done in-memory by `_build_deployed_stream`.)
- Old filenames `actions_trajectory.png` / `horizon_decay.png` — replaced by the numbered five-plot set above.

## Deferred / future-analysis ideas

Still un-implemented; the deprecated docs describe them in more detail and they may be worth revisiting:

- **Full cadence sweep** — `mean_l2` vs every `n_action_steps` k, not just `k = chunk_size`. `_build_deployed_stream` already generalizes to any k, and plot 5's running-mean curve already previews `deployed@k` for all k from a single eval.
- **Per-task breakdown** for the combined-dataset GR00T runs (split episodes by `task` string / source offsets).
- **Language-prompt override** (`--eval-task-prompts`) to probe language-conditioning robustness.
- **Cross-checkpoint comparison** plots (`mean_l2` vs training step).
