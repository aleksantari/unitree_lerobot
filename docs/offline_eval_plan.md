# Robust offline policy evaluation — design reference

> **Status:** reference document for the planned redesign of [`eval_g1_dataset.py`](../unitree_lerobot/eval_robot/eval_g1_dataset.py). Implementation is **incremental** — sections will be built, observed, and iterated. Treat this doc as the north star, not a contract; deviations are expected and welcome as we see what's actually useful.

## Context

We have a finished ACT run on `aleksantari/g1_dex1_tool_0_sorting` (95k steps) and an in-progress GR00T-N1.5 run on `aleksantari/g1_dex1_tools_combined`. The current offline eval at [`unitree_lerobot/eval_robot/eval_g1_dataset.py`](../unitree_lerobot/eval_robot/eval_g1_dataset.py) is a one-episode debug script — it iterates one episode, calls `select_action` per frame, and saves a hardcoded `figure.png`. It also conflates offline analysis with real-robot execution paths (`send_real_robot`, IK, EE control, Rerun).

The goal is to rewrite this file into a **purely offline analysis tool** — load a checkpoint, run inference on a user-specified set of held-out episodes, and produce metrics + plots that diagnose how the policy performs and *why*. This is the first line of defense before any robot or sim eval. The on-robot script `eval_g1.py` is intentionally left alone — that's a later, separate adaptation.

Design decisions:

- **Pure offline.** Strip robot/IK/EE/Rerun paths from this file. The other two scripts already cover those.
- **Both inference modes per episode:** open-loop (`predict_action_chunk` every frame) AND closed-loop (`select_action` every frame at deployment cadence). Open-loop measures instantaneous policy quality + horizon decay; closed-loop measures deployment behavior including staleness of the queued chunk.
- **Episodes specified explicitly** via `--episodes 0 10 20 30` on the CLI (matches the known hold-out and stays trivial to override).
- **Outputs**: metrics JSON + per-episode plots + aggregate plots, under `outputs/train/<run>/eval/<dataset>/` adjacent to the checkpoint.

## File to modify

[`unitree_lerobot/eval_robot/eval_g1_dataset.py`](../unitree_lerobot/eval_robot/eval_g1_dataset.py) — full rewrite. New length ~280–350 lines.

No other files need to be touched. We will reuse:

- `extract_observation(step)` at [`unitree_lerobot/eval_robot/utils/utils.py:20-32`](../unitree_lerobot/eval_robot/utils/utils.py#L20-L32) — handles HWC→CHW transpose
- `make_policy` and `make_pre_post_processors` from `lerobot.policies.factory` — checkpoint load + pre/postprocessor pipelines
- `LeRobotDataset` + `dataset.meta.episodes` (the `dataset_from_index` / `dataset_to_index` arrays per episode)
- Both `policy.select_action(batch)` and `policy.predict_action_chunk(batch)` on `ACTPolicy` ([`modeling_act.py:98-133`](../unitree_lerobot/lerobot/src/lerobot/policies/act/modeling_act.py#L98-L133)) and `GrootPolicy` ([`modeling_groot.py:124-163`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/modeling_groot.py#L124-L163)). GR00T's postprocessor strips the 32-dim padding back to the real 16-dim action — no special handling at the call site.

We will **not** reuse `predict_action(...)` in [`unitree_lerobot/eval_robot/utils/utils.py:35-79`](../unitree_lerobot/eval_robot/utils/utils.py#L35-L79) — it wraps only `select_action` and embeds robot-flow assumptions. The new file inlines small `_run_open_loop` and `_run_closed_loop` helpers that share an input-prep function but call the two policy methods directly.

## New CLI / config

A small `OfflineEvalConfig` dataclass lives in the same file:

```python
@dataclass
class OfflineEvalConfig:
    policy: PreTrainedConfig          # loaded via --policy.path=<ckpt_dir>
    repo_id: str                       # e.g. aleksantari/g1_dex1_tool_0_sorting
    episodes: list[int]                # required, explicit
    output_dir: str | None = None      # default: <ckpt_dir>/../../eval/<dataset_safe>/
    modes: list[str] = field(default_factory=lambda: ["open_loop", "closed_loop"])
    save_predictions: bool = True      # raw GT/pred arrays alongside plots
    root: str | None = None            # passthrough to LeRobotDataset
```

`@parser.wrap()` keeps lerobot's existing CLI / yaml / dataclass parsing.

Example invocation:

```bash
bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.eval_robot.eval_g1_dataset \
    --policy.path=outputs/train/2026-05-19/19-07-27_act_g1_dex1_tool_0_sorting/checkpoints/095000/pretrained_model \
    --repo_id=aleksantari/g1_dex1_tool_0_sorting \
    --episodes 0 10 20 30'
```

## Inference flow per episode

```python
for ep_idx in cfg.episodes:
    from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
    to_idx   = dataset.meta.episodes["dataset_to_index"][ep_idx]

    # ------- open-loop: predict_action_chunk every frame -------
    policy.reset()
    open_chunks = []          # list of (chunk_size, action_dim) arrays, len = T
    for t in range(from_idx, to_idx):
        step = dataset[t]
        batch = _prep_batch(step, device)            # adds batch dim, attaches task string
        batch = preprocessor(batch)
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)        # (1, chunk_size, A_pad-or-A)
        chunk = postprocessor(chunk).squeeze(0).cpu().numpy() # (chunk_size, action_dim)
        open_chunks.append(chunk)

    # ------- closed-loop: deployment cadence -------
    policy.reset()
    closed_actions = []       # (T, action_dim)
    for t in range(from_idx, to_idx):
        step = dataset[t]
        batch = _prep_batch(step, device)
        batch = preprocessor(batch)
        with torch.inference_mode():
            action = policy.select_action(batch)               # uses internal queue
        action = postprocessor(action).squeeze(0).cpu().numpy()
        closed_actions.append(action)

    gt_actions = np.stack([dataset[t]["action"].numpy() for t in range(from_idx, to_idx)])

    _compute_and_save(ep_idx, gt_actions, open_chunks, np.stack(closed_actions), out_dir)
```

Key behaviours:

- `policy.reset()` is called between modes and between episodes so the action queue starts empty.
- Closed-loop is cheap for ACT (chunk=100 → 1 forward pass per 100 frames). Open-loop is the expensive arm (1 per frame).
- `_prep_batch` builds `{ "observation.state": ..., "observation.images.*": ..., "task": step["task"] }` with batch dim 1 on the policy device. Task string passes through harmlessly for ACT and is required for GR00T.

## Metrics computed

Per-mode, per-episode (stored in `metrics.json` under `episodes.<N>.<mode>`):

| Metric | Shape | Definition |
| --- | --- | --- |
| `mse_per_dim` | `(action_dim,)` | mean squared error per action dimension |
| `mae_per_dim` | `(action_dim,)` | mean absolute error per dim |
| `rmse_per_dim` | `(action_dim,)` | sqrt of `mse_per_dim` |
| `l2_per_step` | `(T,)` | per-timestep L2 norm of (pred − gt) — drives error-over-time plot |
| `mean_l2` | scalar | mean of `l2_per_step` — single-number episode score |

Additional for open-loop only (uses the full predicted chunks):

| Metric | Shape | Definition |
| --- | --- | --- |
| `horizon_decay_mse` | `(chunk_size,)` | for each position `k` in the chunk, mean over all frames `t` of `MSE(open_chunks[t][k], gt[t+k])` (truncated at episode end) |

Aggregate across episodes (stored top-level in `metrics.json`):

- Mean and std of `mean_l2` across episodes (per mode)
- Mean per-dim MSE/MAE/RMSE across episodes (per mode)
- Mean `horizon_decay_mse` across episodes (open-loop)
- A `comparison` block: `closed_loop.mean_l2 / open_loop.mean_l2` — diagnoses staleness penalty (≥ 1.0 in healthy runs)

## Output layout

```
outputs/train/<run>/eval/<dataset_safe_name>/
├── metrics.json
├── aggregate/
│   ├── per_dim_error.png            # bar: per-dim MSE — open vs closed, episode error bars
│   ├── horizon_decay.png            # line: mean error vs chunk-position k (open-loop)
│   └── episode_summary.png          # bar: mean_l2 per episode, open vs closed
└── episodes/
    └── episode_<NNN>/
        ├── actions_trajectory.png   # action_dim subplots: GT vs open-loop-step0 vs closed-loop-deployed
        ├── error_over_time.png      # L2 norm per timestep, open vs closed overlaid
        └── predictions.npz          # gt, open_first_step, closed, open_chunks — for re-analysis
```

`<dataset_safe_name>` = `repo_id.replace("/", "__")` for filesystem safety.

## Plotting

`matplotlib.pyplot` only (no Rerun in offline mode). Three plot helpers, each ~30 lines:

- `_plot_per_episode_trajectory(gt, open_first, closed, out_path)` — adapts the existing 16-subplot pattern at [`eval_g1_dataset.py:145-171`](../unitree_lerobot/eval_robot/eval_g1_dataset.py#L145-L171) but plots three lines per subplot (GT solid, open-loop dashed, closed-loop dotted) and adds a per-dim MSE annotation.
- `_plot_aggregate_per_dim(per_episode_metrics, out_path)` — grouped bar chart, per-dim, open vs closed, with min/max whiskers across episodes.
- `_plot_horizon_decay(per_episode_horizon, out_path)` — line plot, x = chunk position, y = mean MSE.

Action dim auto-discovered from `policy.config.output_features["action"].shape[0]`, so plotting works for any action_dim.

## Verification

1. Run the ACT eval (env: `unitree-lerobot`):

   ```bash
   bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.eval_robot.eval_g1_dataset \
       --policy.path=outputs/train/2026-05-19/19-07-27_act_g1_dex1_tool_0_sorting/checkpoints/095000/pretrained_model \
       --repo_id=aleksantari/g1_dex1_tool_0_sorting \
       --episodes 0 10 20 30'
   ```

   Expected:
   - Completes 4 episodes × 2 modes with progress bars (open-loop dominates wall-clock).
   - Writes `outputs/train/2026-05-19/19-07-27_act_g1_dex1_tool_0_sorting/eval/aleksantari__g1_dex1_tool_0_sorting/metrics.json` and the plot tree above.

2. Sanity-check `metrics.json`:
   - All four episodes present under `episodes`.
   - Per-dim MSE is small for the gripper dims (small range) and larger for the wrist/elbow dims.
   - `comparison.closed_to_open_ratio` ≥ 1.0 (closed-loop should never beat open-loop on average — staleness only hurts).

3. Spot-check `aggregate/horizon_decay.png` — should be monotonically (or near-monotonically) increasing with chunk position. If it's flat, suspect a bug in chunk indexing.

4. Spot-check one `episodes/episode_000/actions_trajectory.png` — the three traces should track each other on the dominant arm dims and diverge most on transient/contact frames.

## GR00T extension (deferred)

The script is policy-agnostic by construction: it only calls `select_action` and `predict_action_chunk`, both of which exist on `GrootPolicy`. The task string is already routed through `_prep_batch`. The GR00T postprocessor slices `max_action_dim=32` → 16, so output shape matches dataset GT.

For the combined-dataset GR00T eval, two small additions will be wanted later:

- Per-task metric breakdown (split episodes by `step["task"]` or by source-of-origin in aggregate index space — sorting offsets 0–112, handover 113–221).
- A flag like `--eval-task-prompts "sort the tools"` to override the dataset's recorded task and test language-conditioning robustness.

Both are additive — they don't change the core flow. Once a GR00T checkpoint at `outputs/train/2026-05-20/15-28-24_groot_g1_dex1_tools_combined/checkpoints/<step>/pretrained_model` is far enough along, the same CLI works (switch env to `lerobot-gr00t`).

## Out of scope (deliberately)

- Real-robot or sim execution paths — they remain in `eval_g1.py` / `eval_g1_sim.py` untouched.
- Rerun visualization — overkill for batch analysis; matplotlib is enough.
- MP4 videos with action overlays — defer until the metrics view is in heavy use and we know what's worth animating.
- Auto-detecting held-out from `train_config.json` — explicit `--episodes` is simpler and matches the current mental model.
- Per-joint plots in physical units (degrees, etc.) — GT and pred share normalized units so the comparison is already correct; unit conversion is a later polish item.
