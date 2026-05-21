# Robust offline policy evaluation — design reference

> **Status:** reference document for the planned redesign of [`eval_g1_dataset.py`](../unitree_lerobot/eval_robot/eval_g1_dataset.py). Implementation is **incremental** — sections will be built, observed, and iterated. Treat this doc as the north star, not a contract; deviations are expected and welcome as we see what's actually useful.

## Context

We have a finished ACT run on `aleksantari/g1_dex1_tool_0_sorting` (95k steps) and an in-progress GR00T-N1.5 run on `aleksantari/g1_dex1_tools_combined`. The current offline eval at [`unitree_lerobot/eval_robot/eval_g1_dataset.py`](../unitree_lerobot/eval_robot/eval_g1_dataset.py) is a one-episode debug script — it iterates one episode, calls `select_action` per frame, and saves a hardcoded `figure.png`. It also conflates offline analysis with real-robot execution paths (`send_real_robot`, IK, EE control, Rerun).

The goal is to rewrite this file into a **purely offline analysis tool** — load a checkpoint, run inference on a user-specified set of held-out episodes, and produce metrics + plots that diagnose how the policy performs and *why*. This is the first line of defense before any robot or sim eval. The on-robot script `eval_g1.py` is intentionally left alone — that's a later, separate adaptation.

Design decisions:

- **Pure offline.** Strip robot/IK/EE/Rerun paths from this file. The other two scripts already cover those.
- **Single inference path: `predict_chunk` every frame.** Call `policy.predict_action_chunk()` once per frame, save the full `(chunk_size, action_dim)` chunk. Both "open-loop fresh-prediction" and any "closed-loop deployed-at-`n_action_steps=k`" analysis are recovered in post-processing from the saved chunks — no second inference run needed. See [`docs/action_generation.md`](action_generation.md) for the research that grounds this decision; the key fact is that the deployed action at any cadence k is `chunks[(t // k) * k][t % k]`, so one eval run unlocks every cadence's analysis.
- **Episodes specified explicitly** via `--episodes "[0,10,20,30]"` on the CLI (matches the known hold-out and stays trivial to override).
- **Outputs**: metrics JSON + per-episode plots + aggregate plots, eventually under `outputs/train/<run>/eval/<dataset>/` adjacent to the checkpoint. Today's increment writes per-episode `figure_*.png`, `horizon_decay_*.png`, and `predictions_*.npz` to CWD; the structured layout lands later.

## Files involved

Primary script: [`unitree_lerobot/eval_robot/eval_g1_dataset.py`](../unitree_lerobot/eval_robot/eval_g1_dataset.py) — incrementally refactored, not a single big-bang rewrite.

Shared helpers we reuse / extend in [`unitree_lerobot/eval_robot/utils/utils.py`](../unitree_lerobot/eval_robot/utils/utils.py):

- `extract_observation(step)` — HWC→CHW transpose for dataset frames.
- `predict_action(...)` — single-step wrapper around `policy.select_action`. **Still used by `eval_g1.py` and `eval_g1_sim.py`**, not by the offline eval anymore.
- `predict_chunk(...)` — **the chunk-returning sibling**. Calls `policy.predict_action_chunk`, returns `(chunk_size, action_dim)` on CPU. This is the load-bearing call for the offline path.
- `OfflineEvalConfig` — minimal config for the offline eval (repo_id, episodes, policy, root, visualization, rename_map). Future fields (output_dir, modes, save_predictions) defer to later increments.

Other reused machinery:

- `make_policy` and `make_pre_post_processors` from `lerobot.policies.factory` — checkpoint load + pre/postprocessor pipelines.
- `LeRobotDataset` + `dataset.meta.episodes["dataset_from_index"/"dataset_to_index"]` for per-episode frame ranges.
- `policy.predict_action_chunk(batch)` on `ACTPolicy` ([`modeling_act.py:124-133`](../unitree_lerobot/lerobot/src/lerobot/policies/act/modeling_act.py#L124-L133)) and `GrootPolicy` ([`modeling_groot.py:124-153`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/modeling_groot.py#L124-L153)). GR00T's postprocessor strips the 32-dim padding back to the real 16-dim action — no special handling at the call site.

## New CLI / config

A small `OfflineEvalConfig` dataclass lives in the same file:

```python
@dataclass
class OfflineEvalConfig:
    policy: PreTrainedConfig          # loaded via --policy.path=<ckpt_dir>
    repo_id: str                       # e.g. aleksantari/g1_dex1_tool_0_sorting
    episodes: list[int]                # required, explicit
    output_dir: str | None = None      # default: <ckpt_dir>/../../eval/<dataset_safe>/
    save_predictions: bool = True      # toggle for the per-episode .npz dump
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

One pass per frame using `predict_chunk`. The full chunk is captured at every step; everything else (fresh-prediction stream, deployed-at-any-cadence stream, horizon decay) is derived from those saved chunks in post-processing.

```python
for ep_idx in cfg.episodes:
    from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
    to_idx   = dataset.meta.episodes["dataset_to_index"][ep_idx]

    policy.reset()                                  # defensive; predict_chunk doesn't touch the queue but other state may exist
    predicted_chunks = []                           # list of (chunk_size, action_dim) np arrays, len = T

    for t in tqdm.tqdm(range(from_idx, to_idx), desc=f"episode {ep_idx}"):
        step = dataset[t]
        observation = extract_observation(step)
        chunk = predict_chunk(observation, policy, device, preprocessor, postprocessor,
                              use_amp=policy.config.use_amp, task=step["task"],
                              use_dataset=True, robot_type=None)
        predicted_chunks.append(chunk.cpu().numpy())

    predicted_chunks = np.stack(predicted_chunks)   # (T, chunk_size, action_dim)
    gt_actions = np.stack([dataset[t]["action"].numpy() for t in range(from_idx, to_idx)])

    # ----- Derive analyses from the saved chunks (no further inference) -----
    first_action_stream = predicted_chunks[:, 0, :]                          # (T, action_dim) — fresh prediction per frame
    # deployed_at_k = np.array([predicted_chunks[(t // k) * k, t % k] for t in range(T)])   # any cadence, any time

    # Horizon decay: for each chunk position k, mean MSE(chunks[t, k], gt[t+k]) across valid t.
    T = len(gt_actions)
    horizon_decay_mse = np.full(predicted_chunks.shape[1], np.nan)
    for k in range(predicted_chunks.shape[1]):
        if T - k <= 0:
            continue
        diff = predicted_chunks[:T-k, k] - gt_actions[k:]
        horizon_decay_mse[k] = float((diff ** 2).mean())

    np.savez_compressed(f"predictions_episode_{ep_idx:03d}.npz",
                        chunks=predicted_chunks, ground_truth=gt_actions,
                        horizon_decay_mse=horizon_decay_mse)
```

Key behaviours:

- **One forward pass per frame.** T forward passes per episode (vs ~T/n_action_steps for the old `select_action` path). For ACT chunk=100 with a 600-frame episode that's 600 instead of 6 — but at offline eval time there's no real-time budget to worry about.
- **`predict_chunk` does not touch the policy's internal action queue** — it bypasses the queue logic entirely. We still call `policy.reset()` between episodes for defensive hygiene (in case any other policy state exists).
- **No second inference run.** The "open-loop vs closed-loop" duality from prior iterations of this doc is recovered in post: open-loop = `predicted_chunks[:, 0, :]`, closed-loop at any cadence k = `predicted_chunks[(t // k) * k, t % k]`.
- **Why saving chunks matters.** `predictions_*.npz` contains the full `(T, chunk_size, action_dim)` array. Future analyses ("what would `n_action_steps=20` have looked like?" / "what's chunk variance at k=50?" / etc.) load the npz and slice — no GPU work.

## Metrics computed

All metrics are **derived from the saved `predicted_chunks` array** (no parallel inference modes). Each is a function of `predicted_chunks: (T, chunk_size, action_dim)` and `ground_truth_actions: (T, action_dim)`. The split below is in-script (canonical, every run) vs in-notebook (exploratory, per-experiment).

**Shipped in-script (`metrics.json` + console log):**

| Metric | Shape | Derivation |
| --- | --- | --- |
| `mean_l2` (per ep) | scalar | mean L2 norm of `(predicted_chunks[:, 0, :] - ground_truth)` rows. The single-number episode score. |
| `mse_per_dim` (per ep) | `(action_dim,)` | per-dim MSE between `predicted_chunks[:, 0, :]` and `ground_truth` |
| `mae_per_dim` (per ep) | `(action_dim,)` | per-dim mean absolute error of same |
| `horizon_decay_mse` (per ep) | `(chunk_size,)` | for each `k`: `mean_{valid t} MSE(predicted_chunks[t, k], ground_truth[t+k])` (positions past episode end are NaN). Also saved into `predictions.npz`. |
| Aggregate `mean_l2_mean` / `mean_l2_std` | scalars | mean and std of per-episode `mean_l2` across all episodes |
| Aggregate `mse_per_dim_mean` | `(action_dim,)` | mean of per-episode `mse_per_dim` across episodes |
| Aggregate `horizon_decay_mse_mean` | `(chunk_size,)` | `nanmean` of per-episode `horizon_decay_mse` across episodes (short episodes leave trailing NaN, which nanmean ignores) |

Headline numbers (`mean_l2` per episode + the aggregate mean ± std) are logged to the console at end-of-episode and end-of-run; full structure is dumped to `metrics.json` at the output dir root.

**To do in notebooks (exploratory, per-experiment — loads `.npz` files, no re-inference):**

| View | Construction |
| --- | --- |
| Deployed-cadence sweep | for k in [1, chunk_size]: build `deployed = chunks[(t // k) * k, t % k] for t in range(T)`; compute mean_l2 vs k. Yields the curve that picks the right `n_action_steps` for deployment. |
| `closed_to_open_ratio` at chosen k | `mean_l2_deployed_at_k / mean_l2_fresh` — quantifies the staleness penalty at any cadence. ≥ 1.0 in healthy runs. |
| Per-task breakdown (combined-dataset GR00T) | split episodes by `task` string or source-of-origin offsets; aggregate metrics within each task |
| Cross-checkpoint comparisons | load multiple `metrics.json` + `.npz` from different runs, contrast |
| Failure-frame hunting | find the 10 frames with highest per-frame L2 error, plot the corresponding camera frames |

The dividing line: things you want to read on *every* eval go in-script; things that change per experiment or require interactive exploration go in a notebook. The script defines the stable schema; the notebook is where new analyses live until they prove worth promoting into the script.

The key reframe vs prior versions of this doc: **the "open-loop vs closed-loop" duality is no longer two inference runs**, it's two views of the same saved tensor. The deployment cadence `k` can be chosen post-hoc, or swept post-hoc, without re-running inference.

## Output layout

**Shipped** (default location is sibling to the policy checkpoint, override with `--output_dir=<path>`):

```
outputs/train/<run>/eval/<dataset_safe_name>/
├── metrics.json                     # per-episode + aggregate headline numbers
└── episodes/
    └── episode_<NNN>/
        ├── actions_trajectory.png   # GT vs fresh-prediction stream (chunk[0])
        ├── horizon_decay.png        # MSE of chunk[k] vs GT[t+k] across t
        └── predictions.npz          # chunks, ground_truth, horizon_decay_mse
```

**To add in later increments:**

```
outputs/train/<run>/eval/<dataset_safe_name>/
└── aggregate/
    ├── per_dim_error.png            # bar: per-dim MSE across episodes
    ├── horizon_decay.png            # line: mean horizon decay across episodes
    └── episode_summary.png          # bar: mean_l2 per episode
```

`<dataset_safe_name>` = `repo_id.replace("/", "__")` for filesystem safety. Resolution lives in `_resolve_output_dir(cfg)` in the script: honors `cfg.output_dir` if set, otherwise walks up from `cfg.policy.pretrained_path` to the run dir and appends `eval/<dataset_safe>`. Falls back to `./eval_outputs/<dataset_safe>` if the policy was trained from scratch (no pretrained path).

## Plotting

`matplotlib.pyplot` only (no Rerun in offline mode). Three plot helpers, each ~30 lines:

- `_plot_per_episode_trajectory(gt, open_first, closed, out_path)` — adapts the existing 16-subplot pattern at [`eval_g1_dataset.py:145-171`](../unitree_lerobot/eval_robot/eval_g1_dataset.py#L145-L171) but plots three lines per subplot (GT solid, open-loop dashed, closed-loop dotted) and adds a per-dim MSE annotation.
- `_plot_aggregate_per_dim(per_episode_metrics, out_path)` — grouped bar chart, per-dim, open vs closed, with min/max whiskers across episodes.
- `_plot_horizon_decay(per_episode_horizon, out_path)` — line plot, x = chunk position, y = mean MSE.

Action dim auto-discovered from `policy.config.output_features["action"].shape[0]`, so plotting works for any action_dim.

## Inference latency monitoring

Per-step wall-clock is timed around the `predict_chunk()` call and summarized per episode + aggregate (mean / std / min / max / p50 / p95 / p99) with a 30 Hz real-time budget check. Because `predict_chunk()` ends with `.to("cpu")`, the timer captures the full GPU work — no manual `torch.cuda.synchronize()` needed.

Status: shipped as `_log_inference_times` in [`eval_g1_dataset.py`](../unitree_lerobot/eval_robot/eval_g1_dataset.py).

**Consequence of the chunk-per-frame design: latency is now unimodal in the offline eval.** Every frame is a real forward pass; there are no queue-pop dequeue frames. `mean ≈ p50 ≈ p95` and `max` is within ~10–20% of those (modulo the warm-up first call). The headline numbers are now directly comparable to the real-time budget without per-bucket gymnastics.

One refinement still worth folding in once we have eval data:

1. **Strip the first frame from stats and report it separately.** cudnn benchmark + CUDA JIT make the first call multiples slower than steady state. The implementation already reports `first=X (incl. warm-up)` alongside the headline numbers, but `max`, `p95`, `p99` and the budget check still include that warm-up frame. Replace those with the slice `arr[1:]` so they reflect steady-state, keep `first` as the explicit cold-start indicator.

The previously-listed **per-bucket queue-pop vs forward-pass split** is now **obsolete for offline eval** — chunks-per-frame eliminates the bimodal pattern. It's still relevant for the on-robot script (`eval_g1.py`), which uses `select_action` at deployment cadence; when we adapt that script, the bucket-split refinement applies there.

## Verification

1. Run the ACT eval (env: `unitree-lerobot`):

   ```bash
   bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.eval_robot.eval_g1_dataset \
       --policy.path=outputs/train/2026-05-19/19-07-27_act_g1_dex1_tool_0_sorting/checkpoints/095000/pretrained_model \
       --repo_id=aleksantari/g1_dex1_tool_0_sorting \
       --episodes "[0,10,20,30]"'
   ```

   Expected on completion, under `outputs/train/2026-05-19/19-07-27_act_g1_dex1_tool_0_sorting/eval/aleksantari__g1_dex1_tool_0_sorting/`:
   - `metrics.json` — per-episode + aggregate headline numbers
   - `episodes/episode_000/actions_trajectory.png` ... `episodes/episode_030/actions_trajectory.png` — GT vs fresh-prediction stream
   - `episodes/episode_000/horizon_decay.png` ... `episodes/episode_030/horizon_decay.png` — curves rising left-to-right
   - `episodes/episode_000/predictions.npz` ... `episodes/episode_030/predictions.npz` — each carries `chunks`, `ground_truth`, `horizon_decay_mse`
   - Latency log lines per episode + aggregate, showing **unimodal stats** (mean ≈ p50 ≈ p95, `first` ≫ rest due to warm-up)
   - `Episode N metrics: mean_l2=... | per-dim MSE min=... max=... mean=...` lines per episode
   - `All episodes mean_l2: X ± Y (n=4 episodes)` summary at the end

2. Sanity-check `metrics.json`:
   - 4 entries under `episodes` (keys "0", "10", "20", "30")
   - `aggregate.n_episodes == 4`
   - `aggregate.mean_l2_mean` and `aggregate.mean_l2_std` are finite
   - `aggregate.horizon_decay_mse_mean` is a list of length `chunk_size` with no nan at index 0 (every episode contributed `k=0`)

3. Sanity-check a saved `.npz`:

   ```python
   d = np.load("episodes/episode_000/predictions.npz")
   d["chunks"].shape           # (T, chunk_size, action_dim) — e.g. (600, 100, 16) for ACT
   d["ground_truth"].shape     # (T, action_dim)
   d["horizon_decay_mse"]      # (chunk_size,)
   d["horizon_decay_mse"][0] < d["horizon_decay_mse"][-1]   # True — error grows with horizon
   ```

4. **Post-processing smoke test** (the design's core payoff): from a saved `.npz`, derive what `n_action_steps=100` would have deployed — without any new inference.

   ```python
   k = 100
   deployed = np.array([d["chunks"][(t // k) * k, t % k] for t in range(len(d["ground_truth"]))])
   ```

5. Spot-check `episodes/episode_000/horizon_decay.png` — should rise from near-zero at `k=0` (fresh-prediction quality) to larger at `k=chunk_size-1`. If it's flat, suspect indexing bugs in the `t+k` truncation.

6. Spot-check `episodes/episode_000/actions_trajectory.png` — dominant arm dims should track GT closely on the fresh-prediction trace. The fresh-prediction stream should be at least as accurate as a `select_action`-based equivalent since it has no staleness.

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

## Related docs

- [`docs/action_generation.md`](action_generation.md) — research report on how this repo (and upstream lerobot) generates actions. Grounds the chunk-based inference design used here: the load-bearing finding is that `predict_action_chunk` extracts the *full* policy output per forward pass at no extra GPU cost, and the deployed action stream at any cadence k is recoverable in post via `chunks[(t // k) * k, t % k]`. Also covers ACT temporal ensembling (built into lerobot, off by default), GR00T's stochasticity and 16-step hard cap, and other design implications.
