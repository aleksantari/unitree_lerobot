# Offline policy evaluation — output reference

> A short reader-friendly guide to the outputs produced by our offline policy evaluation script. The eval runs a trained imitation-learning policy (ACT or GR00T) against held-out episodes from the training dataset and produces per-episode plots and saved arrays. This doc explains what each output represents and how to read it.

## What's being evaluated

**The policy.** A trained imitation-learning model — currently either ACT or GR00T-N1.5 — that takes a robot observation (camera images + proprioceptive state, optionally a language task string) and outputs a *chunk* of predicted actions covering the next 16–100 timesteps depending on the model. ACT's chunk is 100 steps; GR00T-N1.5's is hard-capped at 16.

**The dataset.** A LeRobot-format dataset of human teleoperation demonstrations on a Unitree G1 humanoid with Dex1 grippers. We hold out a fixed set of episode indices (`[0, 10, 20, 30]` from each source dataset) that the policy never saw during training; those are the episodes the eval runs on.

**The setup.** At each frame `t` of a held-out episode, we feed the dataset's observation at time `t` into the policy, capture its full predicted action chunk `(chunk_size, action_dim)`, and compare those predictions against the dataset's ground-truth actions. Importantly, we never *deploy* any of these actions on the real robot — we're just comparing the policy's predictions to what the human demonstrator actually did at the same observation. This is an *offline diagnostic*, not a deployment test.

**Why the chunk and not just one action?** Both ACT and GR00T predict a full chunk per forward pass, but most "naive" evaluations only inspect the first action. By capturing and saving the entire chunk, we get a richer measurement at no extra compute cost: we can see not just "what does the policy think I should do *right now*?" but also "what does it think I should do 50 frames from now, given what it sees right now?"

## The two main per-episode plots

For each held-out episode, the eval produces two PNG files.

### 1. Prediction-stream plot (`figure_episode_NNN.png`)

A multi-subplot figure with one row per action dimension (16 for our G1+Dex1 setup, each labeled with the joint name pulled from the dataset's metadata — `kLeftShoulderPitch`, `kRightElbow`, `kLeftGripper`, etc.) showing two overlaid traces over the episode:

- **Blue (Ground Truth)** — the demonstrator's actual action at each timestep, straight from the dataset.
- **Red dashed (Predicted, chunk[0])** — the *first* action of the chunk the policy produced at each frame. This is what the policy thinks should happen *right now* given what it observes *right now*.

This is the cleanest possible view of policy quality: every prediction is made from a fresh observation that matches the ground-truth action's timestep. There is no deployment-cadence staleness in this trace — it isolates pure "observation → action" mapping quality.

**How to read it:**

- Where blue and red overlay closely → the policy maps observations to actions correctly.
- Where they diverge → the policy is struggling even when shown the right observation. Common divergence points: contact transitions, gripper open/close events, end-of-trajectory deceleration.
- Per-dimension variation matters: arm joints (dims 1–14) often track tightly while gripper dims (15–16) behave more abruptly because they're less continuous by nature.

This trace is the **upper bound** on policy quality. No deployment strategy can do better than this — anything that doesn't re-query the policy every frame introduces staleness, which can only hurt.

<!-- IMAGE PLACEHOLDER: episodes/episode_000/actions_trajectory.png -->
> **\[Example figure to be added\]** — `episodes/episode_000/actions_trajectory.png` from the first ACT eval run on episode 0 of `aleksantari/g1_dex1_tool_0_sorting`.

---

### 2. Horizon-decay plot (`horizon_decay_episode_NNN.png`)

A single-panel curve. X-axis is *chunk position k*, ranging from 0 to `chunk_size − 1`. Y-axis is mean squared error.

For each chunk position `k`, the value is computed by:

1. For every frame `t` in the episode where `t + k` is still inside the episode, take the policy's prediction `chunk_t[k]` — the action it predicted (at time `t`, from observation at time `t`) for what should happen at time `t + k`.
2. Compare it against the ground-truth action `gt[t + k]` — the demonstrator's actual action `k` frames later in the episode.
3. Average the squared error over all those valid `(t, k)` pairs.

So `horizon_decay_mse[k]` answers: ***"Across the whole episode, how good is the policy at predicting `k` frames ahead from a single observation?"***

**How to read it:**

- **`k = 0`**: "Given this observation, what should happen *right now*?" Easiest prediction. This is exactly the average MSE between the blue and red traces in the prediction-stream plot above.
- **`k = chunk_size/2`**: "Given this observation, what should happen half a chunk from now?" Harder — by then the world has progressed, and if anything unexpected has happened, the policy didn't see it.
- **`k = chunk_size − 1`**: The hardest case. The policy is extrapolating most of a chunk into the future from a single snapshot.

**Shape interpretations:**

- **Sharply rising** — policy predictions degrade fast over the horizon. The policy is good at the immediate next action but loses information quickly. → Deployment should re-query often (small `n_action_steps`).
- **Mostly flat** — policy maintains accuracy across the entire chunk. → Re-querying less often is safe and saves compute.
- **Rising then plateauing** — accurate at short horizons, degrades to a stable error past some point. → The elbow is roughly where re-querying becomes worth it.

This curve is the **principled basis for choosing the deployment cadence** (`n_action_steps`) on the real robot, instead of guessing.

<!-- IMAGE PLACEHOLDER: episodes/episode_000/horizon_decay.png -->
> **\[Example figure to be added\]** — `episodes/episode_000/horizon_decay.png` from the first ACT eval run on episode 0.

## How the two plots relate

They show the same chunk data from two different angles:

| | What it slices | Visual axis |
| --- | --- | --- |
| Prediction-stream plot | All chunk[0]'s across the whole episode | Timestep on the x-axis |
| Horizon-decay plot | All chunk[k]'s averaged, for each k | Chunk position k on the x-axis |

The first point of the horizon-decay curve (`k=0`) is mathematically equal to the per-dim-averaged MSE between the blue and red traces in the prediction-stream plot. Everything to the right of `k=0` on the decay curve is information the prediction-stream plot doesn't show — because that plot only uses the first action of every chunk.

Together they form a minimum-viable diagnostic pair: **"is the policy good at the instantaneous prediction?"** (the stream plot) and **"how far ahead can it accurately predict?"** (the decay curve).

## Inference latency stats

Logged to the console at the end of each episode (and aggregated across all episodes at the end of the run). A typical line looks like:

```
[INFO] Episode 0 inference (ms): mean=23.4 std=2.1 min=22.1 max=147.2
                                 p50=23.3 p95=24.5 p99=147.2
                                 | n=600 | first=147.2 (incl. warm-up)
                                 | 30Hz budget (33.33ms): OK
```

**What this tells us:**

- The eval times every full inference pass (observation → preprocessor → forward pass → postprocessor → CPU copy) end-to-end. Because the last step forces a CUDA synchronization, the timer captures the real wall-clock latency including all GPU work.
- We compare the maximum observed latency against a **30 Hz real-time budget** (33.33 ms per frame). If `max < 33.33`, the policy can sustain 30 Hz on the real robot. If `max > 33.33`, at least one frame will miss the deadline.
- The `first` value is the cold-start time — the first inference includes CUDA kernel autotuning and JIT compilation, so it's usually several times slower than steady state. We report it separately so it doesn't poison the headline numbers.

Because the eval calls the policy on every frame (it doesn't use the queue-mechanism that real-robot deployment uses), the latency distribution is unimodal: every frame is a real forward pass, no fast queue-pop frames. This makes the stats directly comparable to a deployment budget — every observed number is a "real" inference time.

## Headline metrics (`metrics.json`)

At the end of every eval run, a single `metrics.json` is written at the root of the output directory. It is the policy's report card: stable schema, computed automatically, comparable across runs.

Top-level structure:

```json
{
  "action_dim_names": ["kLeftShoulderPitch", "kLeftShoulderRoll", ..., "kRightGripper"],
  "episodes": {
    "0":  { "mean_l2": ..., "mse_per_dim": [...], "mae_per_dim": [...], "horizon_decay_mse": [...] },
    "10": { ... },
    "20": { ... },
    "30": { ... }
  },
  "aggregate": {
    "mean_l2_mean": ...,
    "mean_l2_std": ...,
    "mse_per_dim_mean": [...],
    "horizon_decay_mse_mean": [...],
    "n_episodes": 4
  }
}
```

The top-level `action_dim_names` is a list of joint names whose ordering matches every per-dim array elsewhere in the file (so `mse_per_dim[3]` is the error for joint `action_dim_names[3]`). Names are read from the dataset's own `info.json` (which was populated from `ROBOT_CONFIGS` at conversion time), so the dataset is the source of truth — the JSON describes itself without needing a separate robot config lookup.

**Per-episode entries:**

| Field | Shape | Meaning |
| --- | --- | --- |
| `mean_l2` | scalar | Mean L2 norm of the prediction error across all frames of the episode (using the fresh-prediction stream — chunk[0] per frame). **The single-number "is this checkpoint any good?" score.** |
| `mse_per_dim` | `(action_dim,)` | Per-dimension mean squared error. Useful for spotting "the policy is fine except one gripper joint is way off." |
| `mae_per_dim` | `(action_dim,)` | Per-dimension mean absolute error. Same use as MSE but less sensitive to outliers. |
| `horizon_decay_mse` | `(chunk_size,)` | The curve plotted in the horizon-decay PNG above, as raw numbers. |

**Aggregate entries** (across all held-out episodes):

| Field | Shape | Meaning |
| --- | --- | --- |
| `mean_l2_mean` | scalar | Average of `mean_l2` across episodes. The headline score. |
| `mean_l2_std` | scalar | Standard deviation of `mean_l2` across episodes. Tells us whether the policy is consistently OK or "great on some episodes, terrible on others." |
| `mse_per_dim_mean` | `(action_dim,)` | Per-dim MSE averaged across episodes. |
| `horizon_decay_mse_mean` | `(chunk_size,)` | The cross-episode mean horizon-decay curve. The single most informative number for picking the deployment cadence. |
| `n_episodes` | int | How many episodes contributed to the aggregates. |

These numbers are also logged to the console at the end of the run so you can read them without opening the JSON.

## Saved arrays (`episodes/episode_<NNN>/predictions.npz`)

For each held-out episode, a NumPy archive is saved containing the raw measurement substrate:

| Array | Shape | Contents |
| --- | --- | --- |
| `chunks` | `(T, chunk_size, action_dim)` | The full predicted action chunk for every frame of the episode. |
| `ground_truth` | `(T, action_dim)` | The demonstrator's actual actions, frame by frame. |
| `horizon_decay_mse` | `(chunk_size,)` | The horizon-decay curve plotted above, as raw numbers. |

These files are the "post-processing kicker" — they let us derive any deployment-cadence behavior offline, after the GPU work is done. For example, to see what the policy would deploy if we set `n_action_steps = 20` on the real robot:

```python
import numpy as np
d = np.load("episodes/episode_000/predictions.npz")
k = 20
deployed_actions = np.array([d["chunks"][(t // k) * k, t % k] for t in range(len(d["ground_truth"]))])
# Now compare deployed_actions vs d["ground_truth"] for any metric you like.
```

This means we never need a second eval run to study a different cadence — one `.npz` per episode is enough material for an arbitrary number of post-hoc analyses.

## Output directory layout

Everything from a single eval run lands under one directory, by default adjacent to the policy's training checkpoint. The path is tagged with the checkpoint *step* as the deepest subdirectory so that evaluating multiple checkpoints from the same training run does not overwrite earlier results:

```
outputs/train/<training_run>/eval/<dataset_safe_name>/<step>/
├── metrics.json
└── episodes/
    ├── episode_000/
    │   ├── actions_trajectory.png      # the prediction-stream plot
    │   ├── horizon_decay.png           # the horizon-decay curve
    │   └── predictions.npz             # raw chunks + ground truth + horizon decay
    ├── episode_010/
    │   └── ... (same structure)
    └── ...
```

`<dataset_safe_name>` is the HuggingFace repo id with `/` replaced by `__` (so `aleksantari/g1_dex1_tool_0_sorting` becomes `aleksantari__g1_dex1_tool_0_sorting`). `<step>` is the checkpoint step (e.g., `005000`, `095000`) inferred from the canonical lerobot layout `checkpoints/<step>/pretrained_model`; if the policy path doesn't match that layout the step subdir is omitted.

Listing all evaluated checkpoints for one dataset is therefore one `ls` away:

```bash
ls outputs/train/<training_run>/eval/<dataset_safe_name>/
# → 005000  010000  095000  ...
```

Override the output location with `--output_dir=<path>` on the CLI if you need to write somewhere else.

## What's next (current state of the tooling)

What's already shipped:
- Per-episode prediction-stream plot (`actions_trajectory.png`)
- Per-episode horizon-decay plot (`horizon_decay.png`)
- Per-episode raw `.npz` dump
- Per-episode + aggregate inference latency stats with real-time-budget check
- Per-episode + aggregate headline metrics (`mean_l2`, per-dim MSE / MAE, horizon-decay curve) collected into a single `metrics.json` at the root of the output directory
- Structured output layout: one directory per eval run, sibling to the checkpoint that produced it

What's coming in subsequent iterations:
- Cross-episode aggregate plots (per-dim error bars, mean horizon-decay curve across episodes, per-episode mean-L2 summary).
- A three-trace per-episode plot adding the deployed-at-some-cadence trace alongside ground truth and fresh prediction.
- An "error over time" per-episode plot (L2 norm of prediction error per timestep).
- A "deployment cadence sweep" chart: average policy error plotted against `n_action_steps`, computed entirely from the saved chunks. This is the chart we'd use to actually pick the right cadence for real-robot deployment.
- For GR00T eval on the multi-task combined dataset: per-task metric breakdowns and a flag to override the language-instruction string at eval time.

The current outputs are sufficient for both qualitative assessment (the plots) and quantitative cross-checkpoint comparison (the `metrics.json`). The next iteration is mostly about derived analyses on the data we already collect.
