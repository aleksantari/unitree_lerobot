# Training pipeline — future improvements

> A living doc of proposed improvements to the training pipeline, captured for later so we don't lose the reasoning. Each entry states the problem, the proposal, the evidence that justifies it, and the open questions. **Nothing here is implemented yet** — revisit before the next big training run.

---

## 1. Log sampler-faithful eval metrics to wandb during training

**Status:** proposed, not implemented (2026-06-12).

### The problem: the training loss is a proxy, blind to deployed behavior

The wandb training loss for GR00T-N1.5 is **not** "MSE of the predicted chunk vs the GT chunk." The action head is **flow-matching**, so the loss is a **velocity-regression** loss at a random noise level:

```
sample noise level τ ~ Beta(1.5, 1.0),  sample noise ε
interpolant     x_τ = τ·a_gt + (1−τ)·ε
target velocity u   = a_gt − ε
loss = ‖ v_θ(x_τ, τ, obs) − u ‖²      ← ONE forward pass, at a random τ
```

(Training path: `flow_matching_action_head.py` `forward()` / Beta-sampled `τ`. Inference path: `get_action()` integrates the velocity field over `num_inference_timesteps` (=4) steps — a different computation.)

Three consequences, all of which matter:

1. **It never runs the sampler.** The loss is a single forward pass at one noise level; the action we actually deploy comes from integrating the field over 4 steps. The loss never sees the sampled chunk.
2. **It's in velocity space, not action space.** Its scale depends on the noise schedule; it isn't interpretable as radians of error and isn't cleanly comparable across configs.
3. **It pools all 16 chunk positions and all τ** — no horizon resolution, no notion of deployment cadence or boundary jerk.

So a falling loss tells you the velocity field is fitting; it does **not** tell you whether the *integrated rollout* (what the robot executes) is getting more coherent, less stale, or smoother at chunk boundaries.

### The proposal

Add a **periodic, held-out, sampler-based** eval hook to the training loop that logs a few compact scalars to wandb every `save_freq` steps:

- `hd[0]` (= fresh `mean_l2`) — loss-aligned, for calibration.
- `deployed@chunk_size` — realistic deployment error at the deployment cadence.
- **drift slope** `hd[K-1] − hd[0]` — chunk coherence (how fast the prediction degrades over the horizon); the quantity the loss structurally cannot see.
- **mean chunk-boundary jerk** (deg) — deployment smoothness.
- optionally the full horizon-decay curve as a periodic wandb line plot.

### Why it's genuinely new information

| | what it measures | runs sampler? | space | resolved by |
|---|---|---|---|---|
| **train loss** | velocity regression at random τ | **no** | velocity | nothing (pooled) |
| **horizon decay** `hd[k]` | sampled chunk error vs GT | **yes** | action | position k |
| **deployed@k** | sampled deployment error at cadence k | **yes** | action | (mean of hd over 0..k−1) |
| **boundary jerk** | action discontinuity at re-query | **yes** | action (deg) | per arm joint |

The gap between row 1 and rows 2–4 is the gap between "is the velocity field well-fit" and "does the integrated rollout behave well." These can diverge — and we have measured them diverging (below).

### Empirical justification (the clincher)

In the **Tier 1 (projector-only) @12.5k vs Tier 2_fixed (proj+diffusion) @27.5k** comparison (episodes 3, 13, 26), the **loss-aligned quantity was nearly flat while the sampler-based metrics showed a large, consistent difference**:

| metric | Tier 1 | Tier 2_fixed | Δ |
|---|---|---|---|
| `fresh mean_l2` (= `hd[0]`, most loss-like) | 0.1496 | 0.1469 | **−1.8%** (flat) |
| `deployed@16` | 0.2222 | 0.1973 | −11% |
| `hd[15]` (far-horizon error) | 0.3032 | 0.2530 | **−16.6%** |
| drift slope `hd[15] − hd[0]` | 0.1537 | 0.1061 | **31% flatter** |
| mean boundary jerk | 2.22° | 2.03° | −8.4% |

The two models predict the *immediate* action equally well (flat `hd[0]`), but Tier 2_fixed's chunks **drift far less over the horizon** — the horizon-decay curves start together and fan apart (see `docs/images/horizon_decay_tier1_vs_tier2_fixed_ep3-13-26.png`). A scalar velocity-space loss would have shown "about the same"; the deployable behavior (coherence, boundary jerk) was materially different. **That difference is exactly what these metrics surface and the loss hides.**

Supporting evidence — the metrics also produce a **clean, monotone training trend**: across Tier 2_fixed checkpoints 17.5k → 22.5k → 27.5k, mean boundary jerk fell 2.64° → 2.21° → 2.03° (every one of 14 arm joints improved) and `mean_l2` 0.1946 → 0.1469 — i.e. these are stable enough to track over training, not just noise.

### Decisions it would change (the "would it change anything?" test)

- **Checkpoint selection** by `deployed@k` / boundary jerk instead of lowest loss.
- **Overfitting / early-stopping**: held-out `hd`/`deployed` rising while train loss still falls — the loss on training data cannot show this.
- **Config diagnosis**: did a change improve *chunk coherence* specifically (the axis the loss can't see)?

### Conditions / caveats (important — these bound the value)

1. **Eval-time, not per-step.** It runs the full sampler on a batch of episodes (~100s of ms each). Compute every `save_freq`/`eval_freq`, not every gradient step. Cheap vs a 2500-step interval, expensive vs the loss.
2. **Held-out, not training data.** On *training* episodes it tracks "fitting the training rollouts better," keeps improving through overfitting, and adds little beyond the loss. On the **hold-out** (`[0,10,20,30,113,123,133,143]`) it becomes a generalization + deployment-quality tracker and is the thing that catches overfitting the loss can't. (Note: the assessments that motivated this were on training-data episodes — on the hold-out is where the metric earns its keep.)
3. **Compact scalars, avoid the noisy `max`-type stats** (per-joint max is a single extreme event — very noisy). The `mean`/curve/slope quantities are stable.
4. **Enough episodes for a stable signal vs cost** — a trade-off to tune.

### Feasibility (low effort)

The functions already exist in `unitree_lerobot/eval_robot/eval_g1_dataset.py` and are pure-numpy on data we'd have in hand:

- `_horizon_decay_l2(predicted_chunks, ground_truth)`
- `_build_deployed_stream(predicted_chunks, ground_truth, k)`
- `_boundary_jump_stats(deployed, action_dim_names, k)`

Add a periodic hook in the lerobot training loop (currently `eval_freq=0`, disabled — there's no sim env) that, every `save_freq`: loads a held-out batch, runs `policy.predict_action_chunk` per frame, calls the three functions, and `wandb.log({...})`. Same code, new caller.

### Open questions to resolve before implementing

- How many held-out episodes give a stable-enough signal at acceptable cost?
- Which exact scalars to commit to as the standing wandb panel?
- Does the lerobot train loop expose a clean periodic hook, or do we compute this out-of-band (e.g. a callback / separate process reading the latest checkpoint)?
- Is the GR00T forward-pass cost at `save_freq` cadence acceptable, or do we subsample frames per episode?

### See also

- `docs/offline_eval_current.md` — the offline eval that defines all these metrics (plots 1–5, `metrics.json` schema).
- `docs/action_generation.md` — `select_action` vs `predict_action_chunk`, the flow-matching sampler, the chunk queue.
