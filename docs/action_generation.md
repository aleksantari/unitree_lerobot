# Action generation — research report

> **Status:** research reference. Maps how this repo currently produces actions from a trained policy, what alternatives lerobot already exposes, and what the implications are for our next-step wrapper redesign. No design committed yet.

## TL;DR

- **Today, every eval script in this repo produces single-step actions only.** All three (`eval_g1.py`, `eval_g1_sim.py`, `eval_g1_dataset.py`) call [`predict_action()`](../unitree_lerobot/eval_robot/utils/utils.py) at [utils.py:35-79](../unitree_lerobot/eval_robot/utils/utils.py#L35-L79), which at its critical line ([utils.py:70](../unitree_lerobot/eval_robot/utils/utils.py#L70)) calls `policy.select_action()` and nothing else. The chunk interface that ACT and GR00T both expose is unreachable from any of our scripts.
- **Both policies expose chunks.** `ACTPolicy.predict_action_chunk` and `GrootPolicy.predict_action_chunk` return shape `(B, chunk_size, action_dim)` and are decorated `@torch.no_grad()`. They're the same forward pass that `select_action` triggers internally to refill its action queue — we just never read the rest of the chunk today.
- **ACT temporal ensembling is already implemented in lerobot.** The `ACTTemporalEnsembler` class is on the vendored submodule; it implements Algorithm 2 of the original ACT paper (exponentially-weighted online average across overlapping chunk predictions). Enabling it is a single config flag (`temporal_ensemble_coeff`) plus a `n_action_steps=1` constraint. **It's off by default.** We're not using it.
- **GR00T has no ensembling and is stochastic.** It uses a diffusion model under the hood — same observation in, potentially different chunks out. There's no analogous ensembler class for GR00T; adding one would mean external wrapping or subclassing. Also: GR00T's chunk is hard-capped at **16** timesteps by the pretrained architecture, regardless of `chunk_size` config.

In one line: *chunk-level inference is a free upgrade for both policies, ACT ensembling is a free upgrade as a config knob, GR00T smoothing is a project.*

---

## 1. Current state — how this repo generates actions today

### 1.1 The `predict_action` wrapper

Location: [`unitree_lerobot/eval_robot/utils/utils.py:35-79`](../unitree_lerobot/eval_robot/utils/utils.py#L35-L79).

```python
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
    ...
    action = policy.select_action(observation)    # line 70 — the only policy call
    action = postprocessor(action)
    action = action.squeeze(0).to("cpu")
    return action
```

**Inputs**: an observation dict, the loaded policy, target device, pre/post processor pipelines, an AMP flag, optional task string, and two toggles:
- `use_dataset` — when `True` (offline eval), skips the image normalization/CHW transpose because the dataset already provides tensors. When `False` (live robot), runs `image / 255.0` + `permute(2,0,1)`.
- `robot_type` — a string field consumed downstream by the preprocessor pipeline; defaults to `""` if not passed.

**Output**: a single action tensor of shape `(action_dim,)`, `float32`, on CPU.

**The chunk side effect**: when `select_action` is called and the policy's internal action queue is empty, the policy *does* run a full chunk forward pass internally — but it only returns the first action to the caller. The remaining `n_action_steps - 1` actions sit in the policy's `deque` until subsequent calls pop them. The wrapper has no visibility into this; if we wanted those actions, we'd have to call `predict_action_chunk` ourselves.

### 1.2 Every call site in the repo

| File | Lines | Context |
| --- | --- | --- |
| [`eval_g1.py:127-137`](../unitree_lerobot/eval_robot/eval_g1.py#L127-L137) | per-frame in the real-robot main loop at 30 Hz |
| [`eval_g1_sim.py:164-174`](../unitree_lerobot/eval_robot/eval_g1_sim.py#L164-L174) | per-frame in the sim eval loop |
| [`eval_g1_dataset.py:115-125`](../unitree_lerobot/eval_robot/eval_g1_dataset.py#L115-L125) | per-frame in the offline dataset eval loop |

All three pass `robot_type=None` and the same policy/processor objects; they differ only in `use_dataset` (`False` for live, `True` for offline). The wrapper is the single source of policy-call code we have.

### 1.3 No other action paths in the repo

A grep for direct uses of `select_action` and `predict_action_chunk` across `unitree_lerobot/` turned up nothing outside of the vendored submodule. `replay_robot.py` indexes dataset actions directly (no policy inference). Upstream lerobot's `async_inference/policy_server.py` does call `predict_action_chunk` directly to populate a server-side queue — useful prior art for "what does a chunk-oriented inference loop look like" but not used by us.

**Conclusion for §1**: chunk-level access is currently unreachable from any of our eval paths. Every frame of every eval is a single-step `select_action` call hiding a partially-discarded chunk.

---

## 2. Available state — what lerobot actually exposes

### 2.1 The two-method pattern (ACT and GR00T)

Both policies expose the *same* pair of methods. The mechanics differ only in the model's internals.

| Aspect | `ACTPolicy` | `GrootPolicy` |
| --- | --- | --- |
| `select_action(batch)` body | [`modeling_act.py:99-121`](../unitree_lerobot/lerobot/src/lerobot/policies/act/modeling_act.py#L99-L121) | [`modeling_groot.py:155-163`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/modeling_groot.py#L155-L163) |
| `select_action` return shape | `(B, action_dim)` | `(B, action_dim)` |
| `predict_action_chunk(batch)` body | [`modeling_act.py:124-133`](../unitree_lerobot/lerobot/src/lerobot/policies/act/modeling_act.py#L124-L133) | [`modeling_groot.py:124-153`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/modeling_groot.py#L124-L153) |
| `predict_action_chunk` return shape | `(B, chunk_size, action_dim)` | `(B, n_action_steps, action_dim)` — un-padded at line 151 from internal `max_action_dim=32` |
| Internal action queue | `deque(maxlen=n_action_steps)` created in `__init__`/`reset()` | same pattern, same `maxlen` |
| Queue refill trigger | inside `select_action` when empty | same |
| `@torch.no_grad()` on both methods | yes | yes |
| Extra autocast inside? | no | `torch.autocast(bf16)` wraps the GR00T model call at [`modeling_groot.py:145`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/modeling_groot.py#L145) |
| Determinism given fixed input | **deterministic** (transformer decode) | **stochastic** — diffusion reverse sampling, same obs → potentially different chunks |
| Hard horizon cap | none | **16** timesteps — enforced at [`processor_groot.py:100`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/processor_groot.py#L100): `action_horizon = min(config.chunk_size, 16)` |

### 2.2 The queue-refill mechanism

The same eight-line pattern appears in both policies. ACT's version at [`modeling_act.py:115-121`](../unitree_lerobot/lerobot/src/lerobot/policies/act/modeling_act.py#L115-L121):

```python
if len(self._action_queue) == 0:
    actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
    # (transpose so deque elements are per-timestep (B, action_dim) tensors)
    self._action_queue.extend(actions.transpose(0, 1))
return self._action_queue.popleft()
```

So `select_action` and `predict_action_chunk` are not independent paths — `select_action` is a queue-managed wrapper around `predict_action_chunk`. The full chunk is computed on the call where the queue is empty; subsequent `select_action` calls within the same chunk window are pure dequeue operations and *don't run the model at all*.

This is why the inference-latency stats we added in [`docs/offline_eval_plan.md`](offline_eval_plan.md) (the *Inference latency monitoring* section) are bimodal for chunk policies: ~1 in `n_action_steps` calls is a full forward pass; the rest are sub-millisecond dequeue ops.

### 2.3 Config knobs

| Field | ACT default | GR00T default | Meaning |
| --- | --- | --- | --- |
| `chunk_size` | 100 ([`configuration_act.py:94`](../unitree_lerobot/lerobot/src/lerobot/policies/act/configuration_act.py#L94)) | 50 ([`configuration_groot.py:31`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/configuration_groot.py#L31)) | Length of the action sequence the model outputs. **GR00T's effective limit is 16** regardless of this value. |
| `n_action_steps` | 100 ([`configuration_act.py:96`](../unitree_lerobot/lerobot/src/lerobot/policies/act/configuration_act.py#L96)) | 50 ([`configuration_groot.py:33`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/configuration_groot.py#L33)) | How many of the chunk actions get pushed into the deployment queue. |

Constraint: `n_action_steps <= chunk_size` enforced at [`configuration_act.py:153-157`](../unitree_lerobot/lerobot/src/lerobot/policies/act/configuration_act.py#L153-L157) and [`configuration_groot.py:123-126`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/configuration_groot.py#L123-L126).

**The cadence implication** worth burning in: with `n_action_steps=k`, the model forward pass runs once every `k` calls of `select_action`. Intermediate calls are free. So a "per-frame inference cost" number averaged across a full episode is heavily diluted unless you separate forward-pass frames from dequeue frames. The plan doc's *Per-bucket stats* refinement points at exactly this.

### 2.4 Base class

[`PreTrainedPolicy`](../unitree_lerobot/lerobot/src/lerobot/policies/pretrained.py) declares `select_action` and `predict_action_chunk` as abstract. The queue mechanism is **not in the base** — it's duplicated independently in `ACTPolicy` and `GrootPolicy` with identical semantics. Design note: if we ever want to override queue behavior consistently across policies (e.g. expose the cached chunk for inspection), we'd either patch both subclasses or wrap externally. There's no single seam in the base.

### 2.5 Out-of-repo prior art

Upstream lerobot's `src/lerobot/async_inference/policy_server.py` calls `predict_action_chunk` directly to fill a server-side queue, returning the chunk to a remote consumer. That's the closest existing pattern to a chunk-oriented inference loop in lerobot. Worth a read when designing our wrapper, but it solves a different problem (async client-server inference) than ours (synchronous offline eval).

---

## 3. Ensembling — beyond deploy-first-of-chunk

### 3.1 ACT temporal ensembling — already in vendored lerobot

The most useful finding of this investigation: lerobot already implements Algorithm 2 of [Zhao et al. 2023 (ACT)](https://arxiv.org/abs/2304.13705).

- **Class:** `ACTTemporalEnsembler` at [`modeling_act.py:164-252`](../unitree_lerobot/lerobot/src/lerobot/policies/act/modeling_act.py#L164-L252).
- **Algorithm**: at each environment step, the policy predicts a fresh chunk. For each future timestep `t+k` we now have *multiple* predictions (from chunks predicted at `t`, `t-1`, `t-2`, …). The ensembler maintains an exponentially-weighted online average across them: `w_i = exp(-coeff · i)`. Larger `coeff` weights newer predictions more (more reactive); smaller `coeff` weights older predictions more (more stable). The original ACT paper used `coeff = 0.01` — heavily favoring stability.
- **Online, not buffered**: the implementation at [`modeling_act.py:218-251`](../unitree_lerobot/lerobot/src/lerobot/policies/act/modeling_act.py#L218-L251) keeps a running tensor of shape `(B, chunk_size - 1, action_dim)` and folds each new chunk in incrementally, popping the front element to deploy. No need to remember every chunk ever predicted.
- **Test:** `tests/policies/test_policies.py::test_act_temporal_ensembler` verifies the online algorithm matches a naive offline implementation.

### 3.2 Enabling it

Two coordinated changes in the ACT config:

```python
ACTConfig(
    temporal_ensemble_coeff=0.01,   # was None (off); set to a positive float to enable
    n_action_steps=1,                # required when ensembling is enabled
    chunk_size=100,                  # whatever you trained with
    ...
)
```

The `n_action_steps == 1` constraint is enforced at [`configuration_act.py:148-152`](../unitree_lerobot/lerobot/src/lerobot/policies/act/configuration_act.py#L148-L152). Reason: temporal ensembling requires a fresh chunk *every step* so we have a new prediction to fold in; if `n_action_steps > 1`, the queue would serve cached actions on most steps and the ensemble would stagnate.

**Cost implication**: turning ensembling on means **every** frame triggers a forward pass, not every `n_action_steps`-th frame. For ACT at chunk_size=100 this is a 100x increase in forward passes per episode. Whether that fits in a real-time budget is a per-hardware question — RTX 5090 with bf16 on ACT is plenty, but worth measuring.

### 3.3 GR00T has no temporal ensembling

[`GrootPolicy.select_action`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/modeling_groot.py#L155-L163) uses only the linear action-queue pattern. There's no config field, no analogous ensembler class.

Two compounding factors make a GR00T ensembler more interesting than ACT's anyway:

1. **GR00T is stochastic.** Even on the same observation, two `predict_action_chunk` calls won't return identical chunks. So *some* averaging would smooth the diffusion noise — an analog of the way ensembling smooths ACT's chunk-boundary discontinuities, but for a different underlying reason.
2. **The 16-step hard cap.** GR00T's effective chunk length is bounded by the pretrained architecture, so an ensembler maintains a 16-wide running window — small enough to be trivially affordable.

But the implementation cost is higher: there's no upstream class to enable, so we'd either subclass `GrootPolicy` or wrap its calls externally with our own ensembler. Worth flagging as a future option; not the next step.

### 3.4 No other chunk-aggregation strategies in lerobot

Grep across the submodule found:

- No median-across-chunks
- No MPC / receding-horizon replanning logic (the closest analog is `n_action_steps`, which is open-loop replan-after-N rather than true MPC)
- No non-exponential weighting schemes
- No aggregation logic in the pre/post processor pipelines ([`processor_act.py`](../unitree_lerobot/lerobot/src/lerobot/policies/act/processor_act.py), [`processor_groot.py`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/processor_groot.py)) — they only handle normalization, batching, and device placement.

Two strategies exist in lerobot, period: **linear queue** (every policy) and **ACT temporal ensembling** (ACT-only, opt-in).

---

## 4. Design implications (for the wrapper redesign, not committed)

Recording the load-bearing takeaways so they're available when we plan the next step.

1. **A sibling `predict_chunk()` util is a drop-in addition.** Both policies expose `predict_action_chunk` with `@torch.no_grad()` already; we just need to package it like the existing wrapper does for `select_action`. Same observation prep, same pre/post pipelines — just call the chunk method and don't squeeze the chunk dimension.
2. **Per-policy idiosyncrasies to handle:**
   - GR00T's `predict_action_chunk` already strips the `max_action_dim=32` padding at [`modeling_groot.py:151`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/modeling_groot.py#L151), so the call-site sees 16-dim actions. No special handling needed there.
   - GR00T's internal autocast at [`modeling_groot.py:145`](../unitree_lerobot/lerobot/src/lerobot/policies/groot/modeling_groot.py#L145) means we should not pass `use_amp=True` to the wrapper for GR00T (the policy is already doing it) — or alternatively, document that double-autocast is a no-op and accept the cosmetic redundancy.
3. **GR00T stochasticity matters for offline eval reproducibility.** Calling `predict_chunk` twice on the same observation will produce different chunks. For repeatable metrics we'd want to seed the diffusion sampler, or accept and document the variance. ACT is unaffected.
4. **ACT temporal ensembling is free if we want it** — set `temporal_ensemble_coeff` in the ACT config and constrain `n_action_steps=1`. The wrapper should *not* re-implement the math; `select_action` will handle it transparently. We should make sure the new wrapper doesn't interfere with the ensembler's stateful queue (e.g. don't call `policy.reset()` between every frame, only between episodes).
5. **The `n_action_steps == 1` cost cliff** is real: enabling ensembling means a 100x increase in ACT forward passes per episode (at chunk_size=100). The plan doc's *Inference latency monitoring* section already gives us the tooling to measure this — we should baseline before/after.
6. **GR00T's 16-step hard cap is non-negotiable.** Document prominently in any future ACT-vs-GR00T comparison: if a config sets `chunk_size > 16` for GR00T, the model silently uses only the first 16 steps. The remaining bytes are wasted computation.

These bullets become the seed for the next plan — *redesigning the wrapper* — once we've discussed them.

---

## Related docs

- [`docs/offline_eval_plan.md`](offline_eval_plan.md) — the overall offline-eval redesign this research feeds into. The *Inference flow per episode* section calls for both `select_action` (closed-loop / deployment cadence) and `predict_action_chunk` (open-loop / per-frame fresh chunk) modes; the chunk wrapper from this research is the precondition. The *Inference latency monitoring* section's *Per-bucket stats* refinement is the right tool to verify the cost story in §2.2 and §3.2 of this doc.
