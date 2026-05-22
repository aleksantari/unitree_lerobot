#!/usr/bin/env bash
# Command reference / scratchpad. Not meant to be executed top-to-bottom.
# Copy individual blocks into a terminal as needed.

# === Backfill task strings into raw JSON ===
# Writes text.goal into every episode's data.json before conversion.
# The converter then reads text.goal at convert_unitree_json_to_lerobot.py:204
# and writes it into every frame's `task` field, which lerobot dedupes into
# meta/tasks.parquet.
python /tmp/backfill_task.py /home/santari/datasets/dex1_dataset/tool_0_sorting "The left arm picks up the Kerrison rongeur from the tray, hands it to the right arm, and the right arm places it on the tool rack"
python /tmp/backfill_task.py /home/santari/datasets/dex1_dataset/tool_0_handover "The right arm picks up the Kerrison rongeur from the tool rack, hands it to the left arm, and the left arm places it on the tray"

# === Convert Unitree JSON datasets to LeRobot v3.0 format ===
# Sorting (113 episodes). Wipes the local cache for repo-id first and re-pushes to hub.
bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.utils.convert_unitree_json_to_lerobot \
    --raw-dir $HOME/datasets/dex1_dataset/tool_0_sorting \
    --repo-id aleksantari/g1_dex1_tool_0_sorting \
    --robot_type Unitree_G1_Dex1 \
    --push_to_hub'

# Handover (109 episodes).
bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.utils.convert_unitree_json_to_lerobot \
    --raw-dir $HOME/datasets/dex1_dataset/tool_0_handover \
    --repo-id aleksantari/g1_dex1_tool_0_handover \
    --robot_type Unitree_G1_Dex1 \
    --push_to_hub'

# === Run the two conversions in parallel inside dedicated tmux sessions ===
# Then `tmux attach -t convert_sorting` (Ctrl-B D to detach) to watch progress.
tmux new-session -d -s convert_sorting   -c /home/santari/repos/unitree_lerobot
tmux new-session -d -s convert_handover  -c /home/santari/repos/unitree_lerobot
tmux send-keys -t convert_sorting  "bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.utils.convert_unitree_json_to_lerobot --raw-dir \$HOME/datasets/dex1_dataset/tool_0_sorting --repo-id aleksantari/g1_dex1_tool_0_sorting --robot_type Unitree_G1_Dex1 --push_to_hub'" Enter
tmux send-keys -t convert_handover "bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.utils.convert_unitree_json_to_lerobot --raw-dir \$HOME/datasets/dex1_dataset/tool_0_handover --repo-id aleksantari/g1_dex1_tool_0_handover --robot_type Unitree_G1_Dex1 --push_to_hub'" Enter

# === Verify _parse_images parallel vs serial produces byte-identical output ===
# Used to validate the ThreadPoolExecutor change in convert_unitree_json_to_lerobot.py.
bash -ic 'use_conda unitree-lerobot && python test/test_parse_images_equivalence.py'

# === Inspect LeRobot dataset shapes / cameras / episode count ===
# Sanity-check the converted dataset before training.
bash -ic 'use_conda unitree-lerobot && python /tmp/check_dataset_dims.py'

# === Train ACT from scratch ===
# Held-out episodes are excluded via dataset.episodes in the config.
bash -ic 'use_conda unitree-lerobot && python -m lerobot.scripts.lerobot_train \
    --config_path=configs/act_g1_dex1_tool_0_sorting.json'

# === Resume ACT from latest checkpoint ===
# `last` is a lerobot-maintained symlink to the most recent step's dir.
# Loads optimizer + scheduler + step counter from training_state/ alongside pretrained_model/.
bash -ic 'use_conda unitree-lerobot && python -m lerobot.scripts.lerobot_train \
    --config_path=outputs/train/2026-05-19/19-07-27_act_g1_dex1_tool_0_sorting/checkpoints/last/pretrained_model/train_config.json \
    --resume=true'

# === Aggregate sorting + handover into a multi-task LeRobot dataset ===
# Uses the official `lerobot-edit-dataset` CLI (operation.type=merge), which
# wraps the same aggregate_datasets() function we patched locally to apply
# upstream PR #2550 (https://github.com/huggingface/lerobot/pull/2550).
# Source datasets are read-only -- this creates a NEW dataset at --repo_id.
# meta/tasks.parquet ends up with 2 rows (one task string per source).
# The CLI does NOT auto-wipe the destination cache, so we rm -rf first to
# keep re-runs idempotent.

# Dry run (no hub push) -- recommended first to verify the result.
rm -rf ~/.cache/huggingface/lerobot/aleksantari/g1_dex1_tools_combined
bash -ic 'use_conda unitree-lerobot && lerobot-edit-dataset \
    --repo_id aleksantari/g1_dex1_tools_combined \
    --operation.type merge \
    --operation.repo_ids "['"'"'aleksantari/g1_dex1_tool_0_sorting'"'"','"'"'aleksantari/g1_dex1_tool_0_handover'"'"']"'

# Verify the aggregated dataset matches expectations.
bash -ic 'use_conda unitree-lerobot && python test/test_aggregate_datasets.py'

# Publish to hub once smoke test on the combined dataset passes locally.
rm -rf ~/.cache/huggingface/lerobot/aleksantari/g1_dex1_tools_combined
bash -ic 'use_conda unitree-lerobot && lerobot-edit-dataset \
    --repo_id aleksantari/g1_dex1_tools_combined \
    --operation.type merge \
    --operation.repo_ids "['"'"'aleksantari/g1_dex1_tool_0_sorting'"'"','"'"'aleksantari/g1_dex1_tool_0_handover'"'"']" \
    --push_to_hub true'

# === Train GR00T-N1.5 on the single-task sorting dataset ===
# Uses the lerobot-gr00t env (NOT unitree-lerobot) for flash-attn/transformers/peft/decord.
# Config sets chunk_size=16, n_action_steps=16, embodiment_tag="unitree_g1",
# tune_diffusion_model=false (tier-1 projector-only starting point).
bash -ic 'use_conda lerobot-gr00t && python -m lerobot.scripts.lerobot_train \
    --config_path=configs/groot_g1_dex1_tool_0_sorting.json'

# === Train GR00T-N1.5 on the combined sorting+handover dataset (tier 1: projector only) ===
# Multi-task variant -- each batch carries one of two task strings, language conditioning live.
# tune_projector=true, tune_diffusion_model=false. Diffusion head stays at the pretrained prior.
bash -ic 'use_conda lerobot-gr00t && python -m lerobot.scripts.lerobot_train \
    --config_path=configs/groot_g1_dex1_tools_combined.json'

# === Train GR00T-N1.5 tier 2 (projector + diffusion model) on combined dataset ===
# tune_projector=true, tune_diffusion_model=true -- unlocks self.model in the action head.
# Adjustments vs tier 1 because the diffusion transformer (~hundreds of M params) is now trainable:
#   batch_size: 20 -> 12     (tier 1 was at ~85% VRAM; tier 2 adds Adam state + grad + activation memory)
#   optimizer_lr: 1e-4 -> 5e-5 (protect pretrained diffusion head from early-training disruption)
#   steps: 30000 -> 50000     (lower LR + larger trainable set converge slower; also room past the
#                              "tier 2 trails tier 1 early then crosses over ~15-25k" pattern)
# Same dataset.episodes and seed as tier 1 so eval comparison is direct.
bash -ic 'use_conda lerobot-gr00t && python -m lerobot.scripts.lerobot_train \
    --config_path=configs/groot_g1_dex1_tools_combined_tier2.json'

# === Offline eval: predict_chunk against dataset episodes ===
# Loads a checkpoint, calls policy.predict_action_chunk on every frame of the chosen episodes,
# and produces per-episode trajectory + horizon-decay plots, a predictions.npz with the raw
# (T, chunk_size, action_dim) tensor, and a metrics.json with per-dim MSE/MAE, mean_l2, and
# horizon-decay (per episode + cross-episode aggregate). Joint names (e.g. kLeftShoulderPitch)
# are pulled from the dataset's action feature schema and used as subplot ylabels + metrics.json keys.
# Outputs land at <run>/eval/<dataset_safe>/<step>/ -- step-stamped so multi-checkpoint sweeps
# don't clobber each other. Override with --output_dir=<path> if needed.
# --episodes is our hold-out [0,10,20,30] for the sorting dataset; swap to any indices to spot-check.
bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.eval_robot.eval_g1_dataset \
    --policy.path=outputs/train/2026-05-19/19-07-27_act_g1_dex1_tool_0_sorting/checkpoints/095000/pretrained_model \
    --repo_id=aleksantari/g1_dex1_tool_0_sorting \
    --episodes "[0,10,20,30]"'

# === Sweep eval across multiple ACT checkpoints ===
# Each iteration writes to its own <step>/ subfolder. Adjust the step list to whatever
# checkpoints you want to compare (use `ls outputs/train/<run>/checkpoints/` to enumerate).
for step in 005000 020000 050000 095000; do
    bash -ic "use_conda unitree-lerobot && python -m unitree_lerobot.eval_robot.eval_g1_dataset \
        --policy.path=outputs/train/2026-05-19/19-07-27_act_g1_dex1_tool_0_sorting/checkpoints/${step}/pretrained_model \
        --repo_id=aleksantari/g1_dex1_tool_0_sorting \
        --episodes \"[0,10,20,30]\""
done

# === Offline eval GR00T-N1.5 on combined dataset (lerobot-gr00t env) ===
# Same script as the ACT eval -- it's policy-agnostic. Two GR00T-specific notes:
#   1. GR00T sampling is stochastic (diffusion). Pass --seed=N to lock the sampling and
#      get reproducible metrics across re-runs of the same checkpoint.
#   2. Use explicit numeric checkpoint paths (NOT checkpoints/last/) so the eval output
#      gets a clean step subdir; "last" isn't numeric and the step subdir gets skipped.
# Combined dataset has 222 episodes: sorting offsets 0-112, handover offsets 113-221.
# Standard hold-out: [0,10,20,30] from sorting + [113,123,133,143] from handover.
bash -ic 'use_conda lerobot-gr00t && python -m unitree_lerobot.eval_robot.eval_g1_dataset \
    --policy.path=outputs/train/2026-05-20/15-28-24_groot_g1_dex1_tools_combined/checkpoints/017500/pretrained_model \
    --repo_id=aleksantari/g1_dex1_tools_combined \
    --episodes "[0,10,20,30,113,123,133,143]" \
    --seed=42'

# === Sweep eval across multiple GR00T checkpoints ===
# Same shape as the ACT sweep, just with the lerobot-gr00t env and a seed for reproducibility.
for step in 005000 010000 015000 017500; do
    bash -ic "use_conda lerobot-gr00t && python -m unitree_lerobot.eval_robot.eval_g1_dataset \
        --policy.path=outputs/train/2026-05-20/15-28-24_groot_g1_dex1_tools_combined/checkpoints/${step}/pretrained_model \
        --repo_id=aleksantari/g1_dex1_tools_combined \
        --episodes \"[0,10,20,30,113,123,133,143]\" \
        --seed=42"
done

# === Real-robot eval (eval_g1.py) — ACT on the live G1+Dex1 ===
# Requires the robot's image_server to be running (see SSH block at the bottom of this file).
# Uses the dex1-calibrated URDF + teleop-derived IK that match the data-collection setup:
#   urdf: unitree_lerobot/eval_robot/assets/g1/g1_29dof_mode_16_dex1_calib.urdf
#   ik:   G1_29_ArmIK (in robot_arm_ik.py). The hand14 backup is G1_29_ArmIK_Hand14.
# Safety layers active in the policy loop:
#   - NaN/inf guard refuses to send garbage to motors
#   - Per-frame arm-delta cap (_MAX_ARM_DELTA_PER_FRAME = 0.2 rad in eval_g1.py; bump to 0.3-0.5
#     if legit fast motion trips it -- above 0.5 the hardware velocity ceiling takes over anyway)
#   - 'q' keypress (no Enter) for emergency stop during the policy loop
#   - Ctrl+C always aborts; image_client + terminal mode are restored in finally
#   - --max_steps=N bounds the loop length
# With --soft_start=true AND --run_policy=true there are TWO 's' prompts: one before soft-start,
# one before the policy loop, so you can verify the arms reached init_arm_pose before the policy
# starts driving. Either flag alone gives just one prompt.

# --- Stage 0: camera dry-run (zero motor risk) ---
# Pulls one observation, saves cam_left_high / cam_right_high / cam_left_wrist / cam_right_wrist
# as PNGs in ./cam_dryrun/. Verifies head binocular split + 480x640 resize + wrist feeds.
# Run this whenever you change anything in make_robot.py or the image stack.
bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.eval_robot.eval_g1 \
    --policy.path=outputs/train/2026-05-19/19-07-27_act_g1_dex1_tool_0_sorting/checkpoints/095000/pretrained_model \
    --repo_id=aleksantari/g1_dex1_tool_0_sorting \
    --cam_check_only=true'

# --- Stage 1: soft-start only (arms interpolate, policy does NOT run) ---
# Linearly interpolates the arms over 3s from current pose to dataset frame 0 arm pose, then
# writes the dataset's first-frame gripper state to shared memory, then exits. Validates the
# motor command path safely. Eyeball the `current:` / `target:` / `delta:` log lines BEFORE
# the interpolation starts -- Ctrl+C at the prompt if anything looks wrong.
bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.eval_robot.eval_g1 \
    --policy.path=outputs/train/2026-05-19/19-07-27_act_g1_dex1_tool_0_sorting/checkpoints/095000/pretrained_model \
    --repo_id=aleksantari/g1_dex1_tool_0_sorting \
    --soft_start=true --run_policy=false'

# --- Stage 2: policy only (robot must already be at init pose from a prior Stage 1) ---
# Runs only the policy loop with a short cap. Gripper init still fires (independent of soft_start).
# Useful for quick re-runs without re-interpolating the arms each time.
bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.eval_robot.eval_g1 \
    --policy.path=outputs/train/2026-05-19/19-07-27_act_g1_dex1_tool_0_sorting/checkpoints/095000/pretrained_model \
    --repo_id=aleksantari/g1_dex1_tool_0_sorting \
    --soft_start=false --run_policy=true --max_steps=30'

# --- Full eval: soft-start + policy (canonical command for an actual eval session) ---
# Sequence: setup -> 1st 's' prompt -> 3s arm interpolation -> gripper init + 0.3s settle ->
# 2nd 's' prompt (verify robot reached init pose) -> policy loop with 'q' e-stop armed ->
# latency summary -> clean exit. max_steps=600 ≈ 20s at 30Hz; tune for longer rollouts.
bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.eval_robot.eval_g1 \
    --policy.path=outputs/train/2026-05-19/19-07-27_act_g1_dex1_tool_0_sorting/checkpoints/095000/pretrained_model \
    --repo_id=aleksantari/g1_dex1_tool_0_sorting \
    --soft_start=true --run_policy=true --max_steps=600'

# === Real-robot eval (eval_g1.py) — GR00T on the live G1+Dex1 ===
# Uses the unified env `unitree-lerobot-groot` (cloned from unitree-lerobot + transformers/peft/timm/
# flash_attn). The eval_g1.py script is policy-agnostic: same factories, same predict_action path,
# same observation dict -- ACT vs GR00T differs only in the checkpoint and the env that backs it.
# Safety layers are identical to the ACT block above (NaN guard, _MAX_ARM_DELTA_PER_FRAME cap,
# 'q' e-stop, Ctrl+C, max_steps bound, two 's' prompts when both flags set).
#
# Expectations specific to GR00T:
#   - GR00T-N1.5 is ~3B params -- single forward pass on RTX 5090 with bf16 is ~50-200 ms.
#     The script's 30Hz budget check WILL be marked BUSTED on the slow-of-chunk frames. This is
#     a label, not a correctness bug -- policy.select_action uses an internal chunk queue so only
#     every Nth step triggers a forward pass; the rest pop a cached action.
#   - First step is always slowest (CUDA graph compile, kernel autotune). log_inference_times
#     labels it "incl. warm-up".
#   - First-run launch is the chosen tier1 checkpoint (projector-only tune from the combined dataset).
#     Use the single-task sorting repo_id for the obs spec + init pose; the model has seen this
#     task string in training so language conditioning lines up.
#   - If --max_steps trips the per-frame delta cap, the cap (0.25 rad in eval_g1.py) may need a
#     small bump (~0.3-0.4) -- see the ACT block's notes for the rationale.

# --- Stage 0: camera dry-run (zero motor risk) ---
bash -ic 'use_conda unitree-lerobot-groot && python -m unitree_lerobot.eval_robot.eval_g1 \
    --policy.path=outputs/train/2026-05-20/15-28-24_groot_g1_dex1_tools_combined/checkpoints/017500/pretrained_model \
    --repo_id=aleksantari/g1_dex1_tool_0_sorting \
    --cam_check_only=true'

# --- Stage 1: soft-start only (arms interpolate, policy does NOT run) ---
bash -ic 'use_conda unitree-lerobot-groot && python -m unitree_lerobot.eval_robot.eval_g1 \
    --policy.path=outputs/train/2026-05-20/15-28-24_groot_g1_dex1_tools_combined/checkpoints/017500/pretrained_model \
    --repo_id=aleksantari/g1_dex1_tool_0_sorting \
    --soft_start=true --run_policy=false'

# --- Stage 2: policy only (robot must already be at init pose from a prior Stage 1) ---
# max_steps=30 keeps the first GR00T-on-robot test to ~1s of motion -- finger on 'q' / Ctrl+C.
bash -ic 'use_conda unitree-lerobot-groot && python -m unitree_lerobot.eval_robot.eval_g1 \
    --policy.path=outputs/train/2026-05-20/15-28-24_groot_g1_dex1_tools_combined/checkpoints/017500/pretrained_model \
    --repo_id=aleksantari/g1_dex1_tool_0_sorting \
    --soft_start=false --run_policy=true --max_steps=30'

# --- Full eval: soft-start + policy (canonical GR00T-on-robot command) ---
bash -ic 'use_conda unitree-lerobot-groot && python -m unitree_lerobot.eval_robot.eval_g1 \
    --policy.path=outputs/train/2026-05-20/15-28-24_groot_g1_dex1_tools_combined/checkpoints/017500/pretrained_model \
    --repo_id=aleksantari/g1_dex1_tool_0_sorting \
    --soft_start=true --run_policy=true --max_steps=600'

# === Attach to running tmux sessions ===
tmux attach -t train_act          # ACT training
tmux attach -t convert_sorting    # tool_0_sorting conversion
tmux attach -t convert_handover   # tool_0_handover conversion
tmux ls                           # list all sessions




# this is the image server
ssh unitree@192.168.123.164
123
cd teleimager 
ZED_Explorer -a
python -m teleimager.image_server

# if want to use neck joint
# python neck_server.py --bind 0.0.0.0:5555

# check the image connection
conda activate tv
cd teleimager/src/teleimager
python image_client.py