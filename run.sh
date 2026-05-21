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

# === Train GR00T-N1.5 on the combined sorting+handover dataset ===
# Multi-task variant -- each batch carries one of two task strings, language conditioning live.
bash -ic 'use_conda lerobot-gr00t && python -m lerobot.scripts.lerobot_train \
    --config_path=configs/groot_g1_dex1_tools_combined.json'

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

# === Attach to running tmux sessions ===
tmux attach -t train_act          # ACT training
tmux attach -t convert_sorting    # tool_0_sorting conversion
tmux attach -t convert_handover   # tool_0_handover conversion
tmux ls                           # list all sessions
