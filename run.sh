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
# Source datasets are read-only -- this creates a NEW dataset at aggr-repo-id.
# meta/tasks.parquet ends up with 2 rows (one task string per source).
# tyro takes list[str] as space-separated args, NOT a JSON literal.
bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.utils.aggregate_datasets \
    --repo-ids aleksantari/g1_dex1_tool_0_sorting aleksantari/g1_dex1_tool_0_handover \
    --aggr-repo-id aleksantari/g1_dex1_tools_combined \
    --push-to-hub'

# Dry run (no hub push) -- recommended first to verify the result before re-running with --push-to-hub.
bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.utils.aggregate_datasets \
    --repo-ids aleksantari/g1_dex1_tool_0_sorting aleksantari/g1_dex1_tool_0_handover \
    --aggr-repo-id aleksantari/g1_dex1_tools_combined'

# Verify the aggregated dataset matches expectations.
bash -ic 'use_conda unitree-lerobot && python test/test_aggregate_datasets.py'

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

# === Attach to running tmux sessions ===
tmux attach -t train_act          # ACT training
tmux attach -t convert_sorting    # tool_0_sorting conversion
tmux attach -t convert_handover   # tool_0_handover conversion
tmux ls                           # list all sessions
