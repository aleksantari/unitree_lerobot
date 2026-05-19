#!/usr/bin/env bash
set -euo pipefail

# Convert Unitree G1 + Dex1 JSON dataset to LeRobot format. (Already run.)
# bash -ic 'use_conda unitree-lerobot && python -m unitree_lerobot.utils.convert_unitree_json_to_lerobot \
#     --raw-dir $HOME/datasets/dex1_dataset/tool_0_sorting \
#     --repo-id aleksantari/g1_dex1_tool_0_sorting \
#     --robot_type Unitree_G1_Dex1 \
#     --push_to_hub'

# Train ACT on the converted dataset.
# Held-out episodes {0, 10, 20, 30} are excluded via dataset.episodes in the config.
bash -ic 'use_conda unitree-lerobot && python -m lerobot.scripts.lerobot_train \
    --config_path=configs/act_g1_dex1_tool_0_sorting.json'
