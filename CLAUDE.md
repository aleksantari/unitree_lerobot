# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo shape

This repo is a thin wrapper around HuggingFace LeRobot, specialized for Unitree robots (G1 humanoid + various end-effectors, Z1 arm). It has three concerns:

1. **Convert** Unitree's native JSON datasets (from `avp_teleoperate`) into LeRobot format.
2. **Train** standard LeRobot policies (ACT, Diffusion, Pi0, Pi0.5, GR00T) on those datasets — training itself is delegated entirely to upstream `lerobot`.
3. **Evaluate** the trained policy on the real G1 robot (or in `unitree_sim_isaaclab`), including DDS communication to motors and hands.

`unitree_lerobot/lerobot/` is a **git submodule** pinned to HF lerobot. Don't edit files there — they belong to upstream. Initialize with `git submodule update --init --recursive` after cloning.

## Environment

The conda env for this project is **`unitree_lerobot`** (Python 3.10). Always activate it via the lane-based wrapper:

```bash
bash -ic 'use_conda unitree_lerobot && <command>'
```

ROS 2 and conda are kept apart with `use_conda` / `use_ros2` / `reset_lane` (see global CLAUDE.md). Never call `conda activate` directly.

System dep that bites: ffmpeg must come from conda-forge (`conda install ffmpeg=7.1.1 -c conda-forge`) — torchcodec needs `libsvtav1` and the system ffmpeg won't have it.

The submodule's lerobot must be installed first (`pip install -e unitree_lerobot/lerobot`) before `pip install -e .` at the repo root.

## Layout

| Path | Purpose |
| --- | --- |
| `unitree_lerobot/utils/constants.py` | `ROBOT_CONFIGS` dict — single source of truth for motor names, camera names, JSON state/action field names per robot variant. **Adding a new G1 variant starts here.** |
| `unitree_lerobot/utils/convert_unitree_json_to_lerobot.py` | Unitree JSON → LeRobot v2.x dataset. Reads `ROBOT_CONFIGS[robot_type]`. |
| `unitree_lerobot/utils/convert_unitree_json_to_h5.py`, `convert_lerobot_to_h5.py` | Sideways conversions for compatibility with other frameworks. |
| `unitree_lerobot/utils/sort_and_rename_folders.py` | Pre-conversion: makes `episode_0..N` contiguous. |
| `data_editor/data_editor_EN.py` | PyQt5 GUI to trim/delete episodes before conversion (needs `pip install PyQt5`). |
| `unitree_lerobot/eval_robot/eval_g1.py` | Real-robot eval entry point — loads a trained policy and runs inference at fixed Hz. |
| `unitree_lerobot/eval_robot/eval_g1_sim.py` | Same, but for `unitree_sim_isaaclab`. Adds optional data-recording during inference. |
| `unitree_lerobot/eval_robot/eval_g1_dataset.py` | Offline eval: policy vs. dataset trajectories, no robot. |
| `unitree_lerobot/eval_robot/replay_robot.py` | Replays a recorded episode straight to the robot (no policy) — useful for sanity-checking joint mappings. |
| `unitree_lerobot/eval_robot/robot_control/` | Hardware abstraction: `robot_arm.py`, IK (`robot_arm_ik.py`), per-hand modules (`robot_hand_unitree.py` for Dex1/Dex3, `robot_hand_inspire.py`, `robot_hand_brainco.py`), and `mobile_control.py` for moveable-lift base. |
| `unitree_lerobot/eval_robot/image_server/` | Image streaming server used during eval (must be running on the robot — see avp_teleoperate README). |
| `test/` | Smoke tests for dataset loading (`test_load_dataset.py`, `test_load_h5.py`, `test_local_push_to_hub.py`). Not a real test suite. |

## Robot variants

The `--robot_type` flag selects an entry in `ROBOT_CONFIGS`. Each entry pins motor list, camera list, the mapping from raw camera keys (`color_0`, `color_1`, ...) to semantic names (`cam_high`, `cam_left_wrist`, ...), and which fields in `data.json` carry state vs. action. Supported keys today:

```
Unitree_Z1_Single, Unitree_Z1_Dual,
Unitree_G1_Dex1, Unitree_G1_Dex1_Sim, Unitree_G1_Dex3,
Unitree_G1_Brainco, Unitree_G1_Inspire,
Unitree_G1_MoveibleLift_Dex1_UseWaist / _NoUseWaist,
Unitree_G1_Lift_Dex1_UseWaist / _NoUseWaist
```

`Unitree_G1_Dex1_Sim` is the single-head-cam sim variant — use it for `unitree_sim_isaaclab` data only.

## Common workflows

**Convert a freshly-collected dataset:**

```bash
python unitree_lerobot/utils/sort_and_rename_folders.py --data_dir $HOME/datasets/<task>
python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets/<task> \
    --repo-id <hf_user>/<task> \
    --robot_type Unitree_G1_Dex3 \
    [--push_to_hub]
```

**Train (all training delegated to lerobot, run from the submodule dir):**

```bash
cd unitree_lerobot/lerobot
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=<hf_user>/<task> \
    --policy.type=act|diffusion|pi0|pi05|groot \
    --policy.push_to_hub=false
```

Pi05 example in the README uses `--policy.pretrained_path=lerobot/pi05_base`, `--policy.compile_model=true`, `--policy.gradient_checkpointing=true`, `--policy.dtype=bfloat16`. On this machine (RTX 5090, 32GB VRAM, 32 threads, 60GB RAM) defaults are tuned for much smaller hardware — actively raise `num_workers`, `batch_size`, and watch `dataloading_s` vs. `update_s` for under-provisioned dataloaders.

**Real-robot eval:** start image_server on the robot first (see avp_teleoperate), then:

```bash
python unitree_lerobot/eval_robot/eval_g1.py \
    --policy.path=<checkpoint_dir>/pretrained_model \
    --repo_id=<dataset_for_obs_spec> \
    --arm=G1_29 --ee=dex3 --frequency=30 \
    --send_real_robot=true
```

`--send_real_robot=false` is dry-run mode. `--arm` ∈ {G1_29, G1_23}. `--ee` ∈ {dex3, dex1, inspire1, brainco}.

## Lint / format

Pre-commit is configured with `ruff` (format + lint, line-length 120, py310 target), pyupgrade, gitleaks, bandit, and prettier for markdown. Run before committing:

```bash
bash -ic 'use_conda unitree_lerobot && pre-commit run --all-files'
```

`pyproject.toml`'s `[tool.ruff.lint]` block is commented out — only formatting + fixes are enforced today.

## Submodule discipline

`unitree_lerobot/lerobot/` tracks HF lerobot at a pinned commit (README references `0878c68` for the v0.1 baseline; v0.3 moved to LeRobot dataset format v3.0). If you change submodule commits, you are changing what `pip install -e` resolves. Keep upstream changes in upstream; put Unitree-specific overrides in this repo instead.
