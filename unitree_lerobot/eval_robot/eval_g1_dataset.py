"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import torch
import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
from pprint import pformat
from typing import Any
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)

from unitree_lerobot.eval_robot.utils.utils import (
    extract_observation,
    predict_action,
    OfflineEvalConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data


import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def eval_policy(
    cfg: OfflineEvalConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() != "s":
        return

    for ep_idx in cfg.episodes:
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]

        policy.reset()

        ground_truth_actions = []
        predicted_actions = []

        for step_idx in tqdm.tqdm(range(from_idx, to_idx), desc=f"episode {ep_idx}"):
            step = dataset[step_idx]
            observation = extract_observation(step)

            action = predict_action(
                observation,
                policy,
                get_safe_torch_device(policy.config.device),
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                step["task"],
                use_dataset=True,
                robot_type=None,
            )
            action_np = action.cpu().numpy()

            ground_truth_actions.append(step["action"].numpy())
            predicted_actions.append(action_np)

            if cfg.visualization:
                visualization_data(step_idx, observation, observation["observation.state"], action_np, rerun_logger)

        ground_truth_actions = np.array(ground_truth_actions)
        predicted_actions = np.array(predicted_actions)

        n_timesteps, n_dims = ground_truth_actions.shape

        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims), sharex=True)
        fig.suptitle(f"Ground Truth vs Predicted Actions — Episode {ep_idx}")

        for i in range(n_dims):
            ax = axes[i] if n_dims > 1 else axes

            ax.plot(ground_truth_actions[:, i], label="Ground Truth", color="blue")
            ax.plot(predicted_actions[:, i], label="Predicted", color="red", linestyle="--")
            ax.set_ylabel(f"Dim {i + 1}")
            ax.legend()

        axes[-1].set_xlabel("Timestep")

        plt.tight_layout()
        plt.savefig(f"figure_episode_{ep_idx:03d}.png")
        plt.close(fig)


@parser.wrap()
def eval_main(cfg: OfflineEvalConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
