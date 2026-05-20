"""
Aggregate multiple LeRobot datasets into a single combined dataset.

The output dataset preserves each source's task strings as distinct rows in
meta/tasks.parquet, so frames from each source retain their own task_index.
Source datasets are NOT modified -- only read.

python -m unitree_lerobot.utils.aggregate_datasets \
    --repo-ids '["aleksantari/g1_dex1_tool_0_sorting", "aleksantari/g1_dex1_tool_0_handover"]' \
    --aggr-repo-id aleksantari/g1_dex1_tools_combined \
    --push-to-hub
"""

import shutil

import tyro

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def main(
    repo_ids: list[str],
    aggr_repo_id: str,
    push_to_hub: bool = False,
) -> None:
    aggr_root = HF_LEROBOT_HOME / aggr_repo_id
    if aggr_root.exists():
        shutil.rmtree(aggr_root)

    aggregate_datasets(repo_ids=repo_ids, aggr_repo_id=aggr_repo_id)

    if push_to_hub:
        dataset = LeRobotDataset(repo_id=aggr_repo_id, root=aggr_root)
        dataset.push_to_hub(upload_large_folder=True)


if __name__ == "__main__":
    tyro.cli(main)
