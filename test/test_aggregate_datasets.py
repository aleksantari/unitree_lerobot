"""Verify aggregate_datasets produced a sane combined LeRobot dataset.

Assumes you've already run:
    python -m unitree_lerobot.utils.aggregate_datasets \
        --repo-ids aleksantari/g1_dex1_tool_0_sorting aleksantari/g1_dex1_tool_0_handover \
        --aggr-repo-id aleksantari/g1_dex1_tools_combined

Then this script loads metadata for both sources + the aggregate and asserts:
  - total_episodes(aggregate) == sum(total_episodes(sources))
  - total_frames(aggregate)   == sum(total_frames(sources))
  - features / fps / robot_type all match the sources
  - meta/tasks.parquet has exactly one row per unique source task string
"""

import sys

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

SOURCES = [
    "aleksantari/g1_dex1_tool_0_sorting",
    "aleksantari/g1_dex1_tool_0_handover",
]
AGGREGATE = "aleksantari/g1_dex1_tools_combined"


def load_or_die(repo_id: str) -> LeRobotDatasetMetadata:
    try:
        return LeRobotDatasetMetadata(repo_id)
    except FileNotFoundError as e:
        sys.exit(
            f"FAIL: could not load {repo_id} from local cache.\n"
            f"       Run the aggregator (and the source conversions) first.\n"
            f"       Underlying: {e}"
        )


src_metas = [load_or_die(r) for r in SOURCES]
dst_meta = load_or_die(AGGREGATE)

expected_episodes = sum(m.total_episodes for m in src_metas)
expected_frames = sum(m.total_frames for m in src_metas)
expected_tasks = set()
for m in src_metas:
    expected_tasks.update(m.tasks.index.tolist())

print(f"--- sources ---")
for m in src_metas:
    print(f"  {m.repo_id}: {m.total_episodes} episodes, {m.total_frames} frames, {len(m.tasks)} tasks, fps={m.fps}")
print(f"--- aggregate ---")
print(f"  {dst_meta.repo_id}: {dst_meta.total_episodes} episodes, {dst_meta.total_frames} frames, {len(dst_meta.tasks)} tasks, fps={dst_meta.fps}")
print()

assert dst_meta.total_episodes == expected_episodes, (
    f"episodes mismatch: aggregate={dst_meta.total_episodes} expected={expected_episodes}"
)
assert dst_meta.total_frames == expected_frames, (
    f"frames mismatch: aggregate={dst_meta.total_frames} expected={expected_frames}"
)
assert dst_meta.fps == src_metas[0].fps, (
    f"fps mismatch: aggregate={dst_meta.fps} sources={src_metas[0].fps}"
)
assert dst_meta.robot_type == src_metas[0].robot_type, (
    f"robot_type mismatch: aggregate={dst_meta.robot_type} sources={src_metas[0].robot_type}"
)
assert dst_meta.features.keys() == src_metas[0].features.keys(), (
    f"feature keys differ: aggregate={set(dst_meta.features)} sources={set(src_metas[0].features)}"
)

aggregate_tasks = set(dst_meta.tasks.index.tolist())
assert aggregate_tasks == expected_tasks, (
    f"task strings differ:\n"
    f"  aggregate: {aggregate_tasks}\n"
    f"  expected:  {expected_tasks}\n"
    f"  missing:   {expected_tasks - aggregate_tasks}\n"
    f"  extra:     {aggregate_tasks - expected_tasks}"
)
assert len(dst_meta.tasks) == len(SOURCES), (
    f"expected one task per source ({len(SOURCES)}), got {len(dst_meta.tasks)} task rows"
)

print(f"PASS: aggregate matches sources on episodes, frames, fps, robot_type, features, and tasks.")
print(f"Tasks in meta/tasks.parquet:")
for task_str in dst_meta.tasks.index:
    print(f"  - {task_str!r}")
