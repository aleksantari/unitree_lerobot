"""
Script Json to Lerobot.

# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire)
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_double_hand \
    --robot_type Unitree_G1_Dex3 \
    --push_to_hub
"""

import os
import cv2
import tqdm
import tyro
import json
import glob
import dataclasses
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import ROBOT_CONFIGS


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class JsonDataset:
    def __init__(self, data_dirs: Path, robot_type: str) -> None:
        """
        Initialize the dataset for loading and processing HDF5 files containing robot manipulation data.

        Args:
            data_dirs: Path to directory containing training data
        """
        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = data_dirs
        self.json_file = "data.json"

        # Initialize paths and cache
        self._init_paths()
        self._init_cache()
        self.json_state_data_name = ROBOT_CONFIGS[robot_type].json_state_data_name
        self.json_action_data_name = ROBOT_CONFIGS[robot_type].json_action_data_name
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key

    def _init_paths(self) -> None:
        """Initialize episode and task paths."""

        self.episode_paths = []
        self.task_paths = []

        def is_episode_dir(p: str) -> bool:
            return os.path.isdir(p) and os.path.isfile(os.path.join(p, self.json_file))

        # Support both layouts:
        # - raw_dir/<episode>/data.json
        # - raw_dir/<task>/<episode>/data.json
        top_level = sorted(glob.glob(os.path.join(self.data_dirs, "*")))

        # Case A: episodes directly under raw_dir.
        direct_episode_dirs = [p for p in top_level if is_episode_dir(p)]
        if direct_episode_dirs:
            self.episode_paths.extend(direct_episode_dirs)
            # Keep a single "task" entry for consistency (not used elsewhere).
            self.task_paths.append(str(self.data_dirs))
        else:
            # Case B: tasks under raw_dir.
            for task_path in top_level:
                if not os.path.isdir(task_path):
                    continue

                candidate_episode_paths = sorted(glob.glob(os.path.join(task_path, "*")))
                episode_dirs = [p for p in candidate_episode_paths if is_episode_dir(p)]
                if episode_dirs:
                    self.task_paths.append(task_path)
                    self.episode_paths.extend(episode_dirs)

        self.episode_paths = sorted(self.episode_paths)
        self.episode_ids = list(range(len(self.episode_paths)))

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episode_paths)

    def _init_cache(self) -> list:
        """Initialize data cache if enabled."""

        self.episodes_data_cached = []
        for episode_path in tqdm.tqdm(self.episode_paths, desc="Loading Cache Json"):
            json_path = os.path.join(episode_path, self.json_file)
            with open(json_path, encoding="utf-8") as jsonf:
                self.episodes_data_cached.append(json.load(jsonf))

        print(f"==> Cached {len(self.episodes_data_cached)} episodes")

        return self.episodes_data_cached

    def _extract_data(self, episode_data: dict, key: str, parts: list[str]) -> np.ndarray:
        """
        Extract data from episode dictionary for specified parts.

        Args:
            episode_data: Dictionary containing episode data
            key: Data key to extract ('states' or 'actions')
            parts: List of parts to include ('left_arm', 'right_arm')

        Returns:
            Concatenated numpy array of the requested data
        """
        result = []
        for sample_data in episode_data["data"]:  # interate over every frame in episode
            data_array = np.array([], dtype=np.float32)  # per frame accumulator
            for part in parts:  # left_arm, right_arm..
                key_parts = part.split(".")
                qpos = None
                for key_part in key_parts:
                    if qpos is None and key_part in sample_data[key] and sample_data[key][key_part] is not None:
                        qpos = sample_data[key][key_part]
                    else:
                        if qpos is None:
                            raise ValueError(f"qpos is None for part: {part}")
                        qpos = qpos[key_part]
                if qpos is None:
                    raise ValueError(f"qpos is None for part: {part}")
                if isinstance(qpos, list):
                    qpos = np.array(qpos, dtype=np.float32).flatten()
                else:
                    qpos = np.array([qpos], dtype=np.float32).flatten()
                data_array = np.concatenate([data_array, qpos])
            result.append(data_array)
        return np.array(result)

    @staticmethod
    def _load_and_process_image(image_path: str) -> np.ndarray:
        # Downsample to the schema target (480, 640). INTER_AREA is the right choice
        # for shrinks; head cams at 1280x720 get squished 16:9 -> 4:3.
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image_rgb, (640, 480), interpolation=cv2.INTER_AREA)

    def _parse_images(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack images for each camera, decoding in parallel across files."""

        keys = episode_data["data"][0]["colors"].keys()
        cameras = [key for key in keys if "depth" not in key]

        # Build (image_key, slot_index, image_path) plan, preserving order.
        plan: list[tuple[str, int, str]] = []
        counts: dict[str, int] = {}
        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)
            if image_key is None:
                continue
            counts.setdefault(image_key, 0)
            for sample_data in episode_data["data"]:
                relative_path = sample_data["colors"].get(camera)
                if not relative_path:
                    continue
                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")
                plan.append((image_key, counts[image_key], image_path))
                counts[image_key] += 1

        # Preallocate slots so workers can fill them out-of-order without races.
        images: dict[str, list[np.ndarray | None]] = {k: [None] * n for k, n in counts.items()}

        # cv2.imread releases the GIL, so threads (not processes) are sufficient.
        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = {ex.submit(self._load_and_process_image, p): (k, i) for (k, i, p) in plan}
            for fut in futures:
                k, i = futures[fut]
                images[k][i] = fut.result()

        return images

    def get_item(
        self,
        index: int | None = None,
    ) -> dict:
        """Get a training sample from the dataset."""

        file_path = np.random.choice(self.episode_paths) if index is None else self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]

        # Load state and action data
        action = self._extract_data(episode_data, "actions", self.json_action_data_name)
        state = self._extract_data(episode_data, "states", self.json_state_data_name)
        episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else state.shape[0]

        # Load task description
        task = episode_data.get("text", {}).get("goal", "")

        # Load camera images
        cameras = self._parse_images(file_path, episode_data)

        # Extract camera configuration
        cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            "camera_names": list(cameras.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }

        return {
            "episode_index": index,
            "episode_length": episode_length,
            "state": state,
            "action": action,
            "cameras": cameras,
            "task": task,
            "data_cfg": data_cfg,
        }


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = ROBOT_CONFIGS[robot_type].motors
    cameras = ROBOT_CONFIGS[robot_type].cameras

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (480, 640, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    robot_type: str,
) -> LeRobotDataset:
    json_dataset = JsonDataset(raw_dir, robot_type)
    for i in tqdm.tqdm(range(len(json_dataset))):
        episode = json_dataset.get_item(i)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]

        num_frames = episode_length
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in cameras.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            frame["task"] = task

            dataset.add_frame(frame)
        dataset.save_episode()

    return dataset


def json_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,  # e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
    )

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)


def local_push_to_hub(
    repo_id: str,
    root_path: Path,
):
    dataset = LeRobotDataset(repo_id=repo_id, root=root_path)
    dataset.push_to_hub(upload_large_folder=True)


if __name__ == "__main__":
    tyro.cli(json_to_lerobot)
