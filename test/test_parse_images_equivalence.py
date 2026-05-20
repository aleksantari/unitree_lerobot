"""Verify the parallel _parse_images produces byte-identical output to the serial version."""

import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, "/home/santari/repos/unitree_lerobot")
from unitree_lerobot.utils.convert_unitree_json_to_lerobot import JsonDataset


def parse_images_serial(self, episode_path, episode_data):
    """Original sequential implementation, inlined here as the reference."""
    images = defaultdict(list)
    keys = episode_data["data"][0]["colors"].keys()
    cameras = [key for key in keys if "depth" not in key]
    for camera in cameras:
        image_key = self.camera_to_image_key.get(camera)
        if image_key is None:
            continue
        for sample_data in episode_data["data"]:
            relative_path = sample_data["colors"].get(camera)
            if not relative_path:
                continue
            image_path = os.path.join(episode_path, relative_path)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path does not exist: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.resize(image_rgb, (640, 480), interpolation=cv2.INTER_AREA)
            images[image_key].append(image_rgb)
    return images


ds = JsonDataset(
    "/home/santari/datasets/dex1_dataset/tool_0_handover",
    "Unitree_G1_Dex1",
)
ep_path = ds.episode_paths[0]
ep_data = ds.episodes_data_cached[0]
n_frames = len(ep_data["data"])

print(f"Episode:    {ep_path}")
print(f"Frames:     {n_frames}")
print()

t0 = time.time()
serial = parse_images_serial(ds, ep_path, ep_data)
t_serial = time.time() - t0

t0 = time.time()
parallel = ds._parse_images(ep_path, ep_data)
t_parallel = time.time() - t0

n_imgs = sum(len(v) for v in serial.values())
print(f"serial:     {t_serial:6.2f}s  ({n_imgs} images, {n_imgs / t_serial:6.1f} img/s)")
print(f"parallel:   {t_parallel:6.2f}s  ({n_imgs} images, {n_imgs / t_parallel:6.1f} img/s)")
print(f"speedup:    {t_serial / t_parallel:5.2f}x")
print()

assert serial.keys() == parallel.keys(), f"key mismatch: serial={list(serial)} parallel={list(parallel)}"

for k in serial:
    assert len(serial[k]) == len(parallel[k]), (
        f"length mismatch for {k}: serial={len(serial[k])} parallel={len(parallel[k])}"
    )
    for i, (a, b) in enumerate(zip(serial[k], parallel[k])):
        assert a.shape == b.shape, f"shape mismatch at {k}[{i}]: {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"dtype mismatch at {k}[{i}]: {a.dtype} vs {b.dtype}"
        assert np.array_equal(a, b), f"pixel mismatch at {k}[{i}]"

print(f"PASS: serial and parallel produce identical output across {n_imgs} images.")
