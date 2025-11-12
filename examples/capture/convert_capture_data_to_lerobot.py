"""
Conversion utility for UMI capture datasets to the LeRobot format.

Example usage:
uv run examples/capture/convert_capture_data_to_lerobot.py \
  --args.data-dir /storage/nfs/xzhang/data/capture/20251106_pipe_insertion_cactus_capture_device_unimanual \
  --args.output-dir /storage/nfs/xzhang/data/lerobot \
  --args.repo-name pipe_insertion_samples

The script expects a directory layout where each episode lives under:
raw/<collection_id>/data/episode_*/data.hdf5
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import h5py
import numpy as np
import tyro

from openpi.utils import rai_lerobot_conversion_utils as utils


def _episode_paths(data_dir: Path) -> list[Path]:
    episodes = sorted(data_dir.glob("raw/*/data/episode_*/data.hdf5"))
    if not episodes:
        raise ValueError(f"No episodes found under {data_dir}")
    return episodes


@dataclasses.dataclass
class Args:
    data_dir: Path
    repo_name: str = "your_hf_username/umi_capture"
    output_dir: Path | None = None
    robot_type: str = "umi"
    task_name: str = "pipe_insertion"
    push_to_hub: bool = False


def main(args: Args) -> None:
    episodes = _episode_paths(args.data_dir)

    gripper_min = math.inf
    gripper_max = -math.inf
    image_shape: tuple[int, int, int] | None = None
    fps_values: list[float] = []

    for episode_path in episodes:
        with h5py.File(episode_path, "r") as f:
            widths = f["gripper_widths"][:, 1]
            gripper_min = min(gripper_min, float(widths.min()))
            gripper_max = max(gripper_max, float(widths.max()))
            if image_shape is None:
                image_shape = tuple(int(x) for x in f["image_frames"].shape[1:])
            timestamps = f["image_timestamps"][:]
            if timestamps.size > 1:
                dt = np.median(np.diff(timestamps))
                fps_values.append(1.0 / dt)

    if image_shape is None:
        raise RuntimeError("Unable to infer image shape from the dataset.")

    fps = int(round(np.median(fps_values))) if fps_values else 30

    features = {
        "image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
        "force": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["force"],
        },
        "gripper_width": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_width"],
        },
    }

    dataset = utils.create_dataset(
        repo_name=args.repo_name,
        output_dir=args.output_dir,
        robot_type=args.robot_type,
        fps=fps,
        features=features,
    )

    for episode_path in episodes:
        with h5py.File(episode_path, "r") as f:
            images = f["image_frames"][:]
            image_times = f["image_timestamps"][:]

            pose_data = f["poses"][:]
            pose_times = pose_data[:, 0]
            positions = pose_data[:, 1:4]
            quaternions = pose_data[:, 4:8]  # stored as xyzw, already normalized

            gripper_data = f["gripper_widths"][:]
            gripper_times = gripper_data[:, 0]
            gripper_values = gripper_data[:, 1]

            force_data = f["forces"][:]
            force_times = force_data[:, 0]
            force_values = force_data[:, 1:]

        if image_times.size < 2:
            continue

        # Align all modalities to image timestamps.
        positions_interp = utils.interp_array(pose_times, positions, image_times)
        quats_interp = utils.interp_quaternions(pose_times, quaternions, image_times)
        gripper_interp = np.interp(image_times, gripper_times, gripper_values)
        gripper_norm = utils.normalize(gripper_interp, gripper_min, gripper_max)
        forces_interp = utils.interp_array(force_times, force_values, image_times)

        actions = utils.compute_actions(positions_interp, quats_interp, gripper_norm)

        for idx in range(image_times.size - 1):
            dataset.add_frame(
                {
                    "image": np.asarray(images[idx], dtype=np.uint8),
                    "state": np.zeros(8, dtype=np.float32),  # UMI capture has no robot state; we store zeros.
                    "actions": actions[idx].astype(np.float32),
                    "force": forces_interp[idx].astype(np.float32),
                    "gripper_width": np.array([gripper_norm[idx]], dtype=np.float32),
                    "task": args.task_name,
                }
            )

        # Drop the last video frame so every stored step has a well-defined next-step action.
        dataset.save_episode()

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["umi", "capture"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

