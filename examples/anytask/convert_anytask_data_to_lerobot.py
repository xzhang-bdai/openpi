"""
Conversion utility for AnyTask datasets to the LeRobot format.

Usage:
uv run examples/anytask/convert_anytask_data_to_lerobot.py --data_dir /path/to/anytask/root

The script expects a directory layout where each episode directory contains
an ``episode_<id>.h5`` file and a ``videos/camera_*/episode_<id>.mp4`` video.
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np
import tyro

from openpi.utils import rai_lerobot_conversion_utils as utils


@dataclasses.dataclass
class Args:
    data_dir: Path
    repo_name: str = "your_hf_username/anytask"
    output_dir: Path | None = None
    robot_type: str = "franka"
    task_name: str = "anytask"
    push_to_hub: bool = False


def _episode_dirs(data_dir: Path) -> list[Path]:
    episodes = sorted(path for path in data_dir.glob("episode_*") if path.is_dir())
    if not episodes:
        raise ValueError(f"No episode directories found under {data_dir}")
    return episodes


def main(args: Args) -> None:
    episodes = _episode_dirs(args.data_dir)

    image_shapes: dict[str, tuple[int, int, int]] = {}
    fps_values: list[float] = []

    for episode_dir in episodes:
        episode_file = episode_dir / f"{episode_dir.name}.h5"
        if not episode_file.exists():
            raise FileNotFoundError(f"HDF5 file missing: {episode_file}")

        with h5py.File(episode_file, "r") as f:
            timestamps = f["timestamp"][:]

        camera_root = episode_dir / "videos"
        cameras = sorted(path for path in camera_root.glob("camera_*") if path.is_dir())
        if not cameras:
            raise ValueError(f"No camera directories found in {camera_root}")

        for camera_dir in cameras:
            cam_id = camera_dir.name
            video_path = camera_dir / f"{episode_dir.name}.mp4"
            if not video_path.exists():
                raise FileNotFoundError(f"Video missing: {video_path}")
            if cam_id not in image_shapes:
                capture = cv2.VideoCapture(str(video_path))
                success, frame = capture.read()
                fps = capture.get(cv2.CAP_PROP_FPS)
                capture.release()
                if not success:
                    raise ValueError(f"Unable to decode frames from {video_path}")
                image_shapes[cam_id] = frame.shape
                fps_values.append(fps)

    if not image_shapes:
        raise RuntimeError("Unable to determine camera shapes.")

    camera_ids = sorted(image_shapes.keys())
    fps = int(round(np.median(fps_values))) if fps_values else 30

    features: dict[str, dict[str, object]] = {
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
        "gripper_width": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_width"],
        },
    }

    for index, cam_id in enumerate(camera_ids):
        feature_name = f"image_camera_{index}"
        shape = image_shapes[cam_id]
        features[feature_name] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }

    dataset = utils.create_dataset(
        repo_name=args.repo_name,
        output_dir=args.output_dir,
        robot_type=args.robot_type,
        fps=fps,
        features=features,
    )

    for episode_dir in episodes:
        episode_start = time.perf_counter()
        print(f"\nProcessing {episode_dir.name}...")

        episode_file = episode_dir / f"{episode_dir.name}.h5"
        with h5py.File(episode_file, "r") as f:
            timestamps = f["timestamp"][:]
            joint_pos = f["joint_pos"][:, :7]
            ee_position = f["ee_position"][:]
            ee_rotation = f["ee_rotation"][:]
            gripper_width = f["gripper_width"][:, 0]

        # Ensure gripper width is within [0, 1].
        gripper_width = np.clip(gripper_width, 0.0, 1.0)

        camera_root = episode_dir / "videos"
        frames_by_camera: dict[str, np.ndarray] = {}

        for cam_id in camera_ids:
            video_path = camera_root / cam_id / f"{episode_dir.name}.mp4"
            load_start = time.perf_counter()
            frames_by_camera[cam_id] = utils.load_video_frames(video_path)
            print(
                f"  Loaded {cam_id} ({frames_by_camera[cam_id].shape[0]} frames) "
                f"in {time.perf_counter() - load_start:.2f}s"
            )

        min_len = min(frames.shape[0] for frames in frames_by_camera.values())
        if min_len < 2 or min_len != timestamps.size:
            print(
                f"  Skipping episode due to mismatched frame count "
                f"(video {min_len}, h5 {timestamps.size})."
            )
            continue

        # Convert quaternion rotations to ensure sign continuity.
        quaternions = ee_rotation
        # Align all modalities to the image timestamps (they already match).
        positions_interp = ee_position
        quats_interp = quaternions
        gripper_norm = gripper_width

        actions = utils.compute_actions(positions_interp, quats_interp, gripper_norm)

        frames_written = False
        for idx in range(min_len - 1):
            frame_dict = {
                "state": np.concatenate(
                    [joint_pos[idx], np.array([gripper_norm[idx]], dtype=np.float64)]
                ).astype(np.float32),
                "actions": actions[idx].astype(np.float32),
                "gripper_width": np.array([gripper_norm[idx]], dtype=np.float32),
                "task": args.task_name,
            }
            for camera_index, cam_id in enumerate(camera_ids):
                feature_name = f"image_camera_{camera_index}"
                frame_dict[feature_name] = frames_by_camera[cam_id][idx].astype(np.uint8)
            dataset.add_frame(frame_dict)
            frames_written = True

        if frames_written:
            dataset.save_episode()
            print(
                f"  Episode saved with {min_len - 1} frames "
                f"in {time.perf_counter() - episode_start:.2f}s"
            )
        else:
            print(f"  No frames written for {episode_dir.name}.")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["anytask", "franka"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

