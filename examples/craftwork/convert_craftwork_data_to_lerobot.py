"""
Conversion utility for Craftwork robot datasets to the LeRobot format.

Usage:
uv run examples/capture/convert_capture_data_to_lerobot.py   --args.data-dir /storage/nfs/xzhang/data/capture/20251106_pipe_insertion_cactus_capture_device_unimanual   --args.output-dir /storage/nfs/xzhang/data/lerobot   --args.repo-name capture_pipe_insertion_samples
The script expects directories of the following form:
raw/<collection_id>/data/demo_*/robot_full.npz
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
import cv2
import numpy as np
import tyro

from openpi.utils import rai_lerobot_conversion_utils as utils


def _episode_dirs(data_dir: Path) -> list[Path]:
    episodes = sorted(data_dir.glob("raw/*/data/demo_*"))
    if not episodes:
        raise ValueError(f"No demo directories found under {data_dir}")
    return episodes


def _object_array_to_float(array: np.ndarray, *, expected_ndim: int) -> np.ndarray:
    """Convert an object-array of numeric sequences to a float np.ndarray."""
    if array.size == 0:
        raise ValueError("Cannot convert empty object array.")
    prototype = None
    for item in array:
        candidate = np.asarray(item, dtype=np.float64)
        if candidate.shape != ():
            prototype = candidate
            break
    if prototype is None:
        values = np.asarray(array, dtype=np.float64)
    else:
        stacked = []
        for idx, item in enumerate(array):
            seq = np.asarray(item, dtype=np.float64)
            if seq.shape == ():
                seq = np.zeros(prototype.shape, dtype=np.float64)
            elif seq.shape != prototype.shape:
                raise ValueError(
                    f"Inconsistent sequence shape at index {idx}: {seq.shape} vs {prototype.shape}"
            )
            stacked.append(seq)
        values = np.stack(stacked, axis=0)
    if values.ndim != expected_ndim:
        raise ValueError(f"Expected {expected_ndim} dims, got {values.ndim}")
    return values


@dataclasses.dataclass
class Args:
    data_dir: Path
    repo_name: str = "your_hf_username/craftwork"
    output_dir: Path | None = None
    robot_type: str = "franka"
    task_name: str = "pipe_insertion"
    push_to_hub: bool = False


def main(args: Args) -> None:
    episodes = _episode_dirs(args.data_dir)

    gripper_min = math.inf
    gripper_max = -math.inf
    fps_values: list[float] = []
    camera_ids: list[str] | None = None
    camera_shapes: dict[str, tuple[int, int, int]] = {}

    for episode in episodes:
        robot_path = episode / "robot_full.npz"
        data = np.load(robot_path, allow_pickle=True)
        gripper = data["arm_gripper_width"].astype(np.float64)
        gripper_min = min(gripper_min, float(gripper.min()))
        gripper_max = max(gripper_max, float(gripper.max()))

        camera_dir = episode / "cameras"
        available_cameras = sorted(
            {
                path.name.replace("_color.mp4", "")
                for path in camera_dir.glob("*_color.mp4")
            }
        )
        if not available_cameras:
            raise ValueError(f"No RGB cameras found in {camera_dir}")

        if camera_ids is None:
            camera_ids = available_cameras
        else:
            camera_ids = [cam for cam in camera_ids if cam in available_cameras]
        if not camera_ids:
            raise ValueError("No common cameras found across Craftwork episodes.")

        for cam_id in available_cameras:
            if cam_id in camera_shapes:
                continue
            color_path = camera_dir / f"{cam_id}_color.mp4"
            capture = cv2.VideoCapture(str(color_path))
            success, frame = capture.read()
            capture.release()
            if not success:
                raise ValueError(f"Failed to decode frames from {color_path}")
            camera_shapes[cam_id] = frame.shape

        reference_cam = available_cameras[0]
        timestamps = np.loadtxt(
            camera_dir / f"{reference_cam}_timestamps.txt", dtype=np.float64
        )
        if timestamps.size > 1:
            fps_values.append(1.0 / np.median(np.diff(timestamps)))

    if camera_ids is None:
        raise RuntimeError("Unable to determine available cameras.")

    camera_entries: list[tuple[str, str]] = []
    features = {
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

    for idx, cam_id in enumerate(camera_ids):
        if cam_id not in camera_shapes:
            raise ValueError(f"Camera {cam_id} missing shape information.")
        key = f"image_camera_{idx}"
        shape = camera_shapes[cam_id]
        features[key] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }
        camera_entries.append((cam_id, key))

    if not fps_values:
        raise RuntimeError("Unable to determine FPS from camera timestamps.")

    fps = int(round(np.median(fps_values)))

    dataset = utils.create_dataset(
        repo_name=args.repo_name,
        output_dir=args.output_dir,
        robot_type=args.robot_type,
        fps=fps,
        features=features,
    )

    print(f"Discovered {len(camera_ids)} RGB cameras: {camera_ids}")

    for episode in episodes:
        episode_start = time.perf_counter()
        print(f"\nProcessing {episode.name}...")
        robot_path = episode / "robot_full.npz"
        load_start = time.perf_counter()
        robot_data = np.load(robot_path, allow_pickle=True)
        print(f"  Loaded robot_full.npz in {time.perf_counter() - load_start:.2f}s")

        joint_positions = _object_array_to_float(
            robot_data["arm_joint_positions"], expected_ndim=2
        )
        ee_pose = np.asarray(robot_data["arm_ee_pose"], dtype=np.float64)
        ee_positions = ee_pose[:, :3, 3]
        ee_quaternions = np.vstack(
            [utils.rotation_matrix_to_quaternion(matrix[:3, :3]) for matrix in ee_pose]
        )
        gripper_width = robot_data["arm_gripper_width"].astype(np.float64)
        timestamps = robot_data["arm_timestamp"].astype(np.float64)

        camera_dir = episode / "cameras"
        frames_by_camera: dict[str, np.ndarray] = {}
        times_by_camera: dict[str, np.ndarray] = {}
        for cam_id, _ in camera_entries:
            color_path = camera_dir / f"{cam_id}_color.mp4"
            timestamps_path = camera_dir / f"{cam_id}_timestamps.txt"
            cam_load_start = time.perf_counter()
            frames_by_camera[cam_id] = utils.load_video_frames(color_path)
            times_by_camera[cam_id] = np.loadtxt(timestamps_path, dtype=np.float64)
            print(
                f"  Loaded {cam_id} ({frames_by_camera[cam_id].shape[0]} frames) "
                f"in {time.perf_counter() - cam_load_start:.2f}s"
            )

        start_time = max(times[0] for times in times_by_camera.values())
        end_time = min(times[-1] for times in times_by_camera.values())
        if start_time >= end_time:
            continue

        trimmed_frames: dict[str, np.ndarray] = {}
        trimmed_times: dict[str, np.ndarray] = {}
        min_len = None
        insufficient_frames = False
        for cam_id in camera_ids:
            mask = (times_by_camera[cam_id] >= start_time) & (
                times_by_camera[cam_id] <= end_time
            )
            times_subset = times_by_camera[cam_id][mask]
            frames_subset = frames_by_camera[cam_id][mask]
            if times_subset.size < 2:
                insufficient_frames = True
                break
            trimmed_frames[cam_id] = frames_subset
            trimmed_times[cam_id] = times_subset
            min_len = times_subset.size if min_len is None else min(min_len, times_subset.size)

        if insufficient_frames or min_len is None or min_len < 2:
            print("  Skipping episode due to insufficient overlapping frames.")
            continue

        for cam_id in camera_ids:
            trimmed_frames[cam_id] = trimmed_frames[cam_id][:min_len]
            trimmed_times[cam_id] = trimmed_times[cam_id][:min_len]

        primary_cam = camera_ids[0]
        image_times = trimmed_times[primary_cam]

        aligned_frames: dict[str, np.ndarray] = {primary_cam: trimmed_frames[primary_cam]}
        tolerance = 0.1

        for cam_id in camera_ids[1:]:
            cam_times = trimmed_times[cam_id]
            indices = np.searchsorted(cam_times, image_times)
            indices = np.clip(indices, 0, cam_times.size - 1)

            # Check if previous index gives closer match.
            prev_indices = np.clip(indices - 1, 0, cam_times.size - 1)
            closer_prev = (
                np.abs(cam_times[prev_indices] - image_times)
                < np.abs(cam_times[indices] - image_times)
            )
            indices = np.where(closer_prev, prev_indices, indices)

            max_delta = np.max(np.abs(cam_times[indices] - image_times))
            if max_delta > tolerance:
                raise ValueError(
                    f"Timestamps for camera {cam_id} deviate by {max_delta:.3f} seconds from primary camera."
                )

            aligned_frames[cam_id] = trimmed_frames[cam_id][indices]

        positions_interp = utils.interp_array(timestamps, ee_positions, image_times)
        quats_interp = utils.interp_quaternions(timestamps, ee_quaternions, image_times)
        gripper_interp = np.interp(image_times, timestamps, gripper_width)
        gripper_norm = utils.normalize(gripper_interp, gripper_min, gripper_max)
        joints_interp = utils.interp_array(timestamps, joint_positions, image_times)
        actions = utils.compute_actions(positions_interp, quats_interp, gripper_norm)

        frames_written = False
        for idx in range(image_times.size - 1):
            frame_dict = {
                "state": np.concatenate(
                    [joints_interp[idx], np.array([gripper_norm[idx]], dtype=np.float64)]
                ).astype(np.float32),
                "actions": actions[idx].astype(np.float32),
                "gripper_width": np.array([gripper_norm[idx]], dtype=np.float32),
                "task": args.task_name,
            }
            for cam_id, feature_key in camera_entries:
                frame_dict[feature_key] = aligned_frames[cam_id][idx].astype(np.uint8)
            dataset.add_frame(frame_dict)
            frames_written = True

        if frames_written:
            dataset.save_episode()
            print(
                f"  Episode saved with {image_times.size - 1} frames "
                f"in {time.perf_counter() - episode_start:.2f}s"
            )
        else:
            print("  No frames written for this episode.")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["craftwork", "franka"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
