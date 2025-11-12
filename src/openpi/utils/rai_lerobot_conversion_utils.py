from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from lerobot.common.datasets import lerobot_dataset


def normalize(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Scale ``values`` linearly into [0, 1] given global ``minimum`` and ``maximum``."""
    if math.isclose(maximum, minimum):
        return np.zeros_like(values)
    return np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)


def interp_array(src_times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Interpolate a time series column-wise onto new timestamps using linear interpolation."""
    output = np.empty((target_times.size, values.shape[1]), dtype=np.float64)
    for idx in range(values.shape[1]):
        output[:, idx] = np.interp(target_times, src_times, values[:, idx])
    return output


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Perform spherical linear interpolation between two unit quaternions."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)

    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def interp_quaternions(src_times: np.ndarray, quaternions: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Interpolate a quaternion trajectory to new timestamps using SLERP between neighbours."""
    result = np.empty((target_times.size, 4), dtype=np.float64)
    for idx, timestamp in enumerate(target_times):
        if timestamp <= src_times[0]:
            result[idx] = quaternions[0]
            continue
        if timestamp >= src_times[-1]:
            result[idx] = quaternions[-1]
            continue
        hi = np.searchsorted(src_times, timestamp)
        lo = hi - 1
        t0, t1 = src_times[lo], src_times[hi]
        alpha = (timestamp - t0) / (t1 - t0) if t1 > t0 else 0.0
        result[idx] = slerp(quaternions[lo], quaternions[hi], float(alpha))
    return result


def quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    """Return the conjugate of an xyzw quaternion."""
    return np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]], dtype=np.float64)


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Multiply two xyzw quaternions."""
    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_to_euler_xyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to roll, pitch, yaw (XYZ intrinsic) angles."""
    x, y, z, w = quaternion
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def compute_actions(positions: np.ndarray, quaternions: np.ndarray, gripper_norm: np.ndarray) -> np.ndarray:
    """Create delta-cartesian + delta-orientation + next-step gripper actions from pose sequence."""
    delta_pos = positions[1:] - positions[:-1]

    delta_quats = np.empty((quaternions.shape[0] - 1, 4), dtype=np.float64)
    for idx in range(quaternions.shape[0] - 1):
        q_curr = quaternions[idx]
        q_next = quaternions[idx + 1]
        if np.dot(q_curr, q_next) < 0.0:
            q_next = -q_next
        dq = quat_multiply(q_next, quat_conjugate(q_curr))
        delta_quats[idx] = dq / np.linalg.norm(dq)

    delta_rpy = np.vstack([quat_to_euler_xyz(quat) for quat in delta_quats])
    next_gripper = gripper_norm[1:, np.newaxis]
    return np.hstack([delta_pos, delta_rpy, next_gripper])


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to an xyzw unit quaternion."""
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    quaternion = np.array([x, y, z, w], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def load_video_frames(video_path: Path) -> np.ndarray:
    """Decode a colour video file into an array of RGB frames."""
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    success, frame = capture.read()
    while success:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        success, frame = capture.read()
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def get_output_path(repo_name: str, output_dir: Path | None) -> Path:
    """Resolve the on-disk directory where a LeRobot dataset should be written."""
    base_dir = output_dir if output_dir is not None else lerobot_dataset.HF_LEROBOT_HOME
    base_dir = base_dir.expanduser().resolve()
    if output_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)
        lerobot_dataset.HF_LEROBOT_HOME = base_dir
    return base_dir / repo_name


def create_dataset(
    repo_name: str,
    output_dir: Path | None,
    robot_type: str,
    fps: int,
    features: Mapping[str, Mapping[str, object]],
) -> lerobot_dataset.LeRobotDataset:
    """Create a fresh LeRobot dataset, clearing any prior contents at the target location."""
    output_path = get_output_path(repo_name, output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    return lerobot_dataset.LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

