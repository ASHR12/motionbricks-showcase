"""Pure-NumPy BVH skeleton & animation clip loader matching NVIDIA MotionBricks conventions.

Positions are converted from centimeters to meters (* 0.01).
Quaternions use XYZW convention (matching Warp wp.quat and Three.js THREE.Quaternion).
SOMA BVH files are natively Y-up (matching both SOMA model space and Three.js).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for XYZW quaternions with broadcasting (..., 4)."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    ).astype(np.float32)


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternion) in XYZW format."""
    out = q.copy()
    out[..., :3] = -out[..., :3]
    return out


def quat_norm(q: np.ndarray) -> np.ndarray:
    """Normalize XYZW quaternions safely."""
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n > 1e-8, n, 1.0)
    return (q / n).astype(np.float32)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector(s) v (..., 3) by unit XYZW quaternion(s) q (..., 4)."""
    qvec = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * np.cross(qvec, v)
    return (v + qw * t + np.cross(qvec, t)).astype(np.float32)


def quat_from_axis_angle(axis: np.ndarray, angle_rad: np.ndarray | float) -> np.ndarray:
    """Build XYZW quaternion from unit axis (..., 3) and angle(s) in radians."""
    half = 0.5 * np.asarray(angle_rad, dtype=np.float32)
    s = np.sin(half)[..., None]
    c = np.cos(half)[..., None]
    ax = np.asarray(axis, dtype=np.float32)
    return quat_norm(np.concatenate([ax * s, c], axis=-1))


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Shortest-arc SLERP for XYZW quaternions."""
    q0 = quat_norm(q0)
    q1 = quat_norm(q1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    t_arr = np.asarray(t, dtype=np.float32)
    if t_arr.ndim < q0.ndim:
        t_arr = t_arr[..., None]
    near = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.where(near, 1.0, np.sin(theta))
    w0 = np.where(near, 1.0 - t_arr, np.sin((1.0 - t_arr) * theta) / sin_theta)
    w1 = np.where(near, t_arr, np.sin(t_arr * theta) / sin_theta)
    return quat_norm(w0 * q0 + w1 * q1)


def transform_mul(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compose rigid transforms (p1, q1) * (p2, q2)."""
    return p1 + quat_rotate(q1, p2), quat_norm(quat_mul(q1, q2))


def transform_inv(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert rigid transform (p, q)."""
    qi = quat_conj(q)
    return -quat_rotate(qi, p), qi


@dataclass
class Skeleton:
    names: list[str]
    parents: np.ndarray  # int32 [J]
    ref_pos: np.ndarray  # float32 [J, 3] (meters)
    ref_quat: np.ndarray  # float32 [J, 4] (xyzw identity)

    @property
    def num_joints(self) -> int:
        return len(self.names)

    def joint_index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            return -1


@dataclass
class AnimationClip:
    name: str
    skeleton: Skeleton
    sample_rate: float
    local_pos: np.ndarray   # float32 [F, J, 3]
    local_quat: np.ndarray  # float32 [F, J, 4]

    @property
    def num_frames(self) -> int:
        return self.local_pos.shape[0]

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.sample_rate if self.num_frames > 1 else 0.0

    def sample_pose(self, time_sec: float) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate local (pos[J,3], quat[J,4]) at time_sec (clamped)."""
        t = float(np.clip(time_sec, 0.0, self.duration))
        exact = t * self.sample_rate
        f0 = int(np.floor(exact))
        f1 = min(f0 + 1, self.num_frames - 1)
        blend = float(exact - f0)
        p = self.local_pos[f0] * (1.0 - blend) + self.local_pos[f1] * blend
        q = quat_slerp(self.local_quat[f0], self.local_quat[f1], blend)
        return p.astype(np.float32), q.astype(np.float32)

    def sample_window_30hz(
        self, time_sec: float, count: int = 4
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample `count` poses at 30 Hz spacing ending/starting around time_sec.

        Matches MotionBricks: samples `time_sec + i / 30.0` for i in 0..count-1.
        """
        dt = 1.0 / 30.0
        ps = np.empty((count, self.skeleton.num_joints, 3), dtype=np.float32)
        qs = np.empty((count, self.skeleton.num_joints, 4), dtype=np.float32)
        for i in range(count):
            ps[i], qs[i] = self.sample_pose(time_sec + i * dt)
        return ps, qs


def compute_world_poses(
    local_pos: np.ndarray,
    local_quat: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics for [..., J, 3] and [..., J, 4] local transforms."""
    world_pos = np.empty_like(local_pos)
    world_quat = np.empty_like(local_quat)
    world_pos[..., 0, :] = local_pos[..., 0, :]
    world_quat[..., 0, :] = quat_norm(local_quat[..., 0, :])
    for j in range(1, len(parents)):
        p_idx = int(parents[j])
        wp, wq = transform_mul(
            world_pos[..., p_idx, :],
            world_quat[..., p_idx, :],
            local_pos[..., j, :],
            local_quat[..., j, :],
        )
        world_pos[..., j, :] = wp
        world_quat[..., j, :] = wq
    return world_pos, world_quat


def _euler_deg_to_quat(angles_deg: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Convert [F, 3] angles in degrees with axis indices `order` [3] -> [F, 4] XYZW."""
    n = angles_deg.shape[0]
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 3] = 1.0
    axes = np.eye(3, dtype=np.float32)
    for k in range(3):
        axis = np.broadcast_to(axes[int(order[k])], (n, 3))
        rad = np.deg2rad(angles_deg[:, k]).astype(np.float32)
        rk = quat_from_axis_angle(axis, rad)
        q = quat_norm(quat_mul(q, rk))
    return q


def load_bvh(path: str | Path, target_skeleton: Skeleton | None = None) -> AnimationClip:
    """Load a BVH file into meters & XYZW quaternions."""
    path = Path(path)
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    joint_channels: list[list[str]] = []
    stack: list[int] = []
    in_end_site = False
    frame_count = 0
    sample_rate = 30.0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "MOTION":
                break
            if tokens[0] in ("ROOT", "JOINT"):
                raw_name = tokens[1].rsplit(":", 1)[-1]
                names.append(raw_name)
                parents.append(stack[-1] if stack else -1)
                stack.append(len(names) - 1)
                offsets.append([0.0, 0.0, 0.0])
                joint_channels.append([])
            elif tokens[:2] == ["End", "Site"]:
                in_end_site = True
            elif tokens[0] == "OFFSET" and not in_end_site:
                offsets[stack[-1]] = [float(v) for v in tokens[1:4]]
            elif tokens[0] == "CHANNELS":
                joint_channels[stack[-1]] = tokens[2:]
            elif tokens[0] == "}":
                if in_end_site:
                    in_end_site = False
                else:
                    stack.pop()

        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "Frames:":
                frame_count = int(tokens[1])
            elif tokens[:2] == ["Frame", "Time:"]:
                sample_rate = round(1.0 / float(tokens[2]), 2)
                break

        samples = np.loadtxt(f, dtype=np.float32, ndmin=2)

    num_joints = len(names)
    ref_pos = np.asarray(offsets, dtype=np.float32) * 0.01
    ref_quat = np.zeros((num_joints, 4), dtype=np.float32)
    ref_quat[:, 3] = 1.0
    file_skel = Skeleton(names, np.asarray(parents, dtype=np.int32), ref_pos, ref_quat)

    local_pos = np.broadcast_to(ref_pos[None, :, :], (frame_count, num_joints, 3)).copy()
    local_quat = np.broadcast_to(ref_quat[None, :, :], (frame_count, num_joints, 4)).copy()

    col = 0
    for j, chs in enumerate(joint_channels):
        has_pos = False
        has_rot = False
        pos_data = np.zeros((frame_count, 3), dtype=np.float32)
        rot_data = np.zeros((frame_count, 3), dtype=np.float32)
        rot_order = np.zeros(3, dtype=np.int32)
        r_idx = 0
        for ch in chs:
            cl = ch.lower()
            if cl.endswith("position"):
                pos_data[:, "xyz".index(cl[0])] = samples[:, col]
                has_pos = True
            elif cl.endswith("rotation"):
                rot_data[:, r_idx] = samples[:, col]
                rot_order[r_idx] = "xyz".index(cl[0])
                has_rot = True
                r_idx += 1
            col += 1
        if has_pos:
            local_pos[:, j, :] = pos_data * 0.01
        if has_rot:
            local_quat[:, j, :] = _euler_deg_to_quat(rot_data, rot_order)

    if target_skeleton is not None and target_skeleton.names != file_skel.names:
        # Remap onto target skeleton by exact joint name
        tgt_j = target_skeleton.num_joints
        mapped_pos = np.broadcast_to(target_skeleton.ref_pos[None, :, :], (frame_count, tgt_j, 3)).copy()
        mapped_quat = np.broadcast_to(target_skeleton.ref_quat[None, :, :], (frame_count, tgt_j, 4)).copy()
        name_to_src = {n: i for i, n in enumerate(file_skel.names)}
        for dst_i, n in enumerate(target_skeleton.names):
            src_i = name_to_src.get(n)
            if src_i is not None:
                mapped_pos[:, dst_i, :] = local_pos[:, src_i, :]
                mapped_quat[:, dst_i, :] = local_quat[:, src_i, :]
        return AnimationClip(path.stem, target_skeleton, sample_rate, mapped_pos, mapped_quat)

    return AnimationClip(path.stem, file_skel, sample_rate, local_pos, local_quat)
