"""Pure-NumPy + ONNX Runtime port of NVIDIA MotionBricks SOMA engine.

Runs the full unquantized `mmm_v6_soma_fp16.onnx` (363 MB, opset 17) directly on
Apple Silicon CPU (~22-25 ms/gen, 30-80x faster than real time) with exact parity
to `motionbricks/model/soma/kernels.py` and `motionbricks/model/soma/engine.py`.

All poses live natively in Y-up SOMA model space (meters, XYZW quaternions), which
matches Three.js directly without any Y<->Z axis conversion.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from bvh_loader import (
    AnimationClip,
    Skeleton,
    compute_world_poses,
    load_bvh,
    quat_conj,
    quat_from_axis_angle,
    quat_mul,
    quat_norm,
    quat_rotate,
    quat_slerp,
    transform_inv,
    transform_mul,
)

MODEL_POSES_PER_TOKEN = 4
MODEL_CONTEXT_AND_TARGET_POSES = 8
MODEL_BOUNDARY_FRAMES = 3
MODEL_SAMPLE_RATE = 30.0
MODEL_MIN_TOKEN_COUNT = 6
MODEL_MAX_TOKEN_COUNT = 16
MODEL_NUM_TOKEN_SLOTS = MODEL_MAX_TOKEN_COUNT - MODEL_MIN_TOKEN_COUNT + 1  # 11
FOOT_CONTACT_COUNT = 4


def strip_root_skeleton(bvh_skeleton: Skeleton) -> Skeleton:
    """Drop the dummy `Root` joint (index 0) so `Hips` becomes joint 0 with parent -1."""
    if bvh_skeleton.names[0] == "Hips":
        return bvh_skeleton
    assert bvh_skeleton.names[0] == "Root" and bvh_skeleton.names[1] == "Hips"
    names = bvh_skeleton.names[1:]
    parents = bvh_skeleton.parents[1:] - 1
    parents[0] = -1
    return Skeleton(
        names=names,
        parents=parents.astype(np.int32),
        ref_pos=bvh_skeleton.ref_pos[1:].copy(),
        ref_quat=bvh_skeleton.ref_quat[1:].copy(),
    )


def _heading_from_hip_positions(right_hip: np.ndarray, left_hip: np.ndarray) -> float:
    """Exact port of `_heading_from_hip_positions` in soma/kernels.py (Y-up)."""
    ax = float(right_hip[0] - left_hip[0])
    az = float(right_hip[2] - left_hip[2])
    length = math.hypot(ax, az)
    if length <= 1.0e-6:
        return 0.0
    ax /= length
    az /= length
    # cross((0, 1, 0), (ax, 0, az)) = (az, 0, -ax)
    return math.atan2(az, -ax)


def _rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Vectorized 3x3 rotation matrix (..., 3, 3) -> unit XYZW quaternion (..., 4)."""
    # Shepperd's method vectorized for stability
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    tr = m00 + m11 + m22
    shape = R.shape[:-2]
    q = np.empty((*shape, 4), dtype=np.float32)

    c0 = tr > 0.0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = (~c0) & (~c1) & (~c2)

    if np.any(c0):
        s = np.sqrt(np.maximum(tr[c0] + 1.0, 1e-12)) * 2.0
        q[c0, 3] = 0.25 * s
        q[c0, 0] = (m21[c0] - m12[c0]) / s
        q[c0, 1] = (m02[c0] - m20[c0]) / s
        q[c0, 2] = (m10[c0] - m01[c0]) / s
    if np.any(c1):
        s = np.sqrt(np.maximum(1.0 + m00[c1] - m11[c1] - m22[c1], 1e-12)) * 2.0
        q[c1, 3] = (m21[c1] - m12[c1]) / s
        q[c1, 0] = 0.25 * s
        q[c1, 1] = (m01[c1] + m10[c1]) / s
        q[c1, 2] = (m02[c1] + m20[c1]) / s
    if np.any(c2):
        s = np.sqrt(np.maximum(1.0 + m11[c2] - m00[c2] - m22[c2], 1e-12)) * 2.0
        q[c2, 3] = (m02[c2] - m20[c2]) / s
        q[c2, 0] = (m01[c2] + m10[c2]) / s
        q[c2, 1] = 0.25 * s
        q[c2, 2] = (m12[c2] + m21[c2]) / s
    if np.any(c3):
        s = np.sqrt(np.maximum(1.0 + m22[c3] - m00[c3] - m11[c3], 1e-12)) * 2.0
        q[c3, 3] = (m10[c3] - m01[c3]) / s
        q[c3, 0] = (m02[c3] + m20[c3]) / s
        q[c3, 1] = (m12[c3] + m21[c3]) / s
        q[c3, 2] = 0.25 * s

    return quat_norm(q)


def _rotation_from_6d(x_raw: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
    """Exact port of `_rotation_from_6d` in soma/kernels.py -> XYZW quat (..., 4)."""
    x_norm = np.linalg.norm(x_raw, axis=-1, keepdims=True)
    x_axis = x_raw / np.where(x_norm > 1e-8, x_norm, 1.0)
    z_raw = np.cross(x_axis, y_raw)
    z_norm = np.linalg.norm(z_raw, axis=-1, keepdims=True)
    z_axis = z_raw / np.where(z_norm > 1e-8, z_norm, 1.0)
    y_axis = np.cross(z_axis, x_axis)
    # matrix_from_cols(x_axis, y_axis, z_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=-1)
    return _rotation_matrix_to_quat_xyzw(R)


def _compute_root_endpoint_delta(
    gen_p: np.ndarray, gen_q: np.ndarray, guide_p: np.ndarray, guide_q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dp = guide_p - gen_p
    dq = quat_norm(quat_mul(quat_conj(quat_norm(gen_q)), quat_norm(guide_q)))
    return dp, dq


def _compute_delta_transform(
    src_p: np.ndarray, src_q: np.ndarray, dst_p: np.ndarray, dst_q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ip, iq = transform_inv(src_p, src_q)
    return transform_mul(ip, iq, dst_p, dst_q)


@dataclass
class GenerationResult:
    local_pos: np.ndarray        # [F, J, 3]
    local_quat: np.ndarray       # [F, J, 4]
    world_pos: np.ndarray        # [F, J, 3]
    world_quat: np.ndarray       # [F, J, 4]
    foot_contacts: np.ndarray    # [F, 4] (LeftFoot, LeftToeBase, RightFoot, RightToeBase)
    num_frames: int
    pred_num_tokens: int
    latency_ms: float
    infer_ms: float


class SOMAOnnxEngine:
    """Full unquantized MotionBricks SOMA ONNX Runtime engine (CPUExecutionProvider)."""

    def __init__(
        self,
        onnx_path: str | Path,
        manifest_path: str | Path,
        skeleton: Skeleton,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.model_joint_names: list[str] = self.manifest["joint_names"]
        self.num_model_joints = len(self.model_joint_names)
        self.skeleton = skeleton
        self.num_pose_joints = skeleton.num_joints

        # Map model joint -> pose skeleton joint
        self.model_to_pose = np.array(
            [skeleton.joint_index(name) for name in self.model_joint_names],
            dtype=np.int32,
        )
        if np.any(self.model_to_pose < 0):
            missing = [
                n for n, i in zip(self.model_joint_names, self.model_to_pose) if i < 0
            ]
            raise ValueError(f"Skeleton missing model joints: {missing}")
        assert self.model_to_pose[0] == 0, "Skeleton joint 0 must be Hips."

        # Connected pose_to_model (exact match to MotionBricksSOMAEngine._bind_pose_skeleton)
        self.pose_to_model = np.full(self.num_pose_joints, -1, dtype=np.int32)
        for mj, pj in enumerate(self.model_to_pose):
            parent = int(skeleton.parents[pj])
            if parent == -1 or self.pose_to_model[parent] >= 0:
                self.pose_to_model[pj] = mj

        self.right_leg_idx = skeleton.joint_index("RightLeg")
        self.left_leg_idx = skeleton.joint_index("LeftLeg")
        assert self.right_leg_idx >= 0 and self.left_leg_idx >= 0

        # Precompute parent model joint indices for connected non-root joints
        self._connected_pose_joints = np.where(
            (np.arange(self.num_pose_joints) > 0) & (self.pose_to_model >= 0)
        )[0]
        self._connected_model_joints = self.pose_to_model[self._connected_pose_joints]
        self._connected_parent_model_joints = self.pose_to_model[
            skeleton.parents[self._connected_pose_joints]
        ]

        # Precompute depth-grouped tree levels for 3.4x faster bit-identical FK
        depths = np.zeros(self.num_pose_joints, dtype=np.int32)
        for j in range(1, self.num_pose_joints):
            depths[j] = depths[int(skeleton.parents[j])] + 1
        self._fk_levels: list[tuple[np.ndarray, np.ndarray]] = []
        for d in range(1, int(depths.max()) + 1):
            js = np.where(depths == d)[0].astype(np.int32)
            ps = skeleton.parents[js].astype(np.int32)
            self._fk_levels.append((js, ps))

        # Build ONNX Runtime CPU session (tuned 8 intra-op threads, sequential -> ~22 ms on Apple Silicon!)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 8
        sess_opts.inter_op_num_threads = 1
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.load_ms = (time.perf_counter() - t0) * 1000.0

    def fast_fk(
        self, local_pos: np.ndarray, local_quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Level-vectorized forward kinematics (bit-identical to compute_world_poses, ~3.4x faster)."""
        wp = np.empty_like(local_pos)
        wq = np.empty_like(local_quat)
        wp[..., 0, :] = local_pos[..., 0, :]
        wq[..., 0, :] = quat_norm(local_quat[..., 0, :])
        for js, ps in self._fk_levels:
            wp[..., js, :], wq[..., js, :] = transform_mul(
                wp[..., ps, :], wq[..., ps, :], local_pos[..., js, :], local_quat[..., js, :]
            )
        return wp, wq

    def generate(
        self,
        context_local_pos: np.ndarray,     # [4, J, 3]
        context_local_quat: np.ndarray,    # [4, J, 4]
        target_local_pos: np.ndarray,      # [N, J, 3] (1 <= N <= 4)
        target_local_quat: np.ndarray,     # [N, J, 4]
        *,
        allowed_token_counts: tuple[int, ...] = (8, 9, 10, 11, 12, 13, 14),
        correct_endpoints: bool = True,
        heading_alignment: tuple[float, float] = (1.0, 0.0),
    ) -> GenerationResult:
        """Run one MotionBricks SOMA generation call (~23 ms on Apple Silicon)."""
        t0 = time.perf_counter()

        num_target = int(target_local_pos.shape[0])
        assert 1 <= num_target <= MODEL_POSES_PER_TOKEN
        assert context_local_pos.shape[0] == MODEL_POSES_PER_TOKEN

        # Pad target to 4 frames by repeating last pose
        tgt_p_pad = np.empty((MODEL_POSES_PER_TOKEN, self.num_pose_joints, 3), dtype=np.float32)
        tgt_q_pad = np.empty((MODEL_POSES_PER_TOKEN, self.num_pose_joints, 4), dtype=np.float32)
        for i in range(MODEL_POSES_PER_TOKEN):
            src = min(i, num_target - 1)
            tgt_p_pad[i] = target_local_pos[src]
            tgt_q_pad[i] = target_local_quat[src]

        guide_local_pos = np.concatenate([context_local_pos, tgt_p_pad], axis=0)
        guide_local_quat = np.concatenate([context_local_quat, tgt_q_pad], axis=0)

        # Forward kinematics in SOMA model space (Y-up)
        guide_wp, guide_wq = self.fast_fk(guide_local_pos, guide_local_quat)

        # 1. Compute guide alignment from last context pose (frame 3)
        last_ctx = MODEL_POSES_PER_TOKEN - 1
        hips_p = guide_wp[last_ctx, 0]
        cur_angle = _heading_from_hip_positions(
            guide_wp[last_ctx, self.right_leg_idx],
            guide_wp[last_ctx, self.left_leg_idx],
        )
        desired_angle = math.atan2(heading_alignment[0], heading_alignment[1])
        align_p = np.array([hips_p[0], 0.0, hips_p[2]], dtype=np.float32)
        align_q = quat_from_axis_angle(
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            cur_angle - desired_angle,
        )
        align_qi = quat_conj(align_q)

        # 2. Encode 8 guide frames into model tensors
        global_root_values = np.zeros((1, 8, 5), dtype=np.float32)
        local_root_values = np.zeros((1, 8, 4), dtype=np.float32)
        local_poses = np.zeros(
            (1, 8, (self.num_model_joints - 1) * 3 + self.num_model_joints * 6),
            dtype=np.float32,
        )

        # Align all 8 frames x all model joints vectorized
        aligned_p = quat_rotate(
            align_qi[None, None, :],
            guide_wp - align_p[None, None, :],
        )
        aligned_q = quat_norm(quat_mul(align_qi[None, None, :], guide_wq))

        x_unit = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_unit = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        for frame in range(MODEL_CONTEXT_AND_TARGET_POSES):
            next_frame = frame if frame == MODEL_POSES_PER_TOKEN - 1 else min(frame + 1, 7)

            cur_hips = aligned_p[frame, 0]
            nxt_hips = aligned_p[next_frame, 0]

            cur_heading = _heading_from_hip_positions(
                aligned_p[frame, self.right_leg_idx],
                aligned_p[frame, self.left_leg_idx],
            )
            nxt_heading = _heading_from_hip_positions(
                aligned_p[next_frame, self.right_leg_idx],
                aligned_p[next_frame, self.left_leg_idx],
            )
            dh = math.atan2(
                math.sin(nxt_heading - cur_heading),
                math.cos(nxt_heading - cur_heading),
            )

            global_root_values[0, frame, 0] = cur_hips[0]
            global_root_values[0, frame, 1] = cur_hips[1]
            global_root_values[0, frame, 2] = cur_hips[2]
            global_root_values[0, frame, 3] = math.cos(cur_heading)
            global_root_values[0, frame, 4] = math.sin(cur_heading)

            local_root_values[0, frame, 0] = dh * MODEL_SAMPLE_RATE
            local_root_values[0, frame, 1] = (nxt_hips[0] - cur_hips[0]) * MODEL_SAMPLE_RATE
            local_root_values[0, frame, 2] = (nxt_hips[2] - cur_hips[2]) * MODEL_SAMPLE_RATE
            local_root_values[0, frame, 3] = cur_hips[1]

            root_xz = np.array([cur_hips[0], 0.0, cur_hips[2]], dtype=np.float32)
            m_pos = aligned_p[frame, self.model_to_pose]  # [30, 3]
            m_rot = aligned_q[frame, self.model_to_pose]  # [30, 4]

            # Child positions [29*3]
            centered = m_pos[1:] - root_xz[None, :]
            local_poses[0, frame, : (self.num_model_joints - 1) * 3] = centered.reshape(-1)

            # 6D rotations [30*6]
            x_axes = quat_rotate(m_rot, np.broadcast_to(x_unit, (self.num_model_joints, 3)))
            y_axes = quat_rotate(m_rot, np.broadcast_to(y_unit, (self.num_model_joints, 3)))
            rot6d = np.concatenate([x_axes, y_axes], axis=-1).reshape(-1)
            local_poses[0, frame, (self.num_model_joints - 1) * 3 :] = rot6d

        # 3. Build conditioning masks
        has_global_root = np.zeros((1, 8), dtype=np.int32)
        has_local_poses = np.zeros((1, 8), dtype=np.int32)
        has_local_root = np.zeros((1, 8), dtype=np.int32)
        for idx in range(MODEL_CONTEXT_AND_TARGET_POSES):
            val = 1 if idx < MODEL_POSES_PER_TOKEN + num_target else 0
            has_global_root[0, idx] = val
            has_local_poses[0, idx] = val
            if idx == MODEL_POSES_PER_TOKEN - 1 or idx == MODEL_POSES_PER_TOKEN + num_target - 1:
                has_local_root[0, idx] = 0
            else:
                has_local_root[0, idx] = val

        allowed_mask = np.zeros((1, MODEL_NUM_TOKEN_SLOTS), dtype=np.int32)
        for tc in allowed_token_counts:
            if MODEL_MIN_TOKEN_COUNT <= tc <= MODEL_MAX_TOKEN_COUNT:
                allowed_mask[0, tc - MODEL_MIN_TOKEN_COUNT] = 1

        # 4. Run ONNX inference!
        t_infer0 = time.perf_counter()
        pred_global_motions, pred_num_tokens_arr = self.session.run(
            ["pred_global_motions", "pred_num_tokens"],
            {
                "global_root_values": global_root_values,
                "local_root_values": local_root_values,
                "local_poses": local_poses,
                "has_global_root_values": has_global_root,
                "has_local_root_values": has_local_root,
                "has_local_poses": has_local_poses,
                "allowed_pred_num_tokens": allowed_mask,
            },
        )
        infer_ms = (time.perf_counter() - t_infer0) * 1000.0

        pred_tokens = int(pred_num_tokens_arr[0])
        raw_count = max(0, min(pred_tokens * MODEL_POSES_PER_TOKEN, pred_global_motions.shape[1]))
        frame_count = max(0, raw_count - 2 * MODEL_BOUNDARY_FRAMES)
        if frame_count < 2:
            raise RuntimeError(f"Invalid generated frame_count={frame_count} (tokens={pred_tokens})")

        # 5. Decode predicted motion into local transforms + foot contacts
        raw_slice = pred_global_motions[0, MODEL_BOUNDARY_FRAMES : MODEL_BOUNDARY_FRAMES + frame_count]
        foot_contacts = np.clip(raw_slice[:, -FOOT_CONTACT_COUNT:], 0.0, 1.0).astype(np.float32)

        rot_offset = 5 + (self.num_model_joints - 1) * 3
        rot6d_all = raw_slice[:, rot_offset : rot_offset + self.num_model_joints * 6].reshape(
            frame_count, self.num_model_joints, 6
        )
        pred_global_quats = _rotation_from_6d(rot6d_all[..., :3], rot6d_all[..., 3:6])  # [F, 30, 4]

        out_local_pos = np.broadcast_to(
            self.skeleton.ref_pos[None, :, :], (frame_count, self.num_pose_joints, 3)
        ).copy()
        out_local_quat = np.broadcast_to(
            self.skeleton.ref_quat[None, :, :], (frame_count, self.num_pose_joints, 4)
        ).copy()

        # Root (Hips) transformed back from aligned space into world space
        hips_aligned_p = raw_slice[:, 0:3].astype(np.float32)
        hips_aligned_q = pred_global_quats[:, 0, :]
        hips_world_p, hips_world_q = transform_mul(
            align_p[None, :], align_q[None, :], hips_aligned_p, hips_aligned_q
        )
        out_local_pos[:, 0, :] = hips_world_p
        out_local_quat[:, 0, :] = hips_world_q

        # Connected non-root joints: local_q = parent_q^-1 * child_q
        child_q = pred_global_quats[:, self._connected_model_joints, :]
        parent_q = pred_global_quats[:, self._connected_parent_model_joints, :]
        out_local_quat[:, self._connected_pose_joints, :] = quat_norm(
            quat_mul(quat_conj(parent_q), child_q)
        )

        # 6. Endpoint stitching (eliminates any pop at chunk boundaries!)
        if correct_endpoints and frame_count >= 2:
            last_f = frame_count - 1
            start_dp0, start_dq0 = _compute_root_endpoint_delta(
                out_local_pos[0, 0], out_local_quat[0, 0],
                context_local_pos[-1, 0], context_local_quat[-1, 0],
            )
            end_dp0, end_dq0 = _compute_root_endpoint_delta(
                out_local_pos[last_f, 0], out_local_quat[last_f, 0],
                target_local_pos[0, 0], target_local_quat[0, 0],
            )

            end_w = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
            start_w = 1.0 - end_w
            ident_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            # Root correction
            out_local_pos[:, 0, :] = (
                out_local_pos[:, 0, :]
                + start_w[:, None] * start_dp0[None, :]
                + end_w[:, None] * end_dp0[None, :]
            )
            sq_root = quat_slerp(
                np.broadcast_to(ident_q, (frame_count, 4)),
                np.broadcast_to(start_dq0, (frame_count, 4)),
                start_w,
            )
            eq_root = quat_slerp(
                np.broadcast_to(ident_q, (frame_count, 4)),
                np.broadcast_to(end_dq0, (frame_count, 4)),
                end_w,
            )
            out_local_quat[:, 0, :] = quat_norm(
                quat_mul(quat_mul(out_local_quat[:, 0, :], sq_root), eq_root)
            )

            # Non-root joint corrections (only rotations differ since ref_pos is constant)
            start_dq = quat_norm(
                quat_mul(quat_conj(out_local_quat[0, 1:]), context_local_quat[-1, 1:])
            )
            end_dq = quat_norm(
                quat_mul(quat_conj(out_local_quat[last_f, 1:]), target_local_quat[0, 1:])
            )
            ident_all = np.broadcast_to(ident_q, (frame_count, self.num_pose_joints - 1, 4))
            sq_j = quat_slerp(ident_all, start_dq[None, :, :], start_w[:, None, None])
            eq_j = quat_slerp(ident_all, end_dq[None, :, :], end_w[:, None, None])
            out_local_quat[:, 1:, :] = quat_norm(
                quat_mul(quat_mul(out_local_quat[:, 1:, :], sq_j), eq_j)
            )

        # 7. Forward kinematics to world space for Three.js rendering
        world_pos, world_quat = self.fast_fk(out_local_pos, out_local_quat)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return GenerationResult(
            local_pos=out_local_pos,
            local_quat=out_local_quat,
            world_pos=world_pos,
            world_quat=world_quat,
            foot_contacts=foot_contacts,
            num_frames=frame_count,
            pred_num_tokens=pred_tokens,
            latency_ms=latency_ms,
            infer_ms=infer_ms,
        )


def canonicalize_target_window(
    clip: AnimationClip,
    sample_time: float,
    right_leg_idx: int,
    left_leg_idx: int,
    desired_speed: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample 4 poses at 30 Hz, center root XZ at (0, 0) facing +X, and optionally scale ground speed.

    Returns (local_pos[4, J, 3], local_quat[4, J, 4]) in Y-up SOMA space.
    """
    lp, lq = clip.sample_window_30hz(sample_time, MODEL_POSES_PER_TOKEN)
    wp, _ = compute_world_poses(lp, lq, clip.skeleton.parents)

    # Determine forward heading in XZ plane (prefer hip travel direction if moving, else hip facing)
    r0 = lp[0, 0]
    r3 = lp[-1, 0]
    dx = float(r3[0] - r0[0])
    dz = float(r3[2] - r0[2])
    travel = math.hypot(dx, dz)
    if travel > 0.03:
        # Travel angle from +Z? Let's align travel vector (dx, dz) onto +X (1, 0)
        # Rotation around +Y by angle theta rotates (x, z) -> (x cos t + z sin t, -x sin t + z cos t)
        # Wait: standard right-handed Y rotation: x' = x cos t + z sin t, z' = -x sin t + z cos t.
        # To map (dx, dz) -> (travel, 0): cos t = dx/travel, sin t = dz/travel => t = atan2(dz, dx)!
        rot_y = math.atan2(dz, dx)
    else:
        # Use hip facing direction (heading = atan2(fwd.x, fwd.z)); map fwd -> +X (pi/2)
        hip_heading = _heading_from_hip_positions(wp[0, right_leg_idx], wp[0, left_leg_idx])
        # fwd = (sin(h), 0, cos(h)); to map to (1, 0, 0): rot_y = pi/2 - h? Let's check:
        # x' = sin(h) cos(t) + cos(h) sin(t) = sin(h + t) = 1 => t = pi/2 - h
        rot_y = (0.5 * math.pi) - hip_heading

    # Apply root transform: translate (-r0.x, 0, -r0.z) then rotate around +Y by rot_y
    q_rot = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), rot_y)
    origin_p = np.array([r0[0], 0.0, r0[2]], dtype=np.float32)
    lp = lp.copy()
    lq = lq.copy()
    for i in range(MODEL_POSES_PER_TOKEN):
        rel = lp[i, 0] - origin_p
        lp[i, 0] = quat_rotate(q_rot, rel)
        lq[i, 0] = quat_norm(quat_mul(q_rot, lq[i, 0]))

    # Optional speed scaling along XZ plane (preserving Y height)
    if desired_speed is not None and desired_speed >= 0.0:
        d = lp[-1, 0] - lp[0, 0]
        src_speed = math.hypot(float(d[0]), float(d[2])) * (MODEL_SAMPLE_RATE / (MODEL_POSES_PER_TOKEN - 1))
        if src_speed > 1e-4:
            scale = desired_speed / src_speed
            start = lp[0, 0].copy()
            for i in range(1, MODEL_POSES_PER_TOKEN):
                lp[i, 0, 0] = start[0] + (lp[i, 0, 0] - start[0]) * scale
                lp[i, 0, 2] = start[2] + (lp[i, 0, 2] - start[2]) * scale

    return lp, lq


def place_target_window(
    canonical_pos: np.ndarray,
    canonical_quat: np.ndarray,
    ground_x: float,
    ground_z: float,
    dir_x: float,
    dir_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place a canonical +X-facing target window at (ground_x, 0, ground_z) facing (dir_x, dir_z)."""
    mag = math.hypot(dir_x, dir_z)
    if mag < 1e-6:
        dir_x, dir_z = 1.0, 0.0
    else:
        dir_x /= mag
        dir_z /= mag
    # Right-handed Y rotation mapping (1, 0, 0) -> (dir_x, 0, dir_z):
    # x' = cos(t) = dir_x, z' = -sin(t) = dir_z => t = atan2(-dir_z, dir_x)
    yaw = math.atan2(-dir_z, dir_x)
    q_place = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), yaw)
    p_place = np.array([ground_x, 0.0, ground_z], dtype=np.float32)

    out_p = canonical_pos.copy()
    out_q = canonical_quat.copy()
    for i in range(out_p.shape[0]):
        out_p[i, 0], out_q[i, 0] = transform_mul(p_place, q_place, out_p[i, 0], out_q[i, 0])
    return out_p, out_q


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    bvh_idle = load_bvh(root_dir / "assets/bvh/idle_loop_001__A268.bvh")
    skel = strip_root_skeleton(bvh_idle.skeleton)
    clip_idle = load_bvh(root_dir / "assets/bvh/idle_loop_001__A268.bvh", skel)
    clip_walk = load_bvh(root_dir / "assets/bvh/Neutral_walk_forward_002__A057.bvh", skel)

    engine = SOMAOnnxEngine(
        root_dir / "models/soma/mmm_v6_soma_fp16.onnx",
        root_dir / "models/soma/soma_n1.json",
        skel,
    )
    print(f"Loaded full FP16 ONNX in {engine.load_ms:.1f} ms | skeleton joints={skel.num_joints}")

    ctx_p, ctx_q = canonicalize_target_window(
        clip_idle, 0.62, engine.right_leg_idx, engine.left_leg_idx
    )
    walk_canon_p, walk_canon_q = canonicalize_target_window(
        clip_walk, 2.55, engine.right_leg_idx, engine.left_leg_idx, desired_speed=1.5
    )
    tgt_p, tgt_q = place_target_window(walk_canon_p, walk_canon_q, 1.5, 0.0, 1.0, 0.0)

    # Warmup + timed runs
    for run_i in range(3):
        res = engine.generate(ctx_p, ctx_q, tgt_p, tgt_q)
        finite = bool(np.isfinite(res.world_pos).all() and np.isfinite(res.world_quat).all())
        hips_start = res.world_pos[0, 0].round(3)
        hips_mid = res.world_pos[res.num_frames // 2, 0].round(3)
        hips_end = res.world_pos[-1, 0].round(3)
        head_y = float(res.world_pos[:, skel.joint_index("Head"), 1].mean())
        lf_min_y = float(res.world_pos[:, skel.joint_index("LeftFoot"), 1].min())
        rf_min_y = float(res.world_pos[:, skel.joint_index("RightFoot"), 1].min())
        fc_max = res.foot_contacts.max(axis=0).round(2)
        print(
            f"Run {run_i}: total={res.latency_ms:.1f} ms (infer={res.infer_ms:.1f} ms) "
            f"| tokens={res.pred_num_tokens} frames={res.num_frames} "
            f"| finite={finite} | hips {hips_start} -> {hips_mid} -> {hips_end} "
            f"| head_y={head_y:.2f}m foot_min_y=({lf_min_y:.3f},{rf_min_y:.3f}) | fc_max={fc_max}"
        )
