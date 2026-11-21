"""Local real-time HTTP API server for NVIDIA MotionBricks SOMA ONNX (full unquantized FP16).

Runs on http://127.0.0.1:8765 using Python stdlib ThreadingHTTPServer (zero extra deps).
Maintains a chunk history ring buffer of exact 77-joint SOMA local poses per client session
so lookahead prefetch, continuous WASD steering, Click-to-Move, zero-shot Style Switching,
Neural Combat Punches (Left Jab / Right Cross / Combat Recover), and Smart Primitive chains
(Vault Box & Sit Bench) stitch at exact `anchor_frame` with zero snap-back at ~22-25 ms/gen.
"""

from __future__ import annotations

import json
import math
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

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
    transform_mul,
)
from soma_onnx import (
    MODEL_POSES_PER_TOKEN,
    MODEL_SAMPLE_RATE,
    GenerationResult,
    SOMAOnnxEngine,
    _heading_from_hip_positions,
    _rotation_matrix_to_quat_xyzw,
    canonicalize_target_window,
    place_target_window,
    strip_root_skeleton,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
BVH_DIR = ROOT_DIR / "assets/bvh"
ONNX_PATH = ROOT_DIR / "models/soma/mmm_v6_soma_fp16.onnx"
MANIFEST_PATH = ROOT_DIR / "models/soma/soma_n1.json"

# 24 clean connected joints sent for world-position skeleton / ghost rendering
RENDER_JOINT_NAMES: list[str] = [
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "HeadEnd",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
]

# Exact 26 connected SOMA model joints from soma_n1.json
# (all 30 except the 4 disconnected finger tips: Left/RightHandThumbEnd, Left/RightHandMiddleEnd)
BONE_JOINT_NAMES: list[str] = [
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
]

CHUNK_RING_BUFFER_MAX = 5

STYLE_SPECS: list[dict[str, Any]] = [
    {
        "id": "walk",
        "label": "Walk",
        "clip": "Neutral_walk_forward_002__A057.bvh",
        "time": 2.55,
        "speed": 1.75,
        "lead_s": 0.68,
        "accent": "#38bdf8",
    },
    {
        "id": "jog",
        "label": "Jog",
        "clip": "neutral_jog_ff_180_R_001__A534.bvh",
        "time": 1.91,
        "speed": 2.45,
        "lead_s": 0.68,
        "accent": "#22d3ee",
    },
    {
        "id": "run",
        "label": "Sprint",
        "clip": "run_loop_180_R_003__A326.bvh",
        "time": 0.41,
        "speed": 3.85,
        "lead_s": 0.66,
        "accent": "#f97316",
    },
    {
        "id": "crouch",
        "label": "Crouch",
        "clip": "crouch_ff_loop_180_R_001__A196.bvh",
        "time": 1.20,
        "speed": 1.45,
        "lead_s": 0.68,
        "accent": "#a3e635",
    },
    {
        "id": "skip",
        "label": "Skip",
        "clip": "skipping_R_002__A360.bvh",
        "time": 3.19,
        "speed": 2.10,
        "lead_s": 0.72,
        "accent": "#c084fc",
    },
    {
        "id": "combat",
        "label": "Combat",
        "clip": "combat_quickstep_ff_idle_180_R_002__A365.bvh",
        "time": 3.66,
        "speed": 1.65,
        "lead_s": 0.68,
        "accent": "#f43f5e",
    },
    {
        "id": "tactical",
        "label": "Tactical",
        "clip": "angry_gun_walk_ff_loop_090_R_001__A393.bvh",
        "time": 1.10,
        "speed": 1.60,
        "lead_s": 0.70,
        "accent": "#fb923c",
    },
    {
        "id": "vogue",
        "label": "Vogue",
        "clip": "dance_vouge_vogue_sequence_180_R_002__A316.bvh",
        "time": 3.13,
        "speed": 1.55,
        "lead_s": 0.72,
        "accent": "#facc15",
    },
    {
        "id": "zombie",
        "label": "Zombie",
        "clip": "zombie_walk_180_R_003__A330.bvh",
        "time": 6.04,
        "speed": 1.10,
        "lead_s": 0.78,
        "accent": "#34d399",
    },
    {
        "id": "injured",
        "label": "Injured",
        "clip": "injured_L_leg_grab_walk_ff_180_002__A549.bvh",
        "time": 5.55,
        "speed": 1.15,
        "lead_s": 0.75,
        "accent": "#fb7185",
    },
    {
        "id": "crawl",
        "label": "Crawl",
        "clip": "crouch_ff_loop_180_R_001__A196.bvh",
        "time": 1.00,
        "speed": 0.74,
        "lead_s": 0.76,
        "accent": "#22d3ee",
    },
]

# Solid inner playable room bounds matching web/src/studioSpec.ts ROOM_BOUNDS (15.4 m x 25.3 m enclosed hall)
ROOM_MIN_X = -7.15
ROOM_MAX_X = 6.70
ROOM_MIN_Z = -16.85
ROOM_MAX_Z = 7.12


def _clamp_room_xz(x: float, z: float) -> tuple[float, float]:
    """Clamp world XZ strictly inside the architectural pavilion walls."""
    return (
        float(min(ROOM_MAX_X, max(ROOM_MIN_X, x))),
        float(min(ROOM_MAX_Z, max(ROOM_MIN_Z, z))),
    )


def _clip_room_step(
    cx: float, cz: float, dx: float, dz: float, max_dist: float
) -> tuple[float, float, float]:
    """Ray-clip a desired forward step against all 4 concrete walls, returning (tx, tz, clipped_dist)."""
    t = max_dist
    if dx < -1e-5:
        t = min(t, max(0.0, (ROOM_MIN_X - cx) / dx))
    elif dx > 1e-5:
        t = min(t, max(0.0, (ROOM_MAX_X - cx) / dx))
    if dz < -1e-5:
        t = min(t, max(0.0, (ROOM_MIN_Z - cz) / dz))
    elif dz > 1e-5:
        t = min(t, max(0.0, (ROOM_MAX_Z - cz) / dz))
    tx, tz = _clamp_room_xz(cx + dx * t, cz + dz * t)
    return tx, tz, max(0.0, t)


# Interactive Gym Smart Primitives: spacious foreground nave + new obstacle types + tight rear stress-test gauntlet
GYM_OBSTACLES: list[dict[str, Any]] = [
    {
        "id": "vault_box",
        "type": "vault",
        "label": "Parkour Vault Plinth",
        "pos": [-2.35, 0.0, -2.5],
        # Spans 2.2 m along Z, 0.34 m thick along X, 0.82 m tall -> vault across X!
        "size": [0.34, 0.82, 2.2],
        "color": "#facc15",
    },
    {
        "id": "sit_bench",
        "type": "sit",
        "label": "Minimalist Gallery Bench",
        "pos": [1.85, 0.0, -2.9],
        # Seat center at (1.85, 0.22, -2.9), angled toward center-front; OBB auto-hurdles if run into, Triangle/E sits
        "size": [0.52, 0.44, 0.72],
        "facing": [-0.8192, 0.5736],
        "color": "#f472b6",
    },
    {
        "id": "duck_beam",
        "type": "duck",
        "label": "Overhead Duck Gantry",
        "pos": [0.0, 0.0, -5.6],
        # Spans 2.2 m wide along X at far end of main runway, clearance y=1.35 m -> duck under along Z!
        "size": [2.2, 0.42, 0.36],
        "clearance": 1.35,
        "color": "#2dd4bf",
    },
    {
        "id": "hurdle_rail",
        "type": "hurdle",
        "label": "Low Steel Hurdle Rail (Left)",
        "pos": [-4.4, 0.0, -6.8],
        # Spacious left wide-lane low athletic hurdle (0.48 m tall) -> OBB auto-hurdles!
        "size": [2.0, 0.48, 0.16],
        "color": "#f59e0b",
    },
    {
        "id": "hurdle_rail_side",
        "type": "hurdle",
        "label": "Low Steel Hurdle Rail (Right Sideways)",
        "pos": [4.3, 0.0, -8.2],
        # Second low athletic hurdle placed sideways on the open right flank (~8.8 m from left hurdle)
        "size": [0.16, 0.48, 2.0],
        "color": "#f59e0b",
    },
    # ---- Center Runway Parkour Gauntlet (Duck -> Vault -> Duck, clean open spacing) ----
    {
        "id": "stress_vault",
        "type": "vault",
        "label": "Center Concrete Vault Wall",
        "pos": [0.0, 0.0, -9.5],
        # Spans 2.1 m along X, 0.34 m thick along Z -> centered 3.9 m between both duck gates!
        "size": [2.1, 0.82, 0.34],
        "color": "#facc15",
    },
    {
        "id": "stress_duck",
        "type": "duck",
        "label": "Stress Gauntlet: Low Tunnel Gate",
        "pos": [0.0, 0.0, -13.4],
        "size": [2.2, 0.42, 0.36],
        "clearance": 1.32,
        "color": "#2dd4bf",
    },
]


def _rotate_window_xz(
    lp: np.ndarray,
    lq: np.ndarray,
    pivot_xz: tuple[float, float],
    rot_y_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate root XZ by -pivot_xz and rotate around +Y by rot_y_rad."""
    out_p = lp.copy()
    out_q = lq.copy()
    q_rot = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), rot_y_rad)
    piv = np.array([pivot_xz[0], 0.0, pivot_xz[1]], dtype=np.float32)
    for i in range(out_p.shape[0]):
        out_p[i, 0] = quat_rotate(q_rot, out_p[i, 0] - piv)
        out_q[i, 0] = quat_norm(quat_mul(q_rot, out_q[i, 0]))
    return out_p, out_q


def _align_window_by_head_gaze(
    lp: np.ndarray,
    lq: np.ndarray,
    skeleton: Skeleton,
) -> tuple[np.ndarray, np.ndarray]:
    """Center root XZ at (0, 0) and rotate around +Y so Head gaze (local +Y) faces +X."""
    wp, wq = compute_world_poses(lp, lq, skeleton.parents)
    head_idx = skeleton.joint_index("Head")
    head_fwd = quat_rotate(wq[0, head_idx], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    # Right-handed Y rotation mapping (head_fwd.x, head_fwd.z) -> (+1, 0): rot_y = atan2(z, x)
    rot_y = math.atan2(float(head_fwd[2]), float(head_fwd[0]))
    return _rotate_window_xz(lp, lq, (float(lp[0, 0, 0]), float(lp[0, 0, 2])), rot_y)


def _apply_world_joint_delta(
    lp: np.ndarray,
    lq: np.ndarray,
    skeleton: Skeleton,
    joint_name: str,
    world_axis: tuple[float, float, float],
    deg: float,
) -> None:
    """Apply an incremental rotation around a world-space axis to `joint_name` in-place."""
    j = skeleton.joint_index(joint_name)
    p = int(skeleton.parents[j])
    _, wq = compute_world_poses(lp, lq, skeleton.parents)
    axis = np.asarray(world_axis, dtype=np.float32)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    q_delta = quat_from_axis_angle(axis, math.radians(deg))
    new_wq = quat_norm(quat_mul(q_delta[None, :], wq[:, j]))
    if p < 0:
        lq[:, j] = new_wq
    else:
        lq[:, j] = quat_norm(quat_mul(quat_conj(wq[:, p]), new_wq))


def _set_world_joint_quat(
    lp: np.ndarray,
    lq: np.ndarray,
    skeleton: Skeleton,
    joint_name: str,
    target_wq: np.ndarray,
) -> None:
    """Set exact world-space orientation `target_wq` ([4] or [F, 4]) on `joint_name` in-place."""
    j = skeleton.joint_index(joint_name)
    p = int(skeleton.parents[j])
    _, wq = compute_world_poses(lp, lq, skeleton.parents)
    t_wq = np.broadcast_to(quat_norm(target_wq), (lq.shape[0], 4))
    if p < 0:
        lq[:, j] = t_wq.copy()
    else:
        lq[:, j] = quat_norm(quat_mul(quat_conj(wq[:, p]), t_wq))


def _make_arm_world_quat(
    bone_dir_world: tuple[float, float, float],
    elbow_hinge_world: tuple[float, float, float],
    is_right: bool = False,
) -> np.ndarray:
    """Build exact world XYZW quaternion for SOMA LeftArm (+X along bone) or RightArm (-X along bone),
    with local +Z aligned to `elbow_hinge_world` (SOMA's positive elbow flexion hinge axis)."""
    d = np.asarray(bone_dir_world, dtype=np.float32)
    d = d / max(float(np.linalg.norm(d)), 1e-8)
    x_axis = -d if is_right else d
    z_hint = np.asarray(elbow_hinge_world, dtype=np.float32)
    z_axis = z_hint - float(np.dot(z_hint, x_axis)) * x_axis
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-8)
    R = np.stack([x_axis, y_axis, z_axis], axis=-1).astype(np.float32)
    return _rotation_matrix_to_quat_xyzw(R)


def _build_combat_poses(
    combat_clip: AnimationClip,
    skeleton: Skeleton,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Synthesize canonical +X-facing combat poses from `combat_quickstep_ff_idle_180_R_002__A365.bvh`:
    - combat_guard: balanced orthodox guard stance (t=6.50s, both feet flat, chin guarded)
    - punch_left:   crisp Left Jab (lead foot stepped forward, clockwise torso snap, right chin guard)
    - punch_right:  heavy Right Cross (full counter-clockwise torso rotation, left cheek guard)

    Verified against SOMA BVH local bone axes:
    - Spine1/Spine2/Chest/Head: +X up, +Y forward, +Z left
    - LeftArm/LeftForeArm/LeftHand: bone along +X, elbow flexion around +Z, thumb along +Y
    - RightArm/RightForeArm/RightHand: bone along -X (mirrored), elbow flexion around +Z, thumb along -Y
    """
    # 1. Tight Orthodox Boxing Guard (t=6.50s base, elbows tucked to ribs ~1.14m, fists at jawline ~1.34m)
    raw_guard_p, raw_guard_q = combat_clip.sample_window_30hz(6.50, MODEL_POSES_PER_TOKEN)
    guard_p, guard_q = _align_window_by_head_gaze(raw_guard_p, raw_guard_q, skeleton)
    q_la_guard_base = _make_arm_world_quat((0.42, -0.78, -0.46), (0.15, -0.35, 0.92), is_right=False)
    q_ra_guard_base = _make_arm_world_quat((0.28, -0.82, 0.48), (-0.15, 0.35, 0.92), is_right=True)
    _set_world_joint_quat(guard_p, guard_q, skeleton, "LeftArm", q_la_guard_base)
    _set_world_joint_quat(guard_p, guard_q, skeleton, "RightArm", q_ra_guard_base)
    guard_q[:, skeleton.joint_index("LeftForeArm")] = quat_from_axis_angle(
        np.array([0.0, 0.0, 1.0], dtype=np.float32), math.radians(118.0)
    )
    guard_q[:, skeleton.joint_index("RightForeArm")] = quat_from_axis_angle(
        np.array([0.0, 0.0, 1.0], dtype=np.float32), math.radians(122.0)
    )

    # Athletic forward-stepped combat stance (t=1.25s: hips dropped to 0.92m, front foot forward +0.25m,
    # rear foot pushing off ball of foot) as base for explosive strikes
    raw_step_p, raw_step_q = combat_clip.sample_window_30hz(1.25, MODEL_POSES_PER_TOKEN)
    step_p, step_q = _align_window_by_head_gaze(raw_step_p, raw_step_q, skeleton)

    # 2. Explosive Straight Left Jab (punch_left — shoulder/chin level y=1.30m, 0.78m reach, rear fist on jaw)
    pl_p, pl_q = step_p.copy(), step_q.copy()
    _apply_world_joint_delta(pl_p, pl_q, skeleton, "Spine1", (0.0, 1.0, 0.0), -10.0)
    _apply_world_joint_delta(pl_p, pl_q, skeleton, "Spine2", (0.0, 1.0, 0.0), -14.0)
    _apply_world_joint_delta(pl_p, pl_q, skeleton, "Chest", (0.0, 1.0, 0.0), -16.0)
    _apply_world_joint_delta(pl_p, pl_q, skeleton, "Spine2", (0.0, 0.0, 1.0), -9.0)  # athletic lean into jab
    _apply_world_joint_delta(pl_p, pl_q, skeleton, "Neck1", (0.0, 1.0, 0.0), 18.0)
    _apply_world_joint_delta(pl_p, pl_q, skeleton, "Head", (0.0, 1.0, 0.0), 20.0)
    _apply_world_joint_delta(pl_p, pl_q, skeleton, "Head", (0.0, 0.0, 1.0), 8.0)     # chin tucked behind shoulder
    _apply_world_joint_delta(pl_p, pl_q, skeleton, "LeftShoulder", (0.0, 1.0, 0.0), -24.0)
    q_la_jab = _make_arm_world_quat((0.995, -0.07, 0.06), (0.0, -0.88, 0.47), is_right=False)
    _set_world_joint_quat(pl_p, pl_q, skeleton, "LeftArm", q_la_jab)
    pl_q[:, skeleton.joint_index("LeftForeArm")] = quat_from_axis_angle(
        np.array([0.0, 0.0, 1.0], dtype=np.float32), math.radians(7.0)
    )
    pl_q[:, skeleton.joint_index("LeftHand")] = quat_from_axis_angle(
        np.array([1.0, 0.0, 0.0], dtype=np.float32), math.radians(35.0)
    )
    _apply_world_joint_delta(pl_p, pl_q, skeleton, "RightShoulder", (0.0, 1.0, 0.0), 12.0)
    q_ra_chin = _make_arm_world_quat((0.25, -0.84, 0.48), (-0.1, 0.35, 0.93), is_right=True)
    _set_world_joint_quat(pl_p, pl_q, skeleton, "RightArm", q_ra_chin)
    pl_q[:, skeleton.joint_index("RightForeArm")] = quat_from_axis_angle(
        np.array([0.0, 0.0, 1.0], dtype=np.float32), math.radians(126.0)
    )

    # 3. Heavy Straight Right Cross (punch_right — full torque, centerline y=1.29m, 0.79m reach, lead fist on cheek)
    pr_p, pr_q = step_p.copy(), step_q.copy()
    _apply_world_joint_delta(pr_p, pr_q, skeleton, "Spine1", (0.0, 1.0, 0.0), 18.0)
    _apply_world_joint_delta(pr_p, pr_q, skeleton, "Spine2", (0.0, 1.0, 0.0), 22.0)
    _apply_world_joint_delta(pr_p, pr_q, skeleton, "Chest", (0.0, 1.0, 0.0), 24.0)
    _apply_world_joint_delta(pr_p, pr_q, skeleton, "Spine2", (0.0, 0.0, 1.0), -11.0)  # drive bodyweight forward +X
    _apply_world_joint_delta(pr_p, pr_q, skeleton, "Neck1", (0.0, 1.0, 0.0), -30.0)
    _apply_world_joint_delta(pr_p, pr_q, skeleton, "Head", (0.0, 1.0, 0.0), -32.0)
    _apply_world_joint_delta(pr_p, pr_q, skeleton, "Head", (0.0, 0.0, 1.0), 10.0)
    _apply_world_joint_delta(pr_p, pr_q, skeleton, "RightShoulder", (0.0, 1.0, 0.0), 28.0)
    q_ra_cross = _make_arm_world_quat((0.995, -0.08, -0.07), (0.0, 0.88, 0.47), is_right=True)
    _set_world_joint_quat(pr_p, pr_q, skeleton, "RightArm", q_ra_cross)
    pr_q[:, skeleton.joint_index("RightForeArm")] = quat_from_axis_angle(
        np.array([0.0, 0.0, 1.0], dtype=np.float32), math.radians(7.0)
    )
    pr_q[:, skeleton.joint_index("RightHand")] = quat_from_axis_angle(
        np.array([1.0, 0.0, 0.0], dtype=np.float32), math.radians(-35.0)
    )
    # Tuck LeftArm tight against left cheekbone as left shoulder retracts
    _apply_world_joint_delta(pr_p, pr_q, skeleton, "LeftShoulder", (0.0, 1.0, 0.0), -14.0)
    q_la_guard = _make_arm_world_quat((0.25, -0.84, -0.48), (0.1, -0.35, 0.93), is_right=False)
    _set_world_joint_quat(pr_p, pr_q, skeleton, "LeftArm", q_la_guard)
    pr_q[:, skeleton.joint_index("LeftForeArm")] = quat_from_axis_angle(
        np.array([0.0, 0.0, 1.0], dtype=np.float32), math.radians(126.0)
    )

    return (guard_p, guard_q), (pl_p, pl_q), (pr_p, pr_q)


def _smoothstep(t: float) -> float:
    """C1 smoothstep ease on [0, 1]."""
    x = float(np.clip(t, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _make_leg_world_quat(
    bone_dir_world: tuple[float, float, float],
    knee_hinge_world: tuple[float, float, float] = (0.0, 0.0, -1.0),
    is_right: bool = False,
) -> np.ndarray:
    """Build exact world XYZW quaternion for SOMA LeftLeg (+X along bone) or RightLeg (-X along bone),
    with local +Z aligned to `knee_hinge_world` (world -Z = left, positive knee flexion bends shin backward)."""
    d = np.asarray(bone_dir_world, dtype=np.float32)
    d = d / max(float(np.linalg.norm(d)), 1e-8)
    x_axis = -d if is_right else d
    z_hint = np.asarray(knee_hinge_world, dtype=np.float32)
    z_axis = z_hint - float(np.dot(z_hint, x_axis)) * x_axis
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-8)
    R = np.stack([x_axis, y_axis, z_axis], axis=-1).astype(np.float32)
    return _rotation_matrix_to_quat_xyzw(R)


def _make_hand_world_quat(
    finger_dir_world: tuple[float, float, float] = (1.0, 0.0, 0.0),
    palm_dir_world: tuple[float, float, float] = (0.0, -1.0, 0.0),
    is_right: bool = False,
) -> np.ndarray:
    """Build exact world XYZW quaternion for SOMA LeftHand (+X fingers, +Z palm normal) or
    RightHand (-X fingers, -Z palm normal), keeping thumb inward toward body centerline."""
    f = np.asarray(finger_dir_world, dtype=np.float32)
    f = f / max(float(np.linalg.norm(f)), 1e-8)
    p = np.asarray(palm_dir_world, dtype=np.float32)
    p = p - float(np.dot(p, f)) * f
    p = p / max(float(np.linalg.norm(p)), 1e-8)
    x_axis = -f if is_right else f
    z_axis = -p if is_right else p
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-8)
    R = np.stack([x_axis, y_axis, z_axis], axis=-1).astype(np.float32)
    return _rotation_matrix_to_quat_xyzw(R)


def _build_roll_and_crawl_poses(
    idle_canon: tuple[np.ndarray, np.ndarray],
    skeleton: Skeleton,
    engine: SOMAOnnxEngine,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Synthesize canonical +X-facing poses from `idle_canon` (NEVER wide-splay `crouch_idle`!):
    - roll_dive_q, roll_tuck_q, roll_plant_q: [J, 4] Hips-relative body-local key poses for the
      single continuous 26-frame 360-deg diagonal parkour shoulder combat roll (`mode == "roll"`).
    - crawl_a_canon, crawl_b_canon: alternating contralateral hands-and-knees low tactical crawl
      with narrow hip-width knees, near-straight pillar arms, and flat palm-down hands.
    """
    z_ax = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    le = skeleton.joint_index("LeftForeArm")
    re = skeleton.joint_index("RightForeArm")
    lk = skeleton.joint_index("LeftShin")
    rk = skeleton.joint_index("RightShin")
    lh = skeleton.joint_index("LeftHand")
    rh = skeleton.joint_index("RightHand")
    finger_idxs = [
        i
        for i, n in enumerate(skeleton.names)
        if any(k in n for k in ("HandThumb", "HandIndex", "HandMiddle", "HandRing", "HandPinky"))
    ]

    # --- 1. Roll Key Poses (built strictly from narrow-knee `idle_canon`) ---
    # 1a. Dive-in key pose (f~4..6): low crouched dive, lead right arm curving forward-inward wheel, chin tucked left
    dp, dq = idle_canon[0][:1].copy(), idle_canon[1][:1].copy()
    _apply_world_joint_delta(dp, dq, skeleton, "Spine1", (0.0, 0.0, 1.0), -16.0)
    _apply_world_joint_delta(dp, dq, skeleton, "Spine2", (0.0, 0.0, 1.0), -20.0)
    _apply_world_joint_delta(dp, dq, skeleton, "Chest", (0.0, 0.0, 1.0), -18.0)
    _apply_world_joint_delta(dp, dq, skeleton, "Neck1", (0.0, 0.0, 1.0), -14.0)
    _apply_world_joint_delta(dp, dq, skeleton, "Neck1", (0.0, 1.0, 0.0), 14.0)
    _apply_world_joint_delta(dp, dq, skeleton, "Head", (0.0, 0.0, 1.0), -20.0)
    _apply_world_joint_delta(dp, dq, skeleton, "Head", (0.0, 1.0, 0.0), 16.0)
    _set_world_joint_quat(
        dp, dq, skeleton, "LeftLeg", _make_leg_world_quat((0.78, -0.62, -0.01), (0.0, 0.0, -1.0), is_right=False)
    )
    _set_world_joint_quat(
        dp, dq, skeleton, "RightLeg", _make_leg_world_quat((0.84, -0.54, 0.01), (0.0, 0.0, -1.0), is_right=True)
    )
    dq[:, lk] = quat_from_axis_angle(z_ax, math.radians(78.0))
    dq[:, rk] = quat_from_axis_angle(z_ax, math.radians(72.0))
    _set_world_joint_quat(
        dp, dq, skeleton, "RightArm", _make_arm_world_quat((0.66, -0.70, -0.26), (0.0, 0.15, 0.99), is_right=True)
    )
    _set_world_joint_quat(
        dp, dq, skeleton, "LeftArm", _make_arm_world_quat((0.52, -0.84, -0.14), (0.0, 0.0, 1.0), is_right=False)
    )
    dq[:, re] = quat_from_axis_angle(z_ax, math.radians(66.0))
    dq[:, le] = quat_from_axis_angle(z_ax, math.radians(48.0))

    # 1b. Tuck key pose (f~9..15): compact diagonal ball, strictly sagittal thighs (~108 deg, knee_sep ~0.22 m),
    # shins tucked ~123 deg, chin locked to left collarbone, arms hugged tightly against outer shins/chest!
    tp, tq = idle_canon[0][:1].copy(), idle_canon[1][:1].copy()
    _apply_world_joint_delta(tp, tq, skeleton, "Spine1", (0.0, 0.0, 1.0), -22.0)
    _apply_world_joint_delta(tp, tq, skeleton, "Spine2", (0.0, 0.0, 1.0), -26.0)
    _apply_world_joint_delta(tp, tq, skeleton, "Chest", (0.0, 0.0, 1.0), -24.0)
    _apply_world_joint_delta(tp, tq, skeleton, "Neck1", (0.0, 0.0, 1.0), -18.0)
    _apply_world_joint_delta(tp, tq, skeleton, "Neck1", (0.0, 1.0, 0.0), 16.0)
    _apply_world_joint_delta(tp, tq, skeleton, "Neck2", (0.0, 0.0, 1.0), -14.0)
    _apply_world_joint_delta(tp, tq, skeleton, "Head", (0.0, 0.0, 1.0), -22.0)
    _apply_world_joint_delta(tp, tq, skeleton, "Head", (0.0, 1.0, 0.0), 18.0)
    _set_world_joint_quat(
        tp, tq, skeleton, "LeftLeg", _make_leg_world_quat((0.94, 0.34, -0.01), (0.0, 0.0, -1.0), is_right=False)
    )
    _set_world_joint_quat(
        tp, tq, skeleton, "RightLeg", _make_leg_world_quat((0.96, 0.28, 0.01), (0.0, 0.0, -1.0), is_right=True)
    )
    tq[:, lk] = quat_from_axis_angle(z_ax, math.radians(124.0))
    tq[:, rk] = quat_from_axis_angle(z_ax, math.radians(122.0))
    _set_world_joint_quat(
        tp, tq, skeleton, "LeftFoot", _make_leg_world_quat((-0.25, -0.97, 0.0), (0.0, 0.0, -1.0), is_right=False)
    )
    _set_world_joint_quat(
        tp, tq, skeleton, "RightFoot", _make_leg_world_quat((-0.25, -0.97, 0.0), (0.0, 0.0, -1.0), is_right=True)
    )
    # Compact body-local tuck arms hugging outer shins/chest (||Hand - Hips|| ~0.38 m -> zero overhead flip!)
    _set_world_joint_quat(
        tp, tq, skeleton, "LeftArm", _make_arm_world_quat((-0.30, -0.94, 0.08), (-0.18, -0.15, 0.97), is_right=False)
    )
    _set_world_joint_quat(
        tp, tq, skeleton, "RightArm", _make_arm_world_quat((-0.30, -0.94, -0.08), (0.18, 0.15, 0.97), is_right=True)
    )
    tq[:, le] = quat_from_axis_angle(z_ax, math.radians(116.0))
    tq[:, re] = quat_from_axis_angle(z_ax, math.radians(116.0))

    # 1c. Athletic spring-plant key pose (f~18): deep crouched narrow-knee recovery matching jog frame 0 stride
    pp, pq = idle_canon[0][:1].copy(), idle_canon[1][:1].copy()
    _apply_world_joint_delta(pp, pq, skeleton, "Spine1", (0.0, 0.0, 1.0), -14.0)
    _apply_world_joint_delta(pp, pq, skeleton, "Spine2", (0.0, 0.0, 1.0), -16.0)
    _apply_world_joint_delta(pp, pq, skeleton, "Chest", (0.0, 0.0, 1.0), -12.0)
    _apply_world_joint_delta(pp, pq, skeleton, "Neck1", (0.0, 0.0, 1.0), 12.0)
    _apply_world_joint_delta(pp, pq, skeleton, "Head", (0.0, 0.0, 1.0), 18.0)
    _set_world_joint_quat(
        pp, pq, skeleton, "LeftLeg", _make_leg_world_quat((0.92, -0.39, -0.01), (0.0, 0.0, -1.0), is_right=False)
    )
    _set_world_joint_quat(
        pp, pq, skeleton, "RightLeg", _make_leg_world_quat((0.76, -0.65, 0.01), (0.0, 0.0, -1.0), is_right=True)
    )
    pq[:, lk] = quat_from_axis_angle(z_ax, math.radians(102.0))
    pq[:, rk] = quat_from_axis_angle(z_ax, math.radians(106.0))
    _set_world_joint_quat(
        pp, pq, skeleton, "LeftFoot", _make_leg_world_quat((0.96, -0.28, 0.0), (0.0, 0.0, -1.0), is_right=False)
    )
    _set_world_joint_quat(
        pp, pq, skeleton, "RightFoot", _make_leg_world_quat((0.68, -0.73, 0.0), (0.0, 0.0, -1.0), is_right=True)
    )
    _set_world_joint_quat(
        pp, pq, skeleton, "LeftArm", _make_arm_world_quat((0.18, -0.96, -0.18), (0.0, 0.0, 1.0), is_right=False)
    )
    _set_world_joint_quat(
        pp, pq, skeleton, "RightArm", _make_arm_world_quat((0.32, -0.93, 0.18), (0.0, 0.0, 1.0), is_right=True)
    )
    pq[:, le] = quat_from_axis_angle(z_ax, math.radians(88.0))
    pq[:, re] = quat_from_axis_angle(z_ax, math.radians(92.0))

    # --- 2. Alternating Low Tactical Hands-and-Knees Crawl (built strictly from `idle_canon`!) ---
    def make_crawl_phase(left_reach: bool) -> tuple[np.ndarray, np.ndarray]:
        cp, cq = idle_canon[0].copy(), idle_canon[1].copy()
        # Flatten fingers completely so palms rest flush on the floor with zero upward scoop
        cq[:, finger_idxs] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        cp[:, 0, 1] = 0.50
        # Horizontal quadruped torso: Hips ~0.50-0.52 m, Chest ~0.52-0.54 m, Head gazing forward ~0.58-0.62 m
        _apply_world_joint_delta(cp, cq, skeleton, "Hips", (0.0, 0.0, 1.0), -89.0)
        _apply_world_joint_delta(cp, cq, skeleton, "Spine1", (0.0, 0.0, 1.0), 1.0)
        _apply_world_joint_delta(cp, cq, skeleton, "Spine2", (0.0, 0.0, 1.0), 1.0)
        _apply_world_joint_delta(cp, cq, skeleton, "Spine2", (0.0, 1.0, 0.0), -5.0 if left_reach else 5.0)
        _apply_world_joint_delta(cp, cq, skeleton, "Chest", (0.0, 0.0, 1.0), 1.0)
        # Subtle scapular elevation so straight supporting pillar arms match grounded knee height cleanly
        _apply_world_joint_delta(cp, cq, skeleton, "LeftShoulder", (1.0, 0.0, 0.0), 18.0)
        _apply_world_joint_delta(cp, cq, skeleton, "RightShoulder", (1.0, 0.0, 0.0), -18.0)
        _apply_world_joint_delta(cp, cq, skeleton, "Neck1", (0.0, 0.0, 1.0), 28.0)
        _apply_world_joint_delta(cp, cq, skeleton, "Neck2", (0.0, 0.0, 1.0), 16.0)
        _apply_world_joint_delta(cp, cq, skeleton, "Head", (0.0, 0.0, 1.0), 22.0)

        # Narrow hip-width contralateral knees (0.21 m lateral, ~0.165 m X-alternation, ~0.265 m 3D sep, grounded ~0.09 m)
        # left_reach=True => Left hand forward + Right knee forward (Left knee trailing)
        l_thigh_dir = (-0.31, -0.95, -0.01) if left_reach else (0.07, -0.997, -0.01)
        r_thigh_dir = (0.07, -0.997, 0.01) if left_reach else (-0.31, -0.95, 0.01)
        _set_world_joint_quat(
            cp, cq, skeleton, "LeftLeg", _make_leg_world_quat(l_thigh_dir, (0.0, 0.0, -1.0), is_right=False)
        )
        _set_world_joint_quat(
            cp, cq, skeleton, "RightLeg", _make_leg_world_quat(r_thigh_dir, (0.0, 0.0, -1.0), is_right=True)
        )
        # Shins bent ~85-105 deg trailing cleanly along -X with ankles/toes grounded safely above floor
        _set_world_joint_quat(
            cp, cq, skeleton, "LeftShin", _make_leg_world_quat((-0.998, 0.03, 0.0), (0.0, 0.0, -1.0), is_right=False)
        )
        _set_world_joint_quat(
            cp, cq, skeleton, "RightShin", _make_leg_world_quat((-0.998, 0.03, 0.0), (0.0, 0.0, -1.0), is_right=True)
        )
        _set_world_joint_quat(
            cp, cq, skeleton, "LeftFoot", _make_leg_world_quat((-0.96, -0.26, 0.0), (0.0, 0.0, -1.0), is_right=False)
        )
        _set_world_joint_quat(
            cp, cq, skeleton, "RightFoot", _make_leg_world_quat((-0.96, -0.26, 0.0), (0.0, 0.0, -1.0), is_right=True)
        )
        _set_world_joint_quat(
            cp, cq, skeleton, "LeftToeBase", _make_leg_world_quat((-0.998, 0.05, 0.0), (0.0, 0.0, -1.0), is_right=False)
        )
        _set_world_joint_quat(
            cp, cq, skeleton, "RightToeBase", _make_leg_world_quat((-0.998, 0.05, 0.0), (0.0, 0.0, -1.0), is_right=True)
        )

        # Near-straight supporting pillar arms (elbow flexion 17-24 deg, elbow_y >= 0.28 m!)
        la_dir = (0.30, -0.945, -0.07) if left_reach else (-0.40, -0.905, -0.14)
        ra_dir = (-0.40, -0.905, 0.14) if left_reach else (0.30, -0.945, 0.07)
        _set_world_joint_quat(
            cp, cq, skeleton, "LeftArm", _make_arm_world_quat(la_dir, (0.0, 0.0, 1.0), is_right=False)
        )
        _set_world_joint_quat(
            cp, cq, skeleton, "RightArm", _make_arm_world_quat(ra_dir, (0.0, 0.0, 1.0), is_right=True)
        )
        cq[:, le] = quat_from_axis_angle(z_ax, math.radians(17.0 if left_reach else 24.0))
        cq[:, re] = quat_from_axis_angle(z_ax, math.radians(24.0 if left_reach else 17.0))

        # Exact flat palm-down hands (fingers straight +X, palm normal straight -Y)
        _set_world_joint_quat(
            cp, cq, skeleton, "LeftHand", _make_hand_world_quat((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), is_right=False)
        )
        _set_world_joint_quat(
            cp, cq, skeleton, "RightHand", _make_hand_world_quat((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), is_right=True)
        )

        # Ground lowest supporting palm flush at y = 0.048 m (keeps knees ~0.09 m, feet >= 0.08 m, toes >= 0.05 m)
        wp_c, _ = engine.fast_fk(cp[:1], cq[:1])
        min_hand = float(min(wp_c[0, lh, 1], wp_c[0, rh, 1]))
        cp[:, 0, 1] += 0.048 - min_hand
        # Steady low crawl cadence velocity across the 4-frame canonical target window
        for i in range(cp.shape[0]):
            cp[i, 0, 0] = i * (0.72 / MODEL_SAMPLE_RATE)
        return cp, cq

    return dq[0], tq[0], pq[0], make_crawl_phase(True), make_crawl_phase(False)


@dataclass
class SessionState:
    chunk_id: int = 0
    crawl_phase: int = 0
    # Ring buffer of recent chunks keyed by chunk_id (evicts oldest when len > CHUNK_RING_BUFFER_MAX):
    # chunk_id -> (local_pos[F, J, 3], local_quat[F, J, 4], prev_history_pos[4, J, 3], prev_history_quat[4, J, 4])
    chunks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    # Latest chunk references
    local_pos: np.ndarray | None = None    # [F, J, 3]
    local_quat: np.ndarray | None = None   # [F, J, 4]
    history_pos: np.ndarray | None = None  # [4, J, 3]
    history_quat: np.ndarray | None = None # [4, J, 4]


class MotionBricksRuntime:
    """Loads full unquantized FP16 ONNX + SOMA BVH seeds and serves real-time chunks."""

    def __init__(self) -> None:
        t0 = time.perf_counter()
        base_clip = load_bvh(BVH_DIR / "idle_loop_001__A268.bvh")
        self.skeleton: Skeleton = strip_root_skeleton(base_clip.skeleton)
        self.engine = SOMAOnnxEngine(ONNX_PATH, MANIFEST_PATH, self.skeleton)
        self.head_idx = self.skeleton.joint_index("Head")
        assert self.head_idx >= 0

        # 24 render joints (world positions for ghost / skeleton viz)
        self.render_indices = np.array(
            [self.skeleton.joint_index(n) for n in RENDER_JOINT_NAMES], dtype=np.int32
        )
        assert np.all(self.render_indices >= 0) and len(self.render_indices) == 24
        idx_map = {int(full_i): rend_i for rend_i, full_i in enumerate(self.render_indices)}
        self.render_parents = [
            -1 if int(self.skeleton.parents[fi]) < 0 else idx_map.get(int(self.skeleton.parents[fi]), 0)
            for fi in self.render_indices
        ]

        # Exact 26 connected SOMA model joints (local XYZW quaternions for direct hierarchical rig drive)
        self.bone_indices = np.array(
            [self.skeleton.joint_index(n) for n in BONE_JOINT_NAMES], dtype=np.int32
        )
        assert np.all(self.bone_indices >= 0) and len(self.bone_indices) == 26
        bone_idx_map = {int(full_i): b_i for b_i, full_i in enumerate(self.bone_indices)}
        self.bone_parents = [
            -1 if int(self.skeleton.parents[fi]) < 0 else bone_idx_map[int(self.skeleton.parents[fi])]
            for fi in self.bone_indices
        ]
        self.bone_ref_pos = np.round(self.skeleton.ref_pos[self.bone_indices], 6).tolist()

        # Cache loaded clips
        self.clips: dict[str, AnimationClip] = {}
        self.styles_canon: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.style_meta: dict[str, dict[str, Any]] = {}

        # Load Idle + Crouch Idle
        idle_clip = self._get_clip("idle_loop_001__A268.bvh")
        self.idle_canon = canonicalize_target_window(
            idle_clip, 0.62, self.engine.right_leg_idx, self.engine.left_leg_idx, desired_speed=0.0
        )
        crouch_idle_clip = self._get_clip("crouch_idle_001__A529.bvh")
        self.crouch_idle_canon = canonicalize_target_window(
            crouch_idle_clip, 0.50, self.engine.right_leg_idx, self.engine.left_leg_idx, desired_speed=0.0
        )

        # Load all locomotion styles
        for spec in STYLE_SPECS:
            clip = self._get_clip(spec["clip"])
            canon = canonicalize_target_window(
                clip,
                spec["time"],
                self.engine.right_leg_idx,
                self.engine.left_leg_idx,
                desired_speed=spec["speed"],
            )
            self.styles_canon[spec["id"]] = canon
            self.style_meta[spec["id"]] = spec

        # Precompute Real Neural Combat Guard + Left Jab + Right Cross
        combat_clip = self._get_clip("combat_quickstep_ff_idle_180_R_002__A365.bvh")
        (
            self.combat_guard_canon,
            self.punch_left_canon,
            self.punch_right_canon,
        ) = _build_combat_poses(combat_clip, self.skeleton)

        # Precompute 360-deg Diagonal Parkour Combat Roll key poses + Narrow-Knee Tactical Crawl (all from `idle_canon`)
        (
            self.roll_dive_q,
            self.roll_tuck_q,
            self.roll_plant_q,
            self.crawl_a_canon,
            self.crawl_b_canon,
        ) = _build_roll_and_crawl_poses(self.idle_canon, self.skeleton, self.engine)
        self.styles_canon["crawl"] = self.crawl_a_canon
        self.crawl_phase = 0
        # Precompute wheel-envelope contact radii for continuous 360-deg combat roll grounding
        self._roll_env_specs: list[tuple[int, float]] = [
            (self.skeleton.joint_index("RightShoulder"), 0.095),
            (self.skeleton.joint_index("LeftShoulder"), 0.095),
            (self.skeleton.joint_index("Spine2"), 0.120),
            (self.skeleton.joint_index("Spine1"), 0.120),
            (self.skeleton.joint_index("Chest"), 0.115),
            (self.skeleton.joint_index("Hips"), 0.140),
            (self.skeleton.joint_index("LeftFoot"), 0.068),
            (self.skeleton.joint_index("RightFoot"), 0.068),
            (self.skeleton.joint_index("LeftToeBase"), 0.035),
            (self.skeleton.joint_index("RightToeBase"), 0.035),
            (self.skeleton.joint_index("LeftToeEnd"), 0.015),
            (self.skeleton.joint_index("RightToeEnd"), 0.015),
            (self.skeleton.joint_index("LeftHand"), 0.045),
            (self.skeleton.joint_index("RightHand"), 0.045),
            (self.skeleton.joint_index("LeftForeArm"), 0.055),
            (self.skeleton.joint_index("RightForeArm"), 0.055),
            (self.skeleton.joint_index("LeftShin"), 0.075),
            (self.skeleton.joint_index("RightShin"), 0.075),
            (self.skeleton.joint_index("Head"), 0.180),
            (self.skeleton.joint_index("HeadEnd"), 0.060),
        ]
        self._lh_idx = self.skeleton.joint_index("LeftHand")
        self._rh_idx = self.skeleton.joint_index("RightHand")
        self._le_idx = self.skeleton.joint_index("LeftForeArm")
        self._re_idx = self.skeleton.joint_index("RightForeArm")

        # Precompute Smart Primitive #1: One-Handed Parkour Vault over 0.82 m Box!
        # In BVH, vault runs along +Z with apex (LHand on box) around z=3.10, x=0.0.
        # Rotate +Z -> +X (rot_y = +pi/2) around pivot (0.0, 3.10) so box is at X=0, vault runs -X -> +X!
        vault_clip = self._get_clip("jump_over_obstacle_0_75m_R_102__A416.bvh")
        v_pivot = (0.0, 3.10)
        v_rot = 0.5 * math.pi  # +Z -> +X
        self.vault_steps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for step_name, t_sec in [("entry", 1.183), ("action", 2.01), ("exit", 2.74)]:
            lp, lq = vault_clip.sample_window_30hz(t_sec, MODEL_POSES_PER_TOKEN)
            self.vault_steps[step_name] = _rotate_window_xz(lp, lq, v_pivot, v_rot)

        # Precompute Smart Primitive #2: Sit & Stand on 0.44 m Block!
        s_pivot = (-0.02, -0.42)
        s_rot = 0.5 * math.pi
        sit_start_clip = self._get_clip("sit_on_chair_start_R_001__A047.bvh")
        sit_loop_clip = self._get_clip("sit_on_chair_loop.bvh")
        sit_stop_clip = self._get_clip("sit_on_chair_stop_R_001__A047.bvh")
        self.sit_steps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for step_name, clip_obj, t_sec in [
            ("to_sit", sit_start_clip, 1.85),
            ("seated", sit_loop_clip, 0.0),
            ("to_stand", sit_stop_clip, 1.99),
        ]:
            lp, lq = clip_obj.sample_window_30hz(t_sec, MODEL_POSES_PER_TOKEN)
            self.sit_steps[step_name] = _rotate_window_xz(lp, lq, s_pivot, s_rot)

        # Precompute Smart Primitive #3: Overhead Duck Beam (bend down under 1.35 m clearance!)
        duck_clip = self._get_clip("neutral_avoid_obstacle_bend_down_jog_ff_180_R_001__A544.bvh")
        d_pivot = (1.90, 2.25)
        d_rot = math.atan2(2.311, 1.716)  # align diagonal travel cleanly onto +X
        self.duck_steps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for step_name, t_sec in [("entry", 1.65), ("action", 2.22), ("exit", 2.95)]:
            lp, lq = duck_clip.sample_window_30hz(t_sec, MODEL_POSES_PER_TOKEN)
            self.duck_steps[step_name] = _rotate_window_xz(lp, lq, d_pivot, d_rot)

        # Precompute No-Hands Parkour Hurdle (for open-air Spacebar jump!)
        hurdle_clip = self._get_clip("jump_over_obstacle_no_hand_0_75m_R_103__A416.bvh")
        h_pivot = (0.0, 3.66)
        h_rot = 0.5 * math.pi
        self.hurdle_steps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for step_name, t_sec in [("entry", 1.42), ("action", 2.05), ("exit", 3.00)]:
            lp, lq = hurdle_clip.sample_window_30hz(t_sec, MODEL_POSES_PER_TOKEN)
            self.hurdle_steps[step_name] = _rotate_window_xz(lp, lq, h_pivot, h_rot)

        # Precompute Idle Emotes: Look Around (Scan) & Warm-Up Stretch
        look_clip = self._get_clip("neutral_looking_around_R_001__A542.bvh")
        self.look_steps = (
            canonicalize_target_window(look_clip, 2.60, self.engine.right_leg_idx, self.engine.left_leg_idx, 0.0),
            canonicalize_target_window(look_clip, 5.40, self.engine.right_leg_idx, self.engine.left_leg_idx, 0.0),
        )
        warm_clip = self._get_clip("warm_up_neck_001__A359.bvh")
        self.warmup_steps = (
            canonicalize_target_window(warm_clip, 4.40, self.engine.right_leg_idx, self.engine.left_leg_idx, 0.0),
            canonicalize_target_window(warm_clip, 10.50, self.engine.right_leg_idx, self.engine.left_leg_idx, 0.0),
        )

        # Warm up ONNX session once so first browser click is instant (~22 ms)
        warm_t_p, warm_t_q = place_target_window(*self.styles_canon["walk"], 1.4, 0.0, 1.0, 0.0)
        _ = self.engine.generate(self.idle_canon[0], self.idle_canon[1], warm_t_p, warm_t_q)

        self.sessions: dict[str, SessionState] = {}
        self.lock = threading.Lock()
        self.boot_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[MotionBricks] Ready in {self.boot_ms:.0f} ms | "
            f"ONNX FP16 Unquantized (363 MB, load={self.engine.load_ms:.0f} ms, 8T seq) | "
            f"{len(self.styles_canon)} styles + Combat Punches + Vault + Sit"
        )

    def _get_clip(self, filename: str) -> AnimationClip:
        if filename not in self.clips:
            self.clips[filename] = load_bvh(BVH_DIR / filename, self.skeleton)
        return self.clips[filename]

    def create_session(self) -> dict[str, Any]:
        sid = uuid.uuid4().hex[:12]
        # Start standing smoothly in right-third foreground facing 3/4 inward athletic stance toward viewer
        init_p, init_q = place_target_window(*self.idle_canon, 1.12, 1.35, -0.46, 0.888)
        wp, wq = self.engine.fast_fk(init_p, init_q)

        st = SessionState(
            chunk_id=0,
            chunks={0: (init_p.copy(), init_q.copy(), init_p.copy(), init_q.copy())},
            local_pos=init_p.copy(),
            local_quat=init_q.copy(),
            history_pos=init_p.copy(),
            history_quat=init_q.copy(),
        )
        with self.lock:
            self.sessions[sid] = st

        return {
            "session_id": sid,
            "model": {
                "name": "NVIDIA MotionBricks SOMA (mmm_v6_soma_fp16.onnx)",
                "precision": "FP16 Unquantized (363 MB)",
                "provider": "ONNX Runtime CPU (Apple Silicon, 8T)",
                "fps": int(MODEL_SAMPLE_RATE),
                "total_joints": self.skeleton.num_joints,
                "render_joints": len(RENDER_JOINT_NAMES),
                "bone_joints": len(BONE_JOINT_NAMES),
            },
            "skeleton": {
                "names": RENDER_JOINT_NAMES,
                "parents": self.render_parents,
                "bone_names": BONE_JOINT_NAMES,
                "bone_parents": self.bone_parents,
                "bone_ref_pos": self.bone_ref_pos,
            },
            "bone_names": BONE_JOINT_NAMES,
            "bone_parents": self.bone_parents,
            "bone_ref_pos": self.bone_ref_pos,
            "styles": [
                {"id": s["id"], "label": s["label"], "speed": s["speed"], "accent": s["accent"]}
                for s in STYLE_SPECS
            ],
            "obstacles": GYM_OBSTACLES,
            "initial_pose": {
                "pos": np.round(wp[-1, self.render_indices], 4).tolist(),
                "quat": np.round(wq[-1, self.render_indices], 4).tolist(),
                "bone_quat": np.round(init_q[-1, self.bone_indices], 5).tolist(),
            },
        }

    def _extract_context_from_chunk(
        self,
        chunk_lp: np.ndarray,
        chunk_lq: np.ndarray,
        hist_p: np.ndarray | None,
        hist_q: np.ndarray | None,
        anchor_frame: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Extract exact 4-frame local pose history ending at clamped `anchor_frame`
        with smooth fractional linear + SLERP interpolation."""
        if chunk_lp is None or chunk_lp.shape[0] == 0:
            assert hist_p is not None and hist_q is not None
            return hist_p.copy(), hist_q.copy(), 0.0

        # Prepend previous 3 history frames before chunk[0] (chunk[0] == hist[-1]) so even
        # anchor_frame < 3 preserves exact continuous velocity across chunk boundaries!
        if hist_p is not None and hist_q is not None:
            full_p = np.concatenate([hist_p[:-1], chunk_lp], axis=0)
            full_q = np.concatenate([hist_q[:-1], chunk_lq], axis=0)
            offset = hist_p.shape[0] - 1
        else:
            full_p = chunk_lp
            full_q = chunk_lq
            offset = 0

        max_f = float(chunk_lp.shape[0] - 1)
        clamped = float(np.clip(anchor_frame, 0.0, max_f))
        end_idx = offset + clamped

        ctx_p = np.empty((MODEL_POSES_PER_TOKEN, self.skeleton.num_joints, 3), dtype=np.float32)
        ctx_q = np.empty((MODEL_POSES_PER_TOKEN, self.skeleton.num_joints, 4), dtype=np.float32)
        n_total = full_p.shape[0]
        for k in range(MODEL_POSES_PER_TOKEN):
            sample_i = float(np.clip(end_idx - (MODEL_POSES_PER_TOKEN - 1 - k), 0.0, n_total - 1))
            f0 = int(math.floor(sample_i))
            f1 = min(f0 + 1, n_total - 1)
            a = sample_i - f0
            if a < 1e-6 or f0 == f1:
                ctx_p[k] = full_p[f0]
                ctx_q[k] = full_q[f0]
            else:
                ctx_p[k] = full_p[f0] * (1.0 - a) + full_p[f1] * a
                ctx_q[k] = quat_slerp(full_q[f0], full_q[f1], a)
        return ctx_p, ctx_q, clamped

    def _root_and_heading_from_world(
        self, wp_last: np.ndarray, wq_last: np.ndarray
    ) -> tuple[float, float, float, float, float, float]:
        """Return (x, z, hip_fx, hip_fz, gaze_fx, gaze_fz) from a single world-pose frame."""
        hips = wp_last[0]
        h = _heading_from_hip_positions(
            wp_last[self.engine.right_leg_idx], wp_last[self.engine.left_leg_idx]
        )
        hip_fx, hip_fz = math.sin(h), math.cos(h)
        head_fwd = quat_rotate(wq_last[self.head_idx], np.array([0.0, 1.0, 0.0], dtype=np.float32))
        g_mag = math.hypot(float(head_fwd[0]), float(head_fwd[2]))
        if g_mag > 1e-4:
            gaze_fx, gaze_fz = float(head_fwd[0]) / g_mag, float(head_fwd[2]) / g_mag
        else:
            gaze_fx, gaze_fz = hip_fx, hip_fz
        return float(hips[0]), float(hips[2]), hip_fx, hip_fz, gaze_fx, gaze_fz

    def _resolve_dir(
        self, req: dict[str, Any], default_fx: float, default_fz: float
    ) -> tuple[float, float]:
        raw_dx = req.get("dir_x")
        raw_dz = req.get("dir_z")
        if raw_dx is None and raw_dz is None:
            return default_fx, default_fz
        dx = float(default_fx if raw_dx is None else raw_dx)
        dz = float(default_fz if raw_dz is None else raw_dz)
        mag = math.hypot(dx, dz)
        if mag < 1e-4:
            return default_fx, default_fz
        return dx / mag, dz / mag

    def _find_obstacle(
        self, req: dict[str, Any], kind: str, cur_x: float, cur_z: float
    ) -> dict[str, Any]:
        """Resolve any obstacle dynamically by `obstacle_id` or nearest matching type (zero hardcoded indices)."""
        req_id = req.get("obstacle_id")
        if req_id:
            for obs in GYM_OBSTACLES:
                if obs["id"] == req_id:
                    return obs
        candidates = [o for o in GYM_OBSTACLES if o["type"] == kind]
        if not candidates:
            return GYM_OBSTACLES[0]
        return min(
            candidates,
            key=lambda o: math.hypot(float(o["pos"][0]) - cur_x, float(o["pos"][2]) - cur_z),
        )

    def _synthesize_combat_roll(
        self,
        ctx_p: np.ndarray,
        ctx_q: np.ndarray,
        dx: float,
        dz: float,
        travel_dist: float,
    ) -> tuple[GenerationResult, np.ndarray, np.ndarray]:
        """Synthesize a single continuous 26-frame (30 Hz, ~0.87 s) hybrid neural-bridged
        360-deg diagonal parkour shoulder combat roll directly in SOMA's 77-joint local-quaternion space.

        - C0-exact entry from live ONNX context `ctx_p[-1], ctx_q[-1]` (zero pop from idle/walk/jog/sprint/combat).
        - Unbroken monotonic 0 -> -360 deg forward pitch + ~18 deg diagonal right-shoulder-to-left-hip cant.
        - Narrow sagittal thighs (`knee_sep` ~ 0.21-0.29 m, zero frog-splay), chin tucked to left collarbone
          (`Head.y >= 0.18 m`), body-local arms wrapped tightly against outer shins/chest (zero overhead flip).
        - Exact wheel-envelope floor grounding + athletic spring recovery into `styles_canon["jog"]` so the
          very next ONNX chunk (`steer` / `idle` / `combat_recover`) stitches seamlessly with zero pop.
        """
        t0 = time.perf_counter()
        num_frames = 26
        num_joints = self.skeleton.num_joints
        lp = np.broadcast_to(self.skeleton.ref_pos[None, :, :], (num_frames, num_joints, 3)).copy()
        lq = np.broadcast_to(self.skeleton.ref_quat[None, :, :], (num_frames, num_joints, 4)).copy()

        cur_x = float(ctx_p[-1, 0, 0])
        cur_y = float(ctx_p[-1, 0, 1])
        cur_z = float(ctx_p[-1, 0, 2])

        yaw_tgt = math.atan2(-dz, dx)
        q_yaw_tgt = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), yaw_tgt)
        q_yaw_inv = quat_conj(q_yaw_tgt)

        entry_q_all = ctx_q[-1].copy()
        entry_q0_canon = quat_norm(quat_mul(q_yaw_inv, entry_q_all[0]))
        idle_q0_canon = self.idle_canon[1][0, 0].copy()
        jog_lp, jog_lq = self.styles_canon["jog"]

        # Smooth monotonic forward travel profile matching `jog` exit cadence across f=22..25
        jog_dx3 = float(jog_lp[3, 0, 0] - jog_lp[0, 0, 0])
        jog_step = jog_dx3 / 3.0
        end_tail_start = max(0.35, travel_dist - jog_dx3)
        s_curve = np.zeros(num_frames, dtype=np.float32)
        for f in range(num_frames):
            if f <= 22:
                u = f / 22.0
                m1 = min(1.8, (jog_step * 22.0) / max(end_tail_start, 0.3))
                m0 = 0.65 * m1
                h01 = 3.0 * u * u - 2.0 * u * u * u
                h10 = u * u * u - 2.0 * u * u + u
                h11 = u * u * u - u * u
                s_curve[f] = end_tail_start * float(np.clip(h01 + h10 * m0 + h11 * m1, 0.0, 1.0))
            else:
                s_curve[f] = end_tail_start + (f - 22) * jog_step
        s_curve = np.maximum.accumulate(s_curve)

        x_ax = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_ax = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        z_ax = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        contacts = np.zeros((num_frames, 4), dtype=np.float32)

        for f in range(num_frames):
            if f == 0:
                lp[0] = ctx_p[-1]
                lq[0] = ctx_q[-1]
                contacts[0] = [1.0, 1.0, 1.0, 1.0]
                continue

            # 1. Body-local non-root joint interpolation across dive -> tuck -> squat-plant -> jog
            if f <= 4:
                q_body = quat_slerp(entry_q_all[1:], self.roll_dive_q[1:], _smoothstep(f / 4.0))
            elif f <= 9:
                q_body = quat_slerp(self.roll_dive_q[1:], self.roll_tuck_q[1:], _smoothstep((f - 4.0) / 5.0))
            elif f <= 15:
                q_body = self.roll_tuck_q[1:].copy()
            elif f <= 18:
                q_body = quat_slerp(self.roll_tuck_q[1:], self.roll_plant_q[1:], _smoothstep((f - 15.0) / 3.0))
            elif f <= 22:
                q_body = quat_slerp(self.roll_plant_q[1:], jog_lq[0, 1:], _smoothstep((f - 18.0) / 4.0))
            else:
                q_body = jog_lq[f - 22, 1:].copy()
            lq[f, 1:] = q_body

            # 2. Monotonic 0 -> -360 deg forward pitch + diagonal right-shoulder cant
            t_p = float(np.clip(f / 21.0, 0.0, 1.0))
            pitch_rad = 2.0 * math.pi * (0.5 - 0.5 * math.cos(math.pi * t_p))
            w_diag = (math.sin(math.pi * t_p)) ** 1.25
            q_pitch = quat_from_axis_angle(z_ax, -pitch_rad)
            q_diag = quat_norm(
                quat_mul(
                    quat_from_axis_angle(x_ax, math.radians(18.0 * w_diag)),
                    quat_from_axis_angle(y_ax, math.radians(16.0 * w_diag)),
                )
            )

            if f <= 5:
                q_base = quat_slerp(entry_q0_canon, idle_q0_canon, _smoothstep(f / 5.0))
            elif f <= 18:
                q_base = idle_q0_canon
            elif f <= 22:
                q_base = quat_slerp(idle_q0_canon, jog_lq[0, 0], _smoothstep((f - 18.0) / 4.0))
            else:
                q_base = jog_lq[f - 22, 0]

            q0_canon = quat_norm(quat_mul(q_pitch, quat_mul(q_diag, q_base)))
            lq[f, 0] = quat_norm(quat_mul(q_yaw_tgt, q0_canon))

            # Enforce hemisphere continuity across consecutive frames for jitter-free SLERP
            dots = np.sum(lq[f] * lq[f - 1], axis=-1)
            lq[f, dots < 0.0] *= -1.0

            # 3. Root XZ along travel ray
            s = float(s_curve[f])
            lp[f, 0, 0] = cur_x + dx * s
            lp[f, 0, 2] = cur_z + dz * s

            # 4. Exact wheel-envelope floor grounding + full-skeleton floor protection
            if f >= 22:
                lp[f, 0, 1] = float(jog_lp[f - 22, 0, 1])
            else:
                lp[f, 0, 1] = 0.55
                wp_f, _ = self.engine.fast_fk(lp[f : f + 1], lq[f : f + 1])
                min_clear = min(float(wp_f[0, ji, 1]) - rad for ji, rad in self._roll_env_specs)
                min_all = float(wp_f[0, :, 1].min()) - 0.005
                y_ground = 0.55 - min(min_clear, min_all)
                if f <= 3:
                    w0 = _smoothstep(f / 3.0)
                    lp[f, 0, 1] = max(cur_y * (1.0 - w0) + y_ground * w0, y_ground)
                elif f >= 19:
                    w1 = _smoothstep((f - 19.0) / 3.0)
                    y_jog0 = float(jog_lp[0, 0, 1])
                    lp[f, 0, 1] = max(y_ground * (1.0 - w1) + y_jog0 * w1, y_ground)
                else:
                    lp[f, 0, 1] = y_ground

            # 5. Smooth foot contact ramp (push-off -> airborne tuck -> staggered foot plant -> jog)
            if f <= 3:
                c = max(0.0, 1.0 - f / 3.0)
                contacts[f] = [c, c, c, c]
            elif 18 <= f < 22:
                c = (f - 17.0) / 4.0
                contacts[f] = [c, c, 0.35 * c * (1.0 - 0.5 * c), 0.35 * c]
            elif f >= 22:
                contacts[f] = [1.0, 1.0, 0.0, 0.0]

        # Gentle vertical smoothing pass over roll body (f=1..21), re-enforcing exact floor clearance
        raw_y = lp[:, 0, 1].copy()
        for _ in range(3):
            sm_y = raw_y.copy()
            for f in range(1, 22):
                sm_y[f] = 0.25 * raw_y[f - 1] + 0.5 * raw_y[f] + 0.25 * raw_y[f + 1]
            raw_y = sm_y
        lp[:, 0, 1] = raw_y
        wp, wq = self.engine.fast_fk(lp, lq)
        for f in range(1, 22):
            min_clear = min(float(wp[f, ji, 1]) - rad for ji, rad in self._roll_env_specs)
            min_all = float(wp[f, :, 1].min()) - 0.005
            deficit = min(min_clear, min_all)
            if deficit < 0.0:
                lp[f, 0, 1] -= deficit
        wp, wq = self.engine.fast_fk(lp, lq)

        # Placed target window for Target (Blue) ghost visualization at exit position
        end_x = cur_x + dx * end_tail_start
        end_z = cur_z + dz * end_tail_start
        tgt_p, tgt_q = place_target_window(*self.styles_canon["jog"], end_x, end_z, dx, dz)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        res = GenerationResult(
            local_pos=lp,
            local_quat=lq,
            world_pos=wp,
            world_quat=wq,
            foot_contacts=contacts,
            num_frames=num_frames,
            pred_num_tokens=8,
            latency_ms=latency_ms,
            infer_ms=0.0,
        )
        return res, tgt_p, tgt_q

    def _stabilize_crawl_palms(self, res: GenerationResult) -> None:
        """Keep hands flat palm-down parallel to the floor and prevent mid-stride wrist floor-clipping
        during ONNX-generated low crawl (zero pop on entry/exit via smooth floor-proximity gating)."""
        changed = False
        for f in range(res.num_frames):
            # Prevent ONNX mid-stride dip from sinking supporting palms into the floor
            min_hy = float(min(res.world_pos[f, self._lh_idx, 1], res.world_pos[f, self._rh_idx, 1]))
            if min_hy < 0.038:
                lift = 0.038 - min_hy
                res.local_pos[f, 0, 1] += lift
                res.world_pos[f, :, 1] += lift
                changed = True

            h = _heading_from_hip_positions(
                res.world_pos[f, self.engine.right_leg_idx],
                res.world_pos[f, self.engine.left_leg_idx],
            )
            fdir = (math.sin(h), 0.0, math.cos(h))
            for jh, je, is_r in (
                (self._lh_idx, self._le_idx, False),
                (self._rh_idx, self._re_idx, True),
            ):
                hy = float(res.world_pos[f, jh, 1])
                if hy < 0.16:
                    w = _smoothstep((0.16 - hy) / 0.06)
                    q_flat_world = _make_hand_world_quat(fdir, (0.0, -1.0, 0.0), is_right=is_r)
                    q_flat_local = quat_norm(quat_mul(quat_conj(res.world_quat[f, je]), q_flat_world))
                    res.local_quat[f, jh] = quat_slerp(res.local_quat[f, jh], q_flat_local, w)
                    changed = True
        if changed:
            res.world_pos, res.world_quat = self.engine.fast_fk(res.local_pos, res.local_quat)

    def _run_generate(
        self, req: dict[str, Any]
    ) -> tuple[dict[str, Any], GenerationResult, np.ndarray, np.ndarray]:
        sid = req.get("session_id", "")
        with self.lock:
            st = self.sessions.get(sid)
            if st is None:
                raise KeyError("Unknown session_id")

            # Resolve base_chunk_id from ring buffer (accepts `base_chunk_id` or legacy `chunk_id`)
            req_base = req.get("base_chunk_id")
            if req_base is None:
                req_base = req.get("chunk_id")
            if req_base is not None and int(req_base) in st.chunks:
                base_chunk_id = int(req_base)
            else:
                base_chunk_id = st.chunk_id

            chunk_lp, chunk_lq, hist_p, hist_q = st.chunks[base_chunk_id]
            max_f = float(chunk_lp.shape[0] - 1) if chunk_lp is not None and chunk_lp.shape[0] > 0 else 0.0

            # Resolve exact anchor_frame (accepts `anchor_frame` or legacy `current_frame`)
            raw_anchor = req.get("anchor_frame")
            if raw_anchor is None:
                raw_anchor = req.get("current_frame")
            req_anchor = max_f if raw_anchor is None else float(raw_anchor)

            ctx_p, ctx_q, anchor_frame = self._extract_context_from_chunk(
                chunk_lp, chunk_lq, hist_p, hist_q, req_anchor
            )
            ctx_wp, ctx_wq = self.engine.fast_fk(ctx_p, ctx_q)
            cur_x, cur_z, cur_fx, cur_fz, gaze_fx, gaze_fz = self._root_and_heading_from_world(
                ctx_wp[-1], ctx_wq[-1]
            )

            mode = str(req.get("mode", "idle"))
            style = str(req.get("style", "walk"))
            if style not in self.styles_canon:
                style = "walk"
            spec = self.style_meta[style]

            allowed_tokens: tuple[int, ...] = (8, 9, 10, 11, 12, 13, 14)
            num_target_poses = 4
            transition_early = 4
            phase_done = False
            custom_res: GenerationResult | None = None

            if mode == "steer":
                dx, dz = self._resolve_dir(req, cur_fx, cur_fz)
                sprint = bool(req.get("sprint", False))
                eff_style = "run" if (sprint and style in ("walk", "jog")) else style
                eff_spec = self.style_meta[eff_style]
                lead = eff_spec["speed"] * eff_spec["lead_s"]
                tx, tz = _clamp_room_xz(cur_x + dx * lead, cur_z + dz * lead)
                sdx, sdz = tx - cur_x, tz - cur_z
                sdist = math.hypot(sdx, sdz)
                if sdist < 0.24:
                    # Head-on against concrete wall: hold balanced idle/guard inside wall instead of penetrating!
                    idle_win = (
                        self.crawl_a_canon
                        if style == "crawl"
                        else (
                            self.crouch_idle_canon
                            if style == "crouch"
                            else (self.combat_guard_canon if style == "combat" else self.idle_canon)
                        )
                    )
                    cx_c, cz_c = _clamp_room_xz(cur_x, cur_z)
                    tgt_p, tgt_q = place_target_window(*idle_win, cx_c, cz_c, dx, dz)
                    allowed_tokens = (6, 7, 8)
                    transition_early = 4
                else:
                    # Smooth wall-slide along boundary if angled against a wall
                    slide_dx, slide_dz = sdx / sdist, sdz / sdist
                    if eff_style == "crawl":
                        crawl_win = self.crawl_a_canon if (st.crawl_phase % 2 == 0) else self.crawl_b_canon
                        st.crawl_phase += 1
                        self.crawl_phase = st.crawl_phase
                        tgt_p, tgt_q = place_target_window(*crawl_win, tx, tz, slide_dx, slide_dz)
                        allowed_tokens = (6, 7)
                        transition_early = 4
                    else:
                        tgt_p, tgt_q = place_target_window(
                            *self.styles_canon[eff_style], tx, tz, slide_dx, slide_dz
                        )
                        allowed_tokens = (6, 7, 8, 9)
                        transition_early = 6

            elif mode == "punch_left":
                # Explosive neural Left Jab: snap torso & lead clenched fist straight at chin height along aim/gaze
                dx, dz = self._resolve_dir(req, gaze_fx, gaze_fz)
                tx, tz = _clamp_room_xz(cur_x + dx * 0.16, cur_z + dz * 0.16)
                tgt_p, tgt_q = place_target_window(*self.punch_left_canon, tx, tz, dx, dz)
                num_target_poses = 4
                allowed_tokens = (6,)
                transition_early = 1
                phase_done = True

            elif mode == "punch_right":
                # Heavy neural Right Cross: step hard into strike, drive torso & rear clenched fist down centerline
                dx, dz = self._resolve_dir(req, gaze_fx, gaze_fz)
                tx, tz = _clamp_room_xz(cur_x + dx * 0.24, cur_z + dz * 0.24)
                tgt_p, tgt_q = place_target_window(*self.punch_right_canon, tx, tz, dx, dz)
                num_target_poses = 4
                allowed_tokens = (6,)
                transition_early = 1
                phase_done = True

            elif mode == "combat_recover":
                # Snappy retraction back to tight orthodox boxing guard facing aim/gaze
                dx, dz = self._resolve_dir(req, gaze_fx, gaze_fz)
                cx_c, cz_c = _clamp_room_xz(cur_x, cur_z)
                tgt_p, tgt_q = place_target_window(*self.combat_guard_canon, cx_c, cz_c, dx, dz)
                num_target_poses = 4
                allowed_tokens = (6,)
                transition_early = 3
                phase_done = True

            elif mode in ("roll", "roll_dive"):
                # Single continuous 26-frame (30 Hz, ~0.87 s) hybrid neural-bridged 360-deg diagonal parkour combat roll
                def_fx, def_fz = (gaze_fx, gaze_fz) if style == "combat" else (cur_fx, cur_fz)
                dx, dz = self._resolve_dir(req, def_fx, def_fz)
                _, _, clipped = _clip_room_step(cur_x, cur_z, dx, dz, 1.95)
                if clipped < 0.70:
                    # Too close to concrete wall: cleanly abort & hold crouch instead of rolling into wall
                    cx_c, cz_c = _clamp_room_xz(cur_x, cur_z)
                    tgt_p, tgt_q = place_target_window(*self.crouch_idle_canon, cx_c, cz_c, dx, dz)
                    mode = "idle"
                    allowed_tokens = (6,)
                    transition_early = 3
                    phase_done = True
                else:
                    mode = "roll"
                    custom_res, tgt_p, tgt_q = self._synthesize_combat_roll(ctx_p, ctx_q, dx, dz, clipped)
                    transition_early = 4
                    phase_done = True

            elif mode == "jump_leap":
                # Compact athletic no-hands parkour hurdle (0.90 m leap + 0.80 m land = 1.70 m span, never overshoots tight obstacles!)
                dx, dz = self._resolve_dir(req, cur_fx, cur_fz)
                apex_x, apex_z, clipped = _clip_room_step(cur_x, cur_z, dx, dz, 0.90)
                if clipped < 0.35:
                    # Blocked directly against wall: hold clean stance inside room instead of leaping through wall
                    cx_c, cz_c = _clamp_room_xz(cur_x, cur_z)
                    tgt_p, tgt_q = place_target_window(*self.idle_canon, cx_c, cz_c, dx, dz)
                    mode = "idle"
                    allowed_tokens = (5, 6)
                    transition_early = 3
                    phase_done = True
                else:
                    tgt_p, tgt_q = place_target_window(*self.hurdle_steps["action"], apex_x, apex_z, dx, dz)
                    allowed_tokens = (5, 6)
                    transition_early = 3
                    phase_done = True

            elif mode == "jump_land":
                dx, dz = self._resolve_dir(req, cur_fx, cur_fz)
                land_x, land_z, clipped = _clip_room_step(cur_x, cur_z, dx, dz, 0.80)
                if clipped < 0.30:
                    tgt_p, tgt_q = place_target_window(*self.idle_canon, land_x, land_z, dx, dz)
                else:
                    tgt_p, tgt_q = place_target_window(*self.hurdle_steps["exit"], land_x, land_z, dx, dz)
                allowed_tokens = (5, 6)
                transition_early = 4
                phase_done = True

            elif mode == "duck_entry":
                obs = self._find_obstacle(req, "duck", cur_x, cur_z)
                bx, bz = float(obs["pos"][0]), float(obs["pos"][2])
                side_dir = float(req.get("duck_dir", 1.0 if cur_z <= bz else -1.0))
                side_dir = 1.0 if side_dir >= 0 else -1.0
                lane_x = float(req.get("duck_x", np.clip(cur_x - bx, -0.65, 0.65) + bx))
                tgt_p, tgt_q = place_target_window(*self.duck_steps["entry"], lane_x, bz, 0.0, side_dir)
                entry_x, entry_z = float(tgt_p[0, 0, 0]), float(tgt_p[0, 0, 2])
                dist_entry = math.hypot(entry_x - cur_x, entry_z - cur_z)
                dist_beam = math.hypot(lane_x - cur_x, bz - cur_z)
                if dist_beam < 2.25 and ((bz - cur_z) * side_dir > 0.15):
                    tgt_p, tgt_q = place_target_window(*self.duck_steps["action"], lane_x, bz, 0.0, side_dir)
                    allowed_tokens = (6, 7, 8)
                    transition_early = 3
                    mode = "duck_action"
                    phase_done = True
                elif dist_entry > 2.0:
                    dx = (entry_x - cur_x) / dist_entry
                    dz = (entry_z - cur_z) / dist_entry
                    step = min(dist_entry - 0.8, 1.8)
                    tgt_p, tgt_q = place_target_window(
                        *self.styles_canon["jog"], cur_x + dx * step, cur_z + dz * step, dx, dz
                    )
                    allowed_tokens = (6, 7, 8, 9)
                    transition_early = 6
                    phase_done = False
                else:
                    allowed_tokens = (6, 7, 8, 9)
                    transition_early = 5
                    phase_done = True

            elif mode == "duck_action":
                obs = self._find_obstacle(req, "duck", cur_x, cur_z)
                bx, bz = float(obs["pos"][0]), float(obs["pos"][2])
                side_dir = 1.0 if float(req.get("duck_dir", -1.0)) >= 0 else -1.0
                lane_x = float(req.get("duck_x", bx))
                tgt_p, tgt_q = place_target_window(*self.duck_steps["action"], lane_x, bz, 0.0, side_dir)
                allowed_tokens = (6, 7, 8)
                transition_early = 2
                phase_done = True

            elif mode == "duck_exit":
                obs = self._find_obstacle(req, "duck", cur_x, cur_z)
                bx, bz = float(obs["pos"][0]), float(obs["pos"][2])
                side_dir = 1.0 if float(req.get("duck_dir", -1.0)) >= 0 else -1.0
                lane_x = float(req.get("duck_x", bx))
                tgt_p, tgt_q = place_target_window(*self.duck_steps["exit"], lane_x, bz, 0.0, side_dir)
                allowed_tokens = (6, 7, 8)
                transition_early = 5
                phase_done = True

            elif mode == "emote_look":
                tgt_p, tgt_q = place_target_window(*self.look_steps[0], cur_x, cur_z, cur_fx, cur_fz)
                allowed_tokens = (7, 8, 9, 10)
                transition_early = 3
                phase_done = True

            elif mode == "emote_look_2":
                tgt_p, tgt_q = place_target_window(*self.look_steps[1], cur_x, cur_z, cur_fx, cur_fz)
                allowed_tokens = (7, 8, 9, 10)
                transition_early = 3
                phase_done = True

            elif mode == "emote_warmup":
                tgt_p, tgt_q = place_target_window(*self.warmup_steps[0], cur_x, cur_z, cur_fx, cur_fz)
                allowed_tokens = (8, 9, 10, 11)
                transition_early = 3
                phase_done = True

            elif mode == "emote_warmup_2":
                tgt_p, tgt_q = place_target_window(*self.warmup_steps[1], cur_x, cur_z, cur_fx, cur_fz)
                allowed_tokens = (8, 9, 10, 11)
                transition_early = 3
                phase_done = True

            elif mode == "goto":
                gx, gz = _clamp_room_xz(
                    float(req.get("target_x", cur_x)),
                    float(req.get("target_z", cur_z)),
                )
                vx, vz = gx - cur_x, gz - cur_z
                dist = math.hypot(vx, vz)
                idle_win = (
                    self.crawl_a_canon
                    if style == "crawl"
                    else (
                        self.crouch_idle_canon
                        if style == "crouch"
                        else (self.combat_guard_canon if style == "combat" else self.idle_canon)
                    )
                )
                if dist < 0.45:
                    fx_use, fz_use = (gaze_fx, gaze_fz) if style == "combat" else (cur_fx, cur_fz)
                    tgt_p, tgt_q = place_target_window(*idle_win, gx, gz, fx_use, fz_use)
                    allowed_tokens = (6, 7, 8, 9)
                    transition_early = 2
                    phase_done = True
                else:
                    dx, dz = vx / dist, vz / dist
                    step_dist = min(dist, spec["speed"] * spec["lead_s"])
                    if dist <= spec["speed"] * 0.95:
                        tgt_p, tgt_q = place_target_window(*idle_win, gx, gz, dx, dz)
                        allowed_tokens = (6, 7, 8, 9, 10)
                        transition_early = 2
                        phase_done = True
                    else:
                        tx = cur_x + dx * step_dist
                        tz = cur_z + dz * step_dist
                        if style == "crawl":
                            crawl_win = self.crawl_a_canon if (st.crawl_phase % 2 == 0) else self.crawl_b_canon
                            st.crawl_phase += 1
                            self.crawl_phase = st.crawl_phase
                            tgt_p, tgt_q = place_target_window(*crawl_win, tx, tz, dx, dz)
                            allowed_tokens = (6, 7)
                            transition_early = 4
                        else:
                            tgt_p, tgt_q = place_target_window(*self.styles_canon[style], tx, tz, dx, dz)
                            allowed_tokens = (6, 7, 8, 9)
                            transition_early = 6

            elif mode == "vault_entry":
                obs = self._find_obstacle(req, "vault", cur_x, cur_z)
                bx, bz = float(obs["pos"][0]), float(obs["pos"][2])
                sx_w, _, sz_w = float(obs["size"][0]), float(obs["size"][1]), float(obs["size"][2])
                cross_z = sz_w < sx_w  # Wall spans along X -> vault across Z!
                if cross_z:
                    side_dir = float(req.get("vault_dir", 1.0 if cur_z <= bz else -1.0))
                    side_dir = 1.0 if side_dir >= 0 else -1.0
                    lane_x = float(req.get("vault_lane", np.clip(cur_x - bx, -0.70, 0.70) + bx))
                    vx_t, vz_t, fvx, fvz = lane_x, bz, 0.0, side_dir
                    approach_sign = (bz - cur_z) * side_dir
                else:
                    side_dir = float(req.get("vault_dir", 1.0 if cur_x <= bx else -1.0))
                    side_dir = 1.0 if side_dir >= 0 else -1.0
                    lane_z = float(req.get("vault_z", req.get("vault_lane", np.clip(cur_z - bz, -0.70, 0.70) + bz)))
                    vx_t, vz_t, fvx, fvz = bx, lane_z, side_dir, 0.0
                    approach_sign = (bx - cur_x) * side_dir
                tgt_p, tgt_q = place_target_window(*self.vault_steps["entry"], vx_t, vz_t, fvx, fvz)
                entry_x = float(tgt_p[0, 0, 0])
                entry_z = float(tgt_p[0, 0, 2])
                dist_entry = math.hypot(entry_x - cur_x, entry_z - cur_z)
                dist_box = math.hypot(vx_t - cur_x, vz_t - cur_z)
                if dist_box < 2.25 and approach_sign > 0.2:
                    tgt_p, tgt_q = place_target_window(*self.vault_steps["action"], vx_t, vz_t, fvx, fvz)
                    allowed_tokens = (6, 7, 8, 9)
                    transition_early = 3
                    mode = "vault_action"
                    phase_done = True
                elif dist_entry > 2.15:
                    dx = (entry_x - cur_x) / dist_entry
                    dz = (entry_z - cur_z) / dist_entry
                    step = min(dist_entry - 0.9, 1.8)
                    tgt_p, tgt_q = place_target_window(
                        *self.styles_canon["jog"], cur_x + dx * step, cur_z + dz * step, dx, dz
                    )
                    allowed_tokens = (6, 7, 8, 9)
                    transition_early = 6
                    phase_done = False
                else:
                    allowed_tokens = (6, 7, 8, 9, 10)
                    transition_early = 5
                    phase_done = True

            elif mode == "vault_action":
                obs = self._find_obstacle(req, "vault", cur_x, cur_z)
                bx, bz = float(obs["pos"][0]), float(obs["pos"][2])
                sx_w, _, sz_w = float(obs["size"][0]), float(obs["size"][1]), float(obs["size"][2])
                cross_z = sz_w < sx_w
                side_dir = 1.0 if float(req.get("vault_dir", 1.0)) >= 0 else -1.0
                if cross_z:
                    lane_x = float(req.get("vault_lane", bx))
                    tgt_p, tgt_q = place_target_window(*self.vault_steps["action"], lane_x, bz, 0.0, side_dir)
                else:
                    lane_z = float(req.get("vault_z", req.get("vault_lane", bz)))
                    tgt_p, tgt_q = place_target_window(*self.vault_steps["action"], bx, lane_z, side_dir, 0.0)
                allowed_tokens = (6, 7, 8, 9)
                transition_early = 2
                phase_done = True

            elif mode == "vault_exit":
                obs = self._find_obstacle(req, "vault", cur_x, cur_z)
                bx, bz = float(obs["pos"][0]), float(obs["pos"][2])
                sx_w, _, sz_w = float(obs["size"][0]), float(obs["size"][1]), float(obs["size"][2])
                cross_z = sz_w < sx_w
                side_dir = 1.0 if float(req.get("vault_dir", 1.0)) >= 0 else -1.0
                if cross_z:
                    lane_x = float(req.get("vault_lane", bx))
                    tgt_p, tgt_q = place_target_window(*self.vault_steps["exit"], lane_x, bz, 0.0, side_dir)
                else:
                    lane_z = float(req.get("vault_z", req.get("vault_lane", bz)))
                    tgt_p, tgt_q = place_target_window(*self.vault_steps["exit"], bx, lane_z, side_dir, 0.0)
                allowed_tokens = (6, 7, 8, 9)
                transition_early = 5
                phase_done = True

            elif mode == "sit_approach":
                obs = self._find_obstacle(req, "sit", cur_x, cur_z)
                sx, sz = float(obs["pos"][0]), float(obs["pos"][2])
                fx, fz = float(obs["facing"][0]), float(obs["facing"][1])
                # Approach point: 1.05 m in front of the bench
                app_x, app_z = sx + fx * 1.05, sz + fz * 1.05
                dist_app = math.hypot(app_x - cur_x, app_z - cur_z)
                if dist_app > 1.85:
                    dx = (app_x - cur_x) / dist_app
                    dz = (app_z - cur_z) / dist_app
                    step = min(dist_app - 0.6, 1.55)
                    tgt_p, tgt_q = place_target_window(
                        *self.styles_canon["walk"], cur_x + dx * step, cur_z + dz * step, dx, dz
                    )
                    allowed_tokens = (8, 9, 10, 11, 12)
                    transition_early = 5
                    phase_done = False
                else:
                    tgt_p, tgt_q = place_target_window(*self.sit_steps["to_sit"], sx, sz, fx, fz)
                    allowed_tokens = (8, 9, 10, 11, 12, 13, 14)
                    transition_early = 6
                    phase_done = True

            elif mode == "sit_down":
                obs = self._find_obstacle(req, "sit", cur_x, cur_z)
                sx, sz = float(obs["pos"][0]), float(obs["pos"][2])
                fx, fz = float(obs["facing"][0]), float(obs["facing"][1])
                tgt_p, tgt_q = place_target_window(*self.sit_steps["seated"], sx, sz, fx, fz)
                num_target_poses = 1
                allowed_tokens = (6, 7, 8, 9, 10, 11, 12)
                transition_early = 1
                phase_done = True

            elif mode == "sit_hold":
                obs = self._find_obstacle(req, "sit", cur_x, cur_z)
                sx, sz = float(obs["pos"][0]), float(obs["pos"][2])
                fx, fz = float(obs["facing"][0]), float(obs["facing"][1])
                tgt_p, tgt_q = place_target_window(*self.sit_steps["seated"], sx, sz, fx, fz)
                num_target_poses = 1
                allowed_tokens = (10, 11, 12, 13, 14)
                transition_early = 2
                phase_done = True

            elif mode == "stand_up":
                obs = self._find_obstacle(req, "sit", cur_x, cur_z)
                sx, sz = float(obs["pos"][0]), float(obs["pos"][2])
                fx, fz = float(obs["facing"][0]), float(obs["facing"][1])
                tgt_p, tgt_q = place_target_window(*self.sit_steps["to_stand"], sx, sz, fx, fz)
                allowed_tokens = (7, 8, 9, 10, 11, 12, 13, 14)
                transition_early = 3
                phase_done = True

            else:
                # Idle in place facing current heading (combat holds orthodox guard, crawl stays low!)
                if style == "crawl":
                    idle_win = self.crawl_a_canon
                    fx_use, fz_use = cur_fx, cur_fz
                elif style == "crouch":
                    idle_win = self.crouch_idle_canon
                    fx_use, fz_use = cur_fx, cur_fz
                elif style == "combat":
                    idle_win = self.combat_guard_canon
                    fx_use, fz_use = gaze_fx, gaze_fz
                else:
                    idle_win = self.idle_canon
                    fx_use, fz_use = cur_fx, cur_fz
                tgt_p, tgt_q = place_target_window(*idle_win, cur_x, cur_z, fx_use, fz_use)
                allowed_tokens = (6, 7, 8, 9)
                transition_early = 2
                phase_done = True

            # Execute full unquantized FP16 ONNX generation (or use continuous 360-deg roll synthesis)!
            if custom_res is not None:
                res = custom_res
            else:
                res = self.engine.generate(
                    ctx_p,
                    ctx_q,
                    tgt_p[:num_target_poses],
                    tgt_q[:num_target_poses],
                    allowed_token_counts=allowed_tokens,
                    correct_endpoints=True,
                )
                if style == "crawl" and mode in ("steer", "idle", "goto"):
                    self._stabilize_crawl_palms(res)

            # Hard Post-ONNX Solid Wall Collision Clamp: zero frames ever cross concrete room perimeter!
            hip_x = res.world_pos[:, 0, 0]
            hip_z = res.world_pos[:, 0, 2]
            clamped_x = np.clip(hip_x, ROOM_MIN_X, ROOM_MAX_X)
            clamped_z = np.clip(hip_z, ROOM_MIN_Z, ROOM_MAX_Z)
            dx_fix = clamped_x - hip_x
            dz_fix = clamped_z - hip_z
            if float(np.max(np.abs(dx_fix))) > 1e-5 or float(np.max(np.abs(dz_fix))) > 1e-5:
                res.local_pos[:, 0, 0] += dx_fix
                res.local_pos[:, 0, 2] += dz_fix
                res.world_pos[:, :, 0] += dx_fix[:, None]
                res.world_pos[:, :, 2] += dz_fix[:, None]

            # Compute world poses for Target (Blue) ghost overlay (Context already computed above)
            tgt_wp, _ = self.engine.fast_fk(tgt_p, tgt_q)

            # Save new chunk into session ring buffer (evicting oldest if > CHUNK_RING_BUFFER_MAX)
            st.chunk_id += 1
            new_chunk_id = st.chunk_id
            st.chunks[new_chunk_id] = (res.local_pos, res.local_quat, ctx_p, ctx_q)
            while len(st.chunks) > CHUNK_RING_BUFFER_MAX:
                oldest_id = next(iter(st.chunks))
                del st.chunks[oldest_id]
            st.history_pos = ctx_p
            st.history_quat = ctx_q
            st.local_pos = res.local_pos
            st.local_quat = res.local_quat

            # Report gaze heading for combat poses (where hips are intentionally bladed ~26 deg)
            if mode in ("punch_left", "punch_right", "combat_recover") or style == "combat":
                end_head_fwd = quat_rotate(
                    res.world_quat[-1, self.head_idx], np.array([0.0, 1.0, 0.0], dtype=np.float32)
                )
                eh_mag = math.hypot(float(end_head_fwd[0]), float(end_head_fwd[2]))
                if eh_mag > 1e-4:
                    end_hx = float(end_head_fwd[0]) / eh_mag
                    end_hz = float(end_head_fwd[2]) / eh_mag
                else:
                    end_h = _heading_from_hip_positions(
                        res.world_pos[-1, self.engine.right_leg_idx],
                        res.world_pos[-1, self.engine.left_leg_idx],
                    )
                    end_hx, end_hz = math.sin(end_h), math.cos(end_h)
            else:
                end_h = _heading_from_hip_positions(
                    res.world_pos[-1, self.engine.right_leg_idx],
                    res.world_pos[-1, self.engine.left_leg_idx],
                )
                end_hx, end_hz = math.sin(end_h), math.cos(end_h)

            meta = {
                "chunk_id": new_chunk_id,
                "base_chunk_id": base_chunk_id,
                "anchor_frame": round(anchor_frame, 4),
                "mode": mode,
                "style": style,
                "phase_done": phase_done,
                "num_frames": res.num_frames,
                "fps": int(MODEL_SAMPLE_RATE),
                "pred_num_tokens": res.pred_num_tokens,
                "transition_early": min(transition_early, max(1, res.num_frames - 4)),
                "latency_ms": round(res.latency_ms, 2),
                "infer_ms": round(res.infer_ms, 2),
                "root_end": [
                    round(float(res.world_pos[-1, 0, 0]), 3),
                    round(float(res.world_pos[-1, 0, 2]), 3),
                ],
                "heading_end": [round(end_hx, 4), round(end_hz, 4)],
                "num_render_joints": len(RENDER_JOINT_NAMES),
                "num_bone_joints": len(BONE_JOINT_NAMES),
            }
            return meta, res, ctx_wp, tgt_wp

    def generate_step(self, req: dict[str, Any]) -> dict[str, Any]:
        meta, res, ctx_wp, tgt_wp = self._run_generate(req)
        return {
            **meta,
            "frames_pos": np.round(res.world_pos[:, self.render_indices], 4).tolist(),
            "frames_quat": np.round(res.world_quat[:, self.render_indices], 4).tolist(),
            "foot_contacts": np.round(res.foot_contacts, 3).tolist(),
            "context_pos": np.round(ctx_wp[:, self.render_indices], 4).tolist(),
            "target_pos": np.round(tgt_wp[:, self.render_indices], 4).tolist(),
        }

    def generate_step_bin(self, req: dict[str, Any]) -> bytes:
        """Ultra-fast binary packet for POST /api/generate_bin (zero JSON array serialization):
        - Bytes 0..4:            uint32 LE `meta_len` (space-padded to multiple of 4 for Float32 alignment)
        - Bytes 4..4+meta_len:   UTF-8 JSON metadata
        - Contiguous LE float32 buffers in exact order:
            1. frames_pos       [F, 24, 3]
            2. frames_bone_quat [F, 26, 4]  (index 0 = Hips world rot, 1..25 = parent-relative local rot)
            3. foot_contacts    [F, 4]
            4. context_pos      [4, 24, 3]
            5. target_pos       [4, 24, 3]
        """
        meta, res, ctx_wp, tgt_wp = self._run_generate(req)
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        pad = (-len(meta_bytes)) % 4
        if pad:
            meta_bytes += b" " * pad
        header = struct.pack("<I", len(meta_bytes))

        frames_pos_b = np.ascontiguousarray(
            res.world_pos[:, self.render_indices], dtype="<f4"
        ).tobytes()
        frames_bone_quat_b = np.ascontiguousarray(
            res.local_quat[:, self.bone_indices], dtype="<f4"
        ).tobytes()
        foot_contacts_b = np.ascontiguousarray(res.foot_contacts, dtype="<f4").tobytes()
        context_pos_b = np.ascontiguousarray(
            ctx_wp[:, self.render_indices], dtype="<f4"
        ).tobytes()
        target_pos_b = np.ascontiguousarray(
            tgt_wp[:, self.render_indices], dtype="<f4"
        ).tobytes()

        return b"".join(
            [
                header,
                meta_bytes,
                frames_pos_b,
                frames_bone_quat_b,
                foot_contacts_b,
                context_pos_b,
                target_pos_b,
            ]
        )

    def reset_session(self, sid: str) -> dict[str, Any]:
        init_p, init_q = place_target_window(*self.idle_canon, 1.12, 1.35, -0.46, 0.888)
        wp, wq = self.engine.fast_fk(init_p, init_q)
        with self.lock:
            st = self.sessions.get(sid)
            if st is None:
                st = SessionState()
                self.sessions[sid] = st
            st.chunk_id += 1
            cid = st.chunk_id
            st.chunks.clear()
            st.chunks[cid] = (init_p.copy(), init_q.copy(), init_p.copy(), init_q.copy())
            st.local_pos = init_p.copy()
            st.local_quat = init_q.copy()
            st.history_pos = init_p.copy()
            st.history_quat = init_q.copy()
        return {
            "chunk_id": cid,
            "pose": {
                "pos": np.round(wp[-1, self.render_indices], 4).tolist(),
                "quat": np.round(wq[-1, self.render_indices], 4).tolist(),
                "bone_quat": np.round(init_q[-1, self.bone_indices], 5).tolist(),
            },
        }


RUNTIME: MotionBricksRuntime | None = None


class APIHandler(BaseHTTPRequestHandler):
    disable_nagle_algorithm = True

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep logs compact & clean
        pass

    def _send_bytes(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_bytes(code, body, "application/json")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        assert RUNTIME is not None
        if self.path == "/api/health":
            self._send_json(200, {"ok": True, "load_ms": round(RUNTIME.engine.load_ms, 1)})
        elif self.path == "/api/init":
            self._send_json(200, RUNTIME.create_session())
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        assert RUNTIME is not None
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return

        if self.path == "/api/generate_bin":
            try:
                body = RUNTIME.generate_step_bin(data)
                self._send_bytes(200, body, "application/octet-stream")
            except KeyError:
                self._send_json(404, {"error": "Unknown session_id"})
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self._send_json(500, {"error": str(exc)})
        elif self.path == "/api/generate":
            try:
                out = RUNTIME.generate_step(data)
                self._send_json(200, out)
            except KeyError:
                self._send_json(404, {"error": "Unknown session_id"})
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self._send_json(500, {"error": str(exc)})
        elif self.path == "/api/reset":
            sid = str(data.get("session_id", ""))
            self._send_json(200, RUNTIME.reset_session(sid))
        else:
            self._send_json(404, {"error": "Not found"})


def main() -> None:
    global RUNTIME
    RUNTIME = MotionBricksRuntime()
    host, port = "127.0.0.1", 8765
    server = ThreadingHTTPServer((host, port), APIHandler)
    print(f"[MotionBricks] API listening on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
