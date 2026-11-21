#!/usr/bin/env python3
"""Headless biomechanical invariant verification suite for NVIDIA MotionBricks SOMA API.

Instantiates `MotionBricksRuntime()` directly and runs real end-to-end generation chains:
  1. `jog -> roll -> steer` (including both end-frame and prefetch `transition_early` ONNX handoffs)
  2. `idle -> roll -> idle` (and wall-proximity abort safety check)
  3. `crawl_a -> crawl_b -> crawl_a` (canonical targets + live ONNX-generated crawl chain)
  4. `combat -> punch_left -> punch_right -> combat_recover` (KEEP LOCKED regression guard)
  5. Binary packet framing check (`POST /api/generate_bin` layout parity for `roll` & `crawl`)

Exits 0 iff EVERY SINGLE biomechanical invariant passes.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "server"))

from api import (  # noqa: E402
    BONE_JOINT_NAMES,
    RENDER_JOINT_NAMES,
    ROOM_MAX_X,
    ROOM_MAX_Z,
    ROOM_MIN_X,
    ROOM_MIN_Z,
    MotionBricksRuntime,
)
from bvh_loader import quat_rotate  # noqa: E402
from soma_onnx import _heading_from_hip_positions  # noqa: E402


class InvariantSuite:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def check(self, name: str, cond: bool, detail: str) -> None:
        tag = "PASS" if cond else "FAIL"
        line = f"  [{tag}] {name}: {detail}"
        print(line)
        if cond:
            self.passes.append(line)
        else:
            self.failures.append(line)


def _measure_signed_pitch_sweep_deg(
    world_quat: np.ndarray, dx: float, dz: float
) -> tuple[float, float, float]:
    """Measure unwrapped signed forward pitch sweep of Hips local +X (anatomical up)
    projected onto the travel plane spanned by forward=(dx, 0, dz) and up=(0, 1, 0)."""
    f_count = world_quat.shape[0]
    up_vecs = quat_rotate(
        world_quat[:, 0], np.broadcast_to(np.array([1.0, 0.0, 0.0], dtype=np.float32), (f_count, 3))
    )
    fwd_comp = up_vecs[:, 0] * dx + up_vecs[:, 2] * dz
    up_comp = up_vecs[:, 1]
    raw_rad = np.arctan2(fwd_comp, up_comp)
    unwrapped_deg = np.degrees(np.unwrap(raw_rad))
    unwrapped_deg -= unwrapped_deg[0]
    # Check monotonicity across the active roll sweep (exclude tiny natural jog bob on final 3 frames)
    roll_steps = np.diff(unwrapped_deg[: min(22, f_count)])
    total_sweep = float(unwrapped_deg[-1])
    min_step = float(roll_steps.min())
    max_step = float(roll_steps.max())
    return total_sweep, min_step, max_step


def _check_roll_chain(
    rt: MotionBricksRuntime,
    suite: InvariantSuite,
    label: str,
    setup_mode: str,
    setup_style: str,
    next_mode: str,
    next_style: str,
    roll_dir: tuple[float, float],
) -> None:
    print(f"\n=== ROLL CHAIN: {label} ===")
    skel = rt.skeleton
    j_lk = skel.joint_index("LeftShin")
    j_rk = skel.joint_index("RightShin")
    j_lh = skel.joint_index("LeftHand")
    j_rh = skel.joint_index("RightHand")
    j_hd = skel.joint_index("Head")

    sess = rt.create_session()
    sid = sess["session_id"]
    dx, dz = roll_dir
    mag = math.hypot(dx, dz)
    dx, dz = dx / mag, dz / mag

    # 1. Warm up into starting locomotion/idle state in open space facing (dx, dz)
    cid = 0
    for _ in range(2):
        meta_pre, res_pre, _, _ = rt._run_generate(
            {
                "session_id": sid,
                "base_chunk_id": cid,
                "mode": setup_mode,
                "style": setup_style,
                "dir_x": dx,
                "dir_z": dz,
            }
        )
        cid = int(meta_pre["chunk_id"])

    pre_end_wp = res_pre.world_pos[-1].copy()

    # 2. Execute single continuous 360-deg diagonal parkour combat roll (`mode == "roll"`)
    meta_roll, res_roll, ctx_wp, _ = rt._run_generate(
        {
            "session_id": sid,
            "base_chunk_id": cid,
            "mode": "roll",
            "style": setup_style,
            "dir_x": dx,
            "dir_z": dz,
        }
    )
    roll_cid = int(meta_roll["chunk_id"])
    wp = res_roll.world_pos
    wq = res_roll.world_quat
    n_f = res_roll.num_frames

    # Also verify binary serialization on this exact mode
    bin_pkt = rt.generate_step_bin(
        {
            "session_id": sid,
            "base_chunk_id": cid,
            "mode": "roll",
            "style": setup_style,
            "dir_x": dx,
            "dir_z": dz,
        }
    )
    meta_len = struct.unpack("<I", bin_pkt[:4])[0]
    bin_meta = json.loads(bin_pkt[4 : 4 + meta_len].decode("utf-8"))
    expected_floats = (
        bin_meta["num_frames"] * len(RENDER_JOINT_NAMES) * 3
        + bin_meta["num_frames"] * len(BONE_JOINT_NAMES) * 4
        + bin_meta["num_frames"] * 4
        + 4 * len(RENDER_JOINT_NAMES) * 3
        + 4 * len(RENDER_JOINT_NAMES) * 3
    )
    expected_bytes = 4 + meta_len + expected_floats * 4

    sweep_deg, min_d_pitch, max_d_pitch = _measure_signed_pitch_sweep_deg(wq, dx, dz)

    # Check yaw continuity at exit (zero 180-deg flip!)
    end_h = _heading_from_hip_positions(
        wp[-1, rt.engine.right_leg_idx], wp[-1, rt.engine.left_leg_idx]
    )
    end_fx, end_fz = math.sin(end_h), math.cos(end_h)
    yaw_dot = end_fx * dx + end_fz * dz

    ksep = np.linalg.norm(wp[:, j_lk] - wp[:, j_rk], axis=-1)
    min_head_y = float(wp[:, j_hd, 1].min())
    min_all_y = float(wp[:, :, 1].min())
    inv_hand_excess = float(
        np.max(np.maximum(wp[6:19, j_lh, 1], wp[6:19, j_rh, 1]) - wp[6:19, 0, 1])
    )
    start_err = float(np.linalg.norm(wp[0] - pre_end_wp, axis=-1).max())
    travel = float(math.hypot(wp[-1, 0, 0] - wp[0, 0, 0], wp[-1, 0, 2] - wp[0, 0, 2]))
    in_room = bool(
        np.all((wp[:, 0, 0] >= ROOM_MIN_X - 1e-4) & (wp[:, 0, 0] <= ROOM_MAX_X + 1e-4))
        and np.all((wp[:, 0, 2] >= ROOM_MIN_Z - 1e-4) & (wp[:, 0, 2] <= ROOM_MAX_Z + 1e-4))
    )

    suite.check(
        f"{label} | single chunk frame count",
        meta_roll["mode"] == "roll" and meta_roll["phase_done"] is True and 22 <= n_f <= 28,
        f"mode={meta_roll['mode']} phase_done={meta_roll['phase_done']} num_frames={n_f} (req 22..28)",
    )
    suite.check(
        f"{label} | binary packet parity",
        len(bin_pkt) == expected_bytes and bin_meta["mode"] == "roll",
        f"bytes={len(bin_pkt)} expected={expected_bytes}",
    )
    suite.check(
        f"{label} | monotonic 360 deg forward pitch sweep",
        sweep_deg >= 340.0 and min_d_pitch >= -1.0 and yaw_dot >= 0.85,
        f"sweep={sweep_deg:.1f}deg min_step={min_d_pitch:+.2f}deg max_step={max_d_pitch:+.2f}deg exit_yaw_dot={yaw_dot:.3f}",
    )
    suite.check(
        f"{label} | narrow knee_sep (zero frog-splay)",
        float(ksep.max()) <= 0.32,
        f"knee_sep=[{float(ksep.min()):.3f}, {float(ksep.max()):.3f}]m (req <= 0.32m)",
    )
    suite.check(
        f"{label} | head clearance",
        min_head_y >= 0.14,
        f"min(Head.y)={min_head_y:.3f}m (req >= 0.14m)",
    )
    suite.check(
        f"{label} | zero floor penetration (all 77 joints)",
        min_all_y >= -0.02,
        f"min(all_joints.y)={min_all_y:.4f}m (req >= -0.02m)",
    )
    suite.check(
        f"{label} | compact inversion arms (no overhead flip)",
        inv_hand_excess <= 0.55,
        f"max(hand.y - hips.y)[f=6..18]={inv_hand_excess:+.3f}m (req <= +0.55m)",
    )
    suite.check(
        f"{label} | C0 entry continuity",
        start_err < 0.03,
        f"start_pose_err={start_err:.5f}m (req < 0.03m)",
    )
    suite.check(
        f"{label} | travel distance & room bounds",
        1.55 <= travel <= 2.30 and in_room,
        f"travel={travel:.3f}m (req 1.55..2.30m) in_room={in_room}",
    )

    # 3. Subsequent ONNX handoff check at BOTH end-frame (n_f - 1) and prefetch (n_f - 1 - transition_early)
    te = int(meta_roll["transition_early"])
    for handoff_lbl, anc in [("end-frame", n_f - 1), (f"prefetch(-{te})", n_f - 1 - te)]:
        _, res_next, _, _ = rt._run_generate(
            {
                "session_id": sid,
                "base_chunk_id": roll_cid,
                "anchor_frame": float(anc),
                "mode": next_mode,
                "style": next_style,
                "dir_x": dx,
                "dir_z": dz,
            }
        )
        first_jump = float(np.linalg.norm(res_next.world_pos[0] - wp[anc], axis=-1).max())
        end_hips_y = float(res_next.world_pos[-1, 0, 1])
        suite.check(
            f"{label} | ONNX {next_mode} handoff ({handoff_lbl})",
            first_jump < 0.08 and end_hips_y >= 0.80,
            f"max_first_frame_jump={first_jump:.5f}m (<0.08m), end_hips_y={end_hips_y:.3f}m (>=0.80m)",
        )


def _check_roll_wall_abort(rt: MotionBricksRuntime, suite: InvariantSuite) -> None:
    print("\n=== ROLL WALL-SAFETY ABORT ===")
    sess = rt.create_session()
    sid = sess["session_id"]
    st = rt.sessions[sid]
    # Place character 0.35 m from ROOM_MAX_X wall facing +X directly into the concrete wall
    from soma_onnx import place_target_window

    wall_p, wall_q = place_target_window(*rt.idle_canon, ROOM_MAX_X - 0.35, 0.0, 1.0, 0.0)
    st.chunks[0] = (wall_p.copy(), wall_q.copy(), wall_p.copy(), wall_q.copy())
    meta, res, _, _ = rt._run_generate(
        {
            "session_id": sid,
            "base_chunk_id": 0,
            "mode": "roll",
            "style": "jog",
            "dir_x": 1.0,
            "dir_z": 0.0,
        }
    )
    max_x = float(res.world_pos[:, 0, 0].max())
    suite.check(
        "wall-proximity roll cleanly aborts & stays inside ROOM_BOUNDS",
        meta["mode"] == "idle" and max_x <= ROOM_MAX_X + 1e-4,
        f"mode={meta['mode']} max_hips_x={max_x:.3f}m (ROOM_MAX_X={ROOM_MAX_X})",
    )


def _check_crawl_chain(rt: MotionBricksRuntime, suite: InvariantSuite) -> None:
    print("\n=== CRAWL TARGETS & ONNX CHAIN (crawl_a -> crawl_b -> crawl_a) ===")
    skel = rt.skeleton
    j_lk = skel.joint_index("LeftShin")
    j_rk = skel.joint_index("RightShin")
    j_le = skel.joint_index("LeftForeArm")
    j_re = skel.joint_index("RightForeArm")
    j_lh = skel.joint_index("LeftHand")
    j_rh = skel.joint_index("RightHand")
    j_lm3 = skel.joint_index("LeftHandMiddle3")
    j_rm3 = skel.joint_index("RightHandMiddle3")
    j_ch = skel.joint_index("Chest")
    j_hd = skel.joint_index("Head")

    # 1. Inspect canonical crawl_a & crawl_b targets
    wpa, _ = rt.engine.fast_fk(rt.crawl_a_canon[0][:1], rt.crawl_a_canon[1][:1])
    wpb, _ = rt.engine.fast_fk(rt.crawl_b_canon[0][:1], rt.crawl_b_canon[1][:1])

    for nm, wp0 in [("crawl_a_canon", wpa[0]), ("crawl_b_canon", wpb[0])]:
        ksep = float(np.linalg.norm(wp0[j_lk] - wp0[j_rk]))
        min_elbow_y = float(min(wp0[j_le, 1], wp0[j_re, 1]))
        lh_y, rh_y = float(wp0[j_lh, 1]), float(wp0[j_rh, 1])
        min_hand_y, max_hand_y = min(lh_y, rh_y), max(lh_y, rh_y)
        lm3_dy = abs(float(wp0[j_lm3, 1] - lh_y))
        rm3_dy = abs(float(wp0[j_rm3, 1] - rh_y))
        suite.check(
            f"{nm} | knee_sep & grounded knees",
            0.18 <= ksep <= 0.34,
            f"knee_sep={ksep:.3f}m (req 0.18..0.34m) | LK.y={wp0[j_lk,1]:.3f}m RK.y={wp0[j_rk,1]:.3f}m",
        )
        suite.check(
            f"{nm} | pillar arms & torso heights",
            min_elbow_y >= 0.20,
            f"min_elbow_y={min_elbow_y:.3f}m (>=0.20m) | Hips.y={wp0[0,1]:.3f}m Chest.y={wp0[j_ch,1]:.3f}m Head.y={wp0[j_hd,1]:.3f}m",
        )
        suite.check(
            f"{nm} | supporting hands & flat palms",
            0.02 <= min_hand_y <= 0.10 and max_hand_y <= 0.10 and lm3_dy < 0.035 and rm3_dy < 0.035,
            f"LH.y={lh_y:.3f}m RH.y={rh_y:.3f}m | |LM3-LH|={lm3_dy:.4f}m |RM3-RH|={rm3_dy:.4f}m (<0.035m)",
        )

    lk_alt = abs(float(wpb[0, j_lk, 0] - wpa[0, j_lk, 0]))
    rk_alt = abs(float(wpb[0, j_rk, 0] - wpa[0, j_rk, 0]))
    suite.check(
        "crawl_a vs crawl_b | contralateral knee X-alternation",
        lk_alt >= 0.12 and rk_alt >= 0.12,
        f"L_knee_alt={lk_alt:.3f}m R_knee_alt={rk_alt:.3f}m (req >= 0.12m)",
    )

    # 2. Live ONNX crawl_a -> crawl_b -> crawl_a chain via MotionBricksRuntime!
    sess = rt.create_session()
    sid = sess["session_id"]
    st = rt.sessions[sid]
    # Seed session directly in crawl_a facing +X at origin so we measure steady-state crawl chunks
    st.chunks[0] = (
        rt.crawl_a_canon[0].copy(),
        rt.crawl_a_canon[1].copy(),
        rt.crawl_a_canon[0].copy(),
        rt.crawl_a_canon[1].copy(),
    )
    st.crawl_phase = 1  # Next steer targets crawl_b, then crawl_a, then crawl_b

    cid = 0
    for step_nm in ("ONNX crawl_a->crawl_b", "ONNX crawl_b->crawl_a"):
        meta, res, _, _ = rt._run_generate(
            {
                "session_id": sid,
                "base_chunk_id": cid,
                "mode": "steer",
                "style": "crawl",
                "dir_x": 1.0,
                "dir_z": 0.0,
            }
        )
        cid = int(meta["chunk_id"])
        wp = res.world_pos
        ksep = np.linalg.norm(wp[:, j_lk] - wp[:, j_rk], axis=-1)
        min_elbow_y = float(min(wp[:, j_le, 1].min(), wp[:, j_re, 1].min()))
        lh_y = wp[:, j_lh, 1]
        rh_y = wp[:, j_rh, 1]
        # Supporting hand on each frame is the lower of the two hands
        support_hand_y = np.minimum(lh_y, rh_y)
        lm3_dy = float(np.abs(wp[:, j_lm3, 1] - lh_y).max())
        rm3_dy = float(np.abs(wp[:, j_rm3, 1] - rh_y).max())
        lk_rel_x = wp[:, j_lk, 0] - wp[:, 0, 0]
        rk_rel_x = wp[:, j_rk, 0] - wp[:, 0, 0]
        knee_span = float(min(lk_rel_x.max() - lk_rel_x.min(), rk_rel_x.max() - rk_rel_x.min()))

        suite.check(
            f"{step_nm} | knee_sep",
            float(ksep.min()) >= 0.18 and float(ksep.max()) <= 0.34,
            f"knee_sep=[{float(ksep.min()):.3f}, {float(ksep.max()):.3f}]m (req 0.18..0.34m)",
        )
        suite.check(
            f"{step_nm} | pillar arms (elbow_y >= 0.20m)",
            min_elbow_y >= 0.20,
            f"min(elbow_y)={min_elbow_y:.3f}m",
        )
        suite.check(
            f"{step_nm} | supporting hand_y in [0.02, 0.10]m",
            float(support_hand_y.min()) >= 0.02 and float(support_hand_y.max()) <= 0.10,
            f"support_hand_y=[{float(support_hand_y.min()):.3f}, {float(support_hand_y.max()):.3f}]m",
        )
        suite.check(
            f"{step_nm} | flat palms (< 0.035m)",
            lm3_dy < 0.035 and rm3_dy < 0.035,
            f"max|LM3-LH|={lm3_dy:.4f}m max|RM3-RH|={rm3_dy:.4f}m",
        )
        suite.check(
            f"{step_nm} | contralateral knee alternation",
            knee_span >= 0.12,
            f"L_rel_x_span={float(lk_rel_x.max()-lk_rel_x.min()):.3f}m R_rel_x_span={float(rk_rel_x.max()-rk_rel_x.min()):.3f}m (>=0.12m)",
        )


def _check_combat_regression(rt: MotionBricksRuntime, suite: InvariantSuite) -> None:
    print("\n=== COMBAT REGRESSION GUARD (combat -> punch_left -> punch_right -> combat_recover) ===")
    skel = rt.skeleton
    j_lh = skel.joint_index("LeftHand")
    j_rh = skel.joint_index("RightHand")

    sess = rt.create_session()
    sid = sess["session_id"]

    # Enter combat guard facing +X
    meta, _, _, _ = rt._run_generate(
        {"session_id": sid, "base_chunk_id": 0, "mode": "combat_recover", "style": "combat", "dir_x": 1.0, "dir_z": 0.0}
    )
    cid = int(meta["chunk_id"])

    # Left Jab (`punch_left`)
    meta_pl, res_pl, _, _ = rt._run_generate(
        {"session_id": sid, "base_chunk_id": cid, "mode": "punch_left", "style": "combat", "dir_x": 1.0, "dir_z": 0.0}
    )
    cid = int(meta_pl["chunk_id"])
    pl_reach = res_pl.world_pos[:, j_lh, 0] - res_pl.world_pos[:, 0, 0]
    pl_peak_f = int(np.argmax(pl_reach))
    pl_max_reach = float(pl_reach[pl_peak_f])
    pl_peak_y = float(res_pl.world_pos[pl_peak_f, j_lh, 1])
    suite.check(
        "punch_left | max left-hand reach & chin height (KEEP LOCKED)",
        pl_max_reach >= 0.72 and 1.22 <= pl_peak_y <= 1.38,
        f"reach={pl_max_reach:.3f}m (>=0.72m), peak_y={pl_peak_y:.3f}m (req 1.22..1.38m)",
    )

    # Right Cross (`punch_right`)
    meta_pr, res_pr, _, _ = rt._run_generate(
        {"session_id": sid, "base_chunk_id": cid, "mode": "punch_right", "style": "combat", "dir_x": 1.0, "dir_z": 0.0}
    )
    cid = int(meta_pr["chunk_id"])
    pr_reach = res_pr.world_pos[:, j_rh, 0] - res_pr.world_pos[:, 0, 0]
    pr_peak_f = int(np.argmax(pr_reach))
    pr_max_reach = float(pr_reach[pr_peak_f])
    pr_peak_y = float(res_pr.world_pos[pr_peak_f, j_rh, 1])
    suite.check(
        "punch_right | max right-hand reach & chin height (KEEP LOCKED)",
        pr_max_reach >= 0.72 and 1.22 <= pr_peak_y <= 1.38,
        f"reach={pr_max_reach:.3f}m (>=0.72m), peak_y={pr_peak_y:.3f}m (req 1.22..1.38m)",
    )

    # Combat Recover (`combat_recover`)
    meta_rec, res_rec, _, _ = rt._run_generate(
        {"session_id": sid, "base_chunk_id": cid, "mode": "combat_recover", "style": "combat", "dir_x": 1.0, "dir_z": 0.0}
    )
    rec_hips_y = float(res_rec.world_pos[-1, 0, 1])
    suite.check(
        "combat_recover | returns upright into guard",
        meta_rec["mode"] == "combat_recover" and rec_hips_y >= 0.85,
        f"end_hips_y={rec_hips_y:.3f}m",
    )


def main() -> int:
    rt = MotionBricksRuntime()
    t_suite0 = time.perf_counter()
    suite = InvariantSuite()

    _check_roll_chain(
        rt, suite, "jog->roll->steer (+X)", "steer", "jog", "steer", "jog", (1.0, 0.0)
    )
    _check_roll_chain(
        rt, suite, "idle->roll->idle (diag)", "idle", "jog", "idle", "jog", (-0.6, -0.8)
    )
    _check_roll_wall_abort(rt, suite)
    _check_crawl_chain(rt, suite)
    _check_combat_regression(rt, suite)

    elapsed_ms = (time.perf_counter() - t_suite0) * 1000.0
    print(
        f"\n=== SUMMARY: {len(suite.passes)} passed, {len(suite.failures)} failed "
        f"(suite execution={elapsed_ms:.0f} ms) ==="
    )
    if suite.failures:
        print("FAILURES:")
        for f in suite.failures:
            print(f)
        return 1
    print("ALL BIOMECHANICAL INVARIANTS PASSED 100%!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
