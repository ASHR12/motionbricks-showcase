#!/usr/bin/env python3
"""
Export all NVIDIA MotionBricks SOMA FP16 neural motion styles & actions from the running
local server (http://127.0.0.1:8765) into a compact, precision-quantized browser binary:
  web/public/models/motion-bank.bin  (~0.45-0.65 MB raw, ~0.30 MB gzipped)

Quantization fidelity:
  - Joint positions (relative to chunk root XZ, absolute Y): Int16 @ 1/4000 m = 0.25 mm precision
  - Bone quaternions (Hips world + 25 local child joints):   Int16 @ 1/32767  = ~0.003 deg precision
  - Foot contacts (4 channels):                              Uint8 @ 1/255

Zero external dependencies beyond stdlib + numpy.
"""
from __future__ import annotations

import json
import math
import os
import struct
import urllib.request
from typing import Any

import numpy as np

API = "http://127.0.0.1:8765"
OUT_PATH = "/Users/ashutosh/development-work/g4-neon/motionbricks-web/web/public/models/motion-bank.bin"
POS_SCALE = 4000.0   # 0.25 mm resolution, range +-8.191 m relative to chunk start root XZ
QUAT_SCALE = 32767.0


def http_get_json(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{API}{path}", timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def http_post_json(path: str, body: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{API}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def http_post_bin(path: str, body: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{API}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read()
    meta_len = struct.unpack_from("<I", raw, 0)[0]
    meta = json.loads(raw[4 : 4 + meta_len].decode("utf-8").strip())
    nf = int(meta["num_frames"])
    nr = int(meta["num_render_joints"])
    nb = int(meta["num_bone_joints"])
    off = 4 + meta_len

    def take_f32(count: int) -> np.ndarray:
        nonlocal off
        arr = np.frombuffer(raw, dtype="<f4", count=count, offset=off).copy()
        off += count * 4
        return arr

    frames_pos = take_f32(nf * nr * 3).reshape(nf, nr, 3)
    frames_bquat = take_f32(nf * nb * 4).reshape(nf, nb, 4)
    foot_contacts = take_f32(nf * 4).reshape(nf, 4)
    context_pos = take_f32(4 * nr * 3).reshape(4, nr, 3)
    target_pos = take_f32(4 * nr * 3).reshape(4, nr, 3)
    return {
        "meta": meta,
        "frames_pos": frames_pos,
        "frames_bquat": frames_bquat,
        "foot_contacts": foot_contacts,
        "context_pos": context_pos,
        "target_pos": target_pos,
    }


def reset_sess(sid: str) -> None:
    http_post_json("/api/reset", {"session_id": sid})


def gen(sid: str, base_id: int, anchor: float, mode: str, style: str = "walk", **kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "session_id": sid,
        "base_chunk_id": base_id,
        "anchor_frame": anchor,
        "mode": mode,
        "style": style,
        "sprint": False,
        **kw,
    }
    return http_post_bin("/api/generate_bin", body)


def advance_chain(
    sid: str,
    prev: dict[str, Any] | None,
    mode: str,
    style: str = "walk",
    **kw: Any,
) -> dict[str, Any]:
    if prev is None:
        return gen(sid, 0, 0.0, mode, style, **kw)
    m = prev["meta"]
    anchor = max(0.0, float(m["num_frames"] - 1 - int(m.get("transition_early", 4))))
    return gen(sid, int(m["chunk_id"]), anchor, mode, style, **kw)


def quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for XYZW quaternions (q1 broadcastable over q2)."""
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
    )


def canonicalize_clip(
    chunk: dict[str, Any],
    desired_heading: tuple[float, float] | None = None,
    keep_world_origin: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """
    Canonicalize a generated chunk so frame 0 Hips sits at XZ=(0, 0) (or relative to `keep_world_origin`)
    and faces canonical forward `-Z` (0, -1), keeping exact world Y heights and local child rotations!
    """
    pos = chunk["frames_pos"].copy()        # [F, 24, 3]
    bquat = chunk["frames_bquat"].copy()    # [F, 26, 4]
    ctx = chunk["context_pos"].copy()       # [4, 24, 3]
    tgt = chunk["target_pos"].copy()        # [4, 24, 3]
    contacts = chunk["foot_contacts"].copy()
    meta = chunk["meta"]

    if keep_world_origin is not None:
        ox, oz = float(keep_world_origin[0]), float(keep_world_origin[1])
    else:
        ox = float(pos[0, 0, 0])
        oz = float(pos[0, 0, 2])

    # Determine heading to align to canonical (0, -1)
    if desired_heading is not None:
        hx, hz = float(desired_heading[0]), float(desired_heading[1])
    else:
        h = meta.get("heading_end", [0.0, -1.0])
        hx, hz = float(h[0]), float(h[1])
    hlen = math.hypot(hx, hz)
    if hlen < 1e-4:
        hx, hz = 0.0, -1.0
    else:
        hx, hz = hx / hlen, hz / hlen

    # Angle to rotate (hx, hz) -> (0, -1) around Y-up axis:
    # Right-handed Y-up rotation by alpha maps (x, z) -> (c*x + s*z, -s*x + c*z).
    # We want alpha = atan2(0, -1) - atan2(hx, hz) = pi - atan2(hx, hz).
    alpha = math.atan2(0.0, -1.0) - math.atan2(hx, hz)
    # Normalize to [-pi, pi]
    alpha = (alpha + math.pi) % (2.0 * math.pi) - math.pi
    c = math.cos(alpha)
    s = math.sin(alpha)
    q_yaw = np.array([0.0, math.sin(alpha * 0.5), 0.0, math.cos(alpha * 0.5)], dtype=np.float32)

    def xform_pos(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        dx = arr[..., 0] - ox
        dz = arr[..., 2] - oz
        out[..., 0] = c * dx + s * dz
        out[..., 2] = -s * dx + c * dz
        return out

    pos_c = xform_pos(pos)
    ctx_c = xform_pos(ctx)
    tgt_c = xform_pos(tgt)

    # Rotate ONLY root bone (Hips, index 0) by q_yaw; child joints 1..25 are parent-relative!
    bquat_c = bquat.copy()
    bquat_c[:, 0, :] = quat_mul_xyzw(q_yaw[None, :], bquat[:, 0, :])
    # Normalize all quaternions cleanly
    norms = np.linalg.norm(bquat_c, axis=-1, keepdims=True)
    bquat_c = np.where(norms > 1e-8, bquat_c / np.maximum(norms, 1e-8), np.array([0, 0, 0, 1], dtype=np.float32))

    # Measure displacement in canonical space from frame 0 to last frame (and end heading)
    disp_x = float(pos_c[-1, 0, 0] - pos_c[0, 0, 0])
    disp_z = float(pos_c[-1, 0, 2] - pos_c[0, 0, 2])
    he = meta.get("heading_end", [hx, hz])
    hex_c = c * float(he[0]) + s * float(he[1])
    hez_c = -s * float(he[0]) + c * float(he[1])
    he_len = math.hypot(hex_c, hez_c) or 1.0

    # Check quantization bounds
    max_abs = float(max(np.max(np.abs(pos_c)), np.max(np.abs(ctx_c)), np.max(np.abs(tgt_c))))
    if max_abs >= 8.15:
        raise ValueError(f"Position overflow for {meta['mode']}/{meta['style']}: max_abs={max_abs:.3f}m")
    if np.isnan(pos_c).any() or np.isnan(bquat_c).any():
        raise ValueError(f"NaN detected in {meta['mode']}/{meta['style']}")

    return {
        "meta": meta,
        "frames_pos": pos_c,
        "frames_bquat": bquat_c,
        "foot_contacts": np.clip(contacts, 0.0, 1.0),
        "context_pos": ctx_c,
        "target_pos": tgt_c,
        "disp_xz": [round(disp_x, 4), round(disp_z, 4)],
        "heading_end": [round(hex_c / he_len, 4), round(hez_c / he_len, 4)],
        "start_root_xz": [round(float(pos_c[0, 0, 0]), 4), round(float(pos_c[0, 0, 2]), 4)],
    }


def quantize_clip(c: dict[str, Any]) -> bytes:
    def q_pos(arr: np.ndarray) -> bytes:
        q = np.clip(np.round(arr * POS_SCALE), -32767, 32767).astype("<i2")
        return np.ascontiguousarray(q).tobytes()

    def q_quat(arr: np.ndarray) -> bytes:
        q = np.clip(np.round(arr * QUAT_SCALE), -32767, 32767).astype("<i2")
        return np.ascontiguousarray(q).tobytes()

    def q_contacts(arr: np.ndarray) -> bytes:
        q = np.clip(np.round(arr * 255.0), 0, 255).astype("u1")
        return np.ascontiguousarray(q).tobytes()

    raw = b"".join(
        [
            q_pos(c["frames_pos"]),
            q_quat(c["frames_bquat"]),
            q_contacts(c["foot_contacts"]),
            q_pos(c["context_pos"]),
            q_pos(c["target_pos"]),
        ]
    )
    pad = (-len(raw)) % 4
    if pad:
        raw += b"\x00" * pad
    return raw


def main() -> None:
    init_data = http_get_json("/api/init")
    sid = init_data["session_id"]
    styles = [s["id"] for s in init_data["styles"]]
    print(f"[ExportBank] Connected to live SOMA FP16 ONNX server. Styles ({len(styles)}): {styles}")

    clips: dict[str, dict[str, Any]] = {}

    # 1. Steady-state forward locomotion chunk for EVERY style (warm up 3 chunks along -Z, capture #3!)
    for st in styles:
        reset_sess(sid)
        prev = None
        for step_i in range(4):
            prev = advance_chain(sid, prev, "steer", st, dir_x=0.0, dir_z=-1.0)
        clips[f"steer:{st}"] = canonicalize_clip(prev, desired_heading=(0.0, -1.0))
        m = prev["meta"]
        print(f"  steer:{st:<10} -> {m['num_frames']}f, disp={clips[f'steer:{st}']['disp_xz']}")

    # 2. Style-specific idles (warm up 2 chunks so posture settles cleanly!)
    idle_styles = ["walk", "crouch", "combat", "zombie", "crawl", "injured", "tactical", "vogue"]
    for st in idle_styles:
        reset_sess(sid)
        # First nudge gently in that style so posture enters style, then settle into idle
        p = advance_chain(sid, None, "steer", st, dir_x=0.0, dir_z=-1.0)
        p = advance_chain(sid, p, "idle", st)
        p = advance_chain(sid, p, "idle", st)
        clips[f"idle:{st}"] = canonicalize_clip(p, desired_heading=(0.0, -1.0))
        print(f"  idle:{st:<11} -> {p['meta']['num_frames']}f")

    # 3. Hurdle Jump Leap + Jump Land (from jog stride along -Z)
    reset_sess(sid)
    p = None
    for _ in range(3):
        p = advance_chain(sid, p, "steer", "jog", dir_x=0.0, dir_z=-1.0)
    leap = advance_chain(sid, p, "jump_leap", "jog", dir_x=0.0, dir_z=-1.0)
    land = advance_chain(sid, leap, "jump_land", "jog", dir_x=0.0, dir_z=-1.0)
    clips["action:jump_leap"] = canonicalize_clip(leap, desired_heading=(0.0, -1.0))
    clips["action:jump_land"] = canonicalize_clip(land, desired_heading=(0.0, -1.0))
    print(f"  jump_leap/land   -> {leap['meta']['num_frames']}f + {land['meta']['num_frames']}f")

    # 4. 360-deg Ground Forward Combat Roll-Over (our Forge R04 WIN 26-frame diagonal parkour shoulder roll!)
    reset_sess(sid)
    p = None
    for _ in range(3):
        p = advance_chain(sid, p, "steer", "jog", dir_x=0.0, dir_z=-1.0)
    roll = advance_chain(sid, p, "roll", "jog", dir_x=0.0, dir_z=-1.0)
    clips["action:roll"] = canonicalize_clip(roll, desired_heading=(0.0, -1.0))
    print(f"  action:roll      -> {roll['meta']['num_frames']}f, disp={clips['action:roll']['disp_xz']}")

    # 5. Combat Boxing: combat_recover, punch_left (Jab), punch_right (Cross)
    reset_sess(sid)
    p = advance_chain(sid, None, "steer", "combat", dir_x=0.0, dir_z=-1.0)
    rec = advance_chain(sid, p, "combat_recover", "combat")
    rec = advance_chain(sid, rec, "combat_recover", "combat")
    jab = advance_chain(sid, rec, "punch_left", "combat", dir_x=0.0, dir_z=-1.0)
    rec2 = advance_chain(sid, jab, "combat_recover", "combat")
    cross = advance_chain(sid, rec2, "punch_right", "combat", dir_x=0.0, dir_z=-1.0)
    clips["action:combat_recover"] = canonicalize_clip(rec, desired_heading=(0.0, -1.0))
    clips["action:punch_left"] = canonicalize_clip(jab, desired_heading=(0.0, -1.0))
    clips["action:punch_right"] = canonicalize_clip(cross, desired_heading=(0.0, -1.0))
    print(f"  boxing           -> rec={rec['meta']['num_frames']}f, jab={jab['meta']['num_frames']}f, cross={cross['meta']['num_frames']}f")

    # 6. Parkour Vault Action + Vault Exit over Center Concrete Vault Wall (0, -9.5)
    reset_sess(sid)
    p = None
    for _ in range(14):
        p = advance_chain(sid, p, "vault_entry", "jog", obstacle_id="stress_vault", vault_dir=-1, vault_z=-9.5, vault_lane=0.0)
        if p["meta"].get("phase_done"):
            break
    v_act = advance_chain(sid, p, "vault_action", "jog", obstacle_id="stress_vault", vault_dir=-1, vault_z=-9.5, vault_lane=0.0)
    v_ext = advance_chain(sid, v_act, "vault_exit", "jog", obstacle_id="stress_vault", vault_dir=-1, vault_z=-9.5, vault_lane=0.0)
    # Canonicalize vault_action relative to obstacle center (0, -9.5) so we know exact start offset before the wall!
    clips["action:vault_action"] = canonicalize_clip(v_act, desired_heading=(0.0, -1.0))
    clips["action:vault_exit"] = canonicalize_clip(v_ext, desired_heading=(0.0, -1.0))
    vault_start_dist = float(v_act["frames_pos"][0, 0, 2] - (-9.5))
    clips["action:vault_action"]["extra"] = {"entry_dist": round(vault_start_dist, 3)}
    print(f"  vault_act/exit   -> {v_act['meta']['num_frames']}f (entry_dist={vault_start_dist:.2f}m) + {v_ext['meta']['num_frames']}f")

    # 7. Overhead Duck Action + Duck Exit under Overhead Duck Gantry (0, -5.6)
    reset_sess(sid)
    p = None
    for _ in range(10):
        p = advance_chain(sid, p, "duck_entry", "jog", obstacle_id="duck_beam", duck_dir=-1, duck_x=0.0)
        if p["meta"].get("phase_done"):
            break
    d_act = advance_chain(sid, p, "duck_action", "jog", obstacle_id="duck_beam", duck_dir=-1, duck_x=0.0)
    d_ext = advance_chain(sid, d_act, "duck_exit", "jog", obstacle_id="duck_beam", duck_dir=-1, duck_x=0.0)
    clips["action:duck_action"] = canonicalize_clip(d_act, desired_heading=(0.0, -1.0))
    clips["action:duck_exit"] = canonicalize_clip(d_ext, desired_heading=(0.0, -1.0))
    duck_start_dist = float(d_act["frames_pos"][0, 0, 2] - (-5.6))
    clips["action:duck_action"]["extra"] = {"entry_dist": round(duck_start_dist, 3)}
    print(f"  duck_act/exit    -> {d_act['meta']['num_frames']}f (entry_dist={duck_start_dist:.2f}m) + {d_ext['meta']['num_frames']}f")

    # 8. Sit Down + Sit Hold + Stand Up on Gallery Bench (1.85, -2.9, facing [-0.8192, 0.5736])
    reset_sess(sid)
    p = None
    for _ in range(10):
        p = advance_chain(sid, p, "sit_approach", "walk")
        if p["meta"].get("phase_done"):
            break
    s_down = advance_chain(sid, p, "sit_down", "walk")
    s_hold = advance_chain(sid, s_down, "sit_hold", "walk")
    s_up = advance_chain(sid, s_hold, "stand_up", "walk")
    bench_facing = (-0.8192, 0.5736)
    # Keep bench-center origin (1.85, -2.9) so playback places hips pixel-exact on sit_bench!
    clips["action:sit_down"] = canonicalize_clip(s_down, desired_heading=bench_facing, keep_world_origin=(1.85, -2.9))
    clips["action:sit_hold"] = canonicalize_clip(s_hold, desired_heading=bench_facing, keep_world_origin=(1.85, -2.9))
    clips["action:stand_up"] = canonicalize_clip(s_up, desired_heading=bench_facing, keep_world_origin=(1.85, -2.9))
    print(f"  sit down/hold/up -> {s_down['meta']['num_frames']}f / {s_hold['meta']['num_frames']}f / {s_up['meta']['num_frames']}f")

    # 9. Emotes (Look Around + Warmup)
    reset_sess(sid)
    p = advance_chain(sid, None, "idle", "walk")
    e_l1 = advance_chain(sid, p, "emote_look", "walk")
    e_l2 = advance_chain(sid, e_l1, "emote_look_2", "walk")
    clips["action:emote_look"] = canonicalize_clip(e_l1, desired_heading=(0.0, -1.0))
    clips["action:emote_look_2"] = canonicalize_clip(e_l2, desired_heading=(0.0, -1.0))
    reset_sess(sid)
    p = advance_chain(sid, None, "idle", "walk")
    e_w1 = advance_chain(sid, p, "emote_warmup", "walk")
    e_w2 = advance_chain(sid, e_w1, "emote_warmup_2", "walk")
    clips["action:emote_warmup"] = canonicalize_clip(e_w1, desired_heading=(0.0, -1.0))
    clips["action:emote_warmup_2"] = canonicalize_clip(e_w2, desired_heading=(0.0, -1.0))
    print(f"  emotes           -> look={e_l1['meta']['num_frames']}+{e_l2['meta']['num_frames']}f, warmup={e_w1['meta']['num_frames']}+{e_w2['meta']['num_frames']}f")

    # Pack binary payload + JSON manifest
    payload_parts: list[bytes] = []
    clip_index: dict[str, Any] = {}
    byte_cursor = 0
    total_frames = 0

    for key, c in clips.items():
        raw_bytes = quantize_clip(c)
        m = c["meta"]
        nf = int(m["num_frames"])
        total_frames += nf
        clip_index[key] = {
            "byteOffset": byte_cursor,
            "byteLength": len(raw_bytes),
            "num_frames": nf,
            "fps": int(m.get("fps", 30)),
            "transition_early": int(m.get("transition_early", 4)),
            "phase_done": bool(m.get("phase_done", True)),
            "mode": str(m.get("mode", "idle")),
            "style": str(m.get("style", "walk")),
            "disp_xz": c["disp_xz"],
            "heading_end": c["heading_end"],
            "start_root_xz": c["start_root_xz"],
            **({"extra": c["extra"]} if "extra" in c else {}),
        }
        payload_parts.append(raw_bytes)
        byte_cursor += len(raw_bytes)

    # Clean initData copy for zero-server browser startup
    init_clean = {
        "session_id": "baked-browser-session",
        "model": {
            **init_data["model"],
            "provider": "Pre-Baked SOMA FP16 Neural Bank (Browser Static)",
        },
        "bone_names": init_data["bone_names"],
        "styles": init_data["styles"],
        "obstacles": init_data["obstacles"],
        "initial_pose": init_data["initial_pose"],
    }

    header_obj = {
        "version": 1,
        "pos_scale": POS_SCALE,
        "quat_scale": QUAT_SCALE,
        "num_render_joints": int(init_data["model"]["render_joints"]),
        "num_bone_joints": int(init_data["model"]["bone_joints"]),
        "init_data": init_clean,
        "clips": clip_index,
    }
    header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    pad = (-len(header_bytes)) % 4
    if pad:
        header_bytes += b" " * pad

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        f.write(b"MBNK")
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        for part in payload_parts:
            f.write(part)

    sz = os.path.getsize(OUT_PATH)
    print(f"\n[ExportBank] DONE! {len(clips)} clips ({total_frames} frames) -> {OUT_PATH}")
    print(f"[ExportBank] Raw binary size: {sz:,} bytes ({sz / 1024:.1f} KB / {sz / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
