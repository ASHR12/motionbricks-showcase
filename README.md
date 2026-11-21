---
title: "SOMA Motion Lab — NVIDIA MotionBricks"
slug: "motionbricks-showcase"
tagline: "60 fps real-time NVIDIA MotionBricks SOMA neural character studio with PS5 DualSense WebHID & 3D auto-parkour."
category: "3D Simulation"
model: "Gemini 4 Neon"
stack: ["Three.js r174", "WebGL2", "NVIDIA MotionBricks SOMA", "ONNX Runtime FP16", "PS5 DualSense WebHID", "Blender 5.2", "Python", "Vite"]
hero_image: "artifacts/captures/W04/w04_guides_runway.png"
showcase_video: "videos/preview.mp4"
live_url: "https://ashutoshai.com/showcase/d/motionbricks-showcase"
date: "2026-11-21"
---

# SOMA Motion Lab — NVIDIA MotionBricks

## The Story (Plain English)

SOMA Motion Lab runs NVIDIA’s full unquantized SIGGRAPH 2026 MotionBricks SOMA neural character motion model (`mmm_v6_soma_fp16.onnx`, 363 MB) live on local Apple Silicon at 60 fps inside a brutalist concrete mocap pavilion — driven directly by a real **PS5 DualSense** controller over WebHID or keyboard and mouse. Instead of playing canned game animations, every footstep, orthodox boxing jab/cross, 360° shoulder roll, hands-and-knees crawl, overhead duck, and one-handed parkour vault is synthesized on the fly in ~24 ms from your live intent and the 3D obstacles ahead. Connect a DualSense over USB or Bluetooth and keyboard input locks out cleanly while every pressed button lights up on the live controller HUD alongside neural footstep haptics, Emerald/Cyan AI Context-to-Target ghosts, and a pure line-art skeletal X-Ray mode.

## How It Was Built (Step-by-Step)

1. **Full Unquantized FP16 Neural Motion Server**: A zero-dependency Python HTTP engine serves NVIDIA’s unquantized 363 MB `mmm_v6_soma_fp16.onnx` model via 8-thread ONNX Runtime CPU (`~23–28 ms` per chunk), maintaining a 5-chunk 77-joint session ring buffer and streaming compact `~20.8 KB` zero-parse binary motion packets plus neural L/R foot-contact probabilities.
2. **Dressed 77-Bone SOMA Rig & Brutalist Mocap Pavilion**: Headless Blender 5.2 skinned fitted athletic techwear (AI Compass front / `ashutoshai.com` back jersey, compression shorts, minimalist training sneakers, and obsidian-black eyes) cleanly onto the official 77-bone SOMA skeleton (`10` multi-material meshes), placed inside an enclosed `15.4 m × 25.3 m` (`389 m²`) concrete pavilion with recessed tungsten skylight baffles, inlaid charcoal rubber runway, and instant Dark ↔ Alabaster Daylight gallery switching.
3. **Universal 3D OBB Spatial Sensor & 60 fps Motion Smoothing**: Because MotionBricks is a pure pose-to-pose neural network with zero vision, a real-time forward-ray 3D oriented bounding box (OBB) sensor scans every registered prop ahead of the runner — automatically classifying overhead gantries (`0.90–1.65 m` → Auto-Duck), concrete walls (`>= 0.60 m` → Auto-Vault), and low obstacles (`0.25–0.59 m` → compact `1.70 m` Auto-Hurdle) — smoothed to 60 fps with lookahead prefetch splicing, 4-point Catmull-Rom splines, shortest-arc quaternion SLERP, 120 ms Hermite inertialization, neural foot-lock, and 3-layer solid 4-wall collision.
4. **First-Class PS5 DualSense WebHID Integration & Verification**: Wired [`ps5-controller-webhid`](https://www.npmjs.com/package/ps5-controller-webhid) for live USB/Bluetooth input, adaptive R2/L2 trigger resistance, ONNX-driven L/R footstep haptics, a theme-adaptive SVG DualSense button-to-action provenance HUD, and Strict PS5 Exclusive Keyboard Lockout (`0` leaks), verified by 43/43 biomechanical invariants and 8/8 automated hardware browser checks.

## Highlights & Interactive Controls

- **Local Unquantized Neural Motion Engine**: Runs NVIDIA’s full unquantized `mmm_v6_soma_fp16.onnx` (`363 MB`, 8-thread CPU sequential inference in `22–28 ms`) over a zero-parse binary protocol (`~20.8 KB/chunk`) with a 5-chunk session ring buffer feeding exact 4-frame (`133 ms`) 77-joint Context windows and 1–4 planned future Target keyframes.
- **Full Action & Style Roster (11 Styles + Combat + Parkour)**: Real-time switching across Walk, Jog, Sprint, Crouch, Crawl (hands-and-knees with flat palms and contralateral knee alternation), Combat Guard, Zombie, Vogue, Skipping, Injured, and Warm-Up — plus orthodox Left Jab (`0.781 m` reach) & Right Cross (`0.795 m` reach), 360° inverted forward shoulder roll (`26` frames, `0.00000 m` ONNX exit continuity), Auto-Vault, Auto-Duck, compact `1.70 m` Auto-Hurdle, Sit/Stand on bench, `0.25×` Slow-Mo, and Emotes.
- **Architectural Mocap Pavilion & Visual AI Guides**: Enclosed `15.4 m × 25.3 m` (`4.55 m` ceiling) 4-wall concrete studio with 3-layer wall-sliding collision (`0.000 m` wall penetration), warm tungsten overhead skylight baffles, Dark Brutalist ↔ Warm Alabaster Daylight theme toggle, live **AI Guides** visualizing the Emerald Context Ghost, Cyan Target Ghost, trajectory path, Gold ONNX foot-contact ring, and Auto-Parkour OBB zones, plus pure translucent **Line-Art Neural X-Ray** (`0` robot body).
- **PS5 DualSense WebHID & Strict Exclusive Lockout**: Powered by `ps5-controller-webhid` over USB/Bluetooth (with Gamepad fallback). Connecting a DualSense immediately engages **PS5 Exclusive · Keys Locked** (`0` keyboard leaks until disconnected), illuminates every pressed button live on the interactive DualSense SVG and action legend, triggers L/R footstep haptics directly from ONNX foot-contact outputs, and applies adaptive R2/L2 trigger tension.
- **User Controls**:
  - **PS5 DualSense (Exclusive when connected)**: `L-Stick` 360° Move, `R-Stick` Orbit Camera, `R1`/`L3` Hold Sprint, `✕` Jump/Vault, `○` Crouch, `□` Left Jab Combo, `R2` Heavy Right Cross, `R3` 360° Combat Roll, `△` Sit/Stand, `L2` Hold `0.25×` Slow-Mo, `L1` Wide/Center Cam, `D-Pad ←/→` Cycle Styles, `D-Pad ↓` Overhead Duck, `D-Pad ↑` Emotes, `Touchpad` Neural X-Ray, `Options` AI Guides, `Create` Reset.
  - **Keyboard & Mouse (when PS5 disconnected)**: `WASD` Move, `Shift` Sprint, `Space` Jump/Vault, `C` Crouch, `J` Punch, `V` Vault, `B` Duck, `F` Roll, `E` Sit, `Tab` Slow-Mo, Click-to-Move / Click Obstacle, drag/wheel orbit camera, `?` System Overview Modal.

## Element-by-Element Build Breakdown

> Tracks every major system in the studio and the exact pipeline used (`NVIDIA MotionBricks ONNX`, `Blender 5.2 Rig & Outfit`, `Python Stdlib Neural Server`, `Three.js r174 WebGL2`, `PS5 DualSense WebHID`, or `4K Showcase Capture`).

| Element | Pipeline / Tooling | How It Was Built | Asset / Source File |
| --- | --- | --- | --- |
| `NVIDIA MotionBricks SOMA Model` | `NVIDIA MotionBricks ONNX (FP16)` | Full unquantized 363 MB SIGGRAPH 2026 pose-to-pose neural motion network synthesizing 30 Hz 77-joint local rotations, root translation, and L/R foot-contact probabilities in `~23–28 ms` on Apple Silicon CPU | `models/soma/mmm_v6_soma_fp16.onnx` · `server/soma_onnx.py` |
| `Dressed 77-Bone SOMA Athlete` | `Blender 5.2 Rig & Outfit` | Official NVIDIA 77-bone SOMA skeleton skinned in Blender with custom fitted multi-material athletic techwear (AI Compass front / `ashutoshai.com` back jersey, compression shorts, sneakers, obsidian eyes; `10` meshes) | `web/public/models/soma_human.glb` · `web/src/somaSkin.ts` |
| `Local Neural Planner & Binary API` | `Python Stdlib Neural Server` | Zero-dependency `ThreadingHTTPServer` managing per-session 5-chunk ring buffers, dynamic obstacle target planning, 4-wall target clipping, post-ONNX hard clamping, and compact `~20.8 KB` zero-parse binary responses | `server/api.py` · `server/bvh_loader.py` |
| `Universal 3D OBB Parkour Sensor` | `Three.js r174 WebGL2` | Real-time 60 fps forward-ray 2D/3D oriented bounding box scanner classifying overhead gantries (`0.90–1.65 m`), concrete vault plinths (`>= 0.60 m`), and low hurdles (`0.25–0.59 m`) with per-obstacle cooldown chaining | `web/src/main.ts` · `web/src/studioProps.ts` |
| `60 fps Neural Motion Smoother` | `Three.js r174 WebGL2` | Lookahead prefetch splicing coupled with 4-point Catmull-Rom position splines, shortest-arc quaternion SLERP, 120 ms Hermite inertialization, neural foot-lock stabilization, and floor/wall clamping | `web/src/motionBuffer.ts` |
| `Brutalist Concrete Mocap Pavilion` | `Three.js r174 WebGL2` | Enclosed `15.4 m × 25.3 m` (`389 m²`) 4-wall architectural hall with 10 colonnade pillars, recessed warm tungsten skylight baffles, inlaid charcoal rubber runway, cast-in-concrete upper-wall branding, and Dark/Light themes | `web/src/pavilion.ts` · `web/src/studioSpec.ts` · `web/src/grid.ts` |
| `AI Context/Target Guides & Line-Art X-Ray` | `Three.js r174 WebGL2` | Real-time Emerald 4-frame Context ghost, Cyan Target ghost, trajectory path, Gold ONNX foot-contact rings, Auto-Parkour OBB outlines, and pure translucent 77-bone skeletal line-art X-Ray mode | `web/src/mannequin.ts` · `web/src/somaSkin.ts` |
| `PS5 DualSense WebHID & Editorial HUD` | `PS5 DualSense WebHID` | Built on `ps5-controller-webhid` (`dualsense-ts`) with Strict PS5 Exclusive Keyboard Lockout, live DualSense SVG illumination, adaptive R2/L2 trigger resistance, ONNX footstep haptics, and `?` Architecture Modal | `web/src/ps5Input.ts` · `web/src/ps5Hud.ts` · `web/src/styles.css` |
| `60-Second 4K Showcase Film` | `4K Capture + Studio Soundtrack` | Native `3840×2160` (`60.0 s`) live studio capture demonstrating PS5 DualSense combat, parkour vaulting, overhead ducking, crawling, AI Context/Target guides, Line-Art X-Ray, and Dark/Light themes paired with stereo `48 kHz` studio music | `videos/preview.mp4` · Release `v1.0-master` |

## Architecture & Verification

- **Coordinate & Metric Standard**: `1 unit = 1 meter`, right-handed `+Y` up inside a single shared blueprint (`15.4 m × 25.3 m` pavilion, `4.55 m` ceiling, solid inner playable bounds `X ∈ [-7.15, +6.70] m`, `Z ∈ [-16.85, +7.12] m`, 77-bone official SOMA hierarchy).
- **Runtime Modules**: Decoupled Python FP16 ONNX inference backend (`server/`) feeding a Vite + TypeScript Three.js r174 frontend (`web/src/`) separated into architectural pavilion & props, 77-bone dressed SOMA skin & X-Ray mannequin, binary prefetch motion buffer, and PS5 WebHID input/HUD layers.
- **Automated Quality Gates**: 43/43 biomechanical & continuity invariant tests passing (`450 ms`) across 360° forward shoulder roll (`360.3°` monotonic sweep, `min Head.y = 0.180 m`, `0.00000 m` ONNX handoff error), hands-and-knees crawl (`flat palms 4.1 mm`, contralateral knee alternation), orthodox Left Jab (`0.781 m`) & Right Cross (`0.795 m`) chin-height punches, and wall-proximity abort clamping; plus 8/8 automated hardware browser checks (`8/8` sabotage proofs) verifying `0` console/page/HTTP errors, 10-mesh dressed SOMA GLB, Strict PS5 Exclusive Keyboard Lockout (`leaks = 0`), clean keyboard restore on disconnect, `10/10` DualSense button-to-action causality, zero UI overlap/truncation at `1440×900` and `1280×720`, and smooth continuous motion (`median step 0.113 m`).
- **Master Showcase Capture**: 60.0-second native `3840×2160` live studio showcase (`48 kHz` stereo AAC studio soundtrack) capturing real-time PS5 DualSense combat, auto-vaulting, overhead ducking, crawling, AI Context/Target visualization, Neural X-Ray, and Dark ↔ Daylight theme transitions.

## Quick Start & Commands

> **Local Runtime Requirement:** Requires NVIDIA’s unquantized `mmm_v6_soma_fp16.onnx` (`363 MB`, place in `models/soma/`) and sample BVH motion seeds (`assets/bvh/`) running locally via Python + `onnxruntime`.

```bash
# 1. Start the local FP16 ONNX neural motion server (http://127.0.0.1:8765)
python3 server/api.py

# 2. Install & launch the Three.js r174 WebGL2 studio (http://localhost:5174)
cd web
npm install
npm run dev

# 3. Run biomechanical invariant suite (43/43) & automated browser verification (8/8)
python3 tools/test_motion_invariants.py
node tools/verify_studio.mjs
```
