---
title: "MotionBrain — Real-Time Neural Character"
slug: "motionbricks-showcase"
tagline: "Zero pre-baked animations, zero NVIDIA GPU — 60 fps real-time neural character brain built on NVIDIA MotionBricks & PS5 DualSense."
category: "3D Simulation"
model: "Gemini 4 Neon"
stack: ["NVIDIA MotionBricks (SOMA)", "ONNX Runtime FP16", "Zero NVIDIA GPU", "Three.js r174", "PS5 DualSense WebHID", "Blender 5.2", "Python", "Vite"]
hero_image: "artifacts/captures/W04/w04_guides_runway.png"
showcase_video: "videos/preview.mp4"
live_url: "https://ashutoshai.com/showcase/d/motionbricks-showcase"
date: "2026-11-21"
---

# MotionBrain — Real-Time Neural Character

> **Live Demo & 4K Showcase:** [ashutoshai.com/showcase/d/motionbricks-showcase](https://ashutoshai.com/showcase/d/motionbricks-showcase)
>
> **Upstream Model Attribution:** Built on **NVIDIA Research’s** open-source SIGGRAPH 2026 **MotionBricks (SOMA)** neural motion model ([`NVlabs/motionbricks`](https://github.com/NVlabs/motionbricks), NVIDIA Open Model Agreement).

## The Story (Plain English)

MotionBrain replaces dozens of hand-animated game loops with a single live neural motion engine — running NVIDIA’s full unquantized SIGGRAPH 2026 MotionBricks (SOMA) FP16 model (`mmm_v6_soma_fp16.onnx`, 363 MB) locally on Apple Silicon at 60 fps with **zero NVIDIA GPU, zero CUDA, and zero TensorRT**. Inside an enclosed brutalist concrete mocap pavilion, a dressed 77-bone athlete responds live to a real **PS5 DualSense** controller over WebHID (or keyboard and mouse) — synthesizing every footstep, orthodox boxing jab or cross, 360° shoulder roll, hands-and-knees crawl, overhead duck, and one-handed parkour vault in ~24 ms from your intent and the 3D obstacles ahead.

For instant zero-server browser play in static deployments, MotionBrain also includes a **`0.38 MB` Pre-Baked SOMA Neural Motion Bank** (`motion-bank.bin`) exported directly from the unquantized FP16 server with a live **`Pre-Baked | Live ONNX`** toggle in the top bar: run **Pre-Baked** immediately in any browser, or start `python3 server/api.py` locally to unlock **Live ONNX** real-time neural synthesis.

## How It Was Built (Step-by-Step)

1. **Full Unquantized FP16 Neural Motion Server (Zero NVIDIA GPU)**: A zero-dependency Python `ThreadingHTTPServer` runs NVIDIA’s unquantized 363 MB `mmm_v6_soma_fp16.onnx` model via 8-thread ONNX Runtime CPU (`~23–28 ms` per chunk on Apple Silicon), maintaining a 5-chunk 77-joint session ring buffer and streaming compact `~20.8 KB` zero-parse binary motion packets plus neural L/R foot-contact probabilities.
2. **0.38 MB Pre-Baked Neural Motion Bank & Dual-Mode Switcher**: `tools/export_motion_bank.py` captures all styles and actions directly from the local FP16 ONNX server into a compact Int16-quantized binary bank (`web/public/models/motion-bank.bin`, `395 KB`, `0.25 mm` position & `~0.003°` quaternion precision). `web/src/motionBank.ts` emits drop-in `BinaryChunk` packets identical to `POST /api/generate_bin`, letting visitors switch between **Pre-Baked** (instant static web demo) and **Live ONNX** (real-time local FP16 server).
3. **Dressed 77-Bone SOMA Rig, Brutalist Pavilion & 3D OBB Auto-Parkour**: Headless Blender 5.2 skinned fitted athletic techwear (AI Compass front / `ashutoshai.com` back jersey, compression shorts, sneakers, and obsidian-black eyes; `10` meshes) onto the official 77-bone SOMA skeleton inside an enclosed `15.4 m × 25.3 m` (`389 m²`) concrete hall with Dark ↔ Alabaster Daylight switching. A real-time forward-ray 3D OBB sensor classifies props purely by geometry (Auto-Duck `0.90–1.65 m`, Auto-Vault `>= 0.60 m`, compact `1.70 m` Auto-Hurdle `0.25–0.59 m`), smoothed to 60 fps with Catmull-Rom splines, shortest-arc SLERP, 120 ms Hermite inertialization, neural foot-lock, and 3-layer solid 4-wall collision.
4. **First-Class PS5 DualSense WebHID & Automated Verification**: Wired [`ps5-controller-webhid`](https://www.npmjs.com/package/ps5-controller-webhid) for live USB/Bluetooth input, adaptive R2/L2 trigger resistance, ONNX-driven L/R footstep haptics, a theme-adaptive SVG DualSense button-to-action provenance HUD, and Strict PS5 Exclusive Keyboard Lockout (`0` leaks), verified by 43/43 biomechanical invariants and 8/8 automated hardware browser checks.

## Highlights & Interactive Controls

- **Dual Runtime Modes (`Pre-Baked` + `Live ONNX`)**: Boots instantly anywhere in the browser at 60 fps using the `0.38 MB` `motion-bank.bin` neural motion bank (`Pre-Baked`), and seamlessly unlocks real-time `~23–28 ms` unquantized FP16 synthesis (`Live ONNX`) when running the local Apple Silicon ONNX server (`http://127.0.0.1:8765`) with zero NVIDIA GPU.
- **Full Action & Style Roster (11 Styles + Combat + Parkour)**: Real-time switching across Walk, Jog, Sprint, Crouch, Crawl (hands-and-knees with flat palms and contralateral knee alternation), Combat Guard, Zombie, Vogue, Skipping, Injured, and Warm-Up — plus orthodox Left Jab (`0.781 m` reach) & Right Cross (`0.795 m` reach), 360° inverted forward shoulder roll (`26` frames, `0.00000 m` ONNX exit continuity), Auto-Vault, Auto-Duck, compact `1.70 m` Auto-Hurdle, Sit/Stand on bench, `0.25×` Slow-Mo, and Emotes.
- **Interactive `Edit Gym` Sandbox (Zero-Hardcoding Proof)**: Toggle **Edit Gym** (`G`) in the top bar to drag any Vault Plinth, Overhead Duck Gantry, Low Hurdle Rail, or Gallery Bench anywhere across the pavilion floor, **Rotate 90°** (`R`), **Shuffle** all 7 obstacles into a fresh randomized arena, or **Reset** to default. Moved and rotated props sync live to the 60 fps 3D OBB ray sensor, the local `Live ONNX` server (`POST /api/obstacles`), and the `Pre-Baked` engine simultaneously so you can prove the character adapts to any obstacle layout with zero hardcoded paths.
- **Architectural Mocap Pavilion & Visual AI Guides**: Enclosed `15.4 m × 25.3 m` (`4.55 m` ceiling) 4-wall concrete studio with 3-layer wall-sliding collision (`0.000 m` wall penetration), warm tungsten overhead skylight baffles, Dark Brutalist ↔ Warm Alabaster Daylight theme toggle, live **AI Guides** visualizing the Emerald Context Ghost, Cyan Target Ghost, trajectory path, Gold ONNX foot-contact ring, and Auto-Parkour OBB zones, plus pure translucent **Line-Art Neural X-Ray** (`0` robot body).
- **PS5 DualSense WebHID & Strict Exclusive Lockout**: Powered by `ps5-controller-webhid` over USB/Bluetooth (with Gamepad fallback). Connecting a DualSense immediately engages **PS5 Exclusive · Keys Locked** (`0` keyboard leaks until disconnected), illuminates every pressed button live on the interactive DualSense SVG and action legend, triggers L/R footstep haptics directly from ONNX foot-contact outputs, and applies adaptive R2/L2 trigger tension.
- **User Controls**:
  - **PS5 DualSense (Exclusive when connected)**: `L-Stick` 360° Move, `R-Stick` Orbit Camera, `R1`/`L3` Hold Sprint, `✕` Jump/Vault, `○` Crouch, `□` Left Jab Combo, `R2` Heavy Right Cross, `R3` 360° Combat Roll, `△` Sit/Stand, `L2` Hold `0.25×` Slow-Mo, `L1` Wide/Center Cam, `D-Pad ←/→` Cycle Styles, `D-Pad ↓` Overhead Duck, `D-Pad ↑` Emotes, `Touchpad` Neural X-Ray, `Options` AI Guides, `Create` Reset.
  - **Keyboard & Mouse (when PS5 disconnected)**: `WASD` Move, `Shift` Sprint, `Space` Jump/Vault, `C` Crouch, `J` Punch, `V` Vault, `B` Duck, `F` Roll, `E` Sit, `Tab` Slow-Mo, `G` Edit Gym (`R` Rotate 90° / `Esc` Done while editing), Click-to-Move / Click Obstacle, drag/wheel orbit camera, `?` System Overview Modal.

## Element-by-Element Build Breakdown

> Tracks every major system in the studio and the exact pipeline used (`NVIDIA MotionBricks ONNX`, `Pre-Baked Neural Motion Bank`, `Blender 5.2 Rig & Outfit`, `Python Stdlib Neural Server`, `Three.js r174 WebGL2`, `PS5 DualSense WebHID`, or `4K Showcase Capture`).

| Element | Pipeline / Tooling | How It Was Built | Asset / Source File |
| --- | --- | --- | --- |
| `NVIDIA MotionBricks (SOMA) Model` | `NVIDIA MotionBricks ONNX (FP16)` | Full unquantized 363 MB SIGGRAPH 2026 pose-to-pose neural motion network synthesizing 30 Hz 77-joint local rotations, root translation, and L/R foot-contact probabilities in `~23–28 ms` on Apple Silicon CPU (zero NVIDIA GPU) | `models/soma/mmm_v6_soma_fp16.onnx` · `server/soma_onnx.py` |
| `0.38 MB Pre-Baked Neural Motion Bank` | `Pre-Baked Neural Motion Bank` | Precision Int16-quantized (`0.25 mm` pos / `0.003°` quat) browser binary exported from the live FP16 server with drop-in `BinaryChunk` parity and a `Pre-Baked / Live ONNX` mode toggle | `web/public/models/motion-bank.bin` · `tools/export_motion_bank.py` · `web/src/motionBank.ts` |
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
- **Runtime Modules**: Decoupled Python FP16 ONNX inference backend (`server/`) and static browser pre-baked motion bank (`web/src/motionBank.ts` + `web/public/models/motion-bank.bin`) feeding a Vite + TypeScript Three.js r174 frontend (`web/src/`) separated into architectural pavilion & props, 77-bone dressed SOMA skin & X-Ray mannequin, binary prefetch motion buffer, and PS5 WebHID input/HUD layers.
- **Automated Quality Gates**: 43/43 biomechanical & continuity invariant tests passing (`450 ms`) across 360° forward shoulder roll (`360.3°` monotonic sweep, `min Head.y = 0.180 m`, `0.00000 m` ONNX handoff error), hands-and-knees crawl (`flat palms 4.1 mm`, contralateral knee alternation), orthodox Left Jab (`0.781 m`) & Right Cross (`0.795 m`) chin-height punches, and wall-proximity abort clamping; plus 8/8 automated hardware browser checks (`8/8` sabotage proofs) verifying `0` console/page/HTTP errors, 10-mesh dressed SOMA GLB, Strict PS5 Exclusive Keyboard Lockout (`leaks = 0`), clean keyboard restore on disconnect, `10/10` DualSense button-to-action causality, zero UI overlap/truncation at `1440×900` and `1280×720`, and smooth continuous motion (`median step 0.113 m`).
- **Master Showcase Capture**: 60.0-second native `3840×2160` live studio showcase (`48 kHz` stereo AAC studio soundtrack) capturing real-time PS5 DualSense combat, auto-vaulting, overhead ducking, crawling, AI Context/Target visualization, Neural X-Ray, and Dark ↔ Daylight theme transitions.

## Quick Start & Commands

### Option A — Instant Browser Play (`Pre-Baked` 0.38 MB Bank, Zero Python Setup)

```bash
cd web
npm install
npm run dev
# Opens at http://localhost:5174 using web/public/models/motion-bank.bin
```

### Option B — Full Real-Time `Live ONNX` Mode (Local Apple Silicon / CPU, Zero NVIDIA GPU)

> Download NVIDIA’s unquantized `mmm_v6_soma_fp16.onnx` (`363 MB`, place in `models/soma/`) and sample BVH motion seeds (`assets/bvh/`) from [`NVlabs/motionbricks`](https://github.com/NVlabs/motionbricks).

```bash
# 1. Start the local FP16 ONNX neural motion server (http://127.0.0.1:8765)
python3 server/api.py

# 2. Launch the Three.js r174 WebGL2 studio (http://localhost:5174) and switch to Live ONNX
cd web
npm install
npm run dev

# 3. Optional: Re-export the 0.38 MB Pre-Baked browser motion bank from the running server
python3 tools/export_motion_bank.py

# 4. Run biomechanical invariant suite (43/43) & automated browser verification (8/8)
python3 tools/test_motion_invariants.py
node tools/verify_studio.mjs
```
