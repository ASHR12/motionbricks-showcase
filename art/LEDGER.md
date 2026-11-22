# Asset & Dependency Provenance Ledger

| Asset / Package | Source | License | Usage & Notes |
| :--- | :--- | :--- | :--- |
| `models/soma/mmm_v6_soma_fp16.onnx` | NVIDIA `NVlabs/motionbricks` | NVIDIA Open Model Agreement | Commercial use & redistribution allowed; excluded from git via `.gitignore` (363 MB). |
| `models/soma/soma_n1.json` | NVIDIA `NVlabs/motionbricks` | NVIDIA Open Model Agreement | SOMA model config metadata (excluded from MIT; see `THIRD_PARTY_NOTICES.md`). |
| `assets/soma_base_skel_minimal_no_root.usd` → `web/public/models/soma_human.glb` | NVIDIA `NVlabs/motionbricks` + original Blender athletic outfit meshes (`SOMA_TechTee`, `SOMA_Shorts`, `SOMA_Sneakers`) | CC-BY-4.0 / Apache-2.0 (base SOMA rig/body) + original project outfit geometry | Official 77-bone SOMA rig with newly authored fitted athletic wear skinned cleanly in Blender (excluded from MIT). |
| `assets/bvh/*.bvh` (20 clips) | NVIDIA `NVlabs/motionbricks` `assets/motions/soma/` | NVIDIA Sample Data Evaluation License | Local evaluation & prototyping only (excluded from git via `.gitignore`; not redistributed). |
| `web/public/models/motion-bank.bin` | Local export via `tools/export_motion_bank.py` | Derived from local evaluation seeds | Excluded from git via `.gitignore` (not redistributed; generate locally for evaluation). |
| `ps5-controller-webhid` (`0.1.1`) | npm (`ASHR12/ps5-controller-webhid`, user's own package) | MIT (wraps `dualsense-ts` LGPL-3.0) | Browser WebHID PS5 DualSense input, adaptive triggers, and dual-motor haptics. |
| `three` (`0.174.0`) | npm (`mrdoob/three.js`) | MIT | WebGL2 renderer, `GLTFLoader`, `RoomEnvironment`. |
| `targets/t01.png` | Generated via `image-generator` | Project internal design target | Visual North Star for Forge Protocol Macro Lock & Blind Review. |
