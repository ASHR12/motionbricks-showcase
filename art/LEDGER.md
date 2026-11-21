# Asset & Dependency Provenance Ledger

| Asset / Package | Source | License | Usage & Notes |
| :--- | :--- | :--- | :--- |
| `models/soma/mmm_v6_soma_fp16.onnx` | NVIDIA `NVlabs/motionbricks` | NVIDIA Open Model Agreement | Commercial use & redistribution allowed; full unquantized FP16 weights running locally via ONNX Runtime CPU. |
| `assets/soma_base_skel_minimal_no_root.usd` → `web/public/models/soma_human.glb` | NVIDIA `NVlabs/motionbricks` + original Blender athletic outfit meshes (`SOMA_TechTee`, `SOMA_Shorts`, `SOMA_Sneakers`) | NVIDIA Open Model Agreement / Apache-2.0 repo + original project geometry | Official 77-bone SOMA rig with newly authored fitted athletic wear skinned cleanly in Blender. |
| `assets/bvh/*.bvh` (20 clips) | NVIDIA `NVlabs/motionbricks` `assets/motions/soma/` | NVIDIA Sample Data Evaluation License | Local evaluation & prototyping only (excluded from git via `.gitignore`; do not redistribute raw BVHs publicly). |
| `ps5-controller-webhid` (`0.1.1`) | npm (`ASHR12/ps5-controller-webhid`, user's own package) | MIT (wraps `dualsense-ts` LGPL-3.0) | Browser WebHID PS5 DualSense input, adaptive triggers, and dual-motor haptics. |
| `three` (`0.174.0`) | npm (`mrdoob/three.js`) | MIT | WebGL2 renderer, `GLTFLoader`, `RoomEnvironment`. |
| `targets/t01.png` | Generated via `image-generator` | Project internal design target | Visual North Star for Forge Protocol Macro Lock & Blind Review. |
