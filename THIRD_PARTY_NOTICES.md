# Third-Party Notices & Asset Licenses

The original source code authored in this repository (`web/src/*`, `web/index.html`, `server/*`, `tools/*`, and documentation authored for this project) is licensed under the [MIT License](LICENSE).

Third-party models, assets, reference code, and dependencies retain their own respective licenses and are **not** relicensed under MIT:

## 1. NVIDIA MotionBricks (`NVlabs/motionbricks`)
- **Upstream Repository:** https://github.com/NVlabs/motionbricks
- **Copyright:** Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- **Upstream Code License:** Apache License 2.0 (referenced by `server/soma_onnx.py` for ONNX tensor interfaces and SOMA pose-space parity).
- **Upstream Documentation & Assets License:** Creative Commons Attribution 4.0 International (CC-BY-4.0).

## 2. Included NVIDIA-Derived Assets (Excluded from MIT)
- **`models/soma/soma_n1.json`**
  - SOMA skeleton/model configuration metadata from `NVlabs/motionbricks/motionbricks/assets/onnx/soma/`.
  - Licensed by NVIDIA Corporation under the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
- **`web/public/models/soma_human.glb`**
  - Derivative 3D character combining NVIDIA's 77-bone SOMA base skeleton and body geometry (`soma_base_skel_minimal_no_root.usd` from `NVlabs/motionbricks`, licensed under CC-BY-4.0 / Apache-2.0) with newly authored athletic techwear meshes (`SOMA_TechTee`, `SOMA_Shorts`, `SOMA_Sneakers`) and materials created in Blender 5.2 by Ashutosh Shrivastava.
  - Modifications made: converted USD skeleton/body to glTF/GLB, added fitted athletic outfit geometry and PBR materials, calibrated rest pose and skin weights for Three.js WebGL2.

## 3. Excluded External Weights & Sample Data (Not Redistributed)
To respect upstream model and evaluation-data terms, the following files are gitignored and **not** distributed in this repository:
- **`models/soma/mmm_v6_soma_fp16.onnx`** (NVIDIA Open Model Agreement): download directly from [`NVlabs/motionbricks`](https://github.com/NVlabs/motionbricks).
- **`assets/bvh/*.bvh`** (NVIDIA Sample Data Evaluation License; internal evaluation only): download directly from [`NVlabs/motionbricks`](https://github.com/NVlabs/motionbricks) after reviewing NVIDIA's evaluation terms.
- **`web/public/models/motion-bank.bin`**: excluded from public redistribution because bank exports seeded from NVIDIA's sample BVHs fall under the Sample Data Evaluation License. You can generate a local evaluation bank via `python3 tools/export_motion_bank.py` (or seed from permissively licensed motion clips).

## 4. Runtime Package Dependencies
- **`three`**: MIT License (https://github.com/mrdoob/three.js)
- **`ps5-controller-webhid`**: MIT License (https://github.com/ASHR12/ps5-controller-webhid), which wraps **`dualsense-ts`** licensed under **LGPL-3.0** (https://github.com/nsfm/dualsense-ts)
- **`onnxruntime`**: MIT License (https://github.com/microsoft/onnxruntime)
- **`numpy`**: BSD-3-Clause License (https://github.com/numpy/numpy)
