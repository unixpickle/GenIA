<!-- Copyright (c) Meta Platforms, Inc. and affiliates. -->

# Third-party components

This repository includes third-party code as git submodules and as local source
under `vendor/` for the macOS port.
Each keeps its own licence; nothing here is relicensed by this project's `LICENSE`
(CC BY-NC 4.0, noncommercial). **Check this table before any commercial use.**

## Included in this repository

The macOS distribution packages the local model implementations and operator
replacements described in [vendor/README.md](vendor/README.md). That inventory
records each source revision and its retained licence under `vendor/licenses/`;
the licence files also ship in the wheel. It includes SAM 3D Objects, MapAnything,
MoGe/DINOv2, ActionMesh/TripoSG, RMBG, UniCeption, LPIPS, PyTorch3D, utils3d,
and the rendering/geometry compatibility packages. The git submodules below
remain part of the original CUDA installation workflow.

| Path | Upstream | Licence |
|---|---|---|
| `submodules/sam-3d-objects` (git submodule at `f91db41`; `submodules/sam-3d-objects.patch` holds our modifications) | [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) | **SAM License** (Meta), including its Acceptable Use Policy |
| `core/inference.py` | Derived from SAM 3D Objects' inference wrapper | **SAM License** (Meta) |
| `submodules/sam-3d-objects/sam3d_objects/model/backbone/tdfy_dit/renderers/gaussian_render.py`, `.../representations/gaussian/general_utils.py` | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), via SAM 3D Objects | **Gaussian-Splatting License** (Inria): non-commercial research and evaluation use only |
| `submodules/map-anything` (git submodule) | [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | Apache-2.0 |
| `submodules/actionmesh` (git submodule: the shapes of dynamic scenes) | [facebookresearch/ActionMesh](https://github.com/facebookresearch/ActionMesh) | **FAIR Noncommercial Research License** — noncommercial research only |
| `submodules/actionmesh/third_party/TripoSG` (nested submodule) | [VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG) | MIT |

The multi-view entropy weighting in `core/entropy.py` follows the formulation of
[MV-SAM3D](https://github.com/devinli123/MV-SAM3D) (a SAM 3D Objects derivative,
SAM License).

## Installed dependencies

The original `install.sh` installs external packages under their own licences, e.g.
[MoGe](https://github.com/microsoft/MoGe) (MIT),
[gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0),
[PyTorch3D](https://github.com/facebookresearch/pytorch3d) (BSD),
[nvdiffrast](https://github.com/NVlabs/nvdiffrast) (NVIDIA Source Code License,
non-commercial).
The macOS port instead redistributes the relevant source or local replacement
operators for these components, as recorded in the linked vendor inventory.
Other dependencies are installed from the pinned `requirements-macos.txt`.

## Model weights

Weights are downloaded at setup time and are **not** covered by this repository's
licence. The SAM 3D Objects checkpoint requires accepting its licence on Hugging Face
before download; map-anything, MoGe and the ActionMesh / TripoSG / DINOv2 / RMBG weights
fetched by ActionMesh carry their own terms.

## Demo inputs

The original scenes under `demo/data/` have a `LICENSE.md` with their source and
terms. The generated macOS test image under `image/generated_elephant/` records
its image-generation prompt and asset provenance in `generation.json`.

| Scene | Source | Licence |
|---|---|---|
| `image/lab_duo`, `image/lab_turtle`, `image/office_buddies`, `image/stutty`, `multiview/lab_duck`, `multiview/pablo` | Captured by the authors | **CC BY-NC 4.0**, as this project's code |
| `dynamic/camel` | [DAVIS dataset](https://davischallenge.org/) (the `camel` sequence, every second frame of frames 0-30, downscaled to 640x360) | **CC BY-NC 4.0** per the DAVIS website (the original repository states BSD) |
| `dynamic/dinosaur` | [ActionMesh Hugging Face Space](https://huggingface.co/spaces/facebook/ActionMesh) demo assets (original source undocumented) | **FAIR Noncommercial Research License** (Meta) |
| `multiview/stuffed_toy` | [MV-SAM3D repository](https://github.com/devinli123/MV-SAM3D) example images (downscaled to 1024x768) | **SAM License** (Meta), the repository's licence |
| `image/sloth` | [4DPM](https://makezur.github.io/4DPM/), provided to us directly by a 4DPM author | **No public licence**; property of the 4DPM authors, shared with us for this project |
