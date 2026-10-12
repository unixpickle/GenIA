# Local model sources

These packages are part of the GenIA distribution, rather than dependencies
fetched at runtime. Upstream copyright and licence notices are retained.

| Package | Source revision | Licence |
| --- | --- | --- |
| sam3d_objects | facebookresearch/sam-3d-objects f91db411c50efee93d8db7aeb323885650f6f722, with the original GenIA patch | licenses/sam-3d-objects.LICENSE |
| mapanything | facebookresearch/map-anything f7ebafb4d8349776705aaa686cf928988d1bd7f4 | licenses/map-anything.LICENSE |
| moge (including its DINOv2 implementation) | microsoft/MoGe 07444410f1e33f402353b99d6ccd26bd31e469e8 | licenses/moge.LICENSE |
| utils3d | EasternJournalist/utils3d 3fab839f0be9931dac7c8488eb0e1600c236e183 | licenses/utils3d.LICENSE |
| pytorch3d transforms, mesh/point-cloud structures, mesh rendering and IO | facebookresearch/pytorch3d v0.7.9, 33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba | licenses/pytorch3d.LICENSE |
| nvdiffrast compatibility API / local operators | NVlabs/nvdiffrast v0.4.0, 253ac4fcea7de5f396371124af597e6cc957bfae | licenses/nvdiffrast.LICENSE |
| diso DiffMC / DiffDMC compatibility API and local operators | DISO 0.1.4 PyPI source distribution | licenses/diso.LICENSE (CC BY-NC 4.0) |
| lpips | LPIPS 0.1.4 PyPI; feature networks adapted from torchvision 0.22.1 | licenses/lpips.LICENSE, licenses/torchvision.LICENSE |
| pytorch_msssim | VainF/pytorch-msssim v1.0.0 | licenses/pytorch_msssim.LICENSE |
| Evaluation SSIM (`core/ops/image_metrics.py`) | Lightning-AI/torchmetrics v1.8.2, default 2D Gaussian metric | licenses/torchmetrics.LICENSE |
| uniception | UniCeption 0.1.6 PyPI, wheel SHA256 ffa919721cca4b69e3d541cab9d94cc74872c4bb8ec0c74b333e8d0c0dfc30e6 | licenses/uniception.LICENSE; third-party notices in source headers |
| vox2seq compatibility API / local operators | microsoft/TRELLIS f17fdf12d8f17a6a09225f01756d141285dc848f | licenses/vox2seq.LICENSE |
| actionmesh | facebookresearch/ActionMesh be4c2905be039d72abe8f1862ca2e1eca1ecc40a | licenses/actionmesh.LICENSE |
| triposg | VAST-AI-Research/TripoSG fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c | licenses/triposg.LICENSE |
| gsconverter PLY readers/writers | gsconverter 99bec766bb074d49da89614d077fba58fdd843ae | licenses/gsconverter.LICENSE |
| HF-layout DINOv2 (`core/models/hf_dinov2.py`) | Transformers 4.57.6 | licenses/transformers.LICENSE |
| RMBG (`core/models/rmbg`) | briaai/RMBG-1.4 2ceba5a5efaec153162aedea169f76caf9b46cf8 | licenses/rmbg.LICENSE, licenses/rmbg.MODEL_CARD.md |

The SAM sparse backend is being ported to local Metal operators. Other CUDA
dependencies still need porting; inclusion here does not yet imply a runnable
macOS pipeline. Model weights retain their original distribution/access terms.

The local `pytorch3d` package includes camera transforms, mesh/point-cloud structures,
mesh shading, textures, and mesh IO. The extension-backed triangle rasterizer is
replaced by `core/ops/mesh.py` / `mesh.metal`; face attributes, normals, and silhouette
blending use native autograd. The original Python rasterizer is retained as a test
oracle. Other extension-backed PyTorch3D APIs are not yet provided. MoGe's utils3d
geometry helpers are vendored; its optional CUDA rasterizer is not used by MoGe.

The local Gaussian renderer in `core/ops/gaussian.py` / `gaussian.metal` follows
gsplat v1.5.3 (937e29912570c372bed6747a5c9bf85fed877bae) equations and thresholds.
The upstream pure-PyTorch file is included as an independent test oracle under
`tests/reference/`, with its Apache 2.0 licence in `licenses/gsplat.LICENSE`.
`core/ops/sh.py` retains the PlenOctree/SAM spherical harmonics implementation
and its original licence notice.

The local `nvdiffrast` namespace exposes clip-space rasterization, attribute
interpolation, silhouette antialiasing and 2D texture filtering through
`core/ops/{raster,antialias,texture}.py`. It uses the local Metal triangle selector
and native PyTorch autograd. Cube textures and depth peeling are not implemented.
Edge coverage is evaluated in floating point, with deterministic first-face ties;
it does not claim bit-for-bit agreement with NVIDIA hardware rasterization.

`core/ops/marching_cubes.py` / `marching_cubes.metal` preserve DISO's DiffMC
threshold, lookup-table order, padding, normalization and deformation rules.
The unchanged CUDA source in `tests/reference/diso` provides lookup tables for
an independent scalar NumPy test oracle. The dual variant in
`core/ops/dual_marching_cubes.py` / `dual_marching_cubes.metal` also preserves
ambiguity resolution, patch averaging, quad winding and minimum-angle
triangulation. Its oracle independently reads the original CUDA dual tables.

`utils3d/torch/legacy.py` restores SAM's mesh connectivity helpers from utils3d
36a9d5cab85b2913b2db11092d6d4eb121480a73 and adapts grouping/unique for Metal
and empty inputs. Camera and raster API aliases retain SAM's older call signatures.
SAM postprocessing uses input devices and float32 cameras on MPS.

LPIPS includes local VGG16, AlexNet and SqueezeNet feature architectures. Only
pretrained tensors are downloaded, using the configured torch cache. Calibration
weights are pinned to PerceptualSimilarity commit
082bb24f84c091ea94de2867d34c4544f68e0963 and checked against SHA256 digests.

MapAnything's UniCeption DINOv2 encoder uses the local MoGe/DINO implementation.
Loading a full MapAnything checkpoint skips redundant pretrained DINO downloads.
Encoder imports are lazy so unused training/alternative model dependencies do not
block inference. CroCo's rotary embedding uses its native PyTorch implementation.
MapAnything completes CPU-to-MPS input transfers before releasing temporary inputs.
LPIPS trunks make channels-last images contiguous before MPS convolution/pooling
to correct a PyTorch 2.7 backward-layout error. Strict SAM checkpoint loading
skips DINO pretraining downloads because the checkpoint includes those tensors.
Modern torch checkpoints are memory-mapped so selecting one component does not
allocate every tensor in a multi-model archive. Legacy pickle loading is retained.
SAM/MoGe NaN-median call sites use the local Metal reduction. SAM's structure
encoder/decoder uses the local Conv3d autocast wrapper to keep input, weight and
bias dtypes consistent on PyTorch 2.7 MPS.

`core/ops/voxel_order.py` / `voxel_order.metal` implement vox2seq's 30-bit Morton
and Hilbert encode/decode, including axis permutations. Coordinates are integers
in [0, 1024). Tests compare every permutation to the unchanged upstream PyTorch
bit-array implementation, and exercise SAM serialized attention and gradients.

ActionMesh and TripoSG retain local model architectures and generic Diffusers
utilities. HF-format DINOv2 and RMBG are also local; runtime loading downloads
weights/configs, without executing remote model source. All dynamic-model
downloads use the configured Hugging Face cache. The checkpoint loader supplies
the local DINO component even when an upstream model index names Transformers.
RMBG retains its upstream source/model terms separately from GenIA's licence.

`core/ops/fps.py` / `fps.metal` implement exact farthest-point selection and
differentiable gathering. TripoSG no longer needs torch-cluster, and ActionMesh
uses the same local operation on CPU/MPS. Chunked/preselected ActionMesh samples
now return indices into the original cloud. The adaptive mesh hierarchy uses
local same-size 3D pooling and nearest-grid upsampling on Metal.
Temporal stage tests cover two autoregressive windows and retained anchor states;
residual mesh decoding passes the XYZ slice of its vertex/normal features.
ActionMesh/TripoSG SDPA calls use `core/ops/attention.py` to bound Metal score
allocations through exact query chunks. Each chunk retains the full key axis;
gradient-enabled chunks checkpoint their score activations. CPU/CUDA and dropout
retain native dispatch. Tests compare values and gradients to unchunked attention
and force chunking inside the complete small dynamic model graphs.
SAM spatial attention also uses the same local module to batch ragged windows
with masked power-of-two padding. Independent per-window references verify
values, row order and gradients across fp32/fp16/bf16; the small complete SAM
decoder tests cover the integrated Metal path.

The gsconverter subset preserves the original standard/compressed PLY codecs,
without importing unrelated format dependencies at package initialization.

Image evaluation uses local PSNR/SSIM and the local AlexNet LPIPS implementation.
It preserves TorchMetrics' reflected SSIM borders, variance clamping and default
LPIPS input convention. Optimization retains pytorch-msssim's distinct valid
window calculation. The optional fused SSIM extension is only selected on CUDA.
