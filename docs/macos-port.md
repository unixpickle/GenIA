# macOS port

Single-image, multiview, and dynamic reconstruction have completed on Apple silicon,
including pretrained models, optimization, and final geometry/video export.
The full multiview pipeline also completed with valid exports. A paired
100-step experiment traced a pillow's geometry regression to mixed-precision
fine-tuning and verified that full FP32 preserves its surfaces. Metal fine-tuning
now defaults to FP32. The final-build generated-image run also passed with that
default. The port uses local model implementations and tested replacements for
CUDA operators.

## Reproduce the current development environment

On Apple silicon, with uv installed:

```bash
bash install_macos.sh
.venv/bin/python -m pytest tests -q
```

No recursive submodule checkout is needed for these tests. `setup.py` packages the
local SAM 3D, MapAnything, MoGe/DINOv2, ActionMesh/TripoSG, RMBG, and geometry
source trees. Metal source files ship in the wheel
and are lazily compiled through PyTorch's `torch.mps.compile_shader`. The macOS
runtime pins PyTorch 2.14.1 and torchvision 0.29.1, using Python 3.12 on Apple
silicon with macOS 14 or newer. The full pipeline and Metal validation recorded
below originally used PyTorch 2.7.1 on macOS 26.
CPU tests also run without an Apple GPU; Metal cases run only with MPS available.
The setup script installs the pinned runtime and test dependencies.

The PyTorch upgrade addresses a reproduced 2.7.1 crash: advanced indexing under
a deliberately low MPS allocator limit exited with `SIGTRAP`, including when its
output was preallocated. Both indexing cases succeeded on 2.14.1; the ordinary
allocation control still raised a catchable OOM exception. This establishes that
the reproducer no longer crashes, not that every allocation failure is fixed.
Upgrade the matching pair together, then run the tests from a terminal with GPU
access:

```bash
uv pip install --python .venv/bin/python -r requirements-macos.txt
.venv/bin/python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__); assert torch.backends.mps.is_available()'
.venv/bin/python -m pytest tests -q
```

Restart the WebUI after upgrading so its workers use the new runtime.

Upgrade validation on 2026-10-12: **709 tests passed, none skipped**, in 53.91 s
with PyTorch 2.14.1 and torchvision 0.29.1 installed together. This includes the
Metal kernels and CPU/MPS value/gradient comparisons, decoder checkpointing,
fine-tuning precision, model imports, and WebUI tests. `uv pip check` passed for
all 149 installed packages; the macOS lock file was regenerated with uv. The
isolated indexing/OOM reproducer also passed again in the installed environment.
Test results and the reproducer log are under `results/macos-port/pytorch-2.14.1/`.
The bundled `generated_elephant` scene also completed on MPS with cached pretrained
models, including shape/pose sampling, rendering-guided appearance, FP32
fine-tuning, final evaluation, and Gaussian/GLB/preview export. Appearance sampling
and guidance used two steps each; pose refinement and fine-tuning used two
iterations each. This is an end-to-end compatibility smoke test, not a generation
quality benchmark. Its outputs are in that directory's `generation/` subdirectory.

### Model cache and demo commands

Accept the [SAM 3D Objects licence](https://huggingface.co/facebook/sam-3d-objects)
on Hugging Face, then log in and select a model-cache location. Set these variables
in the shell used for both downloading and running GenIA; replace the example
external-drive path if needed:

```bash
export HF_HUB_CACHE=/Volumes/MLData4/GenIA/huggingface/hub
export HF_XET_CACHE=/Volumes/MLData4/GenIA/huggingface/xet
export TORCH_HOME=/Volumes/MLData4/GenIA/torch
.venv/bin/hf auth login
.venv/bin/python -m genia.core.download_weights
```

The downloader reuses cached files. Add `--no-actionmesh` if only static scenes
are needed. `--cache-root` sets these paths for the downloader process only;
keep the environment variables set for subsequent model loading too.

The demo commands under validation are:

```bash
.venv/bin/python demo/run_demo.py image --scene generated_elephant processing=moge
.venv/bin/python demo/run_demo.py multiview
.venv/bin/python demo/run_demo.py dynamic
```

They select MPS automatically and retain the original optimization settings.
Results go under `results/`. `--dry-run` prints the resolved command without
loading models. No recursive submodule checkout is needed on macOS.

Verification on 2026-10-11: **607 tests passed**, including the actual MPS cases,
with the wheel installed into a separate uv environment and pytest run from `/tmp`. Runtime import paths
were verified to resolve to that environment's site-packages. The wheel contains the Metal
sources, local model/transform packages, configs, and upstream licences.
This includes area pooling, visibility-biased attention, dynamic attention
chunking, batched spatial-window attention, and temporary CPU offloading of
sampling networks during Metal fine-tuning. The full installed-wheel suite took
73.83 seconds. This includes FP16/BF16 positional-embedding interpolation and
gradient regressions for the local DINO encoder, and SAM dense attention
packing, gradient, and allocation-budget regressions. Nine additional tests cover
Metal fine-tuning precision, decoder-cache invalidation, CPU/MPS gradients, and
reloading full-precision adapters.
The installed wheel also encoded/decoded a four-frame H.264 clip and executed
the Numba voxel-occlusion routine with the pinned runtime dependencies.
`install_macos.sh` also passed in a fresh source copy with no virtual environment
or submodules: 130 installed packages were compatible, all 15 local package
imports resolved into that copy, and a compiled Metal KNN forward/backward check
matched an independent CPU pairwise calculation. CLI help, multiview demo dry-run
and external model-cache dry-run passed. This setup check is recorded in
`results/macos-port/fresh_install/validation.json`; it is separate from the
complete reconstruction and visual-quality checks.

The complete final-output stage also passed on a synthetic 10,242-point sphere:
evaluation, colored GLB and compressed Gaussian PLY export, train/novel-view
renders, world assets, COLMAP, orbit video and world overview. The exported
geometry, metrics, images and video were read back and checked; the six-view
canonical panel was inspected. Results are recorded in
`results/macos-port/final_synthetic/validation.json`. This caught an unnecessary
SAM notebook-visualizer import in scene assembly, which now uses the same local
geometry transform directly. CPU/Metal regression tests cover transformed
positions, rotations, scales and copy behavior. This synthetic export check
does not establish complete pretrained reconstruction success.

## Implemented and tested

Every custom Metal entry point is exercised through its public operator tests:

| Metal source / entry points | Tests and reference |
| --- | --- |
| `knn.metal`: `knn_indices` | `test_knn.py`: direct pairwise distances, padding and gradients |
| `fps.metal`: `farthest_points` | `test_fps.py`: unchanged PyTorch3D naive reference and feature gradients |
| `sparse.metal`: `build_table`, `neighbor_table` | `test_sparse.py`: dense convolution values and gradients |
| `resize.metal`: `resize_axis`, `resize_axis_backward` | `test_resize.py`: CPU PyTorch antialiased resizing |
| `pooling.metal`: `area_pool2d`, `area_pool2d_backward`, `volume_argmax` | `test_area_pooling.py`, `test_volume_pooling.py`: CPU PyTorch pooling and gradients |
| `voxel_order.metal`: `encode_voxels`, `decode_voxels` | `test_voxel_order.py`: unchanged Morton/Hilbert reference |
| `gaussian.metal`: `tile_counts`, `tile_emit`, `tile_ranges`, `splat_forward`, `splat_backward` | `test_gaussian.py`: gsplat projection reference, independent compositing and finite differences |
| `mesh.metal`: `select_faces` | `test_mesh.py`: upstream Python rasterizer and numerical derivatives |
| `marching_cubes.metal`: `classify_cells`, `emit_faces` | `test_marching_cubes.py`: independent scalar oracle using original DISO tables |
| `dual_marching_cubes.metal`: `classify_dual_cells`, `resolve_dual_cells`, `emit_dual_quads` | `test_dual_marching_cubes.py`: independent scalar oracle using original DISO dual tables |
| `indexing.metal`: `unravel_indices` | `test_large_volume_indexing.py`: exact CPU integer coordinates above 2²⁷ and both mesh extractors above 2²⁴ |

The test filenames above are under `tests/`. Native-PyTorch replacement operations
and complete local model graphs have additional dedicated tests in that directory.

- Metal KNN selection with squared L2/L1 distances, deterministic ties, padded
  lengths, empty inputs, and gradients to both clouds. The deformation and pose
  refinement KNN call sites use this implementation. Metal currently supports
  K up to 32. Selection uses float32; CPU double precision supports gradcheck.
- Metal coordinate hash construction and sparse-neighbor lookup. Sparse
  submanifold convolution uses these maps with PyTorch matrix multiplication
  and autograd, preserving spconv checkpoint weight names and layout. Tests
  compare forward and backward against dense convolution with varied kernels,
  dilation, occupancy and batches. Strided/inverse sparse convolutions remain
  explicitly unsupported, rather than silently producing different results.
- Local SAM sparse storage, convolution, subdivision and SDPA attention.
  Packed attention processes individual sequences without a global block mask.
  Visibility-biased cross-attention accepts packed or stacked per-view additive
  biases, including ragged views and shared/per-head bias. Metal query chunks
  bound score memory; tests compare outputs and q/k/v/bias gradients with an
  independent softmax reference for fp32/fp16/bf16.
  Metal spatial self-attention batches ragged windows in masked power-of-two
  buckets, avoiding hundreds of small attention dispatches per decoder layer.
  Padding is excluded from softmax and output rows retain their original order.
  For one synthetic bf16 layer with 20,565 tokens in 512 windows (12 heads,
  64 channels), warmed forward/backward dropped from 4.64 s to 0.17 s while
  a separate reconstruction was active. Output/gradient relative L2 differences
  were 1.61e-5/1.95e-5. This is a layer benchmark, not full-pipeline timing;
  measurements are in `results/macos-port/window_attention/validation.json`.
- Decoder activation checkpointing now defaults to every object on MPS and
  30,000 voxels elsewhere when `decoder_checkpoint_min_voxels` is `null`; explicit
  integer thresholds remain supported. The original 30,000 threshold allowed
  Pablo's 28,049-voxel decoder backward to reach a 43.2 GB process footprint on
  this 32 GB Mac. A subsequent 20,000 threshold let that object complete all
  100 fine-tuning steps, with sampled physical footprints of 25.2–25.3 GB.
  However, a 19,650-voxel pillow still reached 38.6 GB without checkpointing:
  voxel count alone does not predict spatial-attention memory. That run was
  deliberately stopped; a new run resumes from its saved appearance state with
  checkpointing for every Metal object. All optimization iterations are retained.
  Sparse self-attention blocks pass their feature tensor explicitly to PyTorch
  checkpointing so backward recomputation retains MPS autocast. Regression
  tests compare decoder outputs, latent gradients and DoRA parameter gradients
  on CPU/MPS with and without bf16 autocast. A full pretrained-decoder check
  with 28,049 synthetic voxels produced identical outputs and gradients while
  retained forward allocations fell from 25.6 GB to 1.10 GB; isolated
  forward/backward took 22.4 s without checkpointing and 12.7 s with it.
  These are decoder measurements, not full-pipeline peak memory or runtime;
  see `results/macos-port/decoder_checkpoint/validation.json`.
- Fine-tuning uses a full FP32 Gaussian decoder on MPS by default, with
  checkpointing retained. `finetuning.decoder_autocast_bf16=null` selects that
  policy on Metal and retains BF16 autocast elsewhere; `true` explicitly opts
  into mixed precision. `false` also promotes the decoder to FP32 on MPS.
  Adapter caches distinguish decoder precision, and saved adapters record
  full-precision training so re-decoding with a fresh model preserves it.
  CPU/MPS tests cover forward/gradient agreement, overrides, cache invalidation,
  and adapter reloads. The pretrained paired experiment is detailed below.
- Real-valued rotary embeddings remove complex-number operations from the
  attention path. Tests compare values and gradients to complex CPU arithmetic.
- Local NaN-ignoring median uses Metal sorting and gathering for SAM point-map
  normalization and pose decoding. CPU-reference tests cover dimensional/global
  reductions, lower medians, NaNs/infinities, empty/scalar inputs, half/bfloat
  inputs and derivatives, including the global reduction's shared tie gradients.
- SAM's 3D convolutions explicitly cast activations, weights and bias during
  Metal autocast, fixing PyTorch 2.7's mixed-bias failure. Tests cover fp16/bf16
  outputs, parameter/input derivatives, and the complete small structure decoder.
- Metal adaptive area pooling implements the original overlapping-bin averages
  for nondivisible image sizes, including the default 518→129 multiscale loss.
  Forward/backward tests compare CPU PyTorch for down/up sampling, fp32/fp16/bf16,
  noncontiguous inputs, singleton axes and empty batches.
- Gaussian representation allocation and covariance follow input devices.
  Small SAM Gaussian decoder tests compare MPS with CPU for full/windowed
  attention and absolute/rotary positions, including output and latent gradients.
  The dense structure decoder also matches CPU outputs and latent gradients.
  These use random weights and do not establish pretrained-model fidelity.
- Device selection prefers CUDA, then MPS, then CPU, with `GENIA_DEVICE` as an
  override. The reconstruction entry point, shape inversion and ActionMesh use
  this selection; remaining direct CUDA calls in `core/` are guarded device
  helpers or optional memory instrumentation. Full dynamic execution and
  export validation are recorded below.
- Differentiable Gaussian rasterization now uses local Metal tile counting,
  intersection emission, tile ranges, compositing, and reverse compositing
  kernels. GenIA and SAM renderer call sites use this implementation. Tests
  cover all five RGB/depth modes, SH degrees 0–4, pixel-center conventions,
  opacity saturation, termination, tile boundaries, empty scenes, backgrounds,
  camera gradients, and a concurrency regression in the tile-range kernel.
  Projection values and gradients are compared against gsplat 1.5.3's independent
  CPU reference; compositing gradients also pass finite-difference checks.
  The supported API is classic pinhole rendering, first derivatives, unpacked
  metadata, and up to four composited channels. Other camera/antialias modes
  raise explicit errors.
- On this machine, warmed-up 50,000-Gaussian 512×512 renders took 7–8 ms forward
  and 16–25 ms backward for a synthetic scene. Compilation/warmup took longer.
  This is an operator measurement, not full-pipeline performance.
- Local PyTorch3D quaternion, rotation, and transform implementations run on
  CPU/MPS. Tests exposed a PyTorch 2.7 MPS inverse-on-transposed-view issue;
  affected transform and rendering inversions now use contiguous inputs.
- Antialiased bilinear and bicubic resizing uses separable Metal forward/adjoint
  kernels. Tests cover down/up sampling, scale-factor rounding, align_corners,
  unit axes, noncontiguous half inputs, and gradients against CPU PyTorch.
- MoGe and utils3d are vendored at the original installation's pinned revisions.
  MoGe/DINO preprocessing uses the local resize operator. A small random-weight
  MoGe model matches CPU outputs and image gradients and runs public mixed-precision
  inference on MPS. Camera recovery/unprojection tests use known synthetic geometry.
  Pretrained MoGe has also passed the generated-image validation below.
- SAM's DINO encoder now constructs the vendored implementation without fetching
  torch-hub source. A register-token encoder test exercises mask inputs and compares
  MPS output to CPU. Complete SAM checkpoints already include the image-encoder
  weights, so strict checkpoint loading now skips the redundant DINO pretraining
  download. Standalone DINO construction still uses the configured torch cache.

- Local Metal triangle selection supports depth ordering, up to 64 faces per pixel,
  perspective-correct and clipped barycentrics, blurred silhouettes, backface
  culling, and near-plane clipping through the retained PyTorch3D wrapper.
  Depth, barycentrics and signed squared edge distances use native autograd.
  Tests compare against the unchanged upstream Python rasterizer and numerical
  derivatives, and cover multiple tiles, empty/degenerate meshes, batching,
  vertex attributes, OBJ loading, camera projection, and colored flat shading.
  MPS 2.7's expanded-source masked_scatter corrupted background pixels; the local
  hard blender uses elementwise selection, with a value/gradient regression test.
- SAM point-map windows now reuse the local DINO transformer block. Its state keys
  and outputs matched timm 1.0.24 in a separate reference check; CPU/MPS tests
  include invalid points and parameter gradients.
- FlexiCubes and a small complete SAM sparse mesh decoder match CPU topology,
  vertex colors and derivatives. Geometry buffers now follow input devices.
  Dimension-wise unique uses stable MPS sorts without integer-hash overflow;
  tests cover inverse maps, counts, empty inputs and large integer coordinates.
- SAM and GenIA inference modules now import on macOS. Checkpoint-loading tests
  cover torch and safetensors on CPU/MPS. Single-image and multiview pretrained
  and dynamic runs exercised the reconstruction dependencies.
  Open3D is pinned to 0.19.0 because the 0.20 wheel
  required an unavailable external libusb library on this machine.
- Local nvdiffrast-compatible clip-space rendering supports six-plane clipping,
  instance/range batches, perspective attributes and screen-space derivatives,
  silhouette antialiasing, and 2D nearest/bilinear/mipmapped textures. Tests cover
  analytical coverage, derivative checks, CPU/MPS comparisons, and GenIA's actual
  RGB/depth/alpha render functions. Coverage uses floating-point edge tests with
  deterministic first-face ties; NVIDIA hardware's exact boundary rules are not
  reproduced. Cube textures and depth peeling are not implemented.
- Differentiable marching cubes uses local Metal classification/face-emission
  kernels and autograd interpolation. All 256 cases match a scalar reference
  reading the unchanged DISO 0.1.4 tables. Tests include deformation, threshold
  equality, padding, normalization, numerical gradients and a full
  occupancy-to-mesh-to-render backward pass. DiffDMC is also implemented: Metal
  topology kernels preserve ambiguous-cell resolution, dual patch averaging,
  quad winding, and minimum-angle triangulation. Its independent scalar oracle
  reads the original DISO dual tables. All cube cases, NaN adaptive-grid cells,
  deformations, normalization and derivatives are covered. Large-volume tests
  cross the 2²⁴ index boundary: chunked index extraction and a Metal integer
  coordinate kernel avoid MPS rounding in nonzero, division and remainder.
- SAM mesh connectivity, camera construction, hidden-surface removal and texture
  baking run locally on CPU/MPS. Older utils3d helpers are restored from upstream
  source. A synthetic two-shell mesh test verifies that cleanup removes the
  hidden interior. The full 2,500-step texture optimizer ran on MPS in 39.9 seconds
  for an 8×8 texture/16×16 observation, recovering mean RGB
  `[39.31, 99.20, 199.67]` for a `[40, 100, 200]` target.
- Core rendering, pose initialization/refinement, shape inversion, visualization,
  and cache restoration now select the available accelerator or follow input
  devices. Tests cover exported-asset loading, Gaussian/sparse cache roundtrips,
  camera gradients, bf16 SLAT decoding and shape-latent optimization on MPS.
  The local ActionMesh backend is covered by component tests and a complete
  pretrained 16-frame dynamic reconstruction with export readback.
- MPS 2.7 cannot backpropagate bf16 advanced-index gathers. Sparse attention and
  convolution use `index_select`, which supports the required gradients. Tests
  compare half/bf16 sparse-convolution gradients to CPU and exercise a complete
  bf16 decoder backward pass.
- LPIPS includes local VGG16, AlexNet and SqueezeNet feature networks and pinned,
  hash-checked calibration weights. Tests compare values and image gradients
  against original LPIPS 0.1.4 with torchvision, including channels-last inputs.
  MPS 2.7 produced incorrect gradients for that layout; local trunks make inputs
  contiguous before convolution/pooling. Pretrained VGG validation on the 224×224
  elephant image versus a blurred copy gave CPU/MPS scores 0.18244743/0.18244745.
  Flat-region max-pooling ties remain sensitive to backend rounding: the original
  image's relative gradient difference was 10.5%; with a diagnostic 0.01 Gaussian
  perturbation it fell to 0.0192%. No noise is added by the runtime. Measurements
  are in `results/macos-port/lpips/validation.json`.
- MapAnything's UniCeption components are local, including DINOv2 and native
  rotary embeddings. A small two-view model matches CPU forward values and image
  gradients, and public inference supports optional depth/intrinsics/poses and
  minibatched heads on MPS. Input transfers are blocking on Metal to prevent
  temporary CPU inputs being released before the copy finishes. Mixed-precision
  inference produces finite results. Full pretrained MapAnything has also run on
  two real 518×518 views; its stage validation is described below.
  Global attention uses the same bounded query chunks as the dynamic models.
  Complete small-model tests compare 16-view CPU/Metal outputs against native CPU
  attention, with separate two-view gradient and geometry-input checks forcing
  chunked dispatch.
- Metal Morton/Hilbert voxel encode/decode preserves vox2seq's ten-bit axes and
  all six axis permutations. Tests compare against unchanged upstream bit-array
  code and exercise all four SAM serialized-attention modes with gradients,
  including empty batch members and entirely empty sparse inputs.
- Exact Metal farthest-point sampling supports ragged batches, varying sample
  counts, deterministic ties, and differentiable feature gathering. Tests use
  the unchanged PyTorch3D naive reference. An 8,192-to-2,048 sample took 39 ms
  after warmup on this machine. ActionMesh's preselected/chunked sampling now
  returns indices into the original cloud, including uneven chunks.
- ActionMesh and TripoSG model architectures are local. Small complete temporal
  denoiser/autoencoder and TripoSG denoiser/VAE graphs match CPU outputs and input
  gradients on MPS. Offline checkpoint integration exercises image conditioning,
  classifier-free guidance, scheduler steps and VAE decoding with the local
  HF-layout DINO model. Lazy ActionMesh construction uses packaged configs and
  the configured external cache. Full pretrained dynamic reconstruction passed.
  Eight additional small-model checks cover the production precision modes:
  ActionMesh bf16 autocast and TripoSG fp16 weights, with both native and forced
  chunked attention. Outputs agree with fp32 CPU reference graphs within 2.5%
  relative L2. These are included in the installed-wheel suite; they do not
  replace the separate pretrained video validation.
- ActionMesh/TripoSG attention bounds Metal score allocations by splitting queries
  while retaining the full key-axis softmax. Gradient-enabled chunks recompute
  scores during backward instead of retaining them across the entire sequence.
  Tests cover fp32/fp16/bf16, broadcast/boolean/additive/causal masks, masked-out
  rows, empty inputs and gradients, including complete small dynamic models.
  Native dispatch is retained on CPU/CUDA and for dropout.
  During Metal inference with autocast, FP32 query/key inputs are converted
  once before the query loop. ActionMesh's FP32-normalized keys previously
  caused 2,186 full-key conversions per inflated attention layer. On the
  actual `(2, 16, 32784, 128)` attention shape with synthetic activations,
  this change reduced a forward pass from 42.69 s to 9.32 s, with identical
  outputs and the same 16-million-element score budget. Differentiable calls
  retain per-chunk casts and FP32 accumulation of key gradients. Measurements
  are in `results/macos-port/attention_autocast/validation.json`.
  A subsequent operator optimization increases the default inference score budget
  from 16 million elements to at most 256 million, limited by available MPS working
  memory (one sixteenth for a single FP32 score tensor, with the original 16-million
  minimum). Training retains the 16-million budget; explicit budgets are respected.
  The full-shape comparison below measures this change separately from cast hoisting.
  Additional Metal mask fixes return zeros for fully masked inference rows
  and use PyTorch's differentiable math implementation when only an additive
  mask requires gradients. FP16/BF16 tests cover these cases and autocast
  conversion reuse, alongside CPU softmax and gradient references.
- Local HF-layout DINOv2 matches the pinned Transformers architecture's checkpoint
  keys, rectangular/masked image outputs and gradients. Local RMBG matches CPU
  side outputs, features and gradients on MPS, including channels-last inputs;
  checkpoint loading and background-removal preprocessing/postprocessing are tested.
- Both TripoSG adaptive mesh extraction paths run on CPU/MPS test surfaces.
  Same-size 3D pooling uses native 2D pools for inference and a Metal argmax plus
  native gathering when gradients are needed, retaining CPU's tie convention.
  Nearest 3D grid upsampling stays on MPS using native repeat operations.
- Standard/compressed Gaussian PLY IO is local. Roundtrips cover partial chunks,
  SH degrees 0/3/4, position/scale/rotation/opacity quantization and CPU/MPS input.
- ActionMesh stage integration covers 31 frames across two autoregressive windows,
  with retained anchor latents, split/batched classifier-free guidance, latent and
  mesh banks, and direct/residual mesh decoding. CPU/MPS outputs agree with identical
  noise inputs. Residual decoding now slices XYZ from vertex features correctly.
- Strict SAM loading memory-maps modern torch archives, reading the selected
  component's tensors without allocating the entire multi-model checkpoint.
  Legacy pickle checkpoints still load through the non-mapped path.
- Runtime timers synchronize Metal as well as CUDA. MPS reports current memory
  at block entry; unavailable peak counters are marked unmeasured. Shape inversion
  now reduces its batch size after a Metal out-of-memory error, preserving frame
  order and rethrowing unrelated runtime failures.
- During Metal fine-tuning, unused SAM generators and image conditioners move
  temporarily to CPU, freeing their Metal allocations while the decoders stay
  resident. Original devices are restored on completion or failure so a cached
  backbone remains reusable. Unit tests and the complete single-image run pass.
  The resumed multiview run completed the largest object's 100 fine-tuning
  iterations. During the first pillow's checkpointed backward pass, a process
  sample measured a 14.6 GB physical footprint and a 24.2 GB lifetime peak.
  The sample and its scope are recorded in
  `results/macos-port/pablo_resume/run/memory_sampling_first_pillow.json`.
  That pillow subsequently completed all 100 iterations. The second pillow
  began with checkpointing enabled, taking about 16 seconds per iteration
  instead of the previous attempt's 133-second first iteration. An early
  sample measured 15.5 GB and a midpoint sample measured 15.6 GB, with the
  process lifetime peak still 24.2 GB;
  see `results/macos-port/pablo_resume/run/memory_sampling_second_pillow.json`.
  All three objects subsequently completed 100 iterations and the fine-tuning
  cache was saved. The second pillow's PSNR decreased from 27.44 to 23.14 and
  SSIM from 0.9675 to 0.9562, so its final post-refinement quality still needs
  inspection. Iteration coverage, finite logged values, metrics and cache hash
  are recorded in `results/macos-port/pablo_resume/run/finetuning_validation.json`.
  Final pose refinement and export readback subsequently passed. Visual
  inspection confirmed damage to the daisy pillow's unseen surfaces during
  fine-tuning; the later paired precision experiment below resolves this regression. Neither
  these samples nor the isolated decoder benchmark establishes the complete
  pipeline's peak memory.

## Validation coverage

| Workflow | Recorded validation |
| --- | --- |
| Single image | Final installed wheel, generated elephant, fresh reconstruction, FP32 fine-tuning, complete export/readback and visual review |
| Multiview | Full three-object/two-view execution and exports; separate paired 100-step full-object experiment verifies the FP32 correction for the pillow |
| Dynamic | Full pretrained ActionMesh plus 16-frame GenIA reconstruction, deformation/tracks, geometry exports and 160-frame orbit readback/visual review |

The multiview and dynamic reconstructions used cache recovery after diagnosed
failures; their stage provenance is retained. The full dynamic run used the
explicitly supported mixed-precision fine-tuning path before the final FP32
default change. The final policy is additionally covered by the generated-image
run, paired multiview experiment and CPU/MPS regression tests. These checks
establish the recorded workflows on this Mac, not every input, optional mode,
hardware configuration or macOS version. Operator limitations are listed above.

## Generated end-to-end test input

`demo/data/image/generated_elephant/` contains the imagegen-produced transparent
source, its white-background RGB composite (`0000.png`), and binary alpha-derived
mask (`0000_mask.png`). The exact built-in imagegen prompt and asset provenance
are in `generation.json`. Generation and clean single-image reconstruction have
both passed.

SAM access is granted and all SAM checkpoints have downloaded. The final
6,690,136,964-byte `ss_generator.ckpt` was assembled and SHA-256 verified against
its Hugging Face blob identifier. MapAnything, ActionMesh/TripoSG, DINOv2, RMBG
and the LPIPS backbones have also downloaded under `/Volumes/MLData4/GenIA`, as requested.
The earlier clean installed-wheel CLI run completed with MoGe preprocessing, 19,391
occupied voxels and 620,512 Gaussians. It retained the default optimization
settings, including all 100 fine-tuning steps, with caches disabled and no
in-process recovery or intervention. Recorded core reconstruction time was
1,706.7 seconds (about 28.4 minutes, excluding model loading and final exports).
Final object metrics were PSNR **28.04**, SSIM **0.9167**, LPIPS **0.0311**;
full-frame metrics were PSNR 25.26, SSIM 0.9022, LPIPS 0.0516.

`results/macos-port/elephant_clean_2/validation.json` records the exact command,
metrics, timing and file readback. The colored canonical GLB contains 379,767
vertices and 757,552 faces. Canonical/world/background Gaussian files contain
finite attributes and valid scales/opacities. All 60 frames of the 518×518 orbit
video decoded successfully. The training view, world overview, six canonical
views and front/rear orbit frames were inspected and show a coherent red ceramic
elephant. Train renders, synthesized novel views, COLMAP files, poses and the
final pipeline state were also exported.

The final wheel was then tested again from the generated image with cache reuse
disabled and the new default FP32 fine-tuning policy. All 100 fine-tuning steps
completed; final pose refinement stopped normally at iteration 214 of its
500-step maximum, retaining iteration 164. Final object PSNR/SSIM/LPIPS were
**28.86 / 0.9228 / 0.0266**. Core reconstruction took 1,549.3 seconds (25.8 minutes,
excluding model loading and exports). Independent readback and visual review
passed for the meshes, Gaussian clouds, training view, all four novel views,
world overview, six canonical views, and seven samples from the 60-frame orbit.
The saved adapter records FP32 training, and all 620,512 Gaussian attributes are
finite with positive scales. Minor surface artifacts remain, especially around
occluded inner legs; the reconstructed elephant retains coherent volume and
limbs through the orbit. Launch details, logs, checks and review scope are under
`results/macos-port/elephant_final/`.
Hugging Face and torch cache paths are saved
in the ignored local `.env.macos`; the token remains in its existing login store.
The local environment also sets `HF_HUB_DISABLE_XET=1`: the Xet transfers stopped
advancing here, while a direct HTTP probe succeeded. The retried downloads use
standard HTTP in the same cache. This is a local workaround, not a global default.
Use these paths for runtime commands too:

```bash
uv run --no-project --env-file .env.macos .venv/bin/python -m genia.core --help
```

For another machine, the downloader accepts `--cache-root /path/to/models` to
configure the hub, Xet, and torch cache directories together. Its `--dry-run`
mode tests path selection without creating directories or downloading weights.
Interrupted HTTP transfers are now retried from their existing partial files;
client errors such as missing repositories are not retried.
The torch-hub/LPIPS portion uses the local feature architectures; VGG and its
calibration weights have downloaded and passed pretrained CPU/MPS validation.

The complete pretrained 24-block SLAT flow has also run on CPU/MPS with 64
synthetic voxels and 17 synthetic condition tokens. Its first MPS forward took
2.05 seconds, with maximum absolute difference 0.0156 and relative L2 difference
0.207%. This tests checkpoint compatibility and the flow graph, not image-based
reconstruction. Measurements are in `results/macos-port/slat/validation.json`.

The full pretrained SLAT image conditioner has processed the generated elephant's
cropped/full RGB and mask streams, producing `(1, 5496, 1024)` condition tokens.
CPU fp32 took 4.82 seconds, MPS fp32 1.00 second, and MPS fp16 autocast 0.82 seconds.
Relative L2 errors against CPU were 0.000660% and 0.244%, respectively; outputs
were finite. See `results/macos-port/slat_condition/validation.json`.

The original GenIA `run_preprocessing` path also completed with `processing=moge`:
it loaded the elephant image/mask, built a 518×518 frame with predicted depth and
intrinsics, and exported one COLMAP camera with 94,686 depth points. The export
is under `results/macos-port/preprocessing/preprocessing/colmap/`. Its text and
binary models have been read back with the independent COLMAP-format reader and
agree. This caught and fixed NumPy 2 scalar repr strings in the text point writer.

Pretrained local RMBG ran at its original 1024×1024 model resolution on MPS in
0.93 seconds. Its thresholded mask has 99.86% IoU with the generated image's alpha
mask. The inspected cutout and measurements are under `results/macos-port/rmbg/`.
These component validations do not establish full reconstruction completion.

The pretrained Gaussian and mesh decoders completed CPU/MPS forward and latent
backward passes for 64 synthetic occupied voxels. In fp32, Gaussian output
relative L2 differences were at most 1.01e-5 and the latent-gradient difference
was 1.51e-5. Mesh faces matched exactly, with vertex relative L2 difference
3.45e-7 and latent-gradient difference 8.71e-6. With the checkpoint's original
fp16 configuration, Gaussian output differences were 0.33–0.95% and its gradient
difference was 2.81%; mesh vertices differed by 0.0373%, with some triangle
choices differing. These are diagnostic synthetic-latent comparisons, not
reconstructed-image accuracy measurements. Results are in
`results/macos-port/decoders{,_fp32}/validation.json`.

The pretrained structure decoder also ran its full 16³ latent → 64³ occupancy
path with backward on CPU/MPS. Occupied-voxel decisions agreed exactly; output
and latent-gradient relative L2 differences were 4.22e-7 and 2.08e-6. Metal took
1.36 seconds for forward/backward on this synthetic latent. Measurements are in
`results/macos-port/ss_decoder/validation.json`.

Evaluation uses local PSNR, reflected-border SSIM, and AlexNet LPIPS; optimization
retains the original valid-window SSIM and VGG LPIPS. Tests compare evaluation
SSIM to TorchMetrics 1.8.2 and optimization SSIM to an independent 2D convolution,
including image gradients on CPU/MPS. Both LPIPS trunks are included in the
weight downloader. This removes the missing TorchMetrics runtime import while
preserving the original metric defaults.

The four downloaded ActionMesh/TripoSG denoiser and autoencoder checkpoints match
the local model state names and shapes exactly (1,695 tensors combined). This
metadata check constructs models on the meta device; it does not execute trained
inference. See `results/macos-port/dynamic_checkpoint_layout/validation.json`.
The full MapAnything checkpoint likewise matches all 736 stored tensor names and
shapes, accounting for 76 omitted aliases of shared parameters; see
`results/macos-port/mapanything_checkpoint_layout/validation.json`.
Pretrained inference on Pablo's two real views at 518×518 took 3.12 seconds on
Metal after 24.82 seconds of loading. Valid depth covered 95.24% and 91.29% of the
views. A separate fp32 comparison of the same real images at 112×112 passed CPU/MPS
checks, with maximum output relative L2 difference 1.32e-5 and identical binary
masks. The inspected depth images, camera/depth arrays and measurements are in
`results/macos-port/mapanything/`. Full GenIA multiview reconstruction is separate.
With bounded global attention, pretrained MapAnything also processed all 16
dinosaur frames at 518×518 in 19.82 seconds. The valid depth mask covers
12.95–14.16% of each image, following the dinosaur against its white background.
A separate 16-view fp32 CPU/MPS comparison at 112×112 passed, with maximum
relative L2 difference 5.54e-5 and identical binary masks. Depth panels for frames
0, 7 and 15 were inspected; the arrays, panel and report are in
`results/macos-port/mapanything_dynamic_chunked/`. This verifies the depth/camera
stage; the separate completed ActionMesh and GenIA checks are described below.
Separately, ActionMesh's synthetic preview path passed mesh decimation, rendering,
and MP4 encoding/decoding from the installed wheel. CPU/MPS image and normal
differences were at most 1.2e-7, with identical masks; measurements are in
`results/macos-port/actionmesh_render/validation.json`.
The installed wheel also loaded the real 16-frame dinosaur RGB/segmentation input
into intact 512×512 RGBA object crops and constructed the lazy ActionMesh pipeline
with `HF_HUB_OFFLINE=1`. All four dynamic-model directories resolved beneath
`/Volumes/MLData4/GenIA`; no trained forward was run in this check. The report and
inspected first/last crops are in `results/macos-port/actionmesh_input/`.

The normal installed-wheel ActionMesh run subsequently completed all 100 anchor
steps, 30 temporal denoising steps, and 16 decoded meshes in 7,686 seconds.
Temporal denoising took 1:57:07 (234 seconds per step). However, visual inspection
of frames 0/7/15 failed: the exported anchor is an unrecognizable small fragment
near x=-0.86, and the animated meshes retain that malformed geometry. GenIA's
downstream shape inversion was stopped deliberately (exit 143) to investigate
the anchor stage. This is an execution milestone, not a successful dynamic
reconstruction. The original outputs and preview evidence are preserved under
`results/macos-port/dynamic_fixed/run/` and the demo's ActionMesh output directory.
The queued readbacks and pillow precision runs consequently did not start.

The anchor failure was traced to the pretrained DINO image conditioning.
PyTorch 2.7's MPS conversion of a permuted FP16/BF16 positional grid ignored
its nonzero storage offset after removal of the CLS token. Casting the complete
position tensor before slicing, and casting back after concatenation, fixes
both the forward values and gradients. The new regression failed on MPS in
both reduced precisions before the fix and passes afterward. Pretrained MPS
FP16 conditioning error against the CPU FP32 reference fell from relative L2
1.1994 to 0.00347 (CPU FP16 is 0.00313); MPS FP32 is 0.00000769.
The VAE and denoiser predictions agreed across backends when supplied with
identical inputs. Diagnostic evidence is under `results/macos-port/triposg_anchor/`
and `results/macos-port/triposg_conditioning_fixed/`.

The corrected latent produced a recognizable dinosaur with direct uniform-grid
decoding, but adaptive mesh extraction exposed a separate large-index problem.
PyTorch 2.7 MPS `nonzero` and integer division/remainder round large indices
through FP32. The real 505³ field crossed the 2²⁴ exact-integer limit, which the
small synthetic extraction cases did not reach. Chunked nonzero extraction and
a Metal integer-coordinate kernel now preserve those indices in both marching
cubes implementations. Tests cover sparse indices across multiple chunks,
strided coordinates above 2²⁷, and full extraction above the 2²⁴ boundary.
Re-extracting the saved real field gives CPU-identical cells, patches and quads;
maximum vertex difference is 3.58e-7. The processed anchor has 19,985 vertices
and 39,966 faces and passed visual review in three views. Evidence and preview
are under `results/macos-port/triposg_anchor_fixed/`.

The corrected normal installed-wheel ActionMesh run then completed all 16 frames
in 7,687 seconds of inference. Temporal denoising completed all 30 steps in
1:56:32. Visual review passed for frames 0/7/15 in three normal-render views and
for all 16 frames in the side view: the dinosaur retains a coherent body, limbs
and tail through the motion. Previews and the review record are under
`results/macos-port/dynamic_index_fixed/run/`. Independent readback also passed:
all 16 meshes share 19,985 vertices/39,966 faces, exported GLBs agree with the
deformation arrays, geometry is finite and moving, and all preview frames decode.
GenIA reached 16-frame shape inversion, but its process and the follow-up queue
subsequently disappeared while the restricted execution environment could not
discover a Metal device. Restoring execution permissions restored Metal device
discovery and passed an actual PyTorch MPS calculation. Reconstruction has been
restarted using the validated ActionMesh output; its launch is recorded in
`dynamic_index_fixed/run/resume_execution.json`. That attempt's sequential
follow-up queue is recorded under `dynamic_index_resumed_followup/`.

The resumed run completed all 100 shape-inversion steps for the 16 frames,
with per-frame occupancy IoU of 0.967–0.986, and saved
`dynamic_index_fixed/cache/gt_shapes_inversion.pt`. Parallel pose initialization
then failed because SAM's dense attention still called native SDPA directly,
attempting a 14.70 GB score allocation. That call now uses the local bounded
MPS attention operator. `test_sam_attention.py` covers all three input packing
forms against independent CPU values and gradients, and verifies that the
16-frame cross-attention entry point respects the score budget. The allocation
regression failed before the fix; all 77 focused attention tests passed after it.
The rebuilt wheel passed all 598 tests and reconstruction was restarted from
the saved shape cache; launch details are in
`dynamic_index_fixed/run/pose_resume_execution.json`, with the replacement
follow-up queue under `dynamic_sam_attention_followup/`.

The repaired run completed pose initialization (10 steps), appearance consensus
(25 steps), and all 100 fine-tuning steps over 16 frames. Final pose refinement
used the normal early-stopping rule: it stopped at iteration 131 of the configured
500, retaining iteration 81. The final object PSNR/SSIM/LPIPS averaged
22.22/0.9450/0.0638. Independent readback passed for the canonical mesh, all 16
world meshes, canonical/per-frame/world Gaussian files, cameras, render images,
16,384 tracked points across 16 frames, and the 160-frame orbit video. Projected
3D tracks agreed with the saved 2D tracks within 0.000068 pixels.
Visual inspection covered all 16 training renders, six canonical views at frames
0/7/15, 12 perturbed camera views, and nine orbit samples. The dinosaur remains
coherent through the motion, with minor surface artifacts near the head/limbs;
some perturbed views crop the tail. This verifies execution and usable geometry,
not ground-truth 3D accuracy. Results and review scope are recorded in
`dynamic_index_fixed/run/{validation,tracks_validation,final_visual_review}.json`.
The paired pillow precision experiments started after these readbacks passed.

## Full multiview execution and quality follow-up

The fresh source installation completed Pablo's two-view reconstruction with
exit code 0. Recovery used the saved `appearance_init_2` state from the earlier
attempt; all three objects then ran 100 fine-tuning steps and 500 final pose
steps each. The completed fine-tuning and final-pose caches are retained.
The resumed run's timing excludes the cached initialization and appearance
stages, whose separate execution is recorded in `run_provenance.json`.

Independent readback verified all three canonical meshes, the world mesh,
five Gaussian PLY files, two cameras and training-view images, and all 60
518×518 orbit frames. Both source views, both training/world renders, all
three six-view canonical panels, and orbit frames 0/30 were inspected.
Artifacts and execution provenance are under
`results/macos-port/pablo_resume/run/`.

Final per-object PSNR/SSIM were 28.82/0.9581 for the raccoon,
29.21/0.9334 for the vegetable-pattern pillow, and 24.31/0.9664 for the
daisy pillow. The first two objects are coherent across the inspected views.
The daisy pillow retains recognizable front texture, but its unseen surfaces
became incomplete during fine-tuning. A render of the saved pre-fine-tuning
state (`daisy_before_finetuning.png`) is substantially more complete. Mean
per-point opacity fell from 0.91 to 0.19; geometry remained finite with similar
bounds. This recorded mixed-precision run is not an unqualified quality pass.
Its saved state enabled the precision diagnosis and correction below without
repeating initialization.

A CPU decode of the full saved daisy-pillow tokens and trained DoRA weights
reproduced the low opacity (mean 0.185873 on CPU versus 0.185944 in the Metal
cache; about 74.6% of points below 0.01 on both). The checkpoint uses an FP16
transformer body. Relative output differences reached 1.65%, so a strict 0.1%
comparison failed; this diagnostic is not a full precision parity pass.
It shows that the opacity loss is present in the saved trained state, but
does not establish agreement of the CPU/Metal optimization trajectories or
rendering gradients. See `daisy_cpu_decode_validation.json` in that run.

A separate dense-scene compositor check passed on six 4×4 crops from the
saved before/after pillow states. Each crop retained all 1,394–6,289 Gaussian
contributors from its original image tile. CPU autograd and Metal agreed on
RGB/depth, alpha, and gradients for centers, conics, colors, opacity, and
background. Maximum opacity-gradient differences were below 3.5e-7
(relative L2 below 9.1e-6). This narrows the investigation beyond the small
synthetic renderer tests; it does not establish decoder-gradient or complete
optimization-trajectory agreement. Reproduction and results are saved as
`check_daisy_raster.py` and `daisy_raster_validation.json` in the same run.

A CPU FP32 decode separated token changes from the trained DoRA adapter.
The base decoder produced mean opacity 0.91001 with the original tokens and
0.90987 with the trained tokens. The trained adapter reduced those values
to 0.18643 and 0.18584 respectively. All four cases were finite and completed
in 21.37 seconds. Thus the trained adapter is sufficient to reproduce the
collapse, even with the original tokens and without half precision. This
locates the change in the saved trained state; the paired experiment below
tests the precision used during optimization. See `daisy_factors_validation.json` and
`check_daisy_factors_fp32.py`. An earlier CPU FP16 version was stopped after
its base-decoder/original-token case produced non-finite values; its partial
results are preserved separately as `daisy_factors_fp16_validation.json`.

The complete pretrained decoder also underwent an opacity-anchor gradient
comparison on a 512-token spatial neighborhood, using the trained adapter
and checkpointing. FP32 CPU/Metal outputs and token/adapter gradients agreed
within 2.3e-5 relative L2. The checkpoint's FP16 body under BF16 autocast did
not pass a 5% gradient comparison: native CPU/Metal token gradients differed
by 25.0%. Both differed materially from FP32 (22.1% CPU, 17.0% Metal).
Forcing CPU math SDPA reduced the CPU/Metal token-gradient difference to
8.0%, still outside that threshold. This demonstrates precision sensitivity
in the trained decoder; it neither proves a Metal-specific training defect
nor validates complete mixed-precision optimization parity. The initial
failure, calibration, and tensor outputs are preserved in
`daisy_decoder_gradient_validation.json`,
`daisy_decoder_gradient_calibration.json`, and
`daisy_decoder_gradient_calibration_outputs.pt`.

Full CPU decoding of the saved trained adapter also reproduces the collapse
with the training BF16 autocast enabled: mean opacity 0.18671, versus 0.18587
with export-time autocast disabled. Fractions below 0.01 are 74.52% and 74.63%.
Thus the precision switch at export does not account for this geometry loss.
This saved-state check is recorded in `daisy_decode_precision_validation.json`.

The paired experiment then completed all 100 steps with the same initial
checkpoint, both views, all 19,650 tokens / 628,800 Gaussians, and seed 42.
Only object 3 was optimized, so this is an isolated full-object comparison,
not another complete three-object reconstruction. The original losses,
learning rates, visibility rules and optimizer were retained.

| Fine-tuning result | Mixed FP16 body / BF16 autocast | Full FP32 |
| --- | ---: | ---: |
| Initial PSNR | 27.44 | 27.44 |
| Final PSNR | 23.76 | 30.70 |
| Final SSIM | 0.9610 | 0.9736 |
| Mean point opacity | 0.2132 | 0.8623 |
| Points with opacity below 0.01 | 73.89% | 5.81% |

Six identical canonical camera views were rendered from each complete saved
cloud, without point filtering. Mixed precision removed large portions of
the pillow; FP32 retained continuous volume and surfaces. Unseen surface colors
still differ from the initial prior, so this is not ground-truth appearance
validation. FP32 also ran faster here, about 10 seconds per step versus 16.
Single mid-run physical-memory samples were 11.6 GB (FP32) and 12.0 GB (mixed),
with a reported process peak of 24.2 GB in each at sampling; these are not
complete peak-memory profiles. Results are in
`results/macos-port/dynamic_sam_attention_followup/precision_comparison.json`,
and fixed-camera images plus the review are in
`results/macos-port/daisy_precision/comparison/`.

Metal fine-tuning now defaults to full FP32, retaining checkpointing. The
explicit BF16 option remains available; CUDA/CPU defaults remain unchanged.
Saved full-precision adapters retain that choice when loaded with a fresh
decoder. This avoids the reproduced failure without changing the optimizer,
regularization, iteration count, visibility mask, or scene geometry.

## Pretrained MoGe stage validation

The generated 1254×1254 elephant ran through the local pretrained `Ruicheng/moge-vitl`
model on MPS with the default 2,500 tokens. On this machine, model loading took
3.46 seconds and depth inference took 2.30 seconds. A separate fp32 comparison
using 256 tokens and the same image/weights had maximum absolute errors of
1.94e-5 for points and 9.57e-6 for mask logits relative to CPU.

```bash
uv run --no-project --env-file .env.macos .venv/bin/python -m genia.core.validate_moge \
  --image demo/data/image/generated_elephant/0000.png \
  --output results/macos-port/moge --compare-cpu
```

The output directory contains `prediction.npz`, a colored `points.ply`, the
inspected depth visualization `depth.png`, and measurements in `validation.json`.
The valid depth mask covers 35.3% of the image; the 1/50/99 percentiles are
2.23/2.42/2.88 in the model's relative depth units. This validates the depth stage,
not the full SAM/GenIA optimization and reconstruction pipeline.

## Model provenance

See `vendor/README.md` and `vendor/licenses/`. The original GenIA patch was applied
before copying SAM code into this repository. The original CUDA installation
script remains for upstream users; macOS development uses `install_macos.sh`.

## Metal operator optimization comparisons

The optimized operators were compared directly with their previous Metal paths,
using synthetic inputs, one warm-up and the median of three synchronized runs on
this Mac (Apple M2 Max, 32 GiB RAM, PyTorch 2.7.1). These are operator timings, not CUDA speed comparisons or
end-to-end reconstruction speedups. The attention benchmark retains ActionMesh's
actual `(2, 16, 32784, 128)` shape, strided layouts and BF16 autocast. Other timings
use FP32. Sparse neighbors are synthetic, with approximately two thirds present.

| Operator / input | Previous | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| ActionMesh attention, inference | 9.276 s | 5.179 s | 1.79× |
| 3D KNN, 8,192 queries / 8,192 references, K=1, inference | 9.013 ms | 0.879 ms | 10.26× |
| Same KNN, forward + backward | 9.259 ms | 1.903 ms | 4.87× |
| Sparse 3³ convolution, 4,096 sites / 64 channels, forward + backward | 9.822 ms | 4.413 ms | 2.23× |
| Sparse 3³ convolution, 19,650 sites / 128 channels, forward + backward | 46.402 ms | 39.884 ms | 1.16× |
| Sparse 3³ convolution, 19,650 sites / 512 channels, forward + backward | 226.551 ms | 166.092 ms | 1.36× |
| Splat compositor, 10,000 Gaussians / 256² pixels / radius parameter 8, forward + backward | 34.605 ms | 6.106 ms | 5.67× |
| Splat compositor, 1,000 Gaussians / 256² pixels / radius parameter 80, forward + backward | 72.654 ms | 8.454 ms | 8.59× |

Implementation scope:

- Attention uses larger memory-aware native MPS query chunks for inference. A
  custom streaming Metal attention prototype was tested but did not beat the
  larger native chunks, so it is not dispatched or shipped in the runtime.
  Gradient-enabled attention retains its previous checkpointed implementation.
- Sparse convolution combines offsets into the matrix-multiplication reduction
  dimension. Each gather is limited to eight million elements for up to 128 input
  channels, or 64 million elements for wider layers (32/256 MiB at FP32). Autograd
  may retain gathers from multiple chunks; these limits are not total training
  memory limits. This still computes zero rows for missing neighbors and is not a
  full masked implicit-GEMM implementation.
- Default 16×16 splat tiles cooperatively cache Gaussian records. Backward sums
  gradients within each SIMD group before global atomic updates. Other tile sizes
  retain the previous kernels. Timings above isolate compositing and exclude
  projection/binning. The binning sorts and allocation synchronization are unchanged.
- KNN distributes each query's reference scan across a SIMD group for 3D inputs,
  K=1 and at least 256 reference points. Inference returns distances directly from
  that kernel; training retains the existing differentiable distance calculation.
  Other dimensions/K values retain the generic kernel.

The measured attention relative L2 output difference was 1.10e-4. KNN indices
match the generic kernel in the comparison tests; the measured distance difference
was 1.03e-7 relative L2. Sparse output/gradient differences were at most 2.17e-6
relative L2. Splat images and alpha were identical; gradients differed by at most
7.58e-6 relative L2 from the original atomic summation order. Large absolute conic
gradient differences in the broad-splat case reflect correspondingly large
summed gradients; the JSON records both absolute and relative errors.

Thirty additional comparison tests cover FP32/FP16/BF16, autocast, gradients,
multiple sparse chunks, KNN ties/empty clouds/nonfinite inputs, partial image tiles,
multiple splat batches, early termination, masks and attention memory pressure.
The original KNN wrapper and per-pixel compositor are retained under
`tests/reference/`; the generic Metal kernels remain callable.

Reproduce with:

```bash
uv run --no-sync python tests/benchmark_metal_optimizations.py
uv run --no-sync python -m pytest tests/test_metal_optimizations.py -q
```

Detailed samples and numerical errors: `results/macos-port/kernel_optimization/validation.json`.
Benchmark log: `results/macos-port/kernel_optimization/benchmark.log`.

After these changes, the complete CPU/Metal suite passed: **637 tests**, 29 warnings, 74.30 seconds. The suite includes the 30 new operator-comparison tests and the SAM allocation regression under constrained memory.
