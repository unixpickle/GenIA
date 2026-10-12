# Independent numerical reference

`gsplat_torch.py` is the unmodified pure-PyTorch implementation from gsplat
v1.5.3, commit `937e29912570c372bed6747a5c9bf85fed877bae`, originally
`gsplat/cuda/_torch_impl.py`. It is used to compare projection equations against
the original implementation; it is not a runtime dependency.

Source: https://github.com/nerfstudio-project/gsplat/tree/v1.5.3
Licence: Apache 2.0, copied into `vendor/licenses/gsplat.LICENSE`.

`diso/cumc.cu` is unchanged from the DISO 0.1.4 PyPI source distribution.
`diso/reference.py` reads its tables independently and transcribes the CUDA
vertex/face loops using scalar NumPy. It is a topology and geometry test oracle,
not a runtime dependency. Source: https://github.com/SarahWeiii/diso
Licence: CC BY-NC 4.0, retained in `diso/LICENSE` and `vendor/licenses/diso.LICENSE`.

`diso/cudualmc.cu` is likewise unchanged from DISO 0.1.4. The independent scalar
`diso/dual_reference.py` reads those tables and reproduces ambiguity resolution,
dual vertices and quads. Tests cover all 256 cases, including NaN samples used
by TripoSG's adaptive grid, and compare geometry derivatives numerically.

`fps.py` retains PyTorch3D v0.7.9's naive FPS and masked-gather reference routines.
`lpips/` retains LPIPS 0.1.4 test reference code; feature networks there come from
TorchVision 0.22.1. `vox2seq/` retains the original TRELLIS bit-array routines.
These are test oracles, not runtime dependencies; upstream licences are in
`vendor/licenses/` and the corresponding reference directories.

`image_metrics/torchmetrics_{ssim,utils}.py` are unchanged from
Lightning-AI/torchmetrics v1.8.2 (`src/torchmetrics/functional/image/`). Tests
execute the original Gaussian/SSIM functions directly without importing other
TorchMetrics dependencies. The Apache 2.0 licence is retained in
`vendor/licenses/torchmetrics.LICENSE`. Optimization SSIM is additionally checked
against an independent full 2D convolution, including image derivatives.
