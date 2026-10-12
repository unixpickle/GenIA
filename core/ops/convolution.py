"""Metal mixed-precision compatibility for native PyTorch 3D convolution."""
import torch


class Conv3d(torch.nn.Conv3d):
    """Conv3d with consistent activation/weight/bias casting on PyTorch 2.7 MPS.

    MPS autocast otherwise leaves the bias in fp32 for a half input. Explicit
    casts retain autograd and checkpoint parameter names, and still use the
    native Metal convolution. CPU/CUDA use the original PyTorch implementation.
    """

    def forward(self, input):
        if input.device.type == 'mps' and torch.is_autocast_enabled('mps'):
            dtype = torch.get_autocast_dtype('mps')
            with torch.autocast('mps', enabled=False):
                return self._conv_forward(
                    input.to(dtype), self.weight.to(dtype),
                    self.bias.to(dtype) if self.bias is not None else None)
        return super().forward(input)
