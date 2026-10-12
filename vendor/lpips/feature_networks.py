"""Local LPIPS feature networks adapted from torchvision 0.22.1 (BSD-3-Clause).

Classification heads are omitted; only feature parameters are loaded. See
vendor/licenses/torchvision.LICENSE for the retained upstream copyright notice.
"""
from types import SimpleNamespace
from typing import List, Union, cast
import torch
from torch import nn

BACKBONE_URLS = {
    'vgg': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'alex': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
    'squeeze': 'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth',
}


def _finish(features, name, pretrained):
    if pretrained:
        state = torch.hub.load_state_dict_from_url(BACKBONE_URLS[name],
            map_location='cpu', check_hash=True, weights_only=True)
        features.load_state_dict({k.removeprefix('features.'): v for k,v in state.items()
                                 if k.startswith('features.')}, strict=True)
    return SimpleNamespace(features=features)

class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg16(pretrained=True):
    features = make_layers([64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'])
    if not pretrained:
        for module in features.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)
    return _finish(features, 'vgg', pretrained)

def alexnet(pretrained=True):
    features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
    return _finish(features, 'alex', pretrained)

def squeezenet1_1(pretrained=True):
    features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
    if not pretrained:
        for module in features.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    return _finish(features, 'squeeze', pretrained)
