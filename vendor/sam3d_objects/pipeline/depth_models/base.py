# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
from genia.core.device import get_device


class DepthModel:
    def __init__(self, model, device=None):
        self.model = model
        self.device = get_device(device)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, image):
        pass
