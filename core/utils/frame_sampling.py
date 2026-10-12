# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Epoch-based frame sampling shared by pose refinement and fine-tuning."""
import random


class EpochFrameSampler:
    """Epoch-based frame sampler: all frames are seen before any repeats."""

    def __init__(self, frame_indices):
        self._all = list(frame_indices)
        self._queue: list = []

    def sample(self, batch_size: int) -> list:
        batch: list = []
        while len(batch) < batch_size:
            if not self._queue:
                self._queue = list(self._all)
                random.shuffle(self._queue)
            take = min(batch_size - len(batch), len(self._queue))
            batch.extend(self._queue[:take])
            self._queue = self._queue[take:]
        return batch
