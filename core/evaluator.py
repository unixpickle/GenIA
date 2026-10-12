# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from genia.core.ops.image_metrics import psnr, ssim


class Evaluator:
    
    def __init__(self, device: torch.device):
        self.psnr_metric = psnr
        self.ssim_metric = ssim
        self.device = device
        self._lpip_metric = None

    @property
    def lpip_metric(self):
        # Avoid allocating/downloading the evaluation trunk until metrics are used.
        if self._lpip_metric is None:
            import lpips
            self._lpip_metric = lpips.LPIPS(net="alex", verbose=False).to(self.device)
            self._lpip_metric.eval().requires_grad_(False)
        return self._lpip_metric
    
    def evaluate_frame(self, gt_frame: torch.Tensor, pred_frame: torch.Tensor):
        """
        Evaluate a single predicted frame against the ground truth frame with PSNR / SSIM / LPIPS.

        Args:
            gt_frame (torch.Tensor): Ground truth frame.
            pred_frame (torch.Tensor): Predicted frame.

        Returns:
            dict: Dictionary containing PSNR, SSIM, and LPIP values.
        """

        # assert data in [0, 1]
        assert torch.all((gt_frame >= 0) & (gt_frame <= 1)), "Ground truth frame values should be in the range [0, 1]"
        assert torch.all((pred_frame >= 0) & (pred_frame <= 1)), "Predicted frame values should be in the range [0, 1]"
        
        psnr = self.psnr_metric(pred_frame, gt_frame)
        ssim = self.ssim_metric(pred_frame, gt_frame)
        # TorchMetrics' original default is normalize=False, even for [0, 1]
        # inputs. Preserve that convention and its mean reduction over the batch.
        lpip = self.lpip_metric(pred_frame, gt_frame, normalize=False).mean()

        return {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpip": lpip.item(),
        }
    
    def evaluate_sequence(self, gt_frames, pred_frames):
        """
        Evaluate the predicted frames against ground truth frames with PSNR / SSIM / LPIPS.

        Args:
            gt_frames (list of torch.Tensor): List of ground truth frames.
            pred_frames (list of torch.Tensor): List of predicted frames.

        """

        if not gt_frames or not pred_frames:
            raise ValueError(
                "evaluate_sequence called with empty frame lists "
                f"(gt_frames={len(gt_frames)}, pred_frames={len(pred_frames)})"
            )

        psnr_values = []
        ssim_values = []
        lpip_values = []
        for gt, pred in zip(gt_frames, pred_frames):
            metrics = self.evaluate_frame(gt, pred)
            psnr_values.append(metrics["psnr"])
            ssim_values.append(metrics["ssim"])
            lpip_values.append(metrics["lpip"])

        return {
            "psnr_values": psnr_values,
            "ssim_values": ssim_values,
            "lpip_values": lpip_values,
            "psnr_mean": sum(psnr_values) / len(psnr_values),
            "ssim_mean": sum(ssim_values) / len(ssim_values),
            "lpip_mean": sum(lpip_values) / len(lpip_values),
            "psnr_std": torch.std(torch.tensor(psnr_values)).item() if len(psnr_values) > 1 else 0.0,
            "ssim_std": torch.std(torch.tensor(ssim_values)).item() if len(ssim_values) > 1 else 0.0,
            "lpip_std": torch.std(torch.tensor(lpip_values)).item() if len(lpip_values) > 1 else 0.0,
            "psnr_min": min(psnr_values),
            "ssim_min": min(ssim_values),
            "lpip_min": min(lpip_values),
            "psnr_max": max(psnr_values),
            "ssim_max": max(ssim_values),
            "lpip_max": max(lpip_values),
        }
