"""Present-class Dice and cross-entropy losses from the supplied training code."""
import torch
from torch import nn
import torch.nn.functional as F

class PresentDiceLoss3D(nn.Module):

    def __init__(self, ignore_index=None, eps=1e-05, include_background=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps
        self.include_background = include_background

    def forward(self, probs, target, class_weights=None):
        N, C, D, H, W = probs.shape
        device = probs.device
        if self.ignore_index is None:
            valid_mask = torch.ones((N, D, H, W), dtype=torch.bool, device=device)
            tgt = target
        else:
            valid_mask = target != self.ignore_index
            tgt = target.clone()
            tgt[~valid_mask] = 0
        target_1hot = torch.zeros_like(probs)
        target_1hot.scatter_(1, tgt.unsqueeze(1), 1.0)
        valid_mask_ = valid_mask.unsqueeze(1)
        probs = probs * valid_mask_
        target_1hot = target_1hot * valid_mask_
        dims = (2, 3, 4)
        inter = (probs * target_1hot).sum(dim=dims)
        probs_sum = probs.sum(dim=dims)
        target_sum = target_1hot.sum(dim=dims)
        dice = (2.0 * inter + self.eps) / (probs_sum + target_sum + self.eps)
        dice_loss = 1.0 - dice
        present = target_sum > 0
        if not self.include_background and C > 1:
            present[:, 0] = False
        if class_weights is not None:
            w = class_weights.view(1, C).to(device=device, dtype=probs.dtype)
            dice_loss = dice_loss * w
        present_count = present.sum().clamp_min(1)
        loss = dice_loss[present].sum() / present_count
        return loss

class CombinedLoss3D_PresentDice(nn.Module):

    def __init__(self, weight_dice=1.0, weight_ce=1.0, class_weights_ce=None, class_weights_dice=None, ignore_index=None, include_background_in_dice=False, eps=1e-05):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(weight=class_weights_ce, ignore_index=ignore_index if ignore_index is not None else -100, reduction='none')
        self.dice = PresentDiceLoss3D(ignore_index=ignore_index, eps=eps, include_background=include_background_in_dice)
        self.class_weights_dice = class_weights_dice
        self.ignore_index = ignore_index

    def forward(self, logits, target, voxel_weight=None):
        probs = F.softmax(logits, dim=1)
        dice_val = self.dice(probs, target, class_weights=self.class_weights_dice)
        ce_map = self.ce(logits, target)
        if self.ignore_index is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        else:
            valid = target != self.ignore_index
        ce_map = ce_map[valid]
        if voxel_weight is not None:
            vw = voxel_weight[valid].to(dtype=ce_map.dtype)
            ce_val = (ce_map * vw).sum() / (vw.sum() + self.eps)
        else:
            ce_val = ce_map.mean()
        total = self.weight_dice * dice_val + self.weight_ce * ce_val
        return (total, dice_val, ce_val)
