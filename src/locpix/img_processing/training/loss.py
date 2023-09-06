"""Defie loss functions"""

import torch


class dice_loss:
    def __call__(self, logits, target):
        prob = torch.sigmoid(logits)
        int = torch.sum(prob * target)
        union = torch.sum(prob) + torch.sum(target)
        return 1 - (2.0 * int) / (union)
