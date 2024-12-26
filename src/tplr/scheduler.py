# The MIT License (MIT)
# © 2024 templar.tech

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional
from .logging import logger

class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    Args:
        optimizer: Optimizer to schedule learning rate for
        warmup_steps: Number of warmup steps
        alpha_f: Final learning rate multiplier (minimum lr = initial_lr * alpha_f)
        t_max: Maximum number of steps for cosine decay (None = use max_steps)
        warmup_min_lr: Minimum learning rate during warmup (None = 0.1 * initial_lr)
        last_epoch: Last epoch (-1 for fresh start)
    """
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        alpha_f: float = 0.1,
        t_max: Optional[int] = None,
        warmup_min_lr: Optional[float] = None,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.alpha_f = alpha_f
        self.t_max = t_max
        self.warmup_min_lr = warmup_min_lr
        super().__init__(optimizer, last_epoch)

    def _linear_warmup(self, initial_lr: float, step: int) -> float:
        """Calculate learning rate during warmup period"""
        warmup_min_lr = self.warmup_min_lr if self.warmup_min_lr is not None else initial_lr * 0.10
        assert 0 <= warmup_min_lr < initial_lr, "Warmup minimum LR must be less than initial LR"
        return warmup_min_lr + (initial_lr - warmup_min_lr) * min(step, self.warmup_steps) / self.warmup_steps

    def get_lr(self) -> list[float]:
        """Get learning rates for all parameter groups."""
        if not self._get_lr_called_within_step:
            logger.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        step = self.last_epoch + 1  # Adjust for 0-based indexing

        if self.t_max is None:
            max_steps = self.warmup_steps + 1  # Default to warmup_steps + 1 to avoid division by zero
        else:
            max_steps = self.t_max

        if step <= self.warmup_steps:
            # Linear warmup phase
            return [self._linear_warmup(base_lr, step) for base_lr in self.base_lrs]
        elif step >= self.warmup_steps + max_steps:
            # Maintain minimum learning rate after decay
            return [base_lr * self.alpha_f for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            step_in_cosine = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps  # Adjust max_steps for cosine decay
            return [
                base_lr * self.alpha_f
                + (base_lr - base_lr * self.alpha_f)
                * (1 + math.cos(math.pi * step_in_cosine / max_steps))
                / 2
                for base_lr in self.base_lrs
            ] 