import torch
from typing import Any

from .renderer import Renderer
from .mesh_renderer import MeshRenderer, SilhouetteRenderer
from .skeleton_renderer import SkeletonRenderer
from .pose_utils import eval_pose, Evaluator


def recursive_to(x: Any, target: torch.device):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return torch.nn.Linear(*args, **kwargs)


class LinerParameterTuner:
    def __init__(self, start, start_value, end_value, end):
        self.start = start
        self.start_value = start_value
        self.end_value = end_value
        self.end = end
        self.total_steps = self.end - self.start

    def get_value(self, step):
        if step < self.start:
            return self.start_value
        elif step > self.end:
            return self.end_value

        current_step = step - self.start

        ratio = current_step / self.total_steps

        current_value = self.start_value + ratio * (self.end_value - self.start_value)
        return current_value


class StaticParameterTuner:
    def __init__(self, v):
        self.v = v

    def get_value(self, step):
        return self.v