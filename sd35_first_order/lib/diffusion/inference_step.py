"""
Utility functions for diffusion scheduler inference steps.

These functions are designed to be dynamically added to scheduler classes
(e.g., DDIMScheduler, DPMSolverSinglestepScheduler) to provide additional
functionality for prediction and noise computation.
"""

from typing import Optional, Tuple, Union

import math
import torch

try:
    from diffusers.utils import randn_tensor
except ImportError:
    from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler
from diffusers import DPMSolverSinglestepScheduler


def _left_broadcast(t, shape):
    """
    Broadcast tensor `t` to match `shape` by adding dimensions on the right.

    Args:
        t: Input tensor
        shape: Target shape

    Returns:
        Broadcasted tensor
    """
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def get_alpha_prod_t(self, timestep, sample):
    """
    Get cumulative product of alphas (α_t) for given timestep.

    Args:
        self: Scheduler instance (this function is meant to be added to a scheduler class)
        timestep: Current timestep indices
        sample: Sample tensor to match shape

    Returns:
        alpha_prod_t: Cumulative alpha values broadcasted to sample shape
    """
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    return alpha_prod_t


def predict_clean(
        self,
        model_output,
        sample: torch.FloatTensor,
        timestep: int,
        no_jacobian: bool = False,
        strength: float = 1.0
    ):
    """
    Predict clean sample (x_0) from noisy sample and model output.

    Supports different prediction types:
    - 'epsilon': Model predicts noise
    - 'sample': Model predicts clean sample directly
    - 'v_prediction': Model predicts velocity

    Args:
        self: Scheduler instance
        model_output: Output from the diffusion model
        sample: Current noisy sample
        timestep: Current timestep
        no_jacobian: Whether to disable Jacobian computation (unused)
        strength: Scaling factor for model output (default: 1.0)

    Returns:
        pred_clean_sample: Predicted clean sample (x_0)
    """
    with torch.no_grad():
        alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
        alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
        alpha_prod_t = alpha_prod_t.to(sample.dtype)
        beta_prod_t = 1 - alpha_prod_t

        # Handle terminal timestep (t=0)
        beta_prod_t[timestep == 0] = 0
        alpha_prod_t[timestep == 0] = 1

    if self.config.prediction_type == "epsilon":
        # Predict x_0 from noise prediction
        pred_clean_sample = (
            sample - beta_prod_t ** (0.5) * model_output * strength
        ) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        # Model directly predicts x_0
        pred_clean_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        # Predict x_0 from velocity prediction
        pred_clean_sample = (alpha_prod_t ** 0.5) * sample - (
            beta_prod_t ** 0.5
        ) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )
    return pred_clean_sample
