"""
Flow-matching noise schedule for GlyphDiffusion.

This module implements Optimal Transport Flow Matching (OT-FM) as described
in Lipman et al., "Flow Matching for Generative Modeling" (2022).

Unlike DDPM which learns to predict noise, flow matching learns to predict
the velocity field that transports samples from a noise distribution to
the data distribution along straight paths (optimal transport).

Key advantages over DDPM:
- Faster inference (fewer steps needed)
- More stable training
- Better sample quality with fewer NFE (number of function evaluations)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import FlowMatchingConfig


class FlowMatchingSchedule:
    """Flow-matching schedule using optimal transport interpolation.

    In flow matching, we define a time-dependent probability path:
        p_t(x) = interpolate between p_0(x) (data) and p_1(x) (noise)

    Using optimal transport, the interpolation is linear:
        x_t = (1 - t) * x_0 + t * x_1

    where x_0 is data and x_1 is noise (standard normal).

    The velocity field to be learned is:
        v_t(x_t) = x_1 - x_0

    which is constant along each trajectory (straight lines in OT).
    """

    def __init__(self, config: Optional[FlowMatchingConfig] = None):
        """Initialize flow-matching schedule.

        Args:
            config: Flow matching configuration. Uses defaults if not provided.
        """
        self.config = config or FlowMatchingConfig()
        self.sigma_min = self.config.sigma_min
        self.sigma_max = self.config.sigma_max
        self.num_train_steps = self.config.num_train_steps
        self.num_inference_steps = self.config.num_inference_steps

    def get_sigmas(self, num_steps: Optional[int] = None) -> torch.Tensor:
        """Get sigma schedule for inference.

        For flow matching with linear interpolation, sigmas go from 1 to 0.

        Args:
            num_steps: Number of inference steps. Uses default if not provided.

        Returns:
            Tensor of sigma values of shape (num_steps + 1,).
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        # Linear schedule from 1 (pure noise) to 0 (pure data)
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
        return sigmas

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to data using linear interpolation.

        Computes: x_t = (1 - t) * x_0 + t * noise

        Args:
            x_0: Clean data of shape (batch, channels, height, width).
            noise: Standard normal noise of same shape as x_0.
            timesteps: Timesteps in [0, 1] of shape (batch,).

        Returns:
            Noisy data x_t of same shape.
        """
        # Reshape timesteps for broadcasting: (batch, 1, 1, 1)
        t = timesteps.view(-1, 1, 1, 1)

        # Linear interpolation
        x_t = (1 - t) * x_0 + t * noise

        # Add small amount of noise to avoid singular points, but only when t > 0
        # At t=0, x_t should be exactly x_0 (clean data)
        noise_mask = (t > 0).float()
        x_t = x_t + noise_mask * self.sigma_min * torch.randn_like(x_t)

        return x_t

    def get_velocity(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the target velocity for training.

        For optimal transport, the velocity is constant along each path:
            v = x_1 - x_0 = noise - data

        Args:
            x_0: Clean data of shape (batch, channels, height, width).
            noise: Standard normal noise of same shape as x_0.

        Returns:
            Target velocity of same shape.
        """
        return noise - x_0

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample random timesteps for training.

        We use uniform sampling over [0, 1], with small offset to avoid
        exactly t=0 (data) and t=1 (noise).

        Args:
            batch_size: Number of timesteps to sample.
            device: Device to create tensor on.

        Returns:
            Timesteps of shape (batch_size,) in range [sigma_min, 1-sigma_min].
        """
        t = torch.rand(batch_size, device=device)
        # Rescale to avoid boundary issues
        t = t * (1 - 2 * self.sigma_min) + self.sigma_min
        return t

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        sample: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Take one Euler step in the ODE.

        The ODE is: dx/dt = v(x, t)
        Euler step: x_{t-dt} = x_t - dt * v(x_t, t)

        Args:
            model_output: Predicted velocity v(x_t, t).
            timestep: Current timestep t.
            sample: Current sample x_t.
            dt: Time step size (positive, going from t=1 to t=0).

        Returns:
            Updated sample x_{t-dt}.
        """
        # Going from noise (t=1) to data (t=0), so we subtract
        sample = sample - dt * model_output
        return sample


class FlowMatchingScheduler(nn.Module):
    """PyTorch module wrapper for flow-matching schedule.

    This class manages the inference loop for generating samples.
    It handles:
    - Computing the step schedule
    - Stepping through the ODE
    - Optional intermediate saving
    """

    def __init__(self, config: Optional[FlowMatchingConfig] = None):
        """Initialize scheduler.

        Args:
            config: Flow matching configuration.
        """
        super().__init__()
        self.schedule = FlowMatchingSchedule(config)
        self.config = config or FlowMatchingConfig()

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Set the timesteps for inference.

        Args:
            num_inference_steps: Number of denoising steps.
            device: Device to create tensors on.
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps

        # Timesteps go from 1 (noise) to 0 (data)
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        if device is not None:
            self.timesteps = self.timesteps.to(device)

        self.num_inference_steps = num_inference_steps

    def step(
        self,
        model_output: torch.Tensor,
        step_index: int,
        sample: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform one denoising step.

        Args:
            model_output: Model's predicted velocity.
            step_index: Current step index (0 to num_steps-1).
            sample: Current noisy sample.

        Returns:
            Tuple of (denoised sample, predicted original sample).
        """
        # Get current and next timesteps
        t_current = self.timesteps[step_index]
        t_next = self.timesteps[step_index + 1]
        dt = t_current - t_next  # Positive since going 1 -> 0

        # Euler step
        prev_sample = self.schedule.step(
            model_output=model_output,
            timestep=t_current.item(),
            sample=sample,
            dt=dt.item(),
        )

        # Estimate the denoised sample (for visualization)
        # x_0 = x_t - t * v
        pred_original_sample = sample - t_current * model_output

        return prev_sample, pred_original_sample


class FlowMatchingLoss(nn.Module):
    """Loss function for flow matching training.

    The flow matching objective is to minimize:
        E[|| v_theta(x_t, t) - (x_1 - x_0) ||^2]

    where v_theta is the predicted velocity and (x_1 - x_0) is the
    true velocity for optimal transport.
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize loss function.

        Args:
            reduction: How to reduce batch: "mean", "sum", or "none".
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flow matching loss.

        Args:
            predicted_velocity: Model's predicted velocity.
            target_velocity: True velocity (noise - data).

        Returns:
            Loss value.
        """
        # MSE loss
        loss = (predicted_velocity - target_velocity).pow(2)

        # Reduce over all dimensions except batch
        loss = loss.flatten(start_dim=1).mean(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def prepare_training_batch(
    x_0: torch.Tensor,
    schedule: FlowMatchingSchedule,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a batch for training.

    This helper function handles:
    1. Sampling noise
    2. Sampling timesteps
    3. Creating noisy inputs
    4. Computing target velocities

    Args:
        x_0: Clean data of shape (batch, channels, height, width).
        schedule: Flow matching schedule.
        device: Device to use.

    Returns:
        Tuple of (noisy_input, timesteps, noise, target_velocity).
    """
    batch_size = x_0.shape[0]

    # Sample noise
    noise = torch.randn_like(x_0)

    # Sample timesteps
    timesteps = schedule.sample_timesteps(batch_size, device)

    # Create noisy input
    x_t = schedule.add_noise(x_0, noise, timesteps)

    # Compute target velocity
    target_velocity = schedule.get_velocity(x_0, noise)

    return x_t, timesteps, noise, target_velocity


@torch.no_grad()
def sample_euler(
    model: nn.Module,
    noise: torch.Tensor,
    char_indices: torch.Tensor,
    style_embed: torch.Tensor,
    scheduler: FlowMatchingScheduler,
    mask: Optional[torch.Tensor] = None,
    num_steps: Optional[int] = None,
    guidance_scale: float = 1.0,
    save_intermediates: bool = False,
) -> Tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
    """Generate samples using Euler method.

    This is the main sampling function for inference.

    Args:
        model: The GlyphDiffusion model.
        noise: Initial noise of shape (batch, 1, height, width).
        char_indices: Character indices of shape (batch,).
        style_embed: Style embeddings of shape (batch, style_embed_dim).
        scheduler: Flow matching scheduler.
        mask: Optional mask for partial completion.
        num_steps: Number of inference steps.
        guidance_scale: Scale for classifier-free guidance (1.0 = no guidance).
        save_intermediates: Whether to save intermediate steps.

    Returns:
        Tuple of (final samples, list of intermediates if saved).
    """
    device = noise.device

    # Set up timesteps
    scheduler.set_timesteps(num_steps, device)

    # Start with pure noise
    sample = noise
    intermediates = [] if save_intermediates else None

    # Iterate through timesteps
    for i, t in enumerate(scheduler.timesteps[:-1]):  # Skip last (t=0)
        # Get timestep for batch
        timestep = torch.full((noise.shape[0],), t.item(), device=device)

        # Predict velocity
        velocity = model(
            sample,
            timestep,
            char_indices,
            style_embed,
            mask,
        )

        # Classifier-free guidance (if scale > 1)
        if guidance_scale > 1.0:
            # Get unconditional prediction (zero char embedding)
            uncond_velocity = model(
                sample,
                timestep,
                torch.zeros_like(char_indices),
                style_embed,
                mask,
            )
            velocity = uncond_velocity + guidance_scale * (velocity - uncond_velocity)

        # Take Euler step
        sample, pred_x0 = scheduler.step(velocity, i, sample)

        if save_intermediates and pred_x0 is not None:
            intermediates.append(pred_x0)

    return sample, intermediates


if __name__ == "__main__":
    # Test the flow matching schedule
    config = FlowMatchingConfig()
    schedule = FlowMatchingSchedule(config)

    print("Testing FlowMatchingSchedule:")
    print(f"  Num train steps: {schedule.num_train_steps}")
    print(f"  Num inference steps: {schedule.num_inference_steps}")

    # Test add_noise
    batch_size = 4
    x_0 = torch.randn(batch_size, 1, 64, 64)
    noise = torch.randn_like(x_0)
    timesteps = schedule.sample_timesteps(batch_size, torch.device("cpu"))

    print(f"\n  Sampled timesteps: {timesteps}")

    x_t = schedule.add_noise(x_0, noise, timesteps)
    print(f"  x_t shape: {x_t.shape}")

    # Verify interpolation at boundaries
    t_zero = torch.zeros(1)
    t_one = torch.ones(1)

    x_at_0 = schedule.add_noise(x_0[:1], noise[:1], t_zero)
    x_at_1 = schedule.add_noise(x_0[:1], noise[:1], t_one)

    print(f"\n  At t=0, x_t should be close to x_0:")
    print(f"    MSE(x_t, x_0): {((x_at_0 - x_0[:1])**2).mean().item():.6f}")
    print(f"  At t=1, x_t should be close to noise:")
    print(f"    MSE(x_t, noise): {((x_at_1 - noise[:1])**2).mean().item():.6f}")

    # Test velocity
    target_v = schedule.get_velocity(x_0, noise)
    print(f"\n  Target velocity shape: {target_v.shape}")
    print(f"  Velocity = noise - data (constant along path)")

    # Test scheduler
    print("\n\nTesting FlowMatchingScheduler:")
    scheduler = FlowMatchingScheduler(config)
    scheduler.set_timesteps(num_inference_steps=10)
    print(f"  Timesteps: {scheduler.timesteps}")

    # Test step
    model_output = torch.randn_like(x_0)
    prev_sample, pred_x0 = scheduler.step(model_output, step_index=0, sample=x_0)
    print(f"  After step: prev_sample shape = {prev_sample.shape}")

    # Test loss
    print("\n\nTesting FlowMatchingLoss:")
    loss_fn = FlowMatchingLoss()
    pred_v = torch.randn_like(x_0)
    loss = loss_fn(pred_v, target_v)
    print(f"  Loss: {loss.item():.4f}")

    print("\n\nAll tests passed!")
