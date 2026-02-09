"""Unit tests for FlowMatchingSchedule, FlowMatchingScheduler, and prepare_training_batch.

Tests the flow-matching noise schedule components:
- Linear interpolation (add_noise)
- Velocity computation (get_velocity)
- Timestep sampling (sample_timesteps)
- Euler step (step)
- Scheduler timestep management (set_timesteps)
- Scheduler step (step with index)
- Training batch preparation (prepare_training_batch)
"""

import torch

from models.glyph_diffusion import (
    FlowMatchingSchedule,
    FlowMatchingScheduler,
    prepare_training_batch,
)


class TestFlowMatchingSchedule:
    """Tests for the non-nn.Module FlowMatchingSchedule."""

    def test_add_noise_known_values(self):
        """add_noise returns (1-t)*x_0 + t*noise for known values."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()

        x_0 = torch.ones(2, 1, 4, 4) * 3.0
        noise = torch.ones(2, 1, 4, 4) * 7.0
        t = torch.tensor([0.25, 0.75])

        x_t = schedule.add_noise(x_0, noise, t)

        # For t=0.25: (1-0.25)*3 + 0.25*7 = 2.25 + 1.75 = 4.0
        expected_0 = (1 - 0.25) * 3.0 + 0.25 * 7.0
        assert torch.allclose(x_t[0], torch.full((1, 4, 4), expected_0)), (
            f"Expected {expected_0}, got {x_t[0, 0, 0, 0].item()}"
        )

        # For t=0.75: (1-0.75)*3 + 0.75*7 = 0.75 + 5.25 = 6.0
        expected_1 = (1 - 0.75) * 3.0 + 0.75 * 7.0
        assert torch.allclose(x_t[1], torch.full((1, 4, 4), expected_1)), (
            f"Expected {expected_1}, got {x_t[1, 0, 0, 0].item()}"
        )

    def test_add_noise_at_t0_returns_data(self):
        """At t=0, add_noise should return x_0 (pure data)."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()

        x_0 = torch.randn(1, 1, 8, 8)
        noise = torch.randn(1, 1, 8, 8)
        t = torch.tensor([0.0])

        x_t = schedule.add_noise(x_0, noise, t)
        assert torch.allclose(x_t, x_0, atol=1e-6), "At t=0, x_t should equal x_0"

    def test_add_noise_at_t1_returns_noise(self):
        """At t=1, add_noise should return noise (pure noise)."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()

        x_0 = torch.randn(1, 1, 8, 8)
        noise = torch.randn(1, 1, 8, 8)
        t = torch.tensor([1.0])

        x_t = schedule.add_noise(x_0, noise, t)
        assert torch.allclose(x_t, noise, atol=1e-6), "At t=1, x_t should equal noise"

    def test_get_velocity_returns_noise_minus_data(self):
        """get_velocity returns noise - x_0."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()

        x_0 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        noise = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]])

        velocity = schedule.get_velocity(x_0, noise)
        expected = noise - x_0

        assert torch.allclose(velocity, expected), (
            f"Velocity should be noise - x_0. Got {velocity}, expected {expected}"
        )

    def test_get_velocity_shape(self):
        """get_velocity output shape matches inputs."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()

        x_0 = torch.randn(4, 1, 16, 16)
        noise = torch.randn(4, 1, 16, 16)

        velocity = schedule.get_velocity(x_0, noise)
        assert velocity.shape == x_0.shape

    def test_sample_timesteps_shape(self, device):
        """sample_timesteps returns shape (batch_size,)."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()

        for bs in [1, 4, 16]:
            t = schedule.sample_timesteps(bs, device)
            assert t.shape == (bs,), f"Expected shape ({bs},), got {t.shape}"

    def test_sample_timesteps_range(self, device):
        """sample_timesteps values are in [sigma_min, 1 - sigma_min]."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()

        t = schedule.sample_timesteps(1000, device)
        assert t.min() >= schedule.sigma_min, (
            f"Min timestep {t.min().item()} < sigma_min {schedule.sigma_min}"
        )
        assert t.max() <= 1 - schedule.sigma_min, (
            f"Max timestep {t.max().item()} > 1 - sigma_min {1 - schedule.sigma_min}"
        )

    def test_sample_timesteps_values_in_unit_interval(self, device):
        """All sampled timesteps are in [0, 1]."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()

        t = schedule.sample_timesteps(500, device)
        assert (t >= 0).all() and (t <= 1).all(), "All timesteps should be in [0, 1]"

    def test_step_euler_update(self):
        """step computes x_{t-dt} = x_t - dt * velocity."""
        schedule = FlowMatchingSchedule()

        sample = torch.tensor([[[[10.0]]]])
        velocity = torch.tensor([[[[2.0]]]])
        dt = 0.1
        timestep = 0.5  # Not used in the computation, but required by API

        result = schedule.step(velocity, timestep, sample, dt)

        expected = sample - dt * velocity  # 10.0 - 0.1 * 2.0 = 9.8
        assert torch.allclose(result, expected), (
            f"Expected {expected.item()}, got {result.item()}"
        )

    def test_step_preserves_shape(self):
        """step output has same shape as input sample."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()

        sample = torch.randn(4, 1, 16, 16)
        velocity = torch.randn(4, 1, 16, 16)

        result = schedule.step(velocity, 0.5, sample, 0.1)
        assert result.shape == sample.shape


class TestFlowMatchingScheduler:
    """Tests for the nn.Module FlowMatchingScheduler wrapper."""

    def test_set_timesteps_length(self):
        """set_timesteps creates num_steps + 1 timesteps."""
        scheduler = FlowMatchingScheduler()

        for num_steps in [5, 10, 50]:
            scheduler.set_timesteps(num_steps)
            assert len(scheduler.timesteps) == num_steps + 1, (
                f"Expected {num_steps + 1} timesteps, got {len(scheduler.timesteps)}"
            )

    def test_set_timesteps_monotonically_decreasing(self):
        """Timesteps should be monotonically decreasing from 1 to 0."""
        scheduler = FlowMatchingScheduler()
        scheduler.set_timesteps(20)

        ts = scheduler.timesteps
        # First should be 1.0, last should be 0.0
        assert torch.isclose(ts[0], torch.tensor(1.0)), (
            f"First timestep should be 1.0, got {ts[0].item()}"
        )
        assert torch.isclose(ts[-1], torch.tensor(0.0)), (
            f"Last timestep should be 0.0, got {ts[-1].item()}"
        )

        # Monotonically decreasing
        diffs = ts[1:] - ts[:-1]
        assert (diffs < 0).all(), "Timesteps should be monotonically decreasing"

    def test_set_timesteps_start_and_end(self):
        """Timesteps start at 1.0 and end at 0.0."""
        scheduler = FlowMatchingScheduler()
        scheduler.set_timesteps(10)

        assert torch.isclose(scheduler.timesteps[0], torch.tensor(1.0))
        assert torch.isclose(scheduler.timesteps[-1], torch.tensor(0.0))

    def test_step_euler_integration(self):
        """Scheduler step matches expected Euler integration."""
        scheduler = FlowMatchingScheduler()
        scheduler.set_timesteps(10)

        sample = torch.ones(1, 1, 4, 4) * 5.0
        velocity = torch.ones(1, 1, 4, 4) * 2.0

        # dt = timesteps[0] - timesteps[1] = 1.0 - 0.9 = 0.1
        dt = (scheduler.timesteps[0] - scheduler.timesteps[1]).item()

        prev_sample, pred_x0 = scheduler.step(velocity, 0, sample)

        # Expected: sample - dt * velocity = 5.0 - 0.1 * 2.0 = 4.8
        expected_prev = sample - dt * velocity
        assert torch.allclose(prev_sample, expected_prev, atol=1e-5), (
            f"Expected {expected_prev[0, 0, 0, 0].item()}, "
            f"got {prev_sample[0, 0, 0, 0].item()}"
        )

    def test_step_returns_predicted_original(self):
        """Scheduler step returns pred_original_sample = x_t - t * velocity."""
        scheduler = FlowMatchingScheduler()
        scheduler.set_timesteps(10)

        sample = torch.ones(1, 1, 4, 4) * 5.0
        velocity = torch.ones(1, 1, 4, 4) * 2.0

        _, pred_x0 = scheduler.step(velocity, 0, sample)

        # pred_x0 = sample - t_current * velocity = 5.0 - 1.0 * 2.0 = 3.0
        t_current = scheduler.timesteps[0].item()
        expected_x0 = sample - t_current * velocity
        assert torch.allclose(pred_x0, expected_x0, atol=1e-5)

    def test_step_returns_tuple(self):
        """step returns a (prev_sample, pred_original_sample) tuple."""
        scheduler = FlowMatchingScheduler()
        scheduler.set_timesteps(10)

        sample = torch.randn(2, 1, 8, 8)
        velocity = torch.randn(2, 1, 8, 8)

        result = scheduler.step(velocity, 0, sample)
        assert isinstance(result, tuple) and len(result) == 2


class TestPrepareTrainingBatch:
    """Tests for the prepare_training_batch helper."""

    def test_output_shapes(self, device):
        """All outputs have shapes matching the input batch."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()
        batch_size = 4
        x_0 = torch.randn(batch_size, 1, 16, 16, device=device)

        x_t, timesteps, noise, target_velocity = prepare_training_batch(
            x_0, schedule, device
        )

        assert x_t.shape == x_0.shape, f"x_t shape: {x_t.shape} != {x_0.shape}"
        assert timesteps.shape == (batch_size,), (
            f"timesteps shape: {timesteps.shape} != ({batch_size},)"
        )
        assert noise.shape == x_0.shape, f"noise shape: {noise.shape} != {x_0.shape}"
        assert target_velocity.shape == x_0.shape, (
            f"target_velocity shape: {target_velocity.shape} != {x_0.shape}"
        )

    def test_x_t_is_interpolation(self, device):
        """x_t should be (1-t)*x_0 + t*noise for the returned timesteps."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()
        x_0 = torch.randn(4, 1, 8, 8, device=device)

        x_t, timesteps, noise, _ = prepare_training_batch(x_0, schedule, device)

        # Manually recompute the interpolation
        t = timesteps.view(-1, 1, 1, 1)
        expected_x_t = (1 - t) * x_0 + t * noise

        assert torch.allclose(x_t, expected_x_t, atol=1e-6), (
            "x_t should be linear interpolation of x_0 and noise at given timesteps"
        )

    def test_target_velocity_is_noise_minus_data(self, device):
        """target_velocity should equal noise - x_0."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()
        x_0 = torch.randn(4, 1, 8, 8, device=device)

        _, _, noise, target_velocity = prepare_training_batch(x_0, schedule, device)

        expected = noise - x_0
        assert torch.allclose(target_velocity, expected, atol=1e-6), (
            "target_velocity should equal noise - x_0"
        )

    def test_timesteps_in_valid_range(self, device):
        """Returned timesteps should be in valid [sigma_min, 1-sigma_min] range."""
        torch.manual_seed(42)
        schedule = FlowMatchingSchedule()
        x_0 = torch.randn(32, 1, 8, 8, device=device)

        _, timesteps, _, _ = prepare_training_batch(x_0, schedule, device)

        assert (timesteps >= 0).all() and (timesteps <= 1).all(), (
            "Timesteps should be in [0, 1]"
        )
