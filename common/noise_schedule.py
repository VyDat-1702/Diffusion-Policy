"""Noise schedule using diffusers DDPMScheduler (matching original repo)."""

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def create_noise_scheduler(
    num_train_timesteps=100,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="squaredcos_cap_v2",
    variance_type="fixed_small",
    clip_sample=True,
    prediction_type="epsilon",
):
    """Create DDPMScheduler matching original repo config."""
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        variance_type=variance_type,
        clip_sample=clip_sample,
        prediction_type=prediction_type,
    )
    return scheduler


# For backward compatibility
NUM_TIMESTEPS = 100


def get_betas(num_timesteps=NUM_TIMESTEPS, beta_start=0.0001, beta_end=0.02):
    """Get betas from scheduler (for backward compat)."""
    scheduler = create_noise_scheduler(
        num_train_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule="squaredcos_cap_v2",
    )
    return scheduler.betas


def get_alpha_bar(betas):
    """Get cumulative alpha from betas."""
    return torch.cumprod(1.0 - betas, dim=0)


def q_sample(x0, t, noise):
    """Forward diffusion using scheduler."""
    scheduler = create_noise_scheduler()
    return scheduler.add_noise(x0, noise, t)