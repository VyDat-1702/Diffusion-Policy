import torch
import numpy as np
from typing import List, Optional

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from common.normalizer import LinearNormalizer
from models.unet1d import create_unet1d
from paths import BEST_CKPT_PATH, CKPT_PATH, NORMALIZER_PATH


class DiffusionPolicy:
    """Diffusion Policy for action generation via denoising."""

    def __init__(
        self,
        ckpt_path: str = BEST_CKPT_PATH,
        normalizer_path: str = NORMALIZER_PATH,
        action_dim: int = 2,
        obs_dim: int = 5,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        device: str = "cuda",
        num_inference_steps: int = 100,
        use_ddim: bool = True,
    ):
        self.device = device
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps

        # Load normalizers from checkpoint (contains correct 32-dim normalizer)
        checkpoint = torch.load(ckpt_path or CKPT_PATH, map_location=device, weights_only=False)
        self.obs_normalizer = checkpoint['obs_normalizer']
        self.action_normalizer = checkpoint['action_normalizer']

        # Architecture must match the one used at training time.
        net_config = checkpoint.get('net_config', {}) if isinstance(checkpoint, dict) else {}
        self.horizon = net_config.get('horizon', horizon)
        self.n_obs_steps = net_config.get('n_obs_steps', n_obs_steps)
        self.action_dim = net_config.get('action_dim', action_dim)
        self.obs_dim = net_config.get('obs_dim', obs_dim)

        self.model = create_unet1d(
            action_dim=self.action_dim,
            obs_dim=self.obs_dim,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            obs_as_global_cond=True,
            down_dims=net_config.get('down_dims', [256, 512, 1024]),
            diffusion_step_embed_dim=net_config.get('diffusion_step_embed_dim', 256),
            kernel_size=net_config.get('kernel_size', 5),
            n_groups=net_config.get('n_groups', 8),
            cond_predict_scale=net_config.get('cond_predict_scale', True),
        ).to(device)

        # Load model weights
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Create schedulers
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        )
        
        if use_ddim:
            self.scheduler = DDIMScheduler.from_config(self.train_scheduler.config)
            self.scheduler.set_timesteps(num_inference_steps)
        else:
            self.scheduler = self.train_scheduler
            self.scheduler.set_timesteps(num_inference_steps)

    @torch.no_grad()
    def predict_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Generate action via denoising.
        Args:
            obs: (n_obs_steps * obs_dim,) raw observation
        Returns:
            action: (n_action_steps * action_dim,) raw action
        """
        obs_norm = self.obs_normalizer.normalize(obs.reshape(1, -1))
        obs_tensor = torch.from_numpy(obs_norm).to(self.device, dtype=torch.float32)

        B = 1
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_dim
        To = self.n_obs_steps

        # Initialize noisy trajectory (action only, since obs_as_global_cond=True)
        trajectory = torch.randn(B, T, Da, device=self.device)

        # Global conditioning: flatten first To obs steps
        global_cond = obs_tensor[:, :To * Do].reshape(B, -1)

        # Denoising loop
        for t in self.scheduler.timesteps:
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long).expand(B)

            # Predict noise
            pred_noise = self.model(trajectory, t_tensor, global_cond=global_cond)

            # Scheduler step
            trajectory = self.scheduler.step(pred_noise, t, trajectory).prev_sample

        # Unnormalize full horizon first, then slice
        naction_pred = trajectory[0, :, :Da].cpu().numpy()  # (T, Da)
        naction_pred = naction_pred.reshape(-1)  # (T * Da,)
        action_full = self.action_normalizer.unnormalize(naction_pred)  # (T * Da,)
        action_full = action_full.reshape(T, Da)  # (T, Da)
        
        # Get action steps: start at index 0 (current step), matching dataset
        # where action[t : t+horizon] starts at the current timestep.
        start = 0
        end = start + self.n_action_steps
        action = action_full[start:end]  # (n_action_steps, Da)

        return action.reshape(-1)

    @torch.no_grad()
    def predict_action_trajectory(self, obs: np.ndarray) -> List[np.ndarray]:
        """Generate action and return trajectory of denoising steps."""
        obs_norm = self.obs_normalizer.normalize(obs.reshape(1, -1))
        obs_tensor = torch.from_numpy(obs_norm).to(self.device, dtype=torch.float32)

        B = 1
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_dim
        To = self.n_obs_steps

        trajectory = torch.randn(B, T, Da, device=self.device)
        global_cond = obs_tensor[:, :To * Do].reshape(B, -1)

        traj_list = [trajectory.clone().cpu().numpy()]

        for t in self.scheduler.timesteps:
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long).expand(B)

            pred_noise = self.model(trajectory, t_tensor, global_cond=global_cond)
            trajectory = self.scheduler.step(pred_noise, t, trajectory).prev_sample
            traj_list.append(trajectory.clone().cpu().numpy())

        # Convert to raw actions (unnormalize full horizon then slice)
        result = []
        for traj in traj_list:
            naction = traj[0, :, :Da].reshape(-1)
            action_full = self.action_normalizer.unnormalize(naction).reshape(T, Da)
            start = 0
            end = start + self.n_action_steps
            action = action_full[start:end]
            result.append(action.reshape(-1))
        return result


def load_policy(
    ckpt_path: str = BEST_CKPT_PATH,
    normalizer_path: str = NORMALIZER_PATH,
    device: str = "cuda",
    num_inference_steps: int = 100,
    use_ddim: bool = True,
) -> DiffusionPolicy:
    return DiffusionPolicy(
        ckpt_path, normalizer_path, device=device,
        num_inference_steps=num_inference_steps,
        use_ddim=use_ddim,
    )