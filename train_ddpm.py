"""Train Diffusion Policy with U-Net 1D (matching original repo)."""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from einops import reduce
import copy

from common.noise_schedule import create_noise_scheduler
from common.replay_dataset import PushTReplayDataset
from common.normalizer import LinearNormalizer
from models.unet1d import create_unet1d
from paths import CKPT_PATH, NORMALIZER_PATH, ensure_dirs


class EMAModel:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999):
        self.model = model
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.update_after_step = 0
        self.step = 0

    def update(self, model):
        self.step += 1
        if self.step <= self.update_after_step:
            return
        
        decay = self.max_value - (self.max_value - self.min_value) * (self.step ** -self.power)
        decay = min(decay, self.max_value)
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.mul_(decay).add_(param.data, alpha=1 - decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


def train(
    zarr_path: str,
    epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-4,
    obs_horizon: int = 2,
    horizon: int = 16,
    device: str = "cuda",
    save_every: int = 50,
    use_ema: bool = True,
    lr_warmup_steps: int = 500,
    ckpt_dir: str = None,
    val_split: float = 0.1,
    seed: int = 42,
):
    ensure_dirs()

    dataset = PushTReplayDataset(zarr_path, obs_horizon=obs_horizon, horizon=horizon)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    action_dim = 2
    obs_dim = 5

    obs_normalizer = LinearNormalizer()
    action_normalizer = LinearNormalizer()
    all_obs = []
    all_actions = []
    for obs, action in train_dataset:
        all_obs.append(obs)
        all_actions.append(action)
    obs_normalizer.fit(np.stack(all_obs), dim=obs_dim)
    action_normalizer.fit(np.stack(all_actions), dim=action_dim)

    # Create U-Net 1D matching original repo
    net_config = dict(
        action_dim=action_dim,
        obs_dim=obs_dim,
        horizon=horizon,
        n_obs_steps=obs_horizon,
        obs_as_global_cond=True,
        down_dims=[256, 512, 1024],
        diffusion_step_embed_dim=256,
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    )
    model = create_unet1d(**net_config).to(device)

    print(f"Model: U-Net 1D | Params: {sum(p.numel() for p in model.parameters()):,}")

    # Create noise scheduler (cosine schedule)
    noise_scheduler = create_noise_scheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-8)
    
    # LR scheduler with warmup
    def lr_lambda(step):
        if step < lr_warmup_steps:
            return step / lr_warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - lr_warmup_steps) / (epochs * len(dataloader) - lr_warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema = EMAModel(model, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999) if use_ema else None

    # Checkpoint path
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, 'diffusion_policy.pt')
        best_ckpt_path = os.path.join(ckpt_dir, 'diffusion_policy_best.pt')
    else:
        ckpt_path = CKPT_PATH
        best_ckpt_path = os.path.join(os.path.dirname(CKPT_PATH), 'diffusion_policy_best.pt')

    model.train()
    global_step = 0
    best_val_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0.0
        for obs, action in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", colour="#90ee90"):
            obs = torch.from_numpy(obs_normalizer.normalize(obs)).to(device, dtype=torch.float32)
            action = torch.from_numpy(action_normalizer.normalize(action)).to(device, dtype=torch.float32)

            B = obs.shape[0]
            T = horizon
            Da = action_dim
            Do = obs_dim
            To = obs_horizon

            # Reshape action to (B, T, Da)
            action = action.reshape(B, T, Da)

            # When obs_as_global_cond=True, model only predicts action noise
            trajectory = action  # (B, T, Da)

            # Sample noise for action
            noise = torch.randn_like(trajectory)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

            # Add noise to action
            noisy_trajectory = noise_scheduler.add_noise(trajectory, noise, timesteps)

            # Global conditioning: flatten first To obs steps
            global_cond = obs[:, :To * Do].reshape(B, -1)

            # Predict noise
            pred = model(noisy_trajectory, timesteps, global_cond=global_cond)

            # Target is noise (epsilon prediction)
            target = noise

            # Loss
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if use_ema:
                ema.update(model)

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

        eval_model = ema.ema_model if use_ema else model
        eval_model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for obs, action in val_dataloader:
                obs = torch.from_numpy(obs_normalizer.normalize(obs)).to(device, dtype=torch.float32)
                action = torch.from_numpy(action_normalizer.normalize(action)).to(device, dtype=torch.float32)

                B = obs.shape[0]
                T = horizon
                Da = action_dim
                Do = obs_dim
                To = obs_horizon

                action = action.reshape(B, T, Da)
                trajectory = action
                noise = torch.randn_like(trajectory)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
                noisy_trajectory = noise_scheduler.add_noise(trajectory, noise, timesteps)
                global_cond = obs[:, :To * Do].reshape(B, -1)
                pred = eval_model(noisy_trajectory, timesteps, global_cond=global_cond)
                val_loss_sum += F.mse_loss(pred, noise).item()

        avg_val_loss = val_loss_sum / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        save_dict = {
            'model': eval_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
            'net_config': net_config,
            'obs_normalizer': obs_normalizer,
            'action_normalizer': action_normalizer,
        }

        if (epoch + 1) % save_every == 0:
            epoch_ckpt_path = os.path.join(os.path.dirname(ckpt_path), f"diffusion_policy_epoch_{epoch + 1:04d}.pt")
            torch.save(save_dict, epoch_ckpt_path)
            print(f"Saved checkpoint to {epoch_ckpt_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(save_dict, best_ckpt_path)
            print(f"Saved best checkpoint to {best_ckpt_path}")

    # Final save
    save_dict = {
        'model': eval_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epochs - 1,
        'train_loss': avg_loss,
        'val_loss': avg_val_loss,
        'net_config': net_config,
        'obs_normalizer': obs_normalizer,
        'action_normalizer': action_normalizer,
    }
    torch.save(save_dict, ckpt_path)
    print(f"Final checkpoint saved to {ckpt_path}")

    if not os.path.exists(best_ckpt_path):
        torch.save(save_dict, best_ckpt_path)
        print(f"Best checkpoint saved to {best_ckpt_path}")

        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, default="data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory to save checkpoints (e.g., Google Drive path)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of replay data used for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        zarr_path=args.zarr_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        obs_horizon=args.obs_horizon,
        horizon=args.horizon,
        device=args.device,
        use_ema=args.use_ema,
        lr_warmup_steps=args.lr_warmup_steps,
        ckpt_dir=args.ckpt_dir,
        val_split=args.val_split,
        seed=args.seed,
    )