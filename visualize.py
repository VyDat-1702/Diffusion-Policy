import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Optional

from infer_denoise import DiffusionPolicy, load_policy
from common.replay_dataset import PushTReplayDataset
from paths import ZARR_PATH, PLOT_DIR
import os


def visualize_denoising_trajectory(
    policy: DiffusionPolicy,
    obs: np.ndarray,
    save_path: Optional[str] = None,
    num_timesteps_to_show: int = 10,
):
    trajectory = policy.predict_action_trajectory(obs)

    action_horizon = policy.action_horizon
    action_dim = policy.action_dim

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    indices = np.linspace(0, len(trajectory) - 1, num_timesteps_to_show, dtype=int)

    for idx, ax in zip(indices, axes):
        action = trajectory[idx].reshape(action_horizon, action_dim)
        ax.scatter(action[:, 0], action[:, 1], c=range(action_horizon), cmap='viridis', s=50)
        ax.set_xlim(-300, 600)
        ax.set_ylim(-300, 600)
        ax.set_aspect('equal')
        ax.set_title(f"t = {len(trajectory) - 1 - idx}")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Denoising Trajectory: Action Chunk Evolution", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.close()


def visualize_action_distribution(
    policy: DiffusionPolicy,
    obs: np.ndarray,
    num_samples: int = 500,
    save_path: Optional[str] = None,
):
    actions = []
    for _ in range(num_samples):
        action = policy.predict_action(obs)
        actions.append(action[:2])
    actions = np.array(actions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(actions[:, 0], actions[:, 1], alpha=0.5, s=10)
    axes[0].set_xlabel("Action X")
    axes[0].set_ylabel("Action Y")
    axes[0].set_title(f"Action Distribution (n={num_samples})")
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist2d(actions[:, 0], actions[:, 1], bins=30, cmap='Blues')
    axes[1].set_xlabel("Action X")
    axes[1].set_ylabel("Action Y")
    axes[1].set_title("2D Histogram")
    axes[1].set_aspect('equal')

    plt.suptitle("Multimodal Action Distribution", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.close()


def visualize_dataset_actions(
    zarr_path: str = ZARR_PATH,
    save_path: Optional[str] = None,
):
    """Visualize action distribution from dataset."""
    dataset = PushTReplayDataset(zarr_path, obs_horizon=2, action_horizon=8)

    all_actions = []
    for _, action in dataset:
        action = action.reshape(-1, 2)
        all_actions.append(action[0])
    all_actions = np.array(all_actions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(all_actions[:, 0], all_actions[:, 1], alpha=0.3, s=5)
    axes[0].set_xlabel("Action X")
    axes[0].set_ylabel("Action Y")
    axes[0].set_title("Dataset Action Distribution (first step)")
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist2d(all_actions[:, 0], all_actions[:, 1], bins=50, cmap='Blues')
    axes[1].set_xlabel("Action X")
    axes[1].set_ylabel("Action Y")
    axes[1].set_title("2D Histogram")
    axes[1].set_aspect('equal')

    plt.suptitle("Push-T Dataset Actions", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.close()


def visualize_loss_curve(
    loss_history: List[float],
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.close()


def compare_policy_vs_dataset(
    policy: DiffusionPolicy,
    zarr_path: str = ZARR_PATH,
    num_samples: int = 1000,
    save_path: Optional[str] = None,
):
    dataset = PushTReplayDataset(zarr_path, obs_horizon=2, action_horizon=8)

    dataset_actions = []
    for _, action in dataset:
        action = action.reshape(-1, 2)
        dataset_actions.append(action[0])
    dataset_actions = np.array(dataset_actions[:num_samples])

    policy_actions = []
    obs_sample = dataset[0][0]
    for _ in range(num_samples):
        action = policy.predict_action(obs_sample)
        policy_actions.append(action[:2])
    policy_actions = np.array(policy_actions)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(dataset_actions[:, 0], dataset_actions[:, 1], alpha=0.3, s=10, label='Dataset')
    axes[0, 0].set_title("Dataset Actions")
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(policy_actions[:, 0], policy_actions[:, 1], alpha=0.3, s=10, color='orange', label='Policy')
    axes[0, 1].set_title("Policy Actions")
    axes[0, 1].set_aspect('equal')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist2d(dataset_actions[:, 0], dataset_actions[:, 1], bins=30, cmap='Blues')
    axes[1, 0].set_title("Dataset 2D Histogram")
    axes[1, 0].set_aspect('equal')

    axes[1, 1].hist2d(policy_actions[:, 0], policy_actions[:, 1], bins=30, cmap='Oranges')
    axes[1, 1].set_title("Policy 2D Histogram")
    axes[1, 1].set_aspect('equal')

    plt.suptitle("Policy vs Dataset Action Distribution", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint file")
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    visualize_dataset_actions(save_path=os.path.join(PLOT_DIR, "dataset_actions.png"))

    try:
        policy = load_policy(
            device=args.device,
            ckpt_path=args.ckpt_path,
        )
        obs = np.zeros(10)
        visualize_denoising_trajectory(policy, obs, save_path=os.path.join(PLOT_DIR, "denoising_trajectory.png"))
        visualize_action_distribution(policy, obs, save_path=os.path.join(PLOT_DIR, "action_distribution.png"))
        compare_policy_vs_dataset(policy, save_path=os.path.join(PLOT_DIR, "policy_vs_dataset.png"))
    except FileNotFoundError:
        print("Checkpoint not found. Train first.")