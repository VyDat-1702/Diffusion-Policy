import numpy as np
import zarr
from torch.utils.data import Dataset


class PushTReplayDataset(Dataset):

    def __init__(self, zarr_path, obs_horizon=2, horizon=16):
        self.obs_horizon = obs_horizon
        self.horizon = horizon

        root = zarr.open(zarr_path, mode='r')
        self.state = root['data/state'][:]
        self.action = root['data/action'][:]
        self.episode_ends = root['meta/episode_ends'][:]

        self.indices = self._build_indices()

    def _build_indices(self):
        indices = []
        num_episodes = len(self.episode_ends)
        for episode_idx in range(num_episodes):
            episode_start = int(self.episode_ends[episode_idx - 1]) if episode_idx > 0 else 0
            episode_end = int(self.episode_ends[episode_idx])
            for t in range(episode_start, episode_end - self.horizon + 1):
                indices.append((episode_idx, t))
        return indices

    def __len__(self):
        """Total number of valid start positions across all episodes."""
        return len(self.indices)

    def __getitem__(self, idx):
        episode_idx, t = self.indices[idx]
        episode_start = int(self.episode_ends[episode_idx - 1]) if episode_idx > 0 else 0
        obs_idx = np.arange(t - self.obs_horizon + 1, t + 1)
        obs_idx = np.clip(obs_idx, episode_start, t)
        obs = self.state[obs_idx].reshape(-1)
        action = self.action[t:t + self.horizon].reshape(-1)
        return obs, action