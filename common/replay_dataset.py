"""PyTorch Dataset over the PushT zarr replay buffer (matching original repo)."""

import numpy as np
import zarr
from torch.utils.data import Dataset


class PushTReplayDataset(Dataset):
    """Dataset of (obs, action) chunks from a PushT zarr replay buffer.

    Each sample returns RAW (unnormalized) chunks:
        obs:    state[t - obs_horizon + 1 : t + 1] flattened -> (obs_horizon * state_dim,)
        action: action[t : t + horizon] flattened   -> (horizon * action_dim,)

    Action chunks never cross episode boundaries. Obs uses wrap-around
    (fancy) indexing at episode starts so the window always has the full
    obs_horizon rows, matching the reference diffusion-policy dataset.
    """

    def __init__(self, zarr_path, obs_horizon=2, horizon=16):
        self.obs_horizon = obs_horizon
        self.horizon = horizon

        root = zarr.open(zarr_path, mode='r')
        self.state = root['data/state'][:]
        self.action = root['data/action'][:]
        self.episode_ends = root['meta/episode_ends'][:]

        self.indices = self._build_indices()

    def _build_indices(self):
        """Map dataset index -> (episode_idx, start_idx) for every valid start position.

        A start t is valid in an episode when the action chunk
        action[t : t + horizon] stays inside the episode.
        """
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
        """Return (obs, action) raw chunks for the sample at index idx.

        Observation padding at episode starts uses REPEAT padding (the first
        frame of the episode is repeated), matching the reference
        diffusion-policy dataset. No wrap-around (negative index) is used.
        """
        episode_idx, t = self.indices[idx]
        episode_start = int(self.episode_ends[episode_idx - 1]) if episode_idx > 0 else 0
        obs_idx = np.arange(t - self.obs_horizon + 1, t + 1)
        obs_idx = np.clip(obs_idx, episode_start, t)
        obs = self.state[obs_idx].reshape(-1)
        action = self.action[t:t + self.horizon].reshape(-1)
        return obs, action