"""Noise prediction networks for Diffusion Policy (state-based)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class NoisePredMLP(nn.Module):
    """MLP that predicts noise given (noisy_action, state, timestep)."""

    def __init__(
        self,
        action_dim: int = 16,
        state_dim: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_emb_dim: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        input_dim = action_dim + state_dim + time_emb_dim
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Mish())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Mish())
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        noisy_action: torch.Tensor,
        state: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_action: (B, action_dim) - noisy action at timestep t
            state: (B, state_dim) - conditioning state
            timestep: (B,) - timestep indices
        Returns:
            predicted_noise: (B, action_dim)
        """
        t_emb = self.time_mlp(timestep)
        x = torch.cat([noisy_action, state, t_emb], dim=-1)
        return self.net(x)


class NoisePredTransformer(nn.Module):
    """Transformer encoder that predicts noise given (noisy_action, state, timestep).
    
    Treats (noisy_action, state) as a sequence of tokens for self-attention.
    More expressive than MLP for modeling temporal dependencies in action chunks.
    """

    def __init__(
        self,
        action_dim: int = 16,
        state_dim: int = 10,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        time_emb_dim: int = 64,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.d_model = d_model
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, d_model),
        )

        self.action_proj = nn.Linear(action_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(d_model, action_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        noisy_action: torch.Tensor,
        state: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_action: (B, action_dim) - noisy action at timestep t
            state: (B, state_dim) - conditioning state
            timestep: (B,) - timestep indices
        Returns:
            predicted_noise: (B, action_dim)
        """
        B = noisy_action.shape[0]

        action_tokens = self.action_proj(noisy_action).unsqueeze(1)
        state_tokens = self.state_proj(state).unsqueeze(1)

        t_emb = self.time_mlp(timestep).unsqueeze(1)

        tokens = torch.cat([action_tokens, state_tokens], dim=1)
        seq_len = tokens.shape[1]

        tokens = tokens + self.pos_emb[:, :seq_len, :]
        tokens = tokens + t_emb

        tokens = self.transformer(tokens)

        action_out = tokens[:, 0, :]
        return self.out_proj(action_out)


def create_model(
    model_type: str = "mlp",
    action_horizon: int = 8,
    action_dim: int = 2,
    obs_horizon: int = 2,
    state_dim: int = 5,
    hidden_dim: int = 256,
    num_layers: int = 4,
    time_emb_dim: int = 64,
    # Transformer specific
    d_model: int = 256,
    nhead: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory function to create model with flattened dimensions."""
    flat_action_dim = action_horizon * action_dim
    flat_state_dim = obs_horizon * state_dim

    if model_type == "mlp":
        return NoisePredMLP(
            action_dim=flat_action_dim,
            state_dim=flat_state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_emb_dim=time_emb_dim,
        )
    elif model_type == "transformer":
        return NoisePredTransformer(
            action_dim=flat_action_dim,
            state_dim=flat_state_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            time_emb_dim=time_emb_dim,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'mlp' or 'transformer'")