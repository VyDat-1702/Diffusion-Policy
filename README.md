<div align="center">

<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.x-orange?style=for-the-badge&logo=pytorch&logoColor=white"/>

# Diffusion Policy — Push-T (2D State-Based)

**Denoising Diffusion Policy on Push-T** — re-implementation of [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (Chi et al., RSS 2023) using **Conditional U-Net 1D** with **FiLM conditioning**, **cosine noise schedule**, **EMA**, and **DDIM sampling**.

> 💡 **Key idea:** Instead of regressing actions directly (MSE), Diffusion Policy learns to denoise a noisy action trajectory conditioned on observations. This naturally handles multimodal action distributions — the same state can have multiple valid actions.

</div>

---

## Overview

This repository implements a **state-based Diffusion Policy** on the **Push-T** task (2D block pushing with PyMunk physics). The policy learns to push a block to a target position/orientation by denoising action trajectories.

| Component | Implementation |
|-----------|----------------|
| **Architecture** | Conditional U-Net 1D (down_dims=[256,512,1024], kernel=5, FiLM) |
| **Noise Schedule** | Cosine (squaredcos_cap_v2) via diffusers DDPMScheduler |
| **Conditioning** | Global: flatten first `n_obs_steps` observations (obs_as_global_cond=True) |
| **Training** | Epsilon prediction, MSE loss, EMA (inv_gamma=1.0, power=0.75) |
| **Inference** | DDIM sampler (100 steps), oa_step_convention (start at To-1) |
| **Optimizer** | AdamW lr=1e-4, betas=(0.95,0.999), weight_decay=1e-6, warmup=500 steps |

---

## Diffusion Policy vs MSE Regression

| Aspect | MSE Regression | Diffusion Policy |
|--------|----------------|------------------|
| **Model** | Direct obs → action mapping | Denoising U-Net: noisy action + obs → predicted noise |
| **Training** | MSE(pred_action, expert_action) | MSE(pred_noise, true_noise) |
| **Inference** | Single forward pass | Iterative denoising (T steps) |
| **Multimodal data** | Averages modes → invalid action | Naturally covers modes via diffusion |
| **Speed** | Fast (1 forward) | Slower (T denoising steps) |

> ⚠️ **Note:** MSE collapses multimodal actions to their mean. Diffusion Policy models the full conditional distribution implicitly through the denoising process.

---

## Pipeline

```
Push-T Zarr data ──► PushTReplayDataset (horizon=16, obs_horizon=2) ──► Train U-Net 1D ──► Evaluate / Visualize
```

- **Dataset:** `data/pusht/pusht_cchi_v7_replay.zarr` (from diffusion-policy.cs.columbia.edu)
- **Horizon:** 16 (action steps), **Obs horizon:** 2
- **Normalization:** Min-max to [-1, 1] per dimension (LinearNormalizer)

---

## Installation

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `numpy`, `torch` | Core training & inference |
| `einops` | Tensor rearrangements |
| `diffusers` | DDPM/DDIM schedulers |
| `zarr` | Push-T dataset loading |
| `pygame-ce`, `pymunk` | Push-T 2D physics environment |
| `matplotlib` | Visualization |
| `tqdm` | Progress bars |

---

## Usage

### 1. Download Push-T dataset (once)

```bash
mkdir -p data/pusht
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip -O /tmp/pusht.zip
unzip /tmp/pusht.zip -d data/pusht/
```

### 2. Train (500 epochs, ~20-25h on Colab T4)

```bash
# Local
python train_ddpm.py --epochs 500 --batch_size 256 --device cuda --use_ema

# Colab (checkpoints saved to Google Drive)
python train_ddpm.py --epochs 500 --batch_size 256 --device cuda --use_ema \
    --ckpt_dir /content/drive/MyDrive/diffusion_policy_checkpoints
```

**Key args:**
- `--epochs 500` — full training (original config)
- `--batch_size 256` — fits in 16GB VRAM
- `--use_ema` — exponential moving average (crucial for quality)
- `--ckpt_dir` — custom checkpoint directory (e.g., Google Drive on Colab)

### 3. Evaluate

```bash
# Local
python evaluate.py --num_episodes 50 --device cuda

# Load from custom checkpoint
python evaluate.py --num_episodes 50 --device cuda --ckpt_path checkpoints/diffusion_policy.pt
```

### 4. Visualize

```bash
# Dataset actions + denoising trajectory + action distribution + policy vs dataset
python visualize.py --device cuda --ckpt_path checkpoints/diffusion_policy.pt
```

Outputs saved to `plots/`:
- `dataset_actions.png` — expert action distribution
- `denoising_trajectory.png` — action chunk evolution T→0
- `action_distribution.png` — multimodal action samples
- `policy_vs_dataset.png` — policy vs expert comparison

---

## Colab (Free GPU)

1. Open `diffusion_colab.ipynb` in Colab
2. Runtime → Change runtime type → GPU (T4)
3. Run all cells

The notebook:
1. Mounts Google Drive
2. Clones this repo
3. Installs deps & downloads dataset
4. Trains 500 epochs (checkpoints → Drive)
5. Evaluates & visualizes (loads from Drive)

> 💡 **Tip:** Checkpoints saved to `/content/drive/MyDrive/diffusion_policy_checkpoints/` persist across Colab sessions.

---

## Project Structure

```
diffusion-pilicy/
├── models/                      # Network architectures
│   ├── unet1d.py                # Conditional U-Net 1D (FiLM conditioning)
│   └── noise_pred_net.py        # (legacy) MLP/Transformer noise predictor
├── common/                      # Shared utilities
│   ├── noise_schedule.py        # Cosine schedule via diffusers DDPMScheduler
│   ├── replay_dataset.py        # PushTReplayDataset (horizon=16, obs_horizon=2)
│   └── normalizer.py            # LinearNormalizer (min-max to [-1,1])
├── envs/                        # Environments
│   └── pusht_env.py             # Push-T 2D physics (PyMunk + PyGame)
├── train_ddpm.py                # Training loop (EMA, warmup, cosine LR)
├── infer_denoise.py             # DDIM inference, DiffusionPolicy class
├── evaluate.py                  # Closed-loop rollout evaluation
├── visualize.py                 # Plots: trajectory, distribution, comparison
├── paths.py                     # Central path definitions
├── diffusion_colab.ipynb        # Colab notebook (Drive checkpoints)
├── data/                        # Push-T Zarr dataset
├── checkpoints/                 # Trained models (diffusion_policy.pt)
├── plots/                       # PNG visualizations
├── videos/                      # (reserved for animations)
├── requirements.txt
└── .gitignore
```

---

## References

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) — Chi, Feng, Du, Xu, Cousineau, Burchfiel, Song. RSS 2023.
- [Official repo](https://github.com/real-stanford/diffusion_policy) — real-stanford/diffusion_policy
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho, Jain, Abbeel. NeurIPS 2020.
- [DDIM](https://arxiv.org/abs/2010.02502) — Song, Meng, Ermon. ICLR 2021.

---

<div align="center">

</div>