import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
ZARR_PATH = os.path.join(DATA_DIR, 'pusht', 'pusht_cchi_v7_replay.zarr')

# Checkpoints
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'diffusion_policy.pt')
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'diffusion_policy_best.pt')
NORMALIZER_PATH = os.path.join(CHECKPOINT_DIR, 'normalizer.npz')

# Outputs
PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots')
VIDEO_DIR = os.path.join(PROJECT_ROOT, 'videos')


def ensure_dirs():
    for directory in (CHECKPOINT_DIR, PLOT_DIR, VIDEO_DIR):
        os.makedirs(directory, exist_ok=True)