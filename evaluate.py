"""Evaluate Diffusion Policy on Push-T environment."""

import numpy as np
import torch
from typing import Dict, Any
from tqdm import tqdm

from envs.pusht_env import PushTEnv, make_env
from infer_denoise import DiffusionPolicy, load_policy
from paths import CKPT_PATH, NORMALIZER_PATH


def evaluate(
    num_episodes: int = 50,
    max_steps: int = 200,
    n_action_steps: int = 8,
    n_obs_steps: int = 2,
    device: str = "cuda",
    render: bool = False,
    seed: int = 42,
    num_inference_steps: int = 100,
    use_ddim: bool = True,
) -> Dict[str, Any]:
    env = make_env(render_mode="human" if render else None)
    policy = load_policy(
        device=device,
        num_inference_steps=num_inference_steps,
        use_ddim=use_ddim,
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    successes = 0
    episode_lengths = []
    pos_errors = []
    angle_errors = []

    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        obs = env.reset(seed=seed + ep)
        done = False
        step_count = 0
        obs_buffer = [obs] * n_obs_steps

        while not done and step_count < max_steps:
            obs_stack = np.concatenate(obs_buffer[-n_obs_steps:])
            action_chunk = policy.predict_action(obs_stack)

            for i in range(n_action_steps):
                if step_count >= max_steps:
                    break
                action = action_chunk[i * 2:(i + 1) * 2]
                obs, reward, terminated, truncated, info = env.step(action)
                obs_buffer.append(obs)
                step_count += 1
                done = terminated or truncated
                if terminated:
                    successes += 1
                    break

        episode_lengths.append(step_count)
        pos_errors.append(np.linalg.norm(info["block_pos"] - info["target_pos"]))
        angle_errors.append(abs((info["block_angle"] - info["target_angle"] + np.pi) % (2 * np.pi) - np.pi))

    env.close()

    success_rate = successes / num_episodes
    avg_length = np.mean(episode_lengths)
    avg_pos_error = np.mean(pos_errors)
    avg_angle_error = np.mean(angle_errors)

    results = {
        "success_rate": success_rate,
        "avg_episode_length": avg_length,
        "avg_pos_error": avg_pos_error,
        "avg_angle_error": avg_angle_error,
        "num_episodes": num_episodes,
    }

    print(f"\n=== Evaluation Results ===")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Avg Episode Length: {avg_length:.1f}")
    print(f"Avg Position Error: {avg_pos_error:.2f}")
    print(f"Avg Angle Error: {avg_angle_error:.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--use_ddim", action="store_true", default=True)
    args = parser.parse_args()

    evaluate(
        num_episodes=args.num_episodes,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        use_ddim=args.use_ddim,
    )