"""Record demo video of trained Diffusion Policy on Push-T."""

import os
import numpy as np
import torch
import imageio
from tqdm import tqdm

from envs.pusht_env import make_env
from infer_denoise import DiffusionPolicy, load_policy
from paths import VIDEO_DIR


def record_video(
    ckpt_path: str,
    output_path: str = None,
    num_episodes: int = 5,
    max_steps: int = 300,
    fps: int = 10,
    device: str = "cuda",
    num_inference_steps: int = 100,
    use_ddim: bool = True,
    seed: int = 42,
):
    """Record video of policy rollouts."""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    
    if output_path is None:
        output_path = os.path.join(VIDEO_DIR, "demo.mp4")

    # Load policy
    policy = load_policy(
        ckpt_path=ckpt_path,
        device=device,
        num_inference_steps=num_inference_steps,
        use_ddim=use_ddim,
    )

    # Create env with rgb_array render mode for video recording
    env = make_env(render_mode="rgb_array")

    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Recording {num_episodes} episodes to {output_path}...")

    with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
        for ep in tqdm(range(num_episodes), desc="Episodes"):
            obs = env.reset(seed=seed + ep)
            env.start_recording()

            done = False
            step_count = 0
            obs_buffer = [obs] * 2  # n_obs_steps = 2

            while not done and step_count < max_steps:
                obs_stack = np.concatenate(obs_buffer[-2:])
                action_chunk = policy.predict_action(obs_stack)

                for i in range(8):  # n_action_steps = 8
                    if step_count >= max_steps:
                        break
                    action = action_chunk[i * 2:(i + 1) * 2]
                    obs, reward, terminated, truncated, info = env.step(action)
                    obs_buffer.append(obs)
                    step_count += 1
                    done = terminated or truncated
                    if terminated:
                        break

            # Get frames and write to video
            frames = env.stop_recording()
            for frame in frames:
                writer.append_data(frame)

            success = terminated
            pos_err = np.linalg.norm(info["block_pos"] - info["target_pos"])
            print(f"Episode {ep}: {'SUCCESS' if success else 'FAIL'} | coverage={info['coverage']:.3f} | pos_err={pos_err:.1f} | frames={len(frames)}")

    env.close()
    print(f"Video saved to {output_path}")
    return output_path


def record_comparison_video(
    ckpt_path: str,
    output_path: str = None,
    num_episodes: int = 3,
    fps: int = 30,
    device: str = "cuda",
):
    """Record side-by-side comparison: policy vs oracle (expert)."""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    
    if output_path is None:
        output_path = os.path.join(VIDEO_DIR, "comparison.mp4")

    policy = load_policy(ckpt_path=ckpt_path, device=device)
    env = make_env(render_mode="rgb_array")

    print(f"Recording comparison video to {output_path}...")

    with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
        for ep in tqdm(range(num_episodes), desc="Comparison episodes"):
            obs = env.reset(seed=1000 + ep)
            env.start_recording()

            # Run policy
            done = False
            step_count = 0
            obs_buffer = [obs] * 2

            while not done and step_count < 300:
                obs_stack = np.concatenate(obs_buffer[-2:])
                action_chunk = policy.predict_action(obs_stack)

                for i in range(8):
                    if step_count >= 300:
                        break
                    action = action_chunk[i * 2:(i + 1) * 2]
                    obs, reward, terminated, truncated, info = env.step(action)
                    obs_buffer.append(obs)
                    step_count += 1
                    done = terminated or truncated
                    if done:
                        break
            frames = env.stop_recording()
            for frame in frames:
                writer.append_data(frame)

            print(f"Episode {ep}: frames={len(frames)}")
            

    env.close()
    print(f"Comparison video saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--no_ddim", action="store_true", help="Use DDPM sampling instead of DDIM")
    parser.add_argument("--comparison", action="store_true", help="Record comparison video")
    args = parser.parse_args()

    if args.comparison:
        record_comparison_video(
            ckpt_path=args.ckpt_path,
            output_path=args.output,
            num_episodes=args.num_episodes,
            fps=args.fps,
            device=args.device,
        )
    else:
        record_video(
            ckpt_path=args.ckpt_path,
            output_path=args.output,
            num_episodes=args.num_episodes,
            fps=args.fps,
            device=args.device,
            num_inference_steps=args.num_inference_steps,
            use_ddim=not args.no_ddim,
        )