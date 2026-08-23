"""Suy ra target_pos / target_angle thật sự từ dữ liệu demo (pusht_cchi_v7_replay.zarr).

Ý tưởng: nếu toàn bộ demo đều push khối về CÙNG một target cố định, thì
state (block_x, block_y, block_angle) ở CUỐI mỗi episode phải hội tụ quanh
một điểm chung. Script này in ra thống kê đó.

Chạy: python inspect_target.py --zarr_path data/pusht/pusht_cchi_v7_replay.zarr
"""
import argparse
import numpy as np
import zarr


def main(zarr_path):
    root = zarr.open(zarr_path, mode='r')
    state = root['data/state'][:]          # (N, 5) = [agent_x, agent_y, block_x, block_y, block_angle]
    episode_ends = root['meta/episode_ends'][:]

    final_block_pos = []
    final_block_angle = []

    start = 0
    for end in episode_ends:
        end = int(end)
        last_idx = end - 1  # last timestep of this episode
        block_pos = state[last_idx, 2:4]
        block_angle = state[last_idx, 4]
        final_block_pos.append(block_pos)
        final_block_angle.append(block_angle)
        start = end

    final_block_pos = np.array(final_block_pos)      # (num_episodes, 2)
    final_block_angle = np.array(final_block_angle)  # (num_episodes,)

    # Angle is circular -> wrap to (-pi, pi] before stats, and also report
    # mod 2*pi/4 = pi/2 in case the block is a symmetric square (every 90
    # degrees looks the same visually).
    wrapped_angle = (final_block_angle + np.pi) % (2 * np.pi) - np.pi

    print(f"Số episode: {len(episode_ends)}")
    print()
    print("=== Block position lúc kết thúc mỗi episode ===")
    print(f"Mean:   x={final_block_pos[:,0].mean():.1f}, y={final_block_pos[:,1].mean():.1f}")
    print(f"Std:    x={final_block_pos[:,0].std():.1f}, y={final_block_pos[:,1].std():.1f}")
    print(f"Median: x={np.median(final_block_pos[:,0]):.1f}, y={np.median(final_block_pos[:,1]):.1f}")
    print(f"Min/Max x: {final_block_pos[:,0].min():.1f} / {final_block_pos[:,0].max():.1f}")
    print(f"Min/Max y: {final_block_pos[:,1].min():.1f} / {final_block_pos[:,1].max():.1f}")
    print()
    print("=== Block angle lúc kết thúc mỗi episode (rad, đã wrap về [-pi, pi]) ===")
    print(f"Mean:   {wrapped_angle.mean():.3f} rad ({np.degrees(wrapped_angle.mean()):.1f}°)")
    print(f"Std:    {wrapped_angle.std():.3f} rad ({np.degrees(wrapped_angle.std()):.1f}°)")
    print(f"Median: {np.median(wrapped_angle):.3f} rad ({np.degrees(np.median(wrapped_angle)):.1f}°)")
    print()
    print("=== Angle mod 90° (phòng trường hợp khối vuông đối xứng 4 lần) ===")
    mod90 = np.degrees(wrapped_angle) % 90
    print(f"Mean mod 90°:   {mod90.mean():.1f}°")
    print(f"Std mod 90°:    {mod90.std():.1f}°")
    print(f"Median mod 90°: {np.median(mod90):.1f}°")
    print()
    print("Diễn giải:")
    print("- Nếu std position/angle NHỎ (vài px, vài độ) => tất cả demo hội tụ")
    print("  về 1 target cố định -> dùng giá trị Mean/Median ở trên làm target_pos/target_angle.")
    print("- Nếu std LỚN (hàng chục-trăm px, hàng chục độ) => target KHÔNG cố định")
    print("  giữa các episode (có thể random per-episode như agent/target gốc thật),")
    print("  và _check_success() cần nhận target theo từng episode thay vì hard-code 1 giá trị.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, default="data/pusht/pusht_cchi_v7_replay.zarr")
    args = parser.parse_args()
    main(args.zarr_path)
