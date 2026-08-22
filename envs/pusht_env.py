"""Push-T environment using PyMunk (2D physics)."""

import numpy as np
import pymunk
import pygame
from typing import Tuple, Optional, Dict, Any, List


class PushTEnv:
    """Push-T environment: push a T-shaped block to target position/orientation."""

    def __init__(
        self,
        render_mode: Optional[str] = None,
        window_size: int = 512,
        max_episode_steps: int = 200,
        dt: float = 1.0 / 60.0,
    ):
        self.render_mode = render_mode
        self.window_size = window_size
        self.max_episode_steps = max_episode_steps
        self.dt = dt

        self.agent_radius = 15.0
        self.block_mass = 1.0
        self.block_moment = pymunk.moment_for_box(self.block_mass, (80, 80))
        self.friction = 0.5
        self.agent_force = 1000.0

        self.space = None
        self.agent_body = None
        self.block_body = None
        self.screen = None
        self.clock = None
        self.step_count = 0

        self.target_pos = np.array([256.0, 256.0], dtype=np.float32)
        self.target_angle = 0.0

        # Video recording
        self._frames: List[np.ndarray] = []
        self._recording = False

        self._init_pygame()

    def _init_pygame(self):
        if self.render_mode in ("human", "rgb_array"):
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Push-T")
            else:
                # Offscreen surface for rgb_array
                self.screen = pygame.Surface((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def _create_space(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.1

        walls = [
            pymunk.Segment(self.space.static_body, (0, 0), (self.window_size, 0), 5),
            pymunk.Segment(self.space.static_body, (self.window_size, 0), (self.window_size, self.window_size), 5),
            pymunk.Segment(self.space.static_body, (self.window_size, self.window_size), (0, self.window_size), 5),
            pymunk.Segment(self.space.static_body, (0, self.window_size), (0, 0), 5),
        ]
        for wall in walls:
            wall.friction = self.friction
            wall.elasticity = 0.0
        self.space.add(*walls)

        self.agent_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.agent_body.position = (100, 100)
        agent_shape = pymunk.Circle(self.agent_body, self.agent_radius)
        agent_shape.friction = self.friction
        agent_shape.elasticity = 0.0
        agent_shape.collision_type = 1
        self.space.add(self.agent_body, agent_shape)

        self.block_body = pymunk.Body(self.block_mass, self.block_moment)
        self.block_body.position = (256, 256)
        self.block_body.angle = 0.0
        block_shape = pymunk.Poly.create_box(self.block_body, (80, 80))
        block_shape.friction = self.friction
        block_shape.elasticity = 0.0
        block_shape.collision_type = 2
        self.space.add(self.block_body, block_shape)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        if self.space is not None:
            self.space.remove(*list(self.space.bodies), *list(self.space.shapes), *list(self.space.constraints))

        self._create_space()

        self.agent_body.position = (100, 100)
        self.block_body.position = (256, 256)
        self.block_body.angle = 0.0
        self.block_body.velocity = (0, 0)
        self.block_body.angular_velocity = 0.0

        self.target_pos = np.array([
            np.random.uniform(150, 362),
            np.random.uniform(150, 362)
        ], dtype=np.float32)
        self.target_angle = np.random.uniform(-np.pi, np.pi)

        self.step_count = 0
        self._frames = []
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        agent_pos = np.array(self.agent_body.position, dtype=np.float32)
        block_pos = np.array(self.block_body.position, dtype=np.float32)
        block_angle = np.array([self.block_body.angle], dtype=np.float32)
        return np.concatenate([agent_pos, block_pos, block_angle])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        target_pos = np.clip(action, self.agent_radius, self.window_size - self.agent_radius)
        current_pos = np.array(self.agent_body.position)
        direction = target_pos - current_pos
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            direction = direction / dist
            force = direction * self.agent_force
            self.agent_body.velocity = (float(force[0]), float(force[1]))

        self.space.step(self.dt)
        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_success()
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "block_pos": np.array(self.block_body.position),
            "block_angle": self.block_body.angle,
            "target_pos": self.target_pos,
            "target_angle": self.target_angle,
        }

        # Capture frame if recording
        if self._recording:
            self._capture_frame()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self) -> float:
        block_pos = np.array(self.block_body.position)
        pos_error = np.linalg.norm(block_pos - self.target_pos)
        angle_error = abs(self._normalize_angle(self.block_body.angle - self.target_angle))
        reward = -pos_error - 10.0 * angle_error
        return float(reward)

    def _check_success(self) -> bool:
        block_pos = np.array(self.block_body.position)
        pos_error = np.linalg.norm(block_pos - self.target_pos)
        angle_error = abs(self._normalize_angle(self.block_body.angle - self.target_angle))
        return pos_error < 30.0 and angle_error < 0.3

    def _normalize_angle(self, angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def start_recording(self):
        """Start recording frames for video."""
        self._recording = True
        self._frames = []
        self._capture_frame()  # Capture initial frame

    def stop_recording(self) -> List[np.ndarray]:
        """Stop recording and return frames."""
        self._recording = False
        frames = self._frames
        self._frames = []
        return frames

    def _capture_frame(self):
        """Capture current frame as RGB array."""
        if self.screen is not None:
            # Render current state
            self._render_frame()
            # Convert pygame surface to numpy array
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))  # (H, W, C)
            self._frames.append(frame.copy())

    def _render_frame(self):
        """Render current state to screen surface."""
        self.screen.fill((255, 255, 255))

        block_pos = self.block_body.position
        block_angle = self.block_body.angle
        block_verts = [
            pymunk.Vec2d(-40, -40).rotated(block_angle) + block_pos,
            pymunk.Vec2d(40, -40).rotated(block_angle) + block_pos,
            pymunk.Vec2d(40, 40).rotated(block_angle) + block_pos,
            pymunk.Vec2d(-40, 40).rotated(block_angle) + block_pos,
        ]
        pygame.draw.polygon(self.screen, (200, 100, 100), [(int(v.x), int(v.y)) for v in block_verts])

        target_verts = [
            pymunk.Vec2d(-40, -40).rotated(self.target_angle) + self.target_pos,
            pymunk.Vec2d(40, -40).rotated(self.target_angle) + self.target_pos,
            pymunk.Vec2d(40, 40).rotated(self.target_angle) + self.target_pos,
            pymunk.Vec2d(-40, 40).rotated(self.target_angle) + self.target_pos,
        ]
        pygame.draw.polygon(self.screen, (100, 200, 100), [(int(v.x), int(v.y)) for v in target_verts], 3)

        agent_pos = self.agent_body.position
        pygame.draw.circle(self.screen, (100, 100, 200), (int(agent_pos.x), int(agent_pos.y)), int(self.agent_radius))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)

    def render(self):
        if self.render_mode not in ("human", "rgb_array") or self.screen is None:
            return
        self._render_frame()
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


def make_env(render_mode: Optional[str] = None) -> PushTEnv:
    return PushTEnv(render_mode=render_mode)