"""Push-T environment (PyMunk 2D physics).

Re-implementation of the environment that produced ``pusht_cchi_v7_replay.zarr``
(Chi et al., RSS 2023):

* T-shaped block (two disjoint convex quads, ``scale=30``, ``length=4``)
* Kinematic agent driven by a PD controller at ``sim_hz=100`` / ``control_hz=10``
  (10 physics sub-steps per action)
* ``space.damping = 0`` -> quasi-static pushing
* Fixed goal pose ``(256, 256, pi/4)``, success when goal coverage > 0.95
* Observation ``[agent_x, agent_y, block_x, block_y, block_angle % 2*pi]``
"""

import numpy as np
import pymunk
import pygame
from typing import Tuple, Optional, Dict, Any, List


def _polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _to_ccw(pts: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    signed = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return pts if signed >= 0 else pts[::-1]


def _edge_intersect(a: np.ndarray, e: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    s = q - p
    denom = e[0] * s[1] - e[1] * s[0]
    if abs(denom) < 1e-12:
        return q
    t = ((p[0] - a[0]) * s[1] - (p[1] - a[1]) * s[0]) / denom
    return a + t * e


def _convex_intersection_area(subject: np.ndarray, clip: np.ndarray) -> float:
    """Area of the intersection of two CONVEX polygons (Sutherland-Hodgman)."""
    output = [row for row in _to_ccw(np.asarray(subject, dtype=np.float64))]
    clip = _to_ccw(np.asarray(clip, dtype=np.float64))
    n = len(clip)
    for i in range(n):
        if not output:
            return 0.0
        a = clip[i]
        e = clip[(i + 1) % n] - a
        inp, output = output, []
        prev = inp[-1]
        prev_side = e[0] * (prev[1] - a[1]) - e[1] * (prev[0] - a[0])
        for cur in inp:
            cur_side = e[0] * (cur[1] - a[1]) - e[1] * (cur[0] - a[0])
            if cur_side >= 0:
                if prev_side < 0:
                    output.append(_edge_intersect(a, e, prev, cur))
                output.append(cur)
            elif prev_side >= 0:
                output.append(_edge_intersect(a, e, prev, cur))
            prev, prev_side = cur, cur_side
    if len(output) < 3:
        return 0.0
    return _polygon_area(np.asarray(output))


class PushTEnv:
    """Push a T-shaped block onto a fixed goal pose."""

    def __init__(
        self,
        render_mode: Optional[str] = None,
        window_size: int = 512,
        max_episode_steps: int = 300,
        success_threshold: float = 0.95,
    ):
        self.render_mode = render_mode
        self.window_size = window_size
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold

        self.sim_hz = 100
        self.control_hz = 10
        self.k_p, self.k_v = 100.0, 20.0
        self.agent_radius = 15.0
        self.block_scale = 30.0
        self.block_length = 4

        self.space = None
        self.agent = None
        self.block = None
        self.screen = None
        self.clock = None
        self.step_count = 0

        self.goal_pose = np.array([256.0, 256.0, np.pi / 4], dtype=np.float64)
        self.target_pos = self.goal_pose[:2].copy()
        self.target_angle = float(self.goal_pose[2])

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

    def _add_tee(self, position, angle):
        scale, length = self.block_scale, self.block_length
        mass = 1.0
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        # The reference implementation passes vertices1 here too; replicated
        # on purpose so the rotational dynamics match the demo dataset.
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        self.space.add(body, shape1, shape2)
        return body

    def _create_space(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0

        ws = self.window_size
        walls = [
            pymunk.Segment(self.space.static_body, (5, ws - 6), (5, 5), 2),
            pymunk.Segment(self.space.static_body, (5, 5), (ws - 6, 5), 2),
            pymunk.Segment(self.space.static_body, (ws - 6, 5), (ws - 6, ws - 6), 2),
            pymunk.Segment(self.space.static_body, (5, ws - 6), (ws - 6, ws - 6), 2),
        ]
        self.space.add(*walls)

        self.agent = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.agent.position = (256, 400)
        agent_shape = pymunk.Circle(self.agent, self.agent_radius)
        self.space.add(self.agent, agent_shape)

        self.block = self._add_tee((256, 300), 0.0)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._create_space()

        # Same sampling ranges as the demo collection script, so evaluation
        # states stay inside the training distribution.
        rs = np.random.RandomState(seed=seed)
        state = np.array([
            rs.randint(50, 450), rs.randint(50, 450),
            rs.randint(100, 400), rs.randint(100, 400),
            rs.randn() * 2 * np.pi - np.pi,
        ], dtype=np.float64)

        self.agent.position = (float(state[0]), float(state[1]))
        self.agent.velocity = (0.0, 0.0)
        # Angle first: rotation happens about the center of gravity, which
        # also shifts the geometric position.
        self.block.angle = float(state[4])
        self.block.position = (float(state[2]), float(state[3]))
        self.block.velocity = (0.0, 0.0)
        self.block.angular_velocity = 0.0
        self.space.step(1.0 / self.sim_hz)

        self.step_count = 0
        self._frames = []
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.array(
            tuple(self.agent.position)
            + tuple(self.block.position)
            + (self.block.angle % (2 * np.pi),),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        dt = 1.0 / self.sim_hz
        n_steps = self.sim_hz // self.control_hz

        for _ in range(n_steps):
            agent_pos = np.array(self.agent.position)
            agent_vel = np.array(self.agent.velocity)
            acceleration = self.k_p * (action - agent_pos) + self.k_v * (-agent_vel)
            new_vel = agent_vel + acceleration * dt
            self.agent.velocity = (float(new_vel[0]), float(new_vel[1]))
            self.space.step(dt)

        self.step_count += 1

        coverage = self._compute_coverage()
        reward = float(np.clip(coverage / self.success_threshold, 0.0, 1.0))
        terminated = coverage > self.success_threshold
        truncated = self.step_count >= self.max_episode_steps

        obs = self._get_obs()
        info = self.get_info(coverage)

        # Capture frame if recording
        if self._recording:
            self._capture_frame()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def get_info(self, coverage: Optional[float] = None) -> Dict[str, Any]:
        if coverage is None:
            coverage = self._compute_coverage()
        return {
            "block_pos": np.array(self.block.position),
            "block_angle": self.block.angle % (2 * np.pi),
            "target_pos": self.target_pos,
            "target_angle": self.target_angle,
            "coverage": coverage,
        }

    def _body_polys(self, body, shapes=None) -> List[np.ndarray]:
        shapes = body.shapes if shapes is None else shapes
        polys = []
        for shape in shapes:
            if not isinstance(shape, pymunk.Poly):
                continue
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            polys.append(np.array([[v.x, v.y] for v in verts], dtype=np.float64))
        return polys

    def _block_polys(self) -> List[np.ndarray]:
        return self._body_polys(self.block)

    def _goal_polys(self) -> List[np.ndarray]:
        goal_body = pymunk.Body(1, 1)
        goal_body.position = (float(self.goal_pose[0]), float(self.goal_pose[1]))
        goal_body.angle = float(self.goal_pose[2])
        return self._body_polys(goal_body, shapes=self.block.shapes)

    def _compute_coverage(self) -> float:
        goal_polys = self._goal_polys()
        block_polys = self._block_polys()
        goal_area = sum(_polygon_area(p) for p in goal_polys)
        if goal_area <= 0:
            return 0.0
        # Goal and block are each a union of two disjoint convex quads, so the
        # union intersection is exactly the sum of the pairwise intersections.
        inter = sum(
            _convex_intersection_area(g, b)
            for g in goal_polys
            for b in block_polys
        )
        return float(inter / goal_area)

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

        for poly in self._goal_polys():
            pygame.draw.polygon(self.screen, (144, 238, 144), [(int(x), int(y)) for x, y in poly])
        for poly in self._block_polys():
            pygame.draw.polygon(self.screen, (119, 136, 153), [(int(x), int(y)) for x, y in poly])

        agent_pos = self.agent.position
        pygame.draw.circle(self.screen, (65, 105, 225), (int(agent_pos.x), int(agent_pos.y)), int(self.agent_radius))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.control_hz)

    def render(self):
        if self.render_mode not in ("human", "rgb_array") or self.screen is None:
            return
        self._render_frame()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


def make_env(render_mode: Optional[str] = None) -> PushTEnv:
    return PushTEnv(render_mode=render_mode)