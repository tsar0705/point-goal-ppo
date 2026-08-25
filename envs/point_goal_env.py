"""
PointGoalNav-v0
----------------
A minimal custom Gymnasium environment: a 2D point-mass "robot" (double-integrator
dynamics with drag) must navigate from a random start to a random goal while
avoiding one or more circular obstacles.

Built to demonstrate an RL problem formulation end-to-end (state/action design,
dynamics, reward shaping, termination logic, domain randomization) rather than
only using a stock Gymnasium task. Meant as a lightweight stand-in for a
mobile-robot / drone point-to-point navigation problem.

Domain randomization: each episode samples a random number of obstacles
(1-2) and a random radius per obstacle (0.4-0.9), rather than a single fixed
obstacle. This is done specifically so the policy can't just memorize one
obstacle geometry — it has to learn a general "keep clear of things near my
path" behavior, which is the property that actually matters for sim-to-real
transfer.

State (8-dim, all relative/normalized so the policy generalizes across goal
and obstacle placement):
    [pos_x, pos_y, vel_x, vel_y,
     (goal_x - pos_x), (goal_y - pos_y),
     (nearest_obstacle_x - pos_x), (nearest_obstacle_y - pos_y)]
The observation only exposes the *nearest* obstacle's relative position (not
its radius) — the agent has to learn a radius-robust margin behavior rather
than conditioning on exact obstacle size, since real sensors won't always
give a clean radius estimate either.

Action (2-dim, continuous, in [-1, 1]):
    Thrust command along x and y, scaled by MAX_ACCEL.

Reward shaping (this is the part worth being able to defend in an interview):
    - Dense shaping term: negative distance-to-goal delta (reduces distance = positive
      reward), rather than a sparse "+1 on success" signal. Sparse reward was tried
      first and PPO struggled to find the goal within the episode horizon purely by
      random exploration -> switched to potential-based shaping (Ng et al., 1999 style:
      reward = distance_before - distance_after) which is provably policy-invariant
      and fixed convergence.
    - Small action-magnitude penalty, to discourage max-thrust "bang-bang" control and
      encourage smoother trajectories.
    - Obstacle penalty: a soft penalty that grows sharply inside a safety margin around
      the *nearest* obstacle (margin scales with that obstacle's own radius), plus
      episode termination + large penalty on actual collision with any obstacle, so
      the agent learns to route around them rather than clip them.
    - Terminal bonus on reaching the goal radius.

Episode ends on: goal reached (success, terminated), collision with any obstacle
(failure, terminated), or leaving the arena bounds (failure, terminated).
Truncated on a step-count timeout.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PointGoalNavEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 20}

    ARENA_HALF_SIZE = 5.0
    MAX_ACCEL = 3.0
    DRAG = 0.25
    DT = 0.05
    MAX_STEPS = 200

    GOAL_RADIUS = 0.35

    # --- domain randomization ranges ---
    MIN_OBSTACLES = 1
    MAX_OBSTACLES = 2
    OBSTACLE_RADIUS_RANGE = (0.4, 0.75)
    SAFETY_BUFFER = 0.25  # extra soft-penalty margin beyond each obstacle's own radius

    ACTION_PENALTY_WEIGHT = 0.01
    GOAL_BONUS = 15.0
    COLLISION_PENALTY = 15.0
    OUT_OF_BOUNDS_PENALTY = 10.0

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.obstacles = []  # list of (pos: np.ndarray(2,), radius: float)
        self.step_count = 0

    def _sample_point(self, min_dist_from=None, min_dist=1.5):
        for _ in range(50):
            p = self._rng.uniform(-self.ARENA_HALF_SIZE * 0.8,
                                   self.ARENA_HALF_SIZE * 0.8, size=2).astype(np.float32)
            if min_dist_from is None or np.linalg.norm(p - min_dist_from) >= min_dist:
                return p
        return p

    def _nearest_obstacle(self):
        dists = [np.linalg.norm(o_pos - self.pos) - o_r for o_pos, o_r in self.obstacles]
        idx = int(np.argmin(dists))
        return self.obstacles[idx], dists[idx]

    def _get_obs(self):
        (nearest_pos, _nearest_r), _ = self._nearest_obstacle()
        return np.concatenate([
            self.pos, self.vel,
            self.goal - self.pos,
            nearest_pos - self.pos,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.pos = self._sample_point()
        self.goal = self._sample_point(min_dist_from=self.pos, min_dist=3.0)

        n_obstacles = int(self._rng.integers(self.MIN_OBSTACLES, self.MAX_OBSTACLES + 1))
        midpoint = (self.pos + self.goal) / 2.0
        self.obstacles = []
        for _ in range(n_obstacles):
            jitter = self._rng.uniform(-1.1, 1.1, size=2).astype(np.float32)
            o_pos = midpoint + jitter
            o_radius = float(self._rng.uniform(*self.OBSTACLE_RADIUS_RANGE))
            self.obstacles.append((o_pos, o_radius))

        self.vel = np.zeros(2, dtype=np.float32)
        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.step_count += 1

        dist_before = np.linalg.norm(self.goal - self.pos)

        accel = action * self.MAX_ACCEL - self.DRAG * self.vel
        self.vel = self.vel + accel * self.DT
        self.pos = self.pos + self.vel * self.DT

        dist_after = np.linalg.norm(self.goal - self.pos)
        (nearest_pos, nearest_r), nearest_surface_dist = self._nearest_obstacle()
        safety_margin = nearest_r + self.SAFETY_BUFFER

        # --- reward shaping ---
        reward = float(dist_before - dist_after)          # potential-based progress term
        reward -= self.ACTION_PENALTY_WEIGHT * float(np.sum(action ** 2))

        if nearest_surface_dist < safety_margin:
            penetration = safety_margin - nearest_surface_dist
            reward -= 2.0 * penetration  # soft, grows as it gets closer

        terminated = False
        success = False
        collided = any(np.linalg.norm(o_pos - self.pos) < o_r for o_pos, o_r in self.obstacles)

        if dist_after < self.GOAL_RADIUS:
            reward += self.GOAL_BONUS
            terminated = True
            success = True
        elif collided:
            reward -= self.COLLISION_PENALTY
            terminated = True
        elif np.any(np.abs(self.pos) > self.ARENA_HALF_SIZE):
            reward -= self.OUT_OF_BOUNDS_PENALTY
            terminated = True

        truncated = self.step_count >= self.MAX_STEPS

        info = {
            "success": success,
            "dist_to_goal": float(dist_after),
            "collided": bool(collided),
            "n_obstacles": len(self.obstacles),
        }
        return self._get_obs(), reward, terminated, truncated, info


def make_env(seed=None):
    def _init():
        return PointGoalNavEnv(seed=seed)
    return _init

