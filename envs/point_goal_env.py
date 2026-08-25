"""
PointGoalNav-v0
----------------
A minimal custom Gymnasium environment: a 2D point-mass "robot" (double-integrator
dynamics with drag) must navigate from a random start to a random goal while
avoiding a static circular obstacle.

Built to demonstrate an RL problem formulation end-to-end (state/action design,
dynamics, reward shaping, termination logic) rather than only using a stock
Gymnasium task. Meant as a lightweight stand-in for a mobile-robot / drone
point-to-point navigation problem.

State (8-dim, all relative/normalized so the policy generalizes across goal
and obstacle placements):
    [pos_x, pos_y, vel_x, vel_y,
     (goal_x - pos_x), (goal_y - pos_y),
     (obstacle_x - pos_x), (obstacle_y - pos_y)]

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
      the obstacle, plus episode termination + large penalty on actual collision, so
      the agent learns to route around it rather than clip it.
    - Terminal bonus on reaching the goal radius.

Episode ends on: goal reached (success, terminated), collision (failure, terminated),
or leaving the arena bounds (failure, terminated). Truncated on a step-count timeout.
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
    OBSTACLE_RADIUS = 0.6
    OBSTACLE_SAFETY_MARGIN = 0.9  # soft penalty kicks in inside this radius

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
        self.obstacle = np.zeros(2, dtype=np.float32)
        self.step_count = 0

    def _sample_point(self, min_dist_from=None, min_dist=1.5):
        for _ in range(50):
            p = self._rng.uniform(-self.ARENA_HALF_SIZE * 0.8,
                                   self.ARENA_HALF_SIZE * 0.8, size=2).astype(np.float32)
            if min_dist_from is None or np.linalg.norm(p - min_dist_from) >= min_dist:
                return p
        return p

    def _get_obs(self):
        return np.concatenate([
            self.pos, self.vel,
            self.goal - self.pos,
            self.obstacle - self.pos,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.pos = self._sample_point()
        self.goal = self._sample_point(min_dist_from=self.pos, min_dist=3.0)
        # place obstacle roughly between start and goal so avoidance is actually required
        midpoint = (self.pos + self.goal) / 2.0
        jitter = self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self.obstacle = midpoint + jitter
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
        obstacle_dist = np.linalg.norm(self.obstacle - self.pos)

        # --- reward shaping ---
        reward = float(dist_before - dist_after)          # potential-based progress term
        reward -= self.ACTION_PENALTY_WEIGHT * float(np.sum(action ** 2))

        if obstacle_dist < self.OBSTACLE_SAFETY_MARGIN:
            penetration = self.OBSTACLE_SAFETY_MARGIN - obstacle_dist
            reward -= 2.0 * penetration  # soft, grows as it gets closer

        terminated = False
        success = False

        if dist_after < self.GOAL_RADIUS:
            reward += self.GOAL_BONUS
            terminated = True
            success = True
        elif obstacle_dist < self.OBSTACLE_RADIUS:
            reward -= self.COLLISION_PENALTY
            terminated = True
        elif np.any(np.abs(self.pos) > self.ARENA_HALF_SIZE):
            reward -= self.OUT_OF_BOUNDS_PENALTY
            terminated = True

        truncated = self.step_count >= self.MAX_STEPS

        info = {"success": success, "dist_to_goal": float(dist_after)}
        return self._get_obs(), reward, terminated, truncated, info


def make_env(seed=None):
    def _init():
        return PointGoalNavEnv(seed=seed)
    return _init
