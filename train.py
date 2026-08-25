"""
Train PPO (Stable-Baselines3) on the custom PointGoalNav-v0 environment.

Usage:
    python train.py --timesteps 300000

Logs per-episode reward/length to logs/monitor.csv (via SB3's Monitor wrapper)
and full TensorBoard scalars (policy loss, value loss, entropy, approx KL,
clip fraction, explained variance) to logs/tb/.
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.point_goal_env import PointGoalNavEnv

LOG_DIR = "logs"
MODEL_DIR = "models"


def build_env(rank, log_dir):
    def _init():
        env = PointGoalNavEnv(seed=rank)
        env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}"))
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "tb"), exist_ok=True)

    env = DummyVecEnv([build_env(i, LOG_DIR) for i in range(args.n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, "tb"),
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(os.path.join(MODEL_DIR, "ppo_point_goal_nav"))
    print(f"Saved model to {MODEL_DIR}/ppo_point_goal_nav.zip")


if __name__ == "__main__":
    main()
