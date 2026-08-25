"""
Evaluate a trained PPO policy on PointGoalNav-v0: reports success rate
(reached goal without collision/out-of-bounds/timeout) and mean episode reward
over N deterministic evaluation episodes with fixed, held-out seeds (not the
training seeds).
"""

import argparse

import numpy as np
from stable_baselines3 import PPO

from envs.point_goal_env import PointGoalNavEnv


def evaluate(model_path, n_episodes=100, base_seed=10_000):
    model = PPO.load(model_path)
    successes = 0
    collisions = 0
    out_of_bounds = 0
    timeouts = 0
    rewards = []
    lengths = []

    for ep in range(n_episodes):
        env = PointGoalNavEnv(seed=base_seed + ep)
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        lengths.append(ep_len)
        if info["success"]:
            successes += 1
        elif truncated:
            timeouts += 1
        elif info["dist_to_goal"] < env.OBSTACLE_RADIUS + 0.1:
            collisions += 1
        else:
            out_of_bounds += 1

    print(f"Episodes:            {n_episodes}")
    print(f"Success rate:        {successes / n_episodes:.1%}")
    print(f"Collision rate:      {collisions / n_episodes:.1%}")
    print(f"Out-of-bounds rate:  {out_of_bounds / n_episodes:.1%}")
    print(f"Timeout rate:        {timeouts / n_episodes:.1%}")
    print(f"Mean reward:         {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Mean episode length: {np.mean(lengths):.1f} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_point_goal_nav.zip")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.model, n_episodes=args.episodes)
