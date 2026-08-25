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
    by_n_obstacles = {}  # n_obstacles -> [n_episodes, n_successes]

    for ep in range(n_episodes):
        env = PointGoalNavEnv(seed=base_seed + ep)
        obs, _ = env.reset(seed=base_seed + ep)
        n_obs = len(env.obstacles)
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
        by_n_obstacles.setdefault(n_obs, [0, 0])
        by_n_obstacles[n_obs][0] += 1
        if info["success"]:
            successes += 1
            by_n_obstacles[n_obs][1] += 1
        elif info["collided"]:
            collisions += 1
        elif truncated:
            timeouts += 1
        else:
            out_of_bounds += 1

    print(f"Episodes:            {n_episodes}")
    print(f"Success rate:        {successes / n_episodes:.1%}")
    print(f"Collision rate:      {collisions / n_episodes:.1%}")
    print(f"Out-of-bounds rate:  {out_of_bounds / n_episodes:.1%}")
    print(f"Timeout rate:        {timeouts / n_episodes:.1%}")
    print(f"Mean reward:         {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Mean episode length: {np.mean(lengths):.1f} steps")
    print("\nSuccess rate by obstacle count (generalization check):")
    for n_obs in sorted(by_n_obstacles):
        n, s = by_n_obstacles[n_obs]
        print(f"  {n_obs} obstacle(s): {s}/{n} = {s / n:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_point_goal_nav.zip")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.model, n_episodes=args.episodes)
