"""
Reads the per-env Monitor CSV logs written during training and plots a
smoothed episode-reward-vs-timestep curve (and episode length) to
plots/training_curve.png.
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_monitor_logs(log_dir="logs"):
    frames = []
    for path in sorted(glob.glob(os.path.join(log_dir, "monitor_*.monitor.csv"))):
        df = pd.read_csv(path, skiprows=1)
        df["timesteps"] = df["l"].cumsum()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("timesteps")
    return combined


def rolling_mean(x, window=50):
    return pd.Series(x).rolling(window, min_periods=1).mean()


def main():
    df = load_monitor_logs()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(df["timesteps"], df["r"], alpha=0.25, color="tab:blue", linewidth=0.8)
    axes[0].plot(df["timesteps"], rolling_mean(df["r"]), color="tab:blue", linewidth=2)
    axes[0].set_xlabel("Environment timesteps")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("PPO training reward — PointGoalNav-v0")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["timesteps"], df["l"], alpha=0.25, color="tab:orange", linewidth=0.8)
    axes[1].plot(df["timesteps"], rolling_mean(df["l"]), color="tab:orange", linewidth=2)
    axes[1].set_xlabel("Environment timesteps")
    axes[1].set_ylabel("Episode length (steps)")
    axes[1].set_title("Episode length over training")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/training_curve.png", dpi=150)
    print("Saved plots/training_curve.png")


if __name__ == "__main__":
    main()
