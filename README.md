# PointGoalNav-PPO

A custom Gymnasium environment — a 2D point-robot navigating to a goal while
avoiding a static obstacle — trained with PPO (Stable-Baselines3). Built as a
from-scratch RL problem formulation exercise: designing the state/action
space, dynamics, and reward shaping myself, rather than only running a stock
Gymnasium task.

## The task

A point-mass robot (double-integrator dynamics with drag) starts at a random
position and must reach a randomly placed goal, with a static circular
obstacle placed roughly on the direct path between them — so the agent
actually has to learn to route around it, not just decelerate.

- **Observation (8-dim):** position, velocity, vector to goal, vector to obstacle
  (relative vectors so the policy generalizes across goal/obstacle placement
  rather than memorizing fixed coordinates).
- **Action (2-dim, continuous):** thrust command in x/y.
- **Episode ends on:** reaching the goal (success), hitting the obstacle,
  leaving the arena bounds, or a 200-step timeout.

See `envs/point_goal_env.py` for the full dynamics and reward implementation.

## Reward design (and why it changed)

I did not start with the final reward function. First pass was sparse:
`+1` for reaching the goal, `0` otherwise. With a 200-step horizon and a
continuous action space, PPO's entropy-driven exploration essentially never
stumbled into the goal, so there was no learning signal — reward stayed flat
at ~0 for the first several thousand episodes.

Fix: switched to **potential-based reward shaping**
(`reward = distance_before − distance_after` each step), which is
provably policy-invariant (Ng, Harada & Russell, 1999) — it doesn't change
the optimal policy, it just gives the agent a gradient to climb every step
instead of only at episode end. On top of that:

- small action-magnitude penalty, to discourage bang-bang thrust and
  encourage smoother trajectories,
- a soft penalty that grows as the agent enters the obstacle's safety
  margin (before actual collision), so it learns to steer around early
  rather than react late,
- a hard collision penalty + episode termination on actual contact,
- a terminal bonus on reaching the goal.

This is the debugging story I'd expect to be asked about in interview: it's
the difference between "I called `env.step()` in a loop" and actually
understanding why a reward function does or doesn't produce a learning
signal.

## Results

Trained PPO for 300k timesteps (4 parallel envs, ~76s on a single CPU core).

![training curve](plots/training_curve.png)

Evaluated over 200 held-out episodes (unseen start/goal/obstacle seeds,
deterministic policy):

| Metric | Value |
|---|---|
| Success rate | 87.5% |
| Collision rate | 0.0% |
| Out-of-bounds rate | 12.0% |
| Mean episode reward | 14.6 ± 11.6 |
| Mean episode length | 50.5 steps |

Zero collisions across 200 evaluation episodes — the obstacle-avoidance
shaping term worked. Most failures are out-of-bounds (agent overshoots the
arena on goals near the edge), which is the obvious next thing I'd fix with
a boundary-proximity penalty.

## PPO hyperparameters

`n_steps=1024, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
clip_range=0.2, ent_coef=0.005, lr=3e-4` — see `train.py`. TensorBoard logs
(policy loss, value loss, entropy, approx KL, clip fraction, explained
variance) are in `logs/tb/`.

## Reproduce

```bash
pip install -r requirements.txt
python train.py --timesteps 300000 --n-envs 4
python plot_training_curve.py
python evaluate.py --episodes 200
```

## What I'd add next

- Domain randomization over obstacle count/size to test generalization.
- Swap the hand-written reward for a comparison against a purely sparse
  reward + curiosity/RND exploration bonus, to see if shaping is actually
  necessary or just faster.
- Port the trained policy into a ROS 2 node (`geometry_msgs/Twist` output)
  and test against a Gazebo or MuJoCo version of the same task, since this
  environment intentionally mirrors the structure of a real mobile-robot
  point-to-point navigation problem.
