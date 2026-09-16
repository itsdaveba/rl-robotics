import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.enums import ActionType
from gym_pybullet_drones.envs.HoverAviary import HoverAviary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-folder", default="results")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--n-eval-episodes", default=100, type=int)
    parser.add_argument("--context-visible", action="extend", nargs="+", type=int)
    parser.add_argument("--context-kwargs", action="extend", nargs="+")
    parser.add_argument("--context-low", action="extend", nargs="+", type=float)
    parser.add_argument("--context-high", action="extend", nargs="+", type=float)
    parser.add_argument("--context-low-gen", action="extend", nargs="+", type=float)
    parser.add_argument("--context-high-gen", action="extend", nargs="+", type=float)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--stop", type=float, required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--kwarg", required=True)
    args = parser.parse_args()

    output_folder = args.output_folder
    experiment_id = args.experiment_id
    n_eval_episodes = args.n_eval_episodes
    context_visible = args.context_visible
    context_kwargs = args.context_kwargs
    context_low = args.context_low
    context_high = args.context_high
    context_low_gen = args.context_low_gen
    context_high_gen = args.context_high_gen
    start = args.start
    stop = args.stop
    step = args.step
    kwarg = args.kwarg

    if experiment_id is None:
        with open(os.path.join(output_folder, "experiments.txt"), "r") as file:
            experiment_id = file.readlines()[-1].strip()

    filename = os.path.join(output_folder, experiment_id)
    print(f"[INFO] Loading experiment-id: {experiment_id}")

    model = PPO.load(os.path.join(filename, "best_model"), device="cpu")
    env_kwargs = dict(initial_spawn=0.5, initial_angle=10.0, act=ActionType.RPYT,
                      context_visible=context_visible, context_kwargs=context_kwargs if context_kwargs else [],
                      context_low=context_low, context_high=context_high,
                      context_low_gen=context_low_gen if context_low_gen else [],
                      context_high_gen=context_high_gen if context_high_gen else [])

    for value in np.arange(start, stop, step):
        env_kwargs["context_kwargs"].append(kwarg)
        env_kwargs["context_low_gen"].append(value)
        env_kwargs["context_high_gen"].append(value)
        env = make_vec_env(HoverAviary, n_envs=5, env_kwargs=env_kwargs)
        print(f"[INFO]: mass={env.envs[0].env.M:.3f} kf={env.envs[0].env.KF:.3e}")
        episode_rewards, _ = evaluate_policy(model, env, n_eval_episodes, True, return_episode_rewards=True)
        df = pd.DataFrame(dict(episode_rewards=episode_rewards, mass=env.envs[0].env.M, kf=env.envs[0].env.KF))
        df.to_csv(os.path.join(filename, "evaluations.csv"), header=not os.path.exists(os.path.join(filename, "evaluations.csv")), index=False, mode="a")
