import os
import argparse
import numpy as np
import pandas as pd

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

    learning_curve_dirs = []
    for name in os.listdir(os.path.join(filename, "evaluations")):
        if os.path.isdir(os.path.join(filename, "evaluations", name)):
            learning_curve_dirs.append(int(name))
    learning_curve_dirs.sort()

    best_result = -np.inf
    best_dir = None
    for learning_curve_dir in learning_curve_dirs:
        learning_curve_path = os.path.join(filename, "evaluations", str(learning_curve_dir), "evaluations.npz")
        with np.load(learning_curve_path) as data:
            timesteps = data["timesteps"]
            results = np.mean(data["results"], axis=1)
        if results.max() > best_result:
            best_result = results.max()
            best_dir = learning_curve_dir

    model = PPO.load(os.path.join(filename, "evaluations", str(best_dir), "best_model"), device="cpu")
    env_kwargs = dict(initial_spawn=0.5, initial_angle=10.0, act=ActionType.RPYT,
                      context_visible=context_visible, context_kwargs=context_kwargs if context_kwargs else [],
                      context_low=context_low, context_high=context_high,
                      context_low_gen=context_low_gen if context_low_gen else [],
                      context_high_gen=context_high_gen if context_high_gen else [])

    for value in np.arange(start, stop, step):
        env_kwargs["context_kwargs"].append(kwarg)
        env_kwargs["context_low_gen"].append(value)
        env_kwargs["context_high_gen"].append(value)
        vec_env = make_vec_env(HoverAviary, n_envs=5, env_kwargs=env_kwargs)
        env = vec_env.envs[0].env
        print(f"[INFO]: mass={env.M[0]:.3f} kf={env.KF[0]:.3e}")
        episode_rewards, _ = evaluate_policy(model, vec_env, n_eval_episodes, True, return_episode_rewards=True)
        df = pd.DataFrame(dict(episode_rewards=episode_rewards, mass=env.M[0], kf=env.KF[0]))
        df.to_csv(os.path.join(filename, "evaluations.csv"), header=not os.path.exists(os.path.join(filename, "evaluations.csv")), index=False, mode="a")
