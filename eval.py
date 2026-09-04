import os
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
    args = parser.parse_args()

    output_folder = args.output_folder
    experiment_id = args.experiment_id

    if experiment_id is None:
        with open(os.path.join(output_folder, "experiments.txt"), "r") as file:
            experiment_id = file.readlines()[-1].strip()

    filename = os.path.join(output_folder, experiment_id)
    print(f"[INFO] Loading experiment-id: {experiment_id}")

    model = PPO.load(os.path.join(filename, "best_model"), device="cpu")

    n_eval_episodes = 100
    env_kwargs = dict(initial_spawn=0.5, target_pos=np.array([0.0, 0.0, 0.5]), act=ActionType.RPM)

    for mass in np.arange(0.017, 0.038, 0.001):
        env_kwargs["mass"] = mass
        env = make_vec_env(HoverAviary, n_envs=5, env_kwargs=env_kwargs)
        episode_rewards, _ = evaluate_policy(model, env, n_eval_episodes, True, return_episode_rewards=True)
        df = pd.DataFrame(dict(episode_rewards=episode_rewards, mass=env.envs[0].env.M))
        df.to_csv(os.path.join(filename, "evaluations.csv"), header=not os.path.exists(os.path.join(filename, "evaluations.csv")), index=False, mode="a")

    df = pd.read_csv(os.path.join(filename, "evaluations.csv"))
    sns.lineplot(df, x="mass", y="episode_rewards")
    plt.show()
