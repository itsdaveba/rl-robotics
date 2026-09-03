import os
import argparse
import numpy as np

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

    n_eval_episodes = 30
    target_pos = np.array([0.0, 0.0, 0.5])
    env_kwargs = dict(gui=False, num_drones=1, initial_spawn=0.5, target_pos=target_pos, act=ActionType.RPM)
    env = make_vec_env(HoverAviary, n_envs=n_eval_episodes, env_kwargs=env_kwargs)

    episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes, True, return_episode_rewards=True)
    print(episode_rewards, episode_lengths)
