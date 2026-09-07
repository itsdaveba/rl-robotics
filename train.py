import os
import secrets
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-folder", default="results")
    args = parser.parse_args()

    output_folder = args.output_folder
    n_eval_episodes = 5
    n_envs = 4

    experiment_id = secrets.token_hex(4)
    filename = os.path.join(output_folder, experiment_id)
    os.makedirs(filename)

    with open(os.path.join(output_folder, "experiments.txt"), "a") as file:
        file.write(f"{experiment_id}\n")

    env_kwargs = dict(initial_spawn=0.3, target_pos=np.array([0.0, 0.0, 0.3]), act=ActionType.RPYT)
    train_env = make_vec_env(HoverAviary, n_envs=n_envs, env_kwargs=env_kwargs)
    eval_env = make_vec_env(HoverAviary, n_envs=n_eval_episodes, env_kwargs=env_kwargs)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        device="cpu")

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=n_eval_episodes,
        eval_freq=10000 // n_envs,
        log_path=filename,
        best_model_save_path=filename,
        verbose=1)

    model.learn(int(1e6), eval_callback)
    model.save(os.path.join(filename, "final_model"))
