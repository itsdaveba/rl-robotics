import os
import secrets
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-folder", default="results")
    parser.add_argument("--experiment-id")
    parser.add_argument("--total-timesteps", default=1064960, type=int)
    parser.add_argument("--reward-threshold", default=float("inf"), type=float)
    parser.add_argument("--n-eval-episodes", default=100, type=int)
    parser.add_argument("--context-visible", action="extend", nargs="+", type=int)
    parser.add_argument("--context-kwargs", action="extend", nargs="+")
    parser.add_argument("--context-low", action="extend", nargs="+", type=float)
    parser.add_argument("--context-high", action="extend", nargs="+", type=float)
    parser.add_argument("--context-low-gen", action="extend", nargs="+", type=float)
    parser.add_argument("--context-high-gen", action="extend", nargs="+", type=float)
    args = parser.parse_args()

    output_folder = args.output_folder
    experiment_id = args.experiment_id
    total_timesteps = args.total_timesteps
    reward_threshold = args.reward_threshold
    n_eval_episodes = args.n_eval_episodes
    context_visible = args.context_visible
    context_kwargs = args.context_kwargs
    context_low = args.context_low
    context_high = args.context_high
    context_low_gen = args.context_low_gen
    context_high_gen = args.context_high_gen
    n_envs = 4

    new_experiment = False
    if experiment_id is None:
        experiment_id = secrets.token_hex(4)
        new_experiment = True

    filename = os.path.join(output_folder, experiment_id)
    if new_experiment:
        os.makedirs(filename)
        print(f"[INFO]: Creating experiment-id: {experiment_id}")
        with open(os.path.join(output_folder, "experiments.txt"), "a") as file:
            file.write(f"{experiment_id}\n")
    else:
        print(f"[INFO]: Loading experiment-id: {experiment_id}")

    env_kwargs = dict(initial_spawn=0.5, initial_angle=10.0, act=ActionType.RPYT,
                      context_visible=context_visible, context_kwargs=context_kwargs,
                      context_low=context_low, context_high=context_high,
                      context_low_gen=context_low_gen, context_high_gen=context_high_gen)
    train_env = make_vec_env(HoverAviary, n_envs=n_envs, env_kwargs=env_kwargs)
    eval_env = make_vec_env(HoverAviary, n_envs=n_eval_episodes, env_kwargs=env_kwargs)

    if new_experiment:
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
    else:
        model = PPO.load(os.path.join(filename, "final_model"), train_env, device="cpu")

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        n_eval_episodes=n_eval_episodes,
        eval_freq=8192 // n_envs,
        log_path=os.path.join(filename, "evaluations", str(model.num_timesteps)),
        best_model_save_path=os.path.join(filename, "evaluations", str(model.num_timesteps)),
        verbose=1)

    model.learn(total_timesteps, eval_callback, reset_num_timesteps=False)
    model.save(os.path.join(filename, "final_model"))
