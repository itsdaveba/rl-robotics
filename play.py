import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ActionType
from gym_pybullet_drones.envs.HoverAviary import HoverAviary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-folder", default="results")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--visible-context", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    output_folder = args.output_folder
    experiment_id = args.experiment_id
    visible_context = args.visible_context
    gui = args.gui
    save = args.save
    plot = args.plot

    if experiment_id is None:
        with open(os.path.join(output_folder, "experiments.txt"), "r") as file:
            experiment_id = file.readlines()[-1].strip()

    filename = os.path.join(output_folder, experiment_id)
    print(f"[INFO] Loading experiment-id: {experiment_id}")

    model = PPO.load(os.path.join(filename, "best_model"), device="cpu")
    env_kwargs = dict(gui=gui, num_drones=5, initial_spawn=0.5, act=ActionType.RPYT, visible_context=visible_context)
    env = HoverAviary(**env_kwargs)
    obs, _ = env.reset()
    start = time.time()
    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ), output_folder=filename, num_drones=env.NUM_DRONES)

    for i in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
        action, _ = model.predict(np.expand_dims(obs, axis=1), deterministic=True)
        action = np.squeeze(action)
        obs, reward, _, _, _ = env.step(action)
        for j in range(env.NUM_DRONES):
            logger.log(drone=j, timestamp=i / env.CTRL_FREQ, action=action[j], obs=obs[j][:12], reward=reward[j])
        env.render()
        sync(i, start, env.CTRL_TIMESTEP)

    env.close()

    if save:
        logger.save("simulation")

    if plot:
        logger.plot(show=True)
