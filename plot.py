import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ActionType


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

    learning_curve_path = os.path.join(filename, "evaluations.npz")
    if os.path.exists(learning_curve_path):
        with np.load(learning_curve_path) as data:
            timesteps = data["timesteps"]
            results = np.mean(data["results"], axis=1)
            plt.figure(1)
            plt.plot(timesteps, results, marker='o', linestyle='-', markersize=4)
            plt.title("Learning Curve")
            plt.xlabel("Training Steps")
            plt.ylabel("Episode Reward")
            plt.grid(True, alpha=0.6)

    eval_path = os.path.join(filename, "evaluations.csv")
    if os.path.exists(eval_path):
        df = pd.read_csv(os.path.join(filename, "evaluations.csv"))
        plt.figure(2)
        sns.lineplot(df, x="mass", y="episode_rewards")
        plt.title("Context Evaluation")
        plt.grid(True, alpha=0.6)

    flight_data_path = os.path.join(filename, "flight-data-simulation.npz")
    if os.path.exists(flight_data_path):
        env = HoverAviary(act=ActionType.RPYT)
        logger = Logger(logging_freq_hz=int(env.CTRL_FREQ), output_folder=filename)
        logger.load("flight-data-simulation")
        logger.plot()

    flight_data_path = os.path.join(filename, "flight-data.npz")
    if os.path.exists(flight_data_path):
        env = HoverAviary(act=ActionType.RPYT)
        logger = Logger(logging_freq_hz=int(env.CTRL_FREQ), output_folder=filename)
        logger.load("flight-data")
        logger.plot()

    plt.show()
