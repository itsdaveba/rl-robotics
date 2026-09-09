import os
import sys
import time
import logging
import argparse
from threading import Event

import numpy as np
from stable_baselines3 import PPO
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ActionType

import cflib.crtp
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils.encoding import decompress_quaternion
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander


URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")

DEFAULT_HEIGHT = 0.5

deck_attached_event = Event()
flightmode_event = Event()

logging.basicConfig(level=logging.ERROR)

observation = np.zeros(12)


def log_pose_callback(timestamp, data, logconf):
    pos = np.array([data["stateEstimateZ.x"], data["stateEstimateZ.y"], data["stateEstimateZ.z"]]) / 1000
    quat = data["stateEstimateZ.quat"]
    rot = Rotation.from_quat(decompress_quaternion(quat))
    rpy = rot.as_euler(seq="xyz")
    vel = np.array([data["stateEstimateZ.vx"], data["stateEstimateZ.vy"], data["stateEstimateZ.vz"]]) / 1000
    rpy_rate = np.array([data["stateEstimateZ.rateRoll"], data["stateEstimateZ.ratePitch"], data["stateEstimateZ.rateYaw"]]) / 1000
    rpy_rate[1] = -rpy_rate[1]

    global observation
    observation = np.hstack([pos, rpy, vel, rpy_rate])


def print_obs():
    pos = observation[0:3]
    rpy = observation[3:6]
    vel = observation[6:9]
    rpy_rate = observation[9:12]

    print(f"[{logconf.name}]: ", end="")
    print(f"x={pos[0]:+06.3f}", end=" ")
    print(f"y={pos[1]:+06.3f}", end=" ")
    print(f"z={pos[2]:+06.3f}", end=" ")

    print(f"roll={rpy[0] * 180 / np.pi:+08.3f}", end=" ")
    print(f"pitch={rpy[1] * 180 / np.pi:+08.3f}", end=" ")
    print(f"yaw={rpy[2] * 180 / np.pi:+08.3f}", end=" ")

    print(f"vx={vel[0]:+06.3f}", end=" ")
    print(f"vy={vel[1]:+06.3f}", end=" ")
    print(f"vz={vel[2]:+06.3f}", end=" ")

    print(f"roll_rate={rpy_rate[0] * 180 / np.pi:+08.3f}", end=" ")
    print(f"pitch_rate={rpy_rate[1] * 180 / np.pi:+08.3f}", end=" ")
    print(f"yaw_rate={rpy_rate[2] * 180 / np.pi:+08.3f}")


def print_action(action):
    print(f"[action]: ", end="")
    print(f"ROLL={action[0]:+06.3f}", end=" ")
    print(f"PITCH={action[1]:+06.3f}", end=" ")
    print(f"YAW={action[2]:+06.3f}", end=" ")
    print(f"THRUST={action[2]:+06.3f}")


def param_deck_flow(_, value_str):
    value = int(value_str)
    if value:
        deck_attached_event.set()
        print('Deck is attached')
    else:
        print('Deck is NOT attached')


def param_flightmode(_, value_str):
    value = int(value_str)
    if value == 1:
        flightmode_event.set()
        print('Yaw ANGLE flightmode')
    else:
        print('Yaw RATE flightmode')


def stop(mc):
    pos = observation[0:3]
    vel = observation[6:9]
    rpy_rate = observation[9:12]

    mc._thread._hover_setpoint = [vel[0], vel[1], rpy_rate[2], pos[2]]
    mc._thread._z_base = pos[2]
    mc._thread._z_velocity = vel[2]
    mc._thread._z_base_time = time.time()

    mc.stop()
    time.sleep(1.0)


def move_model(scf, model: PPO, env: HoverAviary, logger: Logger, drone, duration, verbose):
    # 1.0 x 1.0 x 1.0 above the drone (with z >= 0.1 to avoid ground effect)
    target_pos = np.random.uniform(-0.5, 0.5, size=3)
    target_pos[2] += 0.6
    print("[target_position]:", end=" ")
    print(f"x={target_pos[0]:+06.3f}", end=" ")
    print(f"y={target_pos[1]:+06.3f}", end=" ")
    print(f"z={target_pos[2]:+06.3f}")

    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        mc._cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
        time.sleep(0.01)

        total_reward = 0.0
        for i in range(duration * env.CTRL_FREQ):
            try:
                if verbose:
                    print_obs()
                observation[0:3] -= target_pos
                action, _ = model.predict(np.expand_dims(observation, axis=0), deterministic=True)
                action = np.squeeze(action)
                reward = env._computeReward(np.array([observation]))
                logger.log(drone=drone, timestamp=i / env.CTRL_FREQ, action=action, obs=observation, reward=reward)
                total_reward += reward
                if verbose:
                    print_action(action)
                    print(f"[reward]: {reward:+.3f}")
                action[0:3] *= 10  # +- 10 degrees
                roll, pitch = action[0:2]
                thrust = int(action[3] * 18022 + 34406)  # from 25% to 80%
                mc._cf.commander.send_setpoint(roll, pitch, 0.0, thrust)
                time.sleep(env.CTRL_TIMESTEP)
            except KeyboardInterrupt:
                return False

        print(f"[total_reward]: {total_reward:+.3f}")

        stop(mc)

        return True


def get_logconf():
    logconf = LogConfig(name="state", period_in_ms=10)

    logconf.add_variable("stateEstimateZ.x", "int16_t")
    logconf.add_variable("stateEstimateZ.y", "int16_t")
    logconf.add_variable("stateEstimateZ.z", "int16_t")
    logconf.add_variable("stateEstimateZ.quat", "uint32_t")
    logconf.add_variable("stateEstimateZ.vx", "int16_t")
    logconf.add_variable("stateEstimateZ.vy", "int16_t")
    logconf.add_variable("stateEstimateZ.vz", "int16_t")
    logconf.add_variable("stateEstimateZ.rateRoll", "int16_t")
    logconf.add_variable("stateEstimateZ.ratePitch", "int16_t")
    logconf.add_variable("stateEstimateZ.rateYaw", "int16_t")

    return logconf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-folder", default="results")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--num-drones", default=1, type=int)
    parser.add_argument("--duration", default=8, type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    output_folder = args.output_folder
    experiment_id = args.experiment_id
    num_drones = args.num_drones
    duration = args.duration
    verbose = args.verbose
    save = args.save
    plot = args.plot

    if experiment_id is None:
        with open(os.path.join(output_folder, "experiments.txt"), "r") as file:
            experiment_id = file.readlines()[-1].strip()

    filename = os.path.join(output_folder, experiment_id)
    print(f"[INFO] Loading experiment-id: {experiment_id}")

    env = HoverAviary(act=ActionType.RPYT)
    model = PPO.load(os.path.join(filename, "best_model"), device="cpu")

    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        scf.cf.param.add_update_callback(group="deck", name="bcFlow2", cb=param_deck_flow)
        scf.cf.param.add_update_callback(group="flightmode", name="stabModeYaw", cb=param_flightmode)
        scf.cf.param.set_value("flightmode.stabModeYaw", 1)

        logconf = get_logconf()
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pose_callback)

        if not deck_attached_event.wait(timeout=5.0):
            print("No flow deck detected")
            sys.exit(1)

        if not flightmode_event.wait(timeout=5.0):
            print("Incorrect flightmode")
            sys.exit(1)

        scf.cf.supervisor.send_arming_request(True)
        time.sleep(1.0)

        logconf.start()

        logger = Logger(logging_freq_hz=int(env.CTRL_FREQ), output_folder=filename, num_drones=num_drones, duration_sec=duration)
        for drone in range(num_drones):
            if not move_model(scf, model, env, logger, drone, duration, verbose):
                break
        if save:
            logger.save()
        if plot:
            logger.plot(show=True)

        logconf.stop()
