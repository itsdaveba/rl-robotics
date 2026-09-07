import os
import sys
import time
import logging
import argparse
from threading import Event

import numpy as np
from stable_baselines3 import PPO
from scipy.spatial.transform import Rotation

import cflib.crtp
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils.encoding import decompress_quaternion
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander


URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")

DEFAULT_HEIGHT = 0.1
CTRL_TIMESTEP = 1.0 / 30

deck_attached_event = Event()
flightmode_event = Event()

logging.basicConfig(level=logging.ERROR)

observation = np.zeros(12)


def log_pose_callback(timestamp, data, logconf):
    pos = np.array([data["stateEstimateZ.x"], data["stateEstimateZ.y"], data["stateEstimateZ.z"]]) / 1000
    print(f"[{timestamp}][{logconf.name}]: ", end="")
    print(f"x={pos[0]:+06.3f}", end=" ")
    print(f"y={pos[1]:+06.3f}", end=" ")
    print(f"z={pos[2]:+06.3f}", end=" ")

    quat = data["stateEstimateZ.quat"]
    rot = Rotation.from_quat(decompress_quaternion(quat))
    rpy = rot.as_euler(seq="xyz")
    print(f"roll={rpy[0] * 180 / np.pi:+08.3f}", end=" ")
    print(f"pitch={rpy[1] * 180 / np.pi:+08.3f}", end=" ")
    print(f"yaw={rpy[2] * 180 / np.pi:+08.3f}", end=" ")

    vel = np.array([data["stateEstimateZ.vx"], data["stateEstimateZ.vy"], data["stateEstimateZ.vz"]]) / 1000
    print(f"vx={vel[0]:+06.3f}", end=" ")
    print(f"vy={vel[1]:+06.3f}", end=" ")
    print(f"vz={vel[2]:+06.3f}", end=" ")

    rpy_rate = np.array([data["stateEstimateZ.rateRoll"], data["stateEstimateZ.ratePitch"], data["stateEstimateZ.rateYaw"]]) / 1000
    rpy_rate[1] = -rpy_rate[1]
    print(f"wx={rpy_rate[0] * 180 / np.pi:+08.3f}", end=" ")
    print(f"wy={rpy_rate[1] * 180 / np.pi:+08.3f}", end=" ")
    print(f"wz={rpy_rate[2] * 180 / np.pi:+08.3f}", end=" ")
    print()

    global observation
    observation = np.hstack([pos, rpy, vel, rpy_rate])


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


def move_model(scf, model: PPO):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:

        mc._cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
        time.sleep(0.01)

        while True:
            try:
                action, _ = model.predict(np.expand_dims(observation, axis=0), deterministic=True)
                action[0:3] *= 30  # +- 30 degrees
                roll, pitch, yaw = action
                thrust = action[3] * 18022 + 34406,  # from 25% to 80%
                mc._cf.commander.send_setpoint(roll, pitch, yaw, thrust)
                time.sleep(CTRL_TIMESTEP)
            except KeyboardInterrupt:
                break

        stop(mc)


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
    args = parser.parse_args()

    output_folder = args.output_folder
    experiment_id = args.experiment_id

    if experiment_id is None:
        with open(os.path.join(output_folder, "experiments.txt"), "r") as file:
            experiment_id = file.readlines()[-1].strip()

    filename = os.path.join(output_folder, experiment_id)
    print(f"[INFO] Loading experiment-id: {experiment_id}")

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
        move_model(scf, model)
        logconf.stop()
