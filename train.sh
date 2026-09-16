#!/bin/bash

python train.py --total-timesteps 106496

experiment_id=$(tail -n 1 results/experiments.txt)
for i in {1..9}; do
    python train.py --total-timesteps 106496 --experiment-id $experiment_id
done