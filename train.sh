#!/bin/bash

python train.py --total-timesteps 106496 --context-kwargs mass kf --context-low-gen 0.027 4.91e-10 --context-high-gen 0.027 4.91e-10 --context-visible 1 1 --context-low 0.007 1.66e-10 --context-high 0.047 8.16e-10

experiment_id=$(tail -n 1 results/experiments.txt)
for i in {1..9}; do
    mass_low=$(printf "%.4f\n" $(echo "0.027 - $i / 450" | bc -l))
    mass_high=$(printf "%.4f\n" $(echo "0.027 + $i / 450" | bc -l))
    kf_low=$(printf "%.4e\n" $(echo "0.000000000491 - $i / 27692000000" | bc -l))
    kf_high=$(printf "%.4e\n" $(echo "0.000000000491 + $i / 27692000000" | bc -l))
    python train.py --total-timesteps 106496 --experiment-id $experiment_id --context-kwargs mass kf --context-low-gen $mass_low $kf_low --context-high-gen $mass_high $kf_high --context-visible 1 1 --context-low 0.007 1.66e-10 --context-high 0.047 8.16e-10
done