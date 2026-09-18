#!/bin/bash

for kf in $(seq 1.66e-10 3.25e-11 8.16e-10); do
    python eval.py --n-eval-episodes 10 --start 0.007 --stop 0.047 --step 0.002 --kwarg mass --context-kwargs kf --context-low-gen $kf --context-high-gen $kf --context-visible 1 1 --context-low 1.66e-10 0.007 --context-high 8.16e-10 0.047
done