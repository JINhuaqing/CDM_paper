#!/bin/bash

## d = 10, 300
## setting=4 (normal), 5(gamma), 6(nonlocal)
## c [0.0 0.02 0.04 0.06 0.08 0.1 0.2 0.4 0.8 1.2]

setting=4
d=10
c=0.0

python -u ../python_scripts/simu_lcp_hetero.py --setting setting$setting --d $d --n 10000 --epoch 2000 --n_T 400 --lr 1e-2 --c $c

