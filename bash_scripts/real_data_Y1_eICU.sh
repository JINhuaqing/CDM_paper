#!/bin/bash

## c in [0.0 0.02 0.04 0.06 0.08 0.1 0.2 0.4 0.8 1.2]

c=0.0
python -u ../python_scripts/real_data_Y1_eICU.py --n 10000 --epoch 2000 --n_T 400 --lr 1e-2 --c $c

