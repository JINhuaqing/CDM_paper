#!/bin/bash
## d = 10, 300
## c [0.0 0.02 0.04 0.06 0.08 0.1 0.2 0.4 0.8 1.2]
## K (M in the paper): [10 20 40 80 160 320]

d=10
c=0.0
K=40

python -u ../python_scripts/simu_lcp_homo_ablK.py --setting setting1 --d $d --n 10000 --epoch 2000  --c $c --K $K

