This repository is the repo for CDM project.

**In my code,  in some place,  `LDDIM=CDM`, `DDIM=CDM-nolocal`**


To my code, firstly create an env with `conda` via 

```bash
conda env create -f environment.yml --name (env)

```
Then, you need to install `R` and related pkgs including 

- grf, 
- randomForest, 
- gbm
- cfcausal (install from [github](https://github.com/lihualei71/cfcausal?tab=readme-ov-file))



To reproduce the results,  you can run (under `bash_scripts` folder)

```bash

bash simu_lcp_homo.sh # fig1
bash simu_lcp_hetero.sh # fig2
bash simu_lcp_homo_ablK.sh # fig3
bash simu_lcp_hetero_ablK.sh # fig4
bash real_data_Y1_eICU.sh # real data, table 1

```


To analyse the results, go to `notebooks` folder

For simulation (fig1 and fig2), run `extract_homo_hetero_res.ipynb` first, then 

- get fig 1 with `simu_results_ana_lcp_homo.ipynb`
- get fig 2 with `simu_results_ana_lcp_hetero.ipynb`

For others

- get fig 3 with `simu_results_ana_lcp_homo_ablK.ipynb`
- get fig 4 with `simu_results_ana_lcp_hetero_ablK.ipynb`
- get table 4 with `real_data_ana.ipynb`
