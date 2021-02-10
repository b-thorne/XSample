#!/bin/bash

module load tensorflow/gpu-2.2.0-py37
module load texlive

srun python xsample/xsample.py --gin_config configs/MAF_5layer_base_omegak.gin --results_dir results/MAF_5layer_base_omegak