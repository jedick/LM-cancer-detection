#!/bin/sh

# Run the UC-CAP pipeline for different parameter combinations
# 20260421 jmd

#python run_uc_cap_pipeline.py --n-uc 500 --n-clusters 1000 --n-cap 2000
#python run_uc_cap_pipeline.py --n-uc 1000 --n-clusters 1000 --n-cap 2000
python run_uc_cap_pipeline.py --n-uc 2000 --n-clusters 1000 --n-cap 2000

#python run_uc_cap_pipeline.py --n-uc 500 --n-clusters 2000 --n-cap 2000
#python run_uc_cap_pipeline.py --n-uc 1000 --n-clusters 2000 --n-cap 2000
python run_uc_cap_pipeline.py --n-uc 2000 --n-clusters 2000 --n-cap 2000

python run_uc_cap_pipeline.py --n-uc 500 --n-clusters 5000 --n-cap 2000
python run_uc_cap_pipeline.py --n-uc 1000 --n-clusters 5000 --n-cap 2000
python run_uc_cap_pipeline.py --n-uc 2000 --n-clusters 5000 --n-cap 2000

#python run_uc_cap_pipeline.py --n-uc 500 --n-clusters 1000 --n-cap 5000
#python run_uc_cap_pipeline.py --n-uc 1000 --n-clusters 1000 --n-cap 5000
python run_uc_cap_pipeline.py --n-uc 2000 --n-clusters 1000 --n-cap 5000

#python run_uc_cap_pipeline.py --n-uc 500 --n-clusters 2000 --n-cap 5000
#python run_uc_cap_pipeline.py --n-uc 1000 --n-clusters 2000 --n-cap 5000
python run_uc_cap_pipeline.py --n-uc 2000 --n-clusters 2000 --n-cap 5000

python run_uc_cap_pipeline.py --n-uc 500 --n-clusters 5000 --n-cap 5000
python run_uc_cap_pipeline.py --n-uc 1000 --n-clusters 5000 --n-cap 5000
python run_uc_cap_pipeline.py --n-uc 2000 --n-clusters 5000 --n-cap 5000

python run_uc_cap_pipeline.py --n-uc 500 --n-clusters 1000 --n-cap 10000
python run_uc_cap_pipeline.py --n-uc 1000 --n-clusters 1000 --n-cap 10000
python run_uc_cap_pipeline.py --n-uc 2000 --n-clusters 1000 --n-cap 10000

python run_uc_cap_pipeline.py --n-uc 500 --n-clusters 2000 --n-cap 10000
python run_uc_cap_pipeline.py --n-uc 1000 --n-clusters 2000 --n-cap 10000
python run_uc_cap_pipeline.py --n-uc 2000 --n-clusters 2000 --n-cap 10000

python run_uc_cap_pipeline.py --n-uc 500 --n-clusters 5000 --n-cap 10000
python run_uc_cap_pipeline.py --n-uc 1000 --n-clusters 5000 --n-cap 10000
python run_uc_cap_pipeline.py --n-uc 2000 --n-clusters 5000 --n-cap 10000

