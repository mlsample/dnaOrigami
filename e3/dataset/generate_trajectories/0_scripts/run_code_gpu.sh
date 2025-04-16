#!/bin/bash
#SBATCH -p run
#SBATCH -N 1                     # number of GPUs
#SBATCH -n 58                    # number of cores
#SBATCH -t 7-0:00              # wall time (D-HH:MM)
#SBATCH --gres=gpu:8
#SBATCH -o slurm.%j.out          # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err          # STDERR (%j = JobId)
#SBATCH --job-name="jup"
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=mlsample@asu.edu     #send to my email

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_$SLURM_TASK_PID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_$SLURM_TASK_PID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d

python3 run_nanobase_data.py
#jupyter lab --ip 0.0.0.0  --no-browser --port=8888
