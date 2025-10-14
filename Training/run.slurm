#!/bin/bash
#SBATCH --partition=gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --ntasks-per-node=12 
#SBATCH --job-name=python-gpu
#SBATCH --output=logs/ball_train.out
#SBATCH --error=logs/ball_train.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # number of cores needed
#SBATCH --gres=gpu:1g.10gb:1                # up to 8; only request what you need
#SBATCH --mem=15gb  
#SBATCH --export=ALL 
#SBATCH --time=0-07:00:00                  

# Load required modules
module load python
module load cuda
module load gnu10

# Run the training script
python3 train.py
