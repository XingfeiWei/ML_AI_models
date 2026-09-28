#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1 
#SBATCH --job-name="pnnrun"
#SBATCH --output="pnnrun.%j.%N.out"
#SBATCH --account=che150015p
#SBATCH --no-requeue
#SBATCH -t 48:00:00

ml cuda/12.6.1
module load gcc
module load openmpi
module load anaconda3

conda activate mytorch

#pip install torch torchvision torchaudio pandas numpy scikit-learn matplotlib scipy tensorflow
# Activate the vitual environment
#source /path/to/venv/bin/activate
#v100-16, v100-32, and h100-80
# Run your Python script
python run400_data4.py

# Deactivate the virtual environment
conda deactivate



