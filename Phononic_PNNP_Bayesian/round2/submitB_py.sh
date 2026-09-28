#!/bin/bash
#SBATCH --job-name="PNNr2d1"
#SBATCH --output="PNNrun.%j.%N.out"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=che150015p
#SBATCH -p RM-shared

#SBATCH --no-requeue
#SBATCH -t 12:00:00

ml cuda/12.6.
module load gcc
module load openmpi
module load anaconda3

conda activate mytorch

#pip install torch torchvision torchaudio pandas numpy scikit-learn matplotlib scipy tensorflow
#Activate the vitual environment
#source /path/to/venv/bin/activate

#Run your Python script
#python round2_run400_data1.py
python heat_flux_calculation_v2.py
#Deactivate the virtual environment
conda deactivate

mkdir result
mv plot_heatflux_set* result/
mv plot_heat_set* result/
