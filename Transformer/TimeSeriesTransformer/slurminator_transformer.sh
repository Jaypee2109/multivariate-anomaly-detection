#!/bin/bash
#SBATCH --output=/home/sc.uni-leipzig.de/uj74reda/logs/transformer_%j.out
#SBATCH --error=/home/sc.uni-leipzig.de/uj74reda/logs/transformer_%j.err
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=a30:1
#SBATCH --mem=64G
#SBATCH -t 2-0
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --job-name=TransformerTimeSeries

echo "===== Job starting on $(hostname) at $(date) ====="

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/11.8.0

source /home/sc.uni-leipzig.de/uj74reda/Transformer/transformer_venv/bin/activate
cd /home/sc.uni-leipzig.de/uj74reda/Transformer


python TimeSeriesTransformer/main_electricity_data.py


echo "===== Job ended at $(date) ====="
