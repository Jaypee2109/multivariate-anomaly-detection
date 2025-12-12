#!/bin/bash
#SBATCH --job-name=transformer_mpi_ai_math
#SBATCH --output=/home/sc.uni-leipzig.de/uj74reda/logs/trsnaformer_%j.out
#SBATCH --error=/home/sc.uni-leipzig.de/uj74reda/logs/transformer_%j.err
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a30:4
#SBATCH --mem=128G
#SBATCH -t 1-0
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --job-name=TransformerTimeSeries

echo "===== Job starting on $(hostname) at $(date) ====="

# ---- Load modules ----
module purge
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/11.8.0 

# ---- Activate virtual environment ----
source /home/sc.uni-leipzig.de/uj74reda/Transformer/transformer_venv/bin/activate

# ---- Navigate to project directory ----
cd /home/sc.uni-leipzig.de/uj74reda/Transformer

# ---- Run the training script  ----
python TransformerTimeSeries/SC_main.py \
    --data_dir /home/sc.uni-leipzig.de/uj74reda/Transformer/realAWSCloudwatch \
    --epochs 10 \
    --batch_size 64 \
    --model_dim 128 \
    --num_heads 16 \
    --num_layers 12 \
    --lag 90 \
    --lr 1e-5 \
    --save_dir checkpoints

echo "===== Job ended at $(date) ====="
