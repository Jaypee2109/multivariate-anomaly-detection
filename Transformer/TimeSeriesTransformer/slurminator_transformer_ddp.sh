#!/bin/bash
#SBATCH --output=/home/sc.uni-leipzig.de/uj74reda/logs/transformer_ddp_%j.out
#SBATCH --error=/home/sc.uni-leipzig.de/uj74reda/logs/transformer_ddp_%j.err
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=a30:8
#SBATCH --mem=64G
#SBATCH -t 2-0
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --job-name=TransformerTimeSeriesDDP


echo "===== Job starting on $(hostname) at $(date) ====="

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/11.8.0

source /home/sc.uni-leipzig.de/uj74reda/Transformer/transformer_venv/bin/activate

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_FAMILY=AF_INET

export NCCL_DEBUG=INFO
export PYTORCH_DEBUG=INFO

cd /home/sc.uni-leipzig.de/uj74reda/Transformer

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

srun python TimeSeriesTransformer/main_electricity_data_ddp.py



echo "===== Job ended at $(date) ====="
