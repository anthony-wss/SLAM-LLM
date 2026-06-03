#!/bin/bash
#SBATCH --job-name=slam-omni-inference
#SBATCH --partition=normal
#SBATCH --account=MST115022
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=5:00:00
#SBATCH --output=/work/u3937558/SLAM-LLM/exp/logs/%x-%j.out
#SBATCH --error=/work/u3937558/SLAM-LLM/exp/logs/%x-%j.err

mkdir -p /work/u3937558/SLAM-LLM/exp/logs

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29503

module load cuda
export LD_LIBRARY_PATH=/work/u3937558/miniconda3/lib:$LD_LIBRARY_PATH
bash examples/s2s/scripts/inference/inference_s2s_NatDialog-Real.sh
