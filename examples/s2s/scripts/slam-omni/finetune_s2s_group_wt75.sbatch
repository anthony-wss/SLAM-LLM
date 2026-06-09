#!/bin/bash
#SBATCH --job-name=slam-omni-s2s
#SBATCH --partition=32gpus
#SBATCH --account=MST115022
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/u3937558/SLAM-LLM/exp/logs/%x-%j.out
#SBATCH --error=/work/u3937558/SLAM-LLM/exp/logs/%x-%j.err

mkdir -p /work/u3937558/SLAM-LLM/exp/logs

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29541

module load cuda
export LD_LIBRARY_PATH=/work/u3937558/miniconda3/lib:$LD_LIBRARY_PATH
bash /work/u3937558/SLAM-LLM/examples/s2s/scripts/slam-omni/finetune_s2s_group_wt75.sh
