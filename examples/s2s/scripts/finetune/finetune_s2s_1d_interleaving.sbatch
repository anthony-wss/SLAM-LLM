#!/bin/bash
#SBATCH --job-name=interleaving-s2s
#SBATCH --partition=normal
#SBATCH --account=MST115022
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/u3937558/SLAM-LLM/exp/logs/%x-%j.out
#SBATCH --error=/work/u3937558/SLAM-LLM/exp/logs/%x-%j.err

mkdir -p /work/u3937558/SLAM-LLM/exp/logs

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29503

srun singularity exec --nv -B /work/u3937558:/work/u3937558 \
  /work/u3937558/SLAM-LLM/slam-omni.sif \
  bash examples/s2s/scripts/finetune/finetune_s2s_1d_interleaving.sh
