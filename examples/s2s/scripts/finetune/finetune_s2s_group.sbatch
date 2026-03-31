#!/bin/bash
#SBATCH --job-name=slam-omni-s2s
#SBATCH --partition=normal
#SBATCH --account=MST115022
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/work/u3937558/SLAM-LLM/exp/logs/%x-%j.out
#SBATCH --error=/work/u3937558/SLAM-LLM/exp/logs/%x-%j.err

mkdir -p /work/u3937558/SLAM-LLM/exp/logs

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29503

srun singularity exec --nv -B /work/u3937558:/work/u3937558 \
  -B /work/u3937558/SLAM-LLM/src/slam_llm:/opt/SLAM-LLM/src/slam_llm \
  /work/u3937558/SLAM-LLM/slam-omni.sif \
  bash -c "cd /work/u3937558/SLAM-LLM && bash examples/s2s/scripts/finetune/finetune_s2s_group.sh"
