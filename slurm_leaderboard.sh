#!/bin/bash
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --nodelist=dgx-H100-02
#SBATCH --job-name=test

srun singularity exec --nvccli --no-home "/raid/aluno_daniel/images/pytorch:25.01-py3" /bin/bash -c '

export HOME="/raid/aluno_daniel/projects/BIA-MEGATRON/home"
export HF_HOME="/raid/aluno_daniel/projects/BIA-MEGATRON/huggingface"
#export CUDA_VISIBLE_DEVICES=7

pip install megatron-core transformers datasets dotenv wandb --user

cd /raid/aluno_daniel/projects/BIA-MEGATRON/Biatron

bash leaderboard.sh
'

exit 0
'''
#git clone https://github.com/NVIDIA/Megatron-LM # dd8200e1be22a23469087eb4f8e761387aaee869
#cd Megatron-LM/
#git checkout core_r0.14.0

srun --partition=h100n2 --gres=gpu:1 --cpus-per-task=4 --mem=24G --ntasks=1 --time=UNLIMITED --nodelist=dgx-H100-02 --job-name=Bash --pty singularity exec --nvccli --no-home "/raid/aluno_daniel/images/pytorch:25.01-py3" /bin/bash
'''
