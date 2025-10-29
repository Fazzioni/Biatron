#!/bin/bash
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --nodelist=dgx-H100-02
#SBATCH --job-name=inference


srun singularity exec --nvccli --no-home "/raid/aluno_daniel/images/pytorch:25.01-py3" /bin/bash -c '

export HOME="/raid/aluno_daniel/projects/BIA-MEGATRON/home"
export HF_HOME="/raid/aluno_daniel/projects/BIA-MEGATRON/huggingface"

#pip install megatron-bridge --user
pip install megatron-core transformers datasets dotenv wandb flask flask_restful simpy sentencepiece tiktoken --user


cd /raid/aluno_daniel/projects/BIA-MEGATRON/Biatron

#export CUDA_VISIBLE_DEVICES="7"
#bash gpt_static_inference.sh
bash inference_prompts.sh
'

exit
'''
srun --partition=h100n2 --gres=gpu:0 --cpus-per-task=4 --mem=24G --ntasks=1 --time=01:00:00 --nodelist=dgx-H100-02 --job-name=inference --pty singularity exec --nvccli --no-home "/raid/aluno_daniel/images/pytorch:25.01-py3" /bin/bash

'''