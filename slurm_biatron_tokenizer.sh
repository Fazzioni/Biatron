#!/bin/bash
#SBATCH --partition=h100n2
#SBATCH --nodelist=dgx-H100-02
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --ntasks=1
#SBATCH --time=UNLIMITED
#SBATCH --job-name=Tokenizer


# 'prompt', 'thought', 'answer'
#datasets.load_from_disk("/raid/amadeus/reasoning-v1-20m-emb_tokenized_ordered")
#

srun singularity exec --nvccli \
    --no-home \
    --bind "/raid/amadeus:/data" \
     "/raid/aluno_daniel/images/pytorch:25.01-py3" /bin/bash -c '

export HOME="/raid/aluno_daniel/projects/BIA-MEGATRON/home"
export HF_HOME="/raid/aluno_daniel/projects/BIA-MEGATRON/huggingface"

pip install megatron-core transformers datasets dotenv wandb --user

cd /raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/tokenizer
 ####python create_dataset.py

# disable all gpus
export CUDA_VISIBLE_DEVICES="-1"
python tokenizer_resoaning.py
'

#git clone https://github.com/NVIDIA/Megatron-LM # dd8200e1be22a23469087eb4f8e761387aaee869
#cd Megatron-LM/
#git checkout core_r0.14.0
#
#pip install --pre megatron-core
#PYTHONPATH=/raid/aluno_daniel/images/Megatron-LM/megatron:$PYTHONPATH torchrun --nproc_per_node=2 --master-port 29300 examples/run_simple_mcore_train_loop.py


#srun --partition=h100n2 --gres=gpu:1 --cpus-per-task=12 --mem=80G --ntasks=1 --time=UNLIMITED --nodelist=dgx-H100-02 --job-name=Bash --pty singularity exec --nvccli --no-home --bind "/raid/amadeus:/data" "/raid/aluno_daniel/images/pytorch:25.01-py3" /bin/bash