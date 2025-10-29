
srun --partition=h100n2 --gres=gpu:1 --cpus-per-task=12 --mem=80G --ntasks=1 --time=UNLIMITED --nodelist=dgx-H100-02 --job-name=Bash --pty singularity exec --nvccli --no-home "/raid/aluno_daniel/images/pytorch:25.01-py3" /bin/bash

export HOME="/raid/aluno_daniel/projects/BIA-MEGATRON/home"
export HF_HOME="/raid/aluno_daniel/projects/BIA-MEGATRON/huggingface"

pip install megatron-core transformers datasets dotenv wandb --user


git clone https://github.com/NVIDIA/Megatron-LM # dd8200e1be22a23469087eb4f8e761387aaee869
cd Megatron-LM/
git checkout core_r0.14.0

pip install --pre megatron-core
PYTHONPATH=/raid/aluno_daniel/images/Megatron-LM/megatron:$PYTHONPATH torchrun --nproc_per_node=2 --master-port 29300 examples/run_simple_mcore_train_loop.py



scp -r 10.100.0.113:    /raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/tokenizer 


scp -r /raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/tokenizer 10.100.0.113:/raid/aluno_daniel/aluno_daniel/projects/Biatron/