#!/bin/bash
#SBATCH --partition=h100n3
#SBATCH --nodelist=dgx-H100-03
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --job-name=Megatron
#SBATCH --time=UNLIMITED

#"/raid/aluno_daniel/images/pytorch:25.09-py3.sif" \
#/raid/aluno_daniel/images/pytorch_25.08-py3.sif cuda 13.0
#https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html


#

echo Using  $CUDA_VISIBLE_DEVICES GPUs

srun singularity exec --nv \
    --no-home \
    --bind "/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1" \
    "/raid/aluno_daniel/aluno_daniel/images/nvidia-25.09.py3.sif" \
    /bin/bash -c '

echo USING device: $CUDA_VISIBLE_DEVICES 
export HOME="/raid/aluno_daniel/aluno_daniel/projects/Biatron_old/HOME"
export HF_HOME="/raid/aluno_daniel/aluno_daniel/projects/Biatron_old/HOME/huggingface"

pip install megatron-core transformers datasets dotenv wandb --user
pip install nvidia-modelopt[hf] --user

python3 -m pip install nvidia-ml-py --user

#pip install --upgrade torch transformer_engine --user

source .env
huggingface-cli login --token ${HF_TOKEN}

bash biatron.sh
'

#git clone https://github.com/NVIDIA/Megatron-LM # dd8200e1be22a23469087eb4f8e761387aaee869
#cd Megatron-LM/
#git checkout core_r0.14.0

exit 0
 

srun --partition=h100n3  --nodelist=dgx-H100-03 --cpus-per-task=12 --mem=32G --ntasks=1 --job-name=Test --time=UNLIMITED --pty \
    singularity exec --nv \
    --no-home \
    --bind "/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1" \
    "/raid/aluno_daniel/aluno_daniel/images/nvidia-25.09.py3.sif" \
    /bin/bash 


export HOME="/raid/aluno_daniel/aluno_daniel/projects/Biatron_old/HOME_bridge"

bridge = AutoBridge.from_hf_pretrained("google/gemma-3-1B-it", trust_remote_code=True)





srun --partition=h100n3  --nodelist=dgx-H100-03 --cpus-per-task=12 --mem=32G --ntasks=1 --job-name=Test --time=UNLIMITED --pty \
    singularity exec --nv \
    --no-home \
    --bind "/usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/cuda/compat/lib/libcuda.so.1" \
    "/raid/aluno_daniel/aluno_daniel/images/nemo_25.09.nemotron_nano_v2_vl.sif" \
    /bin/bash 
dir










#https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/174457385/Installing+flash-attention+3+for+hopper    




echo USING device: $CUDA_VISIBLE_DEVICES 
export HOME="/raid/aluno_daniel/aluno_daniel/projects/Biatron_old/HOME2"
export HF_HOME="/raid/aluno_daniel/aluno_daniel/projects/Biatron_old/HOME/huggingface"


#pip uninstall -y torch torchvision torchaudio
#pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

pip install torch>=2.2 packaging ninja wheel setuptools cmake --no-build-isolation --user
pip install megatron-core transformers datasets dotenv wandb --user
pip install nvidia-modelopt[hf] --user
python3 -m pip install nvidia-ml-py --user
python setup.py install --user

