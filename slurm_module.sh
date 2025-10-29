#!/bin/bash
#SBATCH --partition=h100n2
#SBATCH --nodelist=dgx-H100-02
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=Test

module load shared
module load conda
source /cm/shared/apps/conda/etc/profile.d/conda.sh

# --- Ambiente isolado ---
ENV_PATH=/raid/aluno_daniel/projects/BIA-MEGATRON/megatron_venv

# Criar ambiente se ainda não existir
if [ ! -d "$ENV_PATH" ]; then
    conda create --prefix $ENV_PATH python=3.11 -y
fi

export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME/.local/bin" | tr '\n' ':' | sed 's/:$//')


# Ativar o ambiente corretamente
source activate $ENV_PATH



# Instalar mamba dentro do AMBIENTE, não no base
conda install -c conda-forge mamba -y

# Instalar PyTorch compatível com CUDA do cluster
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Instalar Megatron
pip install --pre megatron-core

# Configurar variáveis isoladas
export HF_HOME=/raid/aluno_daniel/projects/BIA-MEGATRON/huggingface
export HOME=/raid/aluno_daniel/projects/BIA-MEGATRON/home

# Verificar se torch está acessível
python - <<'EOF'
import torch
print("Torch versão:", torch.__version__)
print("GPU disponível:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
EOF

# Executar o treinamento
bash train_gpt3_175b_distributed.sh
