#!/bin/bash

source .env
export CUDA_DEVICE_MAX_CONNECTIONS=1

START_PATH="/raid/aluno_daniel/aluno_daniel/projects/Biatron_old"
#START_PATH="/workspace/"

CHECKPOINT_PATH="${START_PATH}/checkpoints"
#CHECKPOINT_PATH="${START_PATH}/checkpoints_test_speed"
mkdir -p $CHECKPOINT_PATH
TENSORBOARD_LOGS_PATH="${CHECKPOINT_PATH}/tensorboard_logs"

GIGA_VERBO="${START_PATH}/dados/GigaVerbo/GigaVerbo"
FINE_MATH="${START_PATH}/dados/finemath/finemath-4plus/finemath-4plus_complet"
INFIWEB="${START_PATH}/dados/finemath/infiwebmath-4plus/infiwebmath-4plus_complet"
RESOANING="${START_PATH}/dados/resoaning/resoaning_complet"

#########################################################
########################
#######################################################
GPUS_PER_NODE=1 #Change for multinode config
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
MASTER_ADDR=localhost
MASTER_PORT=$(($SLURM_JOB_ID + 10000))
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 32
    --hidden-size 960
    --num-attention-heads 15
    --group-query-attention
    --num-query-groups 5
    --seq-length 4096
    --max-position-embeddings 4096
    --position-embedding-type rope
    --rotary-base 10000
    --rotary-percent 1.0
    --tokenizer-type NullTokenizer
    --vocab-size 32100
    --bf16
    --attention-backend fused # Can use (flash/fused/unfused/local)
    #--use-flash-attn

    --enable-cuda-graph
    --no-load-rng
    --cross-entropy-loss-fusion
    --use-fused-weighted-squared-relu
    --grad-reduce-in-bf16
    --cross-entropy-fusion-impl 'te'







    #--no-rope-fusion
    #--apply-layernorm-1p
    #--untie-embeddings-and-output-weights
    #--disable-bias-linear 
)

# 2097152 /16 = 

TRAINING_ARGS=(
    --micro-batch-size 16
    --global-batch-size 512
    #--global-batch-size 480
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --train-iters 152000 # 
#    --ckpt-format 'torch'
    #--auto-detect-ckpt-format
    --no-ckpt-fully-parallel-save
    --decrease-batch-size-if-needed
)

# --rampup-batch-size 16 16 5859375 
    
MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
    --data-parallel-sharding-strategy no_shard
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

DATA_ARGS=(
    --data-path 12 $GIGA_VERBO 1 $FINE_MATH 1 $INFIWEB 6 $RESOANING
    --split 949,50,1
    --seed 1234
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 100
    --eval-interval 500 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --log-progress
    --log-throughput
)

WANDB_ARGS=(
    --wandb-project BiaTron-PT
    --wandb-save-dir $TENSORBOARD_LOGS_PATH
    --wandb-exp-name "GPT-${GPUS_PER_NODE}"
)

#python3 -m pip install "transformer-engine"
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


torchrun ${DISTRIBUTED_ARGS[@]} ./Megatron-LM/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${WANDB_ARGS[@]} \
