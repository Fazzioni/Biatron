#!/bin/bash

source .env
export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH="/raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/checkpoints2"
TENSORBOARD_LOGS_PATH="/raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/checkpoints/tensorboard_logs"
DATA_PATH="/raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/tokenizer/data/fineweb-edu-dedup"

#########################################################
########################
#######################################################
GPUS_PER_NODE=1 #Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6985
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
    --num-layers 30
    --hidden-size 576
    --num-attention-heads 9
    --group-query-attention
    --num-query-groups 3
    --seq-length 2048
    --max-position-embeddings 2048 
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --tokenizer-type NullTokenizer
    --vocab-size 49152
    --bf16
    --attention-backend fused # Can use (flash/fused/unfused/local)
    #--use-flash-attn

    --enable-cuda-graph
    #--grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --use-fused-weighted-squared-relu
    #--no-rope-fusion
    #--apply-layernorm-1p
    #--untie-embeddings-and-output-weights
    #--disable-bias-linear 
)

# 2097152 /16 = 

TRAINING_ARGS=(
    --micro-batch-size 16
    --global-batch-size 1024
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
    #--train-iters 24000 # 50B tokens
    --train-iters 500 # 50B tokens
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
    --data-path $DATA_PATH
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 50
    --eval-interval 500 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 6
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --log-progress
    --log-throughput
)

WANDB_ARGS=(
    --wandb-project BiaTron
    --wandb-save-dir $TENSORBOARD_LOGS_PATH
    --wandb-exp-name "Fineweb-edu-dedup-english-${GPUS_PER_NODE}"
)

#python3 -m pip install "transformer-engine"

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun ${DISTRIBUTED_ARGS[@]} ../Megatron-LM/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${WANDB_ARGS[@]} \
