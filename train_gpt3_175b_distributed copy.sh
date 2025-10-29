#!/bin/bash
source .env
# Runs the "175B" parameter model

# https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html
# https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/nemo_megatron/mcore_customization.html

# config attention

# /raid/aluno_daniel/projects/BIA-MEGATRON/Megatron-LM/megatron/core/transformer/attention.py




export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2 #Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH="/raid/aluno_daniel/projects/BIA-MEGATRON/outputs/checkpoints"
TENSORBOARD_LOGS_PATH="/raid/aluno_daniel/projects/BIA-MEGATRON/outputs/tensorboard_logs"
DATA_PATH="/raid/aluno_daniel/projects/BIA-MEGATRON/Fineweb/test/fineweb"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 24
    --hidden-size 2048
    --num-attention-heads 16
    #--group-query-attention
    #--num-query-groups 1
    --seq-length 4096 
    --max-position-embeddings 4096 
    --position-embedding-type rope
    --rotary-base 1000000 
    --rotary-percent 1.0
    --tokenizer-type NullTokenizer
    --vocab-size 32000
    --bf16
    --attention-backend fused # Can use (flash/fused/unfused/local)
    #--use-flash-attn

    #--fp8-format hybrid
    #--fp8-amax-history-len 1024
    #--fp8-amax-compute-algo max
    #--fp8-param-gather
    #--use-precision-aware-optimizer

    #--enable-cuda-graph
    #--cuda-graph-scope 'full_iteration'
    --grad-reduce-in-bf16
    #--cross-entropy-loss-fusion
    #--use-fused-weighted-squared-relu

    #--apply-layernorm-1p 
    #--untie-embeddings-and-output-weights
    #--disable-bias-linear 
)



#GPT_MODEL_ARGS=(
#    --num-layers 24
#    --hidden-size 2048
#    --num-attention-heads 16
#    --seq-length 4096 
#    --max-position-embeddings 4096 
#    --tokenizer-type NullTokenizer
#    --vocab-size 32000
#)


#--manual-gc 

# 2097152 /16 = 

TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size 192
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
    --train-iters 100

)
# --rampup-batch-size 16 16 5859375 
    
MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
    --data-parallel-sharding-strategy no_shard
    --log-throughput
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
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

WANDB_ARGS=(
    --wandb-project Megatron-LM-Test
    --wandb-save-dir $TENSORBOARD_LOGS_PATH
    --wandb-exp-name "GPT3_110M-${GPUS_PER_NODE}xH100-1.7B"
)

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun ${DISTRIBUTED_ARGS[@]} Megatron-LM/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${WANDB_ARGS[@]} \
