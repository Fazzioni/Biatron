# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Run dynamic batching inference on the 357M GPT model.

set -u


# Environment variables.
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Checkpoint.
CHECKPOINT_DIR="/raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/checkpoints/"
#CHECKPOINT_DIR="/raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/checkpoints2/"

VOCAB_FILE="/raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/tokenizer/vocab.json"
MERGE_FILE="/raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/tokenizer/merge.txt"
TOKENIZER_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"

# Prompts.
: ${NUM_TOKENS_TO_PROMPT="8 32"}
: ${NUM_TOKENS_TO_GENERATE=256}
: ${INCOMING_REQUESTS_DURATION=10.}
: ${INCOMING_REQUESTS_PER_SEC=100.}

# Dynamic context.
: ${BUFFER_SIZE_GB=2.}
: ${BUFFER_OVERFLOW_FACTOR=1.}
: ${BUFFER_GUARANTEED_FRACTION=0.05}

# Cuda graphs.
: ${ENABLE_CUDA_GRAPHS=0}
: ${NUM_CUDA_GRAPHS=16}
: ${CUDA_GRAPH_SHARE_IO_BUFFERS=1}

# Miscellaneous.
#: ${ENGINE=dynamic}
# NSIGHT_PREFIX=/path/to/nsight/profile

# Arguments.
#--tokenizer_model ${TOKENIZER_MODEL} \
ARGS=" \
    --exit-on-missing-checkpoint \
    --transformer-impl local \
    --load ${CHECKPOINT_DIR} \
    --tokenizer-type HuggingFaceTokenizer \
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --exit-on-missing-checkpoint \
    --max-position-embeddings 2048 \
    --seq-length 2048 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 30 \
    --num-attention-heads 9 \
    --group-query-attention \
    --vocab-size 49152 \
    --num-query-groups 3 \
    --hidden-size 576 \
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --bf16 \
    --micro-batch-size 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --seed 42 \
    --inference-rng-tracker \
    \
    --inference-dynamic-batching \
    --inference-dynamic-batching-buffer-size-gb ${BUFFER_SIZE_GB} \
    --inference-dynamic-batching-buffer-overflow-factor ${BUFFER_OVERFLOW_FACTOR} \
    --inference-dynamic-batching-buffer-guaranteed-fraction ${BUFFER_GUARANTEED_FRACTION} \
    \
    --tensor-model-parallel-size 1 \
	--pipeline-model-parallel-size 1
"

# Cuda graphs.
if [ "${ENABLE_CUDA_GRAPHS}" = 1 ]; then
    ARGS+=" \
        --enable-cuda-graph \
        --inference-dynamic-batching-num-cuda-graphs ${NUM_CUDA_GRAPHS} \
    "
fi

# Prompts.
if [[ -v PROMPTS ]]; then
    ARGS+=" \
        --prompts ${PROMPTS} \
        --num-tokens-to-generate ${NUM_TOKENS_TO_GENERATE} \
    "
else
    ARGS+=" \
        --num-tokens-to-prompt ${NUM_TOKENS_TO_PROMPT} \
        --num-tokens-to-generate ${NUM_TOKENS_TO_GENERATE} \
        --incoming-requests-duration ${INCOMING_REQUESTS_DURATION} \
        --incoming-requests-per-sec ${INCOMING_REQUESTS_PER_SEC} \
    "
fi

#ARGS+=" \
#    --ckpt-step 14500
#    "


ARGS+=" \
    --prompt-file /raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/prompt_en.jsonl
    "

# create a dir
mkdir -p /raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/checkpoints/result_prompts


for NUM_CHECKPOINT in 03500 06500 09500 12500 15500 18500 21500 01000 04000 07000 10000 13000 16000 19000 22000 01500 04500 07500 10500 13500 16500 19500 22500 02000 05000 08000 11000 14000 17000 20000 23000 02500 05500 08500 11500 14500 17500 20500 23500 03000 06000 09000 12000 15000 18000 21000 24000; do

#for NUM_CHECKPOINT in 50 100 150 200 250 300 350 400 450 500; do
##for NUM_CHECKPOINT in 24000; do
#    #ARGS+=" \
#    #    --ckpt-step ${arg} \
#    #"

export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM + 10000))

NEW_ARGS="${ARGS} --ckpt-step ${NUM_CHECKPOINT} --output-path /raid/aluno_daniel/projects/BIA-MEGATRON/Biatron/checkpoints/result_prompts/ckpt_${NUM_CHECKPOINT}.json"

CMD="python inference_generate.py ${NEW_ARGS}"
if [[ -v NSIGHT_PREFIX ]]; then
    CMD="nsys profile -s none -t nvtx,cuda --cudabacktrace=all --cuda-graph-trace=node --python-backtrace=cuda --wait all -o ${NSIGHT_PREFIX} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop ${CMD}"
fi

echo "~~~"
echo "CMD ... ${CMD}."
echo "~~~"
eval cd /raid/aluno_daniel/projects/BIA-MEGATRON/Megatron-LM && ${CMD}




'''
--prompts "<|im_start|>With the development of science and technology"
'''
done

json row, with {'text':'...'} entries
