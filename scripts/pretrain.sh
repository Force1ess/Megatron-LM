#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Arguments
GPUS_PER_NODE=1
DATA_PATH=data/fineweb-2-aak_Latn_text_document
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
MODEL_FAMILY=Qwen2.5
MODEL_SIZE=1.5B


TOKENIZER_MODEL=models/${MODEL_FAMILY}-${MODEL_SIZE}
MODEL=${MODEL_FAMILY}-${MODEL_SIZE}-tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-mcore
CHECKPOINT_PATH=models/${MODEL}
CONFIG_PATH=${CHECKPOINT_PATH}/${MODEL_FAMILY}-${MODEL_SIZE}.json
if [ ! -f $CONFIG_PATH ]; then
    echo "Config file $CONFIG_PATH does not exist"
    exit 1
fi

NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)


DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 1000,300,53
)

# warmup steps should less than decay steps
TRAINING_ARGS=(
    --seq-length 512
    --micro-batch-size 1
    --global-batch-size 1
    --train-iters 1000
    --lr-decay-iters 500
    --lr-decay-style cosine
    --lr 1e-4
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 100
    --clip-grad 1.0
    --bf16
    --use-precision-aware-optimizer
)

# below arguments were automatically generated

model_args=$(python autoalign-megatron/get_model_args.py --config-path $CONFIG_PATH)
MODEL_ARGS=()
while IFS= read -r line; do
    # 检查是否是以 "--" 开头的参数行
    if [[ $line == --* ]]; then
        # 如果包含 "="，分割为 key 和 value
        if [[ $line == *"="* ]]; then
            key=${line%%=*}  # 提取等号前的部分
            value=${line#*=} # 提取等号后的部分
            MODEL_ARGS+=("$key" "$value")
        else
            # 如果没有 "="，直接添加到数组
            MODEL_ARGS+=("$line")
        fi
    fi
done <<< "$model_args"

MAGIC_ARGS=(
    --disable-bias-linear
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --no-masked-softmax-fusion
    --rotary-base 1000000
    --use-mcore-models
    --rotary-percent 1.0
    --rotary-seq-len-interpolation-factor 1
    --group-query-attention
)


MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE
    --use-distributed-optimizer
    # --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 10
    --save-interval 500
    --eval-interval 100
    --eval-iters 10
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --no-load-optim
    --no-load-rng
    --no-save-optim
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Autoalign-Megatron"}
        --wandb-exp-name ${WANDB_NAME:-$MODEL}
    )
fi


set -x
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MAGIC_ARGS[@]} \
    ${LOGGING_ARGS[@]}
