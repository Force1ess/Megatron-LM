#!/bin/bash
TP=1
PP=1
SAVER=mcore
ModelPath="models/Qwen2.5-1.5B"

set -x
python autoalign-megatron/model_converter.py --model-type GPT \
   --loader llama_mistral \
   --saver $SAVER \
   --checkpoint-type hf \
   --model-size qwen2.5-1.5B \
   --load-dir $ModelPath \
   --save-dir $ModelPath-tp$TP-pp$PP-$SAVER \
   --tokenizer-model $ModelPath \
   --bf16 \
   --target-tensor-parallel-size $TP \
   --target-pipeline-parallel-size $PP 
cp $ModelPath/config.json $ModelPath-tp$TP-pp$PP-$SAVER/