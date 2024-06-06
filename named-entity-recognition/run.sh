#!/bin/bash

ENTITY=$1
NUM_EPOCHS=$2
RUN_NAME=$3
OPTIMIZER=${4:-"adamw_torch"}

DATA_DIR="../datasets/NER"
SAVE_DIR="./output"

if [ -z $MODEL_USED ]; then
    export MODEL_USED="dmis-lab/biobert-base-cased-v1.1"
fi

if [ -z $BATCH_SIZE ]; then
    export BATCH_SIZE=32
fi

if [ -z $MAX_LENGTH ]; then
    export MAX_LENGTH=192
fi

if [ -z $SAVE_STEPS ]; then
    export SAVE_STEPS=1000
fi

if [ -z $SEED ]; then
    export SEED=1
fi

if [ -z $LOG_STEPS ]; then
    export LOG_STEPS=10
fi

WANDB_PROJECT=BioBERT python run_ner.py --data_dir ${DATA_DIR}/${ENTITY}/ --labels ${DATA_DIR}/${ENTITY}/labels.txt --model_name_or_path "${MODEL_USED}" --output_dir ${SAVE_DIR}/${ENTITY} --max_seq_length ${MAX_LENGTH} --num_train_epochs ${NUM_EPOCHS} --per_device_train_batch_size ${BATCH_SIZE} --save_steps ${SAVE_STEPS} --seed ${SEED} --do_train --do_eval --do_predict     --overwrite_output_dir --report_to wandb --run_name "${RUN_NAME}" --logging_steps ${LOG_STEPS} --evaluation_strategy epoch --optim $OPTIMIZER

