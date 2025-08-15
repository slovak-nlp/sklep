#!/bin/bash
set -e
ALL_TASKS="qa sts nli rte hate sentiment uner wikigold pos"
print_help() {
  echo SKLEP Evaluation Launcher
  echo
  echo Example usage:
  echo
  echo Run all tasks:
  echo sklep_run --tasks=all --model_name=gerulata/slovakbert
  echo
  echo Run qa and nli on card 1
  echo sklep_run --tasks=qa,nli --cuda=1 --model_name=gerulata/slovakbert
  echo
  echo Run with parameter sweep for qa and nli tasks:
  echo MODEL_NAME=gerulata/slovakbert sklep_run --tasks=qa --sweep --num_train_epochs=1 --learning_rate=1e-5 --warmup_ratio=0.05 --dropout=0 --wandb=sklep_qa
  echo
  echo Arguments:
  echo --tasks=[all $ALL_TASKS] Evaluation tasks to run. A comma separated list
  echo --model_name=[path] Name of model. Can be Huggingface HUB name or local path
  echo --out_dir=[path] Output location where models and logs are saved
  echo --wandb=\<project\> Name of WANDB project to log
  echo --cuda=[int] CUDA Cards to use. A comma separated list
  echo --help
  echo
  echo Sweep arguments:
  echo --sweep Enable sweep
  echo --num_train_epochs=\<int\> Number or trining epochs, int
  echo --warmup_ratio=\<float\> Ratio of sampples for the warmup phase
  echo --learning_rate=\<float\> Learning rate of scheduler
  echo --dropout=\<float\> Dropout regularization
}
if [ "$#" -eq 0 ]; then
  print_help
fi

# Reporting to
OUT_DIR=outdir
TASKS_ARG=all
# https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash

# Parse arguments
# TODO check argument value types
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --tasks=*) TASKS_ARG="${1#*=}" ;;
  --model_name=*) MODEL_NAME="${1#*=}" ;;
  --out_dir=*) OUT_DIR="${1#*=}" ;;
  --wandb=*) export WANDB_PROJECT="${1#*=}" ;;
  --cuda=*) export CUDA_VISIBLE_DEVICES="${1#*=}" ;;
  --help) print_help ;;
  --sweep) SWEEP=1 ;;
  --num_train_epochs=*) NUM_TRAIN_EPOCHS_ARG="${1#*=}" ;;
  --warmup_ratio=*) WARMUP_RATIO_ARG="${1#*=}" ;;
  --learning_rate=*) LEARNING_RATE_ARG="${1#*=}" ;;
  --dropout=*) DROPOUT_ARG="${1#*=}" ;;
  *) print_help && echo Error: bad parameter $1 && exit 1 ;;
  esac
  shift
done

# Parse task list, separated by comma
TASKS=$ALL_TASKS
if [[ "$TASKS_ARG" != 'all' ]]; then
  # split according to ,
  TASKS=${TASKS_ARG/,/ }
fi
echo Running $TASKS

# Print CUDA card
# TODO run_glue.py allocates all cards if this variable is not set, but uses only one
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  echo "Using default CUDA card."
else
  echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# Ensure MODEL_NAME is set
if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME is not set."
  exit 1
fi

MODEL_BASENAME=$(basename "$MODEL_NAME")
echo Model $MODEL_NAME
# Set WANDB_ARGS
# Is Wandb Mandatory?
if [ -z "$WANDB_PROJECT" ]; then
  echo "Warning: WANDB_PROJECT  is not set."
else
  WANDB_ARGS="--report_to wandb"
fi

# Task specific args as associative array
declare -A COMMAND
# Hyperparameters can be overriden from the command line
declare -A TASK_LR
declare -A TASK_EPOCHS
declare -A TASK_DROPOUT
declare -A TASK_WARMUP

# Check if "eval/" is in the current path
if [[ "$PWD" == *"eval/"* ]]; then
    PREFIX=""
else
    PREFIX="eval/"
fi

COMMAND[qa]="python ${PREFIX}scripts/run_qa.py
    --learning_rate 5e-5 \
    --num_train_epochs 2 \
    --dropout 0.0 \
    --warmup_ratio 0.3
"
TASK_LR[qa]=5e-5
TASK_EPOCHS[qa]=2
TASK_DROPOUT[qa]=0
TASK_WARMUP[qa]=0.3

COMMAND[sts]="python ${PREFIX}scripts/run_glue.py \
  --dataset_config sts \
  --task_name stsb \
"
TASK_LR[sts]=5e-5
TASK_EPOCHS[sts]=3
TASK_DROPOUT[sts]=0
TASK_WARMUP[sts]=0

COMMAND[nli]="python ${PREFIX}scripts/run_glue.py \
  --dataset_config nli \
  --task_name mnli \
"
TASK_LR[nli]=1e-5
TASK_EPOCHS[nli]=3
TASK_DROPOUT[nli]=0
TASK_WARMUP[nli]=0.3

COMMAND[rte]="python ${PREFIX}scripts/run_glue.py \
  --dataset_config rte \
  --task_name rte \
"
TASK_LR[rte]=2e-5
TASK_EPOCHS[rte]=5
TASK_DROPOUT[rte]=0.1
TASK_WARMUP[rte]=0.1

COMMAND[hate]="python ${PREFIX}scripts/run_classification.py
  --dataset_config hate-speech \
  --metric_name accuracy \
  --text_column_name text \
"
TASK_LR[hate]=5e-5
TASK_EPOCHS[hate]=4
TASK_DROPOUT[hate]=0.0
TASK_WARMUP[hate]=0.1

COMMAND[sentiment]="python ${PREFIX}scripts/run_classification.py \
  --dataset_config sentiment-analysis \
  --metric_name accuracy \
  --text_column_name text
"

TASK_LR[sentiment]=5e-5
TASK_EPOCHS[sentiment]=3
TASK_DROPOUT[sentiment]=0.0
TASK_WARMUP[sentiment]=0.0

COMMAND[uner]="python ${PREFIX}scripts/run_ner.py \
  --dataset_config ner-uner \
  --text_column_name tokens \
  --label_column_name ner_tags \
"

TASK_LR[uner]=5e-5
TASK_EPOCHS[uner]=6
TASK_DROPOUT[uner]=0.0
TASK_WARMUP[uner]=0.1

COMMAND[wikigold]="python ${PREFIX}scripts/run_ner.py \
  --dataset_config ner-wikigoldsk \
  --text_column_name tokens \
  --label_column_name ner_tags \
"

TASK_LR[wikigold]=5e-5
TASK_EPOCHS[wikigold]=6
TASK_DROPOUT[wikigold]=0.0
TASK_WARMUP[wikigold]=0.1

COMMAND[pos]="python ${PREFIX}scripts/run_ner.py \
  --dataset_config pos \
  --text_column_name tokens \
  --label_column_name pos_tags \
"

TASK_LR[pos]=5e-5
TASK_EPOCHS[pos]=6
TASK_DROPOUT[pos]=0.0
TASK_WARMUP[pos]=0.1

# Check if tasks in task list are valid
for TASK_NAME in $TASKS; do
  if [ -z "${COMMAND[$TASK_NAME]}" ]; then
    echo $TASK_NAME is not defined. Use one or multiple of $ALL_TASKS
    exit 1
  fi
done

# For directory name stamp
DATESTRING=$(date -Iminutes)

# NOTE
# In this setting new run creates a new directory according to current time
# If there are multiple results for the same model, which results to take into the final?
# this is resolved in the gather script

# Initialize seed values

# in evaluation run use 3 seeds
SEEDS="12 42 99"
if [ -n "$SWEEP" ]; then
  # in sweep run use only one seed
  SEEDS="42"
fi

# Run all tasks in sequence

for SEED_VALUE in $SEEDS; do
  for TASK_NAME in $TASKS; do
    OUT=$OUT_DIR/$TASK_NAME/$MODEL_BASENAME/$SEED_VALUE\_$DATESTRING
    RUN_NAME="$TASK_NAME--$MODEL_NAME-$SEED_VALUE-$DATESTRING"

    # Set hyperparameters - task default or from command line arguments
    LEARNING_RATE=$LEARNING_RATE_ARG
    if [ -z "$LEARNING_RATE_ARG" ]; then
      LEARNING_RATE=${TASK_LR[$TASK_NAME]}
    fi

    NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS_ARG
    if [ -z "$NUM_TRAIN_EPOCHS_ARG" ]; then
      NUM_TRAIN_EPOCHS=${TASK_EPOCHS[$TASK_NAME]}
    fi

    DROPOUT=$DROPOUT_ARG
    if [ -z "$DROPOUT_ARG" ]; then
      DROPOUT=${TASK_DROPOUT[$TASK_NAME]}
    fi

    WARMUP=$WARMUP_RATIO_ARG
    if [ -z "$WARMUP_RATIO_ARG" ]; then
      WARMUP=${TASK_WARMUP[$TASK_NAME]}
    fi

    # Set RUN_NAME, OUT and SWEEP_ARGS
    if [ -n "$SWEEP" ]; then
      # sweep is set
      # Set sweep args and output dir
      OUT="$OUT-E:$NUM_TRAIN_EPOCHS--W:$WARMUP_RATIO--LR:$LEARNING_RATE--D:$DROPOUT"
      RUN_NAME="$TASK_NAME--$MODEL_NAME--E:$NUM_TRAIN_EPOCHS--W:$WARMUP_RATIO--LR:$LEARNING_RATE--D:$DROPOUT--$DATESTRING"
    fi
    # prepare common args
    LAUNCH_ARGS="--model_name_or_path $MODEL_NAME \
            --dataset_name slovak-nlp/sklep \
            --learning_rate $LEARNING_RATE \
            --num_train_epochs $NUM_TRAIN_EPOCHS \
            --dropout $DROPOUT \
            --warmup_ratio $WARMUP \
            --logging_dir $OUT/logs \
            --output_dir $OUT/model \
	        --seed $SEED_VALUE \
            --run_name $RUN_NAME \
        "

    # Prepare output dir
    # Does the script restart training from checkpoint?
    rm -fR "$OUT"
    mkdir -p "$OUT"

    # TODO LOG Stdout and stderr
    # TODO LOG time of run

    # LAUNCH !
    ${COMMAND[$TASK_NAME]} \
      $LAUNCH_ARGS \
      $WANDB_ARGS \
      --do_train \
      --do_eval \
      --validation_split_name "test" \
      --per_device_train_batch_size 12 \
      --per_device_eval_batch_size 12 \
      --gradient_accumulation_steps 2 \
      --eval_accumulation_steps 2 \
      --log_level debug \
      --log_on_each_node \
      --logging_steps 100 \
      --save_steps 30000 \
      --max_seq_length 512 \
      --fp16 \
      --trust_remote_code True
  done
done

# TODO Call gather script
