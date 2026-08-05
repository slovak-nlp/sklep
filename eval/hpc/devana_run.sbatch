#!/bin/bash
#SBATCH -J sklep-mmbert
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 2-00:00
#SBATCH --account=CHANGE_ME       # fill in with the output of `sprojects`
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Submit with: sbatch eval/hpc/devana_run.sbatch
# (run eval/hpc/devana_setup.sh once on the login node first)

set -e
# git isn't available on compute nodes -- SLURM sets this to the directory
# `sbatch` was invoked from, which must be the repo root.
export REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
mkdir -p logs

module purge
module load Python/3.11.5-GCCcore-13.2.0

# Skip `wandb login` -- this client version's key-length validation rejects
# on-prem key formats. Reading the key from the env at runtime has no such
# check, so just source it from a file you created once (see devana_setup.sh).
if [ -f ~/.config/sklep/wandb.env ]; then
  source ~/.config/sklep/wandb.env
fi

# Model + dataset were pre-downloaded into this cache during setup, in case
# the gpu partition's compute nodes lack outbound internet access.
export HF_HOME="$REPO_ROOT/.hf-cache"
export HF_HUB_OFFLINE=1

MODEL_NAME="${MODEL_NAME:-jhu-clsp/mmBERT-base}"
export RUN_PYTHON="$REPO_ROOT/.venv-devana/bin/python3"

# SEEDS unset -> sklep_run.sh's own default (the paper's 3-seed protocol: 12 42 99)
SEEDS_FLAG=""
if [ -n "$SEEDS" ]; then
  SEEDS_FLAG="--seeds=$SEEDS"
fi

./eval/sklep_run.sh \
  --tasks=all \
  --model_name="$MODEL_NAME" \
  $SEEDS_FLAG \
  --wandb=sklep-mmbert \
  --out_dir=outdir-devana
