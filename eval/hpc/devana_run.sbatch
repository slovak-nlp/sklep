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
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
mkdir -p logs

module purge
module load Python/3.11.5-GCCcore-13.2.0
source .venv-devana/bin/activate

# Model + dataset were pre-downloaded into this cache during setup, in case
# the gpu partition's compute nodes lack outbound internet access.
export HF_HOME="$REPO_ROOT/.hf-cache"
export HF_HUB_OFFLINE=1

MODEL_NAME="${MODEL_NAME:-jhu-clsp/mmBERT-base}"

./eval/sklep_run.sh \
  --tasks=all \
  --model_name="$MODEL_NAME" \
  --seeds=42 \
  --wandb=sklep-mmbert \
  --out_dir=outdir-devana
