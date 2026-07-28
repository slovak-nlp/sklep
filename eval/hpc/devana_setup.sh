#!/bin/bash
# One-time environment setup for the Devana cluster (userdocs.hpc.sav.sk).
# Run this ON THE LOGIN NODE (has internet access; compute nodes may not).
#
# Usage: bash eval/hpc/devana_setup.sh
set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

module purge
module load Python/3.11.5-GCCcore-13.2.0

python3 -m venv .venv-devana
source .venv-devana/bin/activate
pip install -U pip
pip install .

# Pre-download the model(s) and dataset here while the login node has
# internet access, so the compute-node job doesn't need outbound access.
# Adjust MODELS if you add more.
export HF_HOME="$REPO_ROOT/.hf-cache"
MODELS=(jhu-clsp/mmBERT-base jhu-clsp/mmBERT-small)
for M in "${MODELS[@]}"; do
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$M')
"
done
python3 -c "
from datasets import load_dataset
load_dataset('slovak-nlp/sklep')
"

echo
echo "Setup complete. Before submitting jobs:"
echo "  1. Run 'sprojects' to find your SLURM account name."
echo "  2. Edit eval/hpc/devana_run.sbatch and set --account and MODEL_NAME."
echo "  3. Log in to W&B once: source .venv-devana/bin/activate && wandb login"
