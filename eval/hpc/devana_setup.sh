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

# pyproject.toml has no [build-system], and `pip install .` fails on the flat
# eval/ + images/ layout (setuptools auto-discovery refuses to build a wheel).
# uv (with [tool.uv] package = false) sidesteps this; install it if missing.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv .venv-devana --python "$(command -v python3)"
UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv-devana" uv sync

# Pre-download the model(s) and dataset here while the login node has
# internet access, so the compute-node job doesn't need outbound access.
# Adjust MODELS if you add more.
export HF_HOME="$REPO_ROOT/.hf-cache"
MODELS=(jhu-clsp/mmBERT-base jhu-clsp/mmBERT-small)
for M in "${MODELS[@]}"; do
  .venv-devana/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$M')
"
done
.venv-devana/bin/python3 -c "
from datasets import load_dataset
load_dataset('slovak-nlp/sklep')
"

echo
echo "Setup complete. Before submitting jobs:"
echo "  1. Run 'sprojects' to find your SLURM account name, then pass it as"
echo "     sbatch --account=<name> eval/hpc/devana_run.sbatch"
echo "  2. Create ~/.config/sklep/wandb.env with:"
echo "       export WANDB_BASE_URL=\"https://wandb.tool.kinit.sk\""
echo "       export WANDB_API_KEY=\"<your key>\""
echo "     (skip wandb login entirely -- this client version's key-length"
echo "     validation rejects on-prem keys, but reading the env var at"
echo "     runtime has no such check)"
