# skLEP: A Slovak General Language Understanding Benchmark

skLEP is a GLUE-style benchmark for evaluating Slovak natural language understanding (NLU) models.

## Setup

This project uses `uv` for environment management. If you don't have `uv` installed, you can find installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

To create the virtual environment and install the necessary dependencies, run the following command:

```bash
uv pip install .
```

## Evaluation

Before running the evaluation, make sure to activate the virtual environment:

```bash
source .venv/bin/activate
```

The evaluation script `sklep_run.sh` is used to run the benchmark tasks.

### Usage

The script can be run with various arguments to customize the evaluation:

- `--tasks`: A comma-separated list of tasks to run. Available tasks are: `qa`, `sts`, `nli`, `rte`, `hate`, `sentiment`, `uner`, `wikigold`, `pos`. Use `all` to run all tasks.
- `--model_name`: The name or path of the Hugging Face model to evaluate.
- `--out_dir`: The directory to save the output logs and models.
- `--wandb`: The name of the Weights & Biases project to log the results.
- `--cuda`: A comma-separated list of CUDA devices to use.

### Example

To run all tasks with the `gerulata/slovakbert` model, use the following command:

```bash
./eval/sklep_run.sh \
    --tasks=all \
    --model_name=gerulata/slovakbert
```

### Parameter Sweep

The script also supports parameter sweeps for hyperparameter optimization. To enable sweep mode, use the `--sweep` flag and provide the desired hyperparameter values.

```bash
MODEL_NAME=gerulata/slovakbert ./eval/sklep_run.sh \
    --tasks=qa \
    --sweep \
    --num_train_epochs=1 \
    --learning_rate=1e-5 \
    --warmup_ratio=0.05 \
    --dropout=0 \
    --wandb=sklep_qa
```

## License

This project is licensed under the MIT License.

## Citation

If you use this benchmark in your research, please cite it's associated paper (<https://arxiv.org/abs/2506.21508>) as follows:

```bibtex
@misc{suppa2025sklepslovakgenerallanguage,
      title={skLEP: A Slovak General Language Understanding Benchmark},
      author={Marek Šuppa and Andrej Ridzik and Daniel Hládek and Tomáš Javůrek and Viktória Ondrejová and Kristína Sásiková and Martin Tamajka and Marián Šimko},
      year={2025},
      eprint={2506.21508},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.21508},
}
```

