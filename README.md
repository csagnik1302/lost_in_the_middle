# Lost in the Middle

Research scripts for studying how the position of relevant information in a long context affects language-model retrieval. The repository contains controlled factoid experiments and attention-score analysis for long-context prompts.

This is experimental research code, not a packaged library. The scripts retain the original model choices, absolute paths, and output conventions used for the experiments; adapt those settings before running them on another machine.

## Included experiments

### Key-value retrieval

`Factoid_Analysis/KV_Retrieval/` measures whether a model can retrieve the value for a requested key from a JSON object when the target key-value pair is placed at different positions.

- `Model.py` runs greedy generation over selected target positions and scores a response by matching its UUID to the expected value.
- `prompt_creation_kv.py` constructs the JSON retrieval prompt and moves the target pair to the requested index.
- `../Plots/` contains saved generations and accuracy figures for 75-, 140-, and 300-key configurations.

### Multi-document QA

`Factoid_Analysis/QA/` measures answer accuracy as the answer-bearing document moves through a set of retrieved documents.

- `Model.py` evaluates configured JSONL datasets, using greedy generation and answer-substring exact match.
- `Model_Oracle.py` is the corresponding oracle-context experiment.
- `prompt_creation_qa.py` formats documents, questions, and accepted answers into a model prompt.
- `response_matching.py` normalizes predictions and accepted answers before evaluation.

The checked-in plots in `Factoid_Analysis/Plots/` cover contexts with 10, 20, and 30 documents.

### Attention analysis

`attention_sink_analysis/` contains scripts that extract generation-time attention weights from a causal language model and aggregate them across heads, layers, and prompts. Variants are provided for average attention, dropout, and no-dropout analysis.

## Repository layout

```text
Factoid_Analysis/
  KV_Retrieval/                 # Synthetic JSON key-value retrieval
  QA/                           # Multi-document question answering
  Plots/                        # Committed generated outputs and figures
attention_sink_analysis/
  data/                         # Small checked-in probe data and export helper
  *.py                          # Attention aggregation experiments
test.py                         # Local inspection helper for a TREC retrieval file
```

## Requirements

- Python 3.10+
- PyTorch with CUDA support for the supplied model configurations
- An NVIDIA GPU with enough memory for the selected model
- A Hugging Face token with access to the selected model, where required

Install the Python dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch transformers matplotlib tqdm regex
```

The default scripts reference `meta-llama/Meta-Llama-3.1-8B-Instruct` or `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`. You may need additional model-specific dependencies or a compatible quantization backend for AWQ models.

## Data and credentials

The full KV and QA datasets are **not included** in this checkout. Before running an experiment, obtain or recreate the required JSONL files and update the input paths in the relevant script.

The original scripts read a Hugging Face token from an absolute `API_KEY` file. For a portable setup, use an environment variable instead:

```bash
export HF_TOKEN='hf_...'
```

Then replace the token-file read in the script you are running with:

```python
import os

TOKEN = os.environ["HF_TOKEN"]
```

Never commit a token. `API_KEY` is ignored by Git.

## Running an experiment

All experiment scripts currently contain machine-specific paths such as `/home/irlab/sagnik/...`. Update the input-data paths, output paths, and token loading before executing them.

### KV retrieval

In `Factoid_Analysis/KV_Retrieval/Model.py`, set the dataset path, model, output paths, and the `positions` to evaluate. Then run from that directory so the local prompt module resolves correctly:

```bash
cd Factoid_Analysis/KV_Retrieval
python Model.py
```

The script writes one JSONL record per generation, stores aggregate correct counts, and saves an accuracy-by-position figure.

### QA retrieval

In `Factoid_Analysis/QA/Model.py`, configure `PATHS`, the model, token loading, and output locations. Run:

```bash
cd Factoid_Analysis/QA
python Model.py
```

For the oracle configuration, update `Model_Oracle.py` and run `python Model_Oracle.py`. Note that the checked-in oracle script needs `correct_array` initialized before it can append its result.

### Attention scores

Set the QA data path, model, output directory, `prompt_count`, `doc_count`, and `gold_count` in the attention script you choose. For example:

```bash
python attention_sink_analysis/average_attention_score_computation.py
```

Attention extraction is memory-intensive because it requests attention tensors during generation. Start with one prompt and a small number of documents.

## Reproducibility notes

- Keep the model revision, quantization, dataset version, target positions, and generation settings fixed when comparing positions.
- The main KV and QA scripts use greedy decoding (`do_sample=False`) with `max_new_tokens=70`.
- Save new runs to distinct output files so that the committed reference artifacts remain intact.
- Record the PyTorch, Transformers, CUDA, and GPU versions used for every run.

## Known limitations

- Dataset, credential, and output paths are hard-coded in the original scripts.
- There is no lockfile or pinned dependency set.
- Several scripts assume a CUDA device and will not run unchanged on CPU-only systems.
- The repository includes reference outputs and plots, but not the large datasets needed to reproduce them end to end.
