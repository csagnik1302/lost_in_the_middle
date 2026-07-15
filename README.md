# Lost in the Middle: Long-Context Retrieval Analysis

[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Used-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-Used-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20experiments-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Status](https://img.shields.io/badge/Status-Analysis%20in%20progress-4C1?logo=github)](#project-progress)

This repository contains experiments that study **where information appears in a long prompt** and how that position affects a language model's ability to retrieve and use it. The work covers controlled key-value (KV) retrieval, open-domain question answering (QA), attention-sink measurements, and TREC RAG evaluation under discriminator and noise settings.

The repository is research code rather than a packaged library. Scripts are intentionally kept close to the experiments and datasets that produced the saved outputs.

## Tools and Libraries

| Component | Current research configuration | Notes |
| --- | --- | --- |
| Language | Python 3.10–3.12 | Versions observed in the committed experiment artifacts. |
| Deep learning | PyTorch | Tensor computation, model loading, and attention analysis. |
| LLM framework | Hugging Face Transformers | Text generation, tokenization, and causal language-model loading. |
| Hardware | NVIDIA CUDA GPU | Expected for the supplied Llama, Mistral, and attention runs. |
| Quantization | AWQ INT4 and bitsandbytes | Used by selected quantized model configurations. |
| Analysis | Matplotlib | Accuracy and attention plots. |
| Progress | tqdm | Iteration progress for long-running experiments. |

## Project Progress

| Workstream | What has been completed | Main outputs | Status |
| --- | --- | --- | --- |
| KV retrieval | Position-controlled JSON key-value experiments at 75, 140, and 300 keys | Raw generations and accuracy-by-position plots | Complete baseline |
| Multi-document QA | Gold-document position experiments at 10, 20, and 30 documents, plus oracle data | Exact-match counts and accuracy plots | Complete baseline |
| Attention-sink analysis | Per-token attention measurements across model heads and layers | Attention tensors and line charts | Initial analysis complete |
| TREC RAG discriminator | Answer generation, nugget extraction/assignment, and correctness scoring | JSONL pipeline outputs and six score plots | In progress |
| TREC RAG noise | Equivalent evaluation with noise-context conditions | JSONL outputs, error log, and score plots | In progress |
| GINGER nuggets | Passage annotation and nugget extraction workflow | Annotated passages and nugget JSONL | Initial pipeline complete |

> **Reproduction version:** use a commit hash, model revision, dataset copy, and dependency versions together when reporting or comparing a run. The repository currently does not pin these in a lockfile.

## What We Worked On

### 1. Controlled KV retrieval

`lost_in_the_middle/Project/KV_Retrieval/` tests whether a model can return the value for a requested key when that key-value pair appears at different positions in a JSON object.

- Datasets contain contexts with 75, 140, and 300 key-value pairs.
- The target key-value pair is moved through the context while all other pairs remain fixed.
- The model generates greedily; a UUID match is used as the correctness criterion.
- Saved output counts and accuracy plots are in `lost_in_the_middle/Project/Plots/`.

### 2. Multi-document QA

`lost_in_the_middle/Project/QA/` measures position sensitivity when answering Natural Questions-style prompts with 10, 20, or 30 retrieved documents.

- The answer-bearing (gold) document is inserted at selected positions.
- The prompt asks the model to use only the supplied search results.
- Predictions are evaluated with best-subspan exact match against the accepted answers.
- `Model_Oracle.py` supports the corresponding oracle-context experiment.

### 3. Attention-sink analysis

`attention_sink_analysis/` inspects attention patterns for the QA prompts using `output_attentions=True`.

- It averages per-token attention importance across heads, layers, and prompts.
- Both average-attention and no-dropout variants are included.
- Tensor outputs (`.pt`) and rendered plots (`.png`) are saved under `attention_sink_analysis/Plot/`.

### 4. TREC RAG 2024 analysis

`TREC-RAG_2024_Analysis/` extends the positional analysis to retrieval-augmented generation with TREC RAG inputs.

- **Discriminator** and **Noise** tracks vary the arrangement of gold documents and distracting context.
- The correctness pipeline generates an answer, extracts nuggets, scores them, assigns them to the generated response, and computes all, vital, weighted, strict, and non-strict scores.
- `GINGER/` annotates retrieved passages and extracts nuggets with a separate LLM-assisted pipeline.

## Methodology

The experiments follow the same high-level flow:

```text
Choose dataset and context size
        |
Place the gold evidence at one or more positions
        |
Build a prompt containing the full ordered context
        |
Run deterministic (greedy) LLM generation
        |
Score output against the known answer or nuggets
        |
Save raw outputs and plot accuracy / attention by position
```

This design isolates **position** as the main variable. The QA and KV runs use fixed datasets and deterministic decoding (`do_sample=False`), making comparisons between positions more meaningful. TREC runs add a richer, nugget-based evaluation that distinguishes broad, vital, weighted, and strict correctness.

## Results and Artifacts

The repository includes the experiment artifacts needed to inspect the current findings:

- KV retrieval accuracy curves for 75, 140, and 300 keys.
- QA exact-match accuracy curves for 10, 20, and 30 documents.
- Attention-importance curves for selected gold-document positions.
- TREC discriminator/noise score plots and JSONL pipeline outputs.

The plots are the source of record for observed positional trends. Before comparing new runs, use the same model, data split, context size, gold positions, and greedy-decoding configuration; changing any of these changes the experimental condition.

## Repository Layout

```text
lost_in_the_middle/
  Project/
    KV_Retrieval/             # Synthetic JSON key-value position experiments
    QA/                       # Natural Questions multi-document QA experiments
    Plots/                    # KV and QA raw counts, generations, and figures
attention_sink_analysis/      # Attention extraction, metrics, and figures
TREC-RAG_2024_Analysis/
  Discriminator_and_Noise/    # TREC RAG correctness pipelines and datasets
  GINGER/                     # Passage annotation and nugget extraction
```

## Setup

### Prerequisites

- Python 3.10+ (the saved bytecode indicates runs with Python 3.10–3.12)
- Git
- An NVIDIA GPU with CUDA for the supplied Llama/Mistral runs
- A Hugging Face account/token with access to the selected gated model, where applicable

The included configurations use models such as `meta-llama/Meta-Llama-3.1-8B-Instruct`, its AWQ INT4 variant, `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`, and `Qwen/Qwen2.5-14B-Instruct`. The full-precision models can require substantial GPU memory; quantized variants are the more practical starting point.

### Create an environment

```bash
git clone <your-repository-url>
cd lost_in_the_middle

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install torch transformers accelerate bitsandbytes matplotlib tqdm
```

For the TREC correctness scripts, install the Flash Attention build that matches your CUDA, PyTorch, and compiler environment if you keep `attn_implementation="flash_attention_2"`. Otherwise, change that setting to an attention implementation supported by your installed `transformers` and hardware.

### Configure the Hugging Face token

The historical scripts read a token from an absolute `API_KEY` path. For a portable setup, create a local token file outside version control and update the scripts to read it, or use an environment variable.

```bash
# macOS/Linux
export HF_TOKEN='hf_...'

# Windows PowerShell
# $env:HF_TOKEN = 'hf_...'
```

Then replace the token-file reads in the script you run with:

```python
import os
TOKEN = os.environ["HF_TOKEN"]
```

Never commit a token. `.gitignore` already excludes `API_KEY`.

### Make paths portable

Several scripts contain machine-specific paths such as `/home/irlab/sagnik/...`. Before running an experiment, update these paths to your clone or define a common project root:

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # adjust for the current script
DATA_PATH = REPO_ROOT / "lost_in_the_middle" / "Project" / "QA" / "Data"
```

Also create the relevant output directory before a run, for example `lost_in_the_middle/Project/Plots/` or a new run-specific folder under it. Keep generated files separate from the committed reference artifacts when comparing experiments.

## Run an Experiment

Run commands from the repository root after updating the path and token configuration.

### KV retrieval

1. Open `lost_in_the_middle/Project/KV_Retrieval/Model.py`.
2. Select the data file and `positions` to test.
3. Set a model you can run locally and choose an output filename that does not overwrite a reference result.
4. Run:

```bash
python lost_in_the_middle/Project/KV_Retrieval/Model.py
```

The script writes per-example generations, aggregates correct UUID matches, and saves an accuracy-by-position plot.

### QA retrieval

1. Open `lost_in_the_middle/Project/QA/Model.py`.
2. Set `PATHS` to the desired context length and gold-document positions.
3. Configure the model, token, and output paths.
4. Run:

```bash
python lost_in_the_middle/Project/QA/Model.py
```

The script creates prompts with `prompt_creation_qa.py`, computes best-subspan exact match with `response_matching.py`, and plots accuracy against the gold document position.

### Attention analysis

Set `prompt_count`, `doc_count`, `gold_count`, model configuration, and output directory in one of the scripts in `attention_sink_analysis/`, then run it. The average-attention script is a good baseline:

```bash
python attention_sink_analysis/average_attention_score_computation.py
```

Attention extraction is memory intensive because it retains attention tensors for every layer and head. Start with one prompt and a small document count.

### TREC correctness analysis

For either the `Discriminator` or `Noise` pipeline:

1. Configure the repository-relative import paths, data path, model, and token in `Correctness_Analysis/main.py`.
2. Confirm the corresponding `Data/` JSONL input files are available.
3. Run the pipeline:

```bash
python TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/main.py
```

The checkpoint file `misc/itr_index.json` allows this long-running pipeline to resume. Use a new checkpoint/output name for an independent run.

## Reproducibility Checklist

- Record the exact model revision, quantization, `transformers`, PyTorch, and CUDA versions.
- Keep `do_sample=False`, `max_new_tokens`, gold positions, and dataset files fixed when making a positional comparison.
- Save raw generations alongside aggregate scores so incorrect matches can be inspected.
- Use a distinct output directory for each model/configuration.
- Report both the aggregate curve and the underlying sample count for each position.

## Current Limitations

- Paths and token loading are hard-coded in several original experiment scripts and must be adapted per machine.
- There is no lockfile or pinned dependency list yet; environment versions should be recorded for reproducibility.
- The QA/KV notes flag that prompt ordering and batch-boundary handling need further robustness checks.
- Runs are GPU-oriented and may be slow or impractical on CPU-only machines.