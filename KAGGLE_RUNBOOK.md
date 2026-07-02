# Kaggle Experiment Runbook

Use `stanza.ipynb` for substantive training. Local scripts are for smoke tests only.

## Setup

1. Open `stanza.ipynb` on Kaggle.
2. Attach these datasets:
   - `nurikw3/chagatai-test`
   - `nurikw3/charlm-corpus`
3. Enable GPU acceleration. The notebook is prepared for Kaggle GPU runtime and checks `torch.cuda.device_count()`.

## Run Experiments

Run all cells top to bottom. By default, the notebook trains:

- Chagatai-only forward CharLM
- Chagatai+UZS forward CharLM
- two tokenizer variants using the Chagatai+UZS CharLM

For a quick shakedown, set `MAX_EXPERIMENTS = 1` in the experiment matrix cell before running. The notebook will train only the CharLM needed by the selected experiment.

## Collect Results

After completion, download:

- `/kaggle/working/chagatai_sbd/results_run_blocks.txt`
- `/kaggle/working/chagatai_sbd/best_metrics.json`
- any model `.pt` files worth keeping

To preserve the current best completed model (`nocharlm_seed7` from Run 7),
download these exact files:

- `/kaggle/working/chagatai_sbd/nocharlm_seed7/models/nocharlm_seed7.pt`
- `/kaggle/working/chagatai_sbd/nocharlm_seed7/train_config.json`
- `/kaggle/working/chagatai_sbd/metrics/nocharlm_seed7.json`

Together, these store the model weights, training configuration, and evaluated
test metrics needed to reproduce or compare the run later.

Append real Kaggle metrics locally with:

```bash
uv run python append_results.py /path/to/results_run_blocks.txt
```

`append_results.py` refuses duplicate `## Run N:` headings, so adjust `START_RUN_NUMBER` in the notebook if you rerun experiments after appending previous runs.
