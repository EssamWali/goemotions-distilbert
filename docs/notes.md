# Development notes

## Where this came from, and what was wrong with it

[`notebooks/exploration.ipynb`](../notebooks/exploration.ipynb) is the original
working notebook, kept as the record of the run that produced the checkpoint. It
does not execute top to bottom, and the scripts at the repository root are the
runnable version. Three things were fixed on the way out of it:

**It was never evaluated.** The notebook printed a training loss and stopped. A
fine-tune with no held-out numbers is a claim, not a result. `evaluate.py` is
the missing half, and every number in the README comes from it.

**The training loop referenced an undefined name.** `loss_fn(logits, labels)`,
with `labels` never assigned inside the loop. It only ever ran because a stale
global was left over from an out-of-order cell execution. `train.py` takes the
loss against `batch["labels"]`.

**Every example was padded to 512 tokens.** GoEmotions comments have a median
length under twenty tokens, so most of the compute went into padding. Dynamic
padding to the longest sequence in each batch (`DataCollatorWithPadding`) is
mathematically identical, because the attention mask already hides padding from
the model, and several times faster.

## Why the checkpoint is not in the repository

DistilBERT's weights are 256 MB and the AdamW optimiser state another 512 MB,
both over GitHub's 100 MB file limit, and the optimiser state is only good for
resuming a run that finished long ago. Both are ignored by `.gitignore`.
`train.py` reproduces the checkpoint in roughly fifteen minutes on a mid-range
GPU.

## What CI checks

Training needs a GPU and fifteen minutes, so CI does not attempt it. The
workflow in `.github/workflows/ci.yml` installs CPU-only torch, checks that
`train.py`, `evaluate.py` and `predict.py` import, and runs `tests/` (pytest).
The one test file, `tests/test_labels.py`, checks that training and evaluation
derive a single label from the multi-label annotation in the same way. If they
ever disagreed, the reported numbers would not describe the model that was
trained.

## Regenerating the README chart

`scripts/plot_metrics.py` reads `metrics.json` and writes
`docs/img/per-class-f1.png`. It needs matplotlib, which is not in
`requirements.txt` because nothing else uses it.
