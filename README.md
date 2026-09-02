# GoEmotions DistilBERT

DistilBERT fine-tuned to label a sentence with one of 28 emotions, trained on
[GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
— 58k Reddit comments annotated by humans.

```
$ python predict.py "thanks so much, you saved me hours"
  -> gratitude 1.00, amusement 0.00, annoyance 0.00

$ python predict.py "I'm not sure I understand the question"
  -> confusion 0.90, neutral 0.09, annoyance 0.00
```

## Results

Measured on the held-out **test** split, 5,427 examples it never saw.

| | model | majority-class baseline |
| --- | --- | --- |
| accuracy | **0.535** | 0.296 |
| macro F1 | **0.442** | 0.016 |
| weighted F1 | 0.528 | — |

Accuracy is the less interesting number. Nearly a third of this dataset is
`neutral`, so a model that answers `neutral` to everything gets 0.296 and has
learned nothing. Macro F1 weights all 28 classes equally, which is why it is the
number reported first: 0.442 against a baseline of 0.016 says the fine-tune
actually learned the rare emotions rather than just the common one.

Per-class F1 spans a wide range, and the spread is the useful part:

| strongest | | weakest | |
| --- | --- | --- | --- |
| gratitude | 0.803 | relief | 0.167 |
| amusement | 0.784 | pride | 0.182 |
| admiration | 0.665 | realization | 0.192 |
| love | 0.668 | grief | 0.267 |
| fear | 0.621 | nervousness | 0.286 |

The strong classes are the ones with a reliable surface form — "thank you" is
gratitude, "lol" is amusement. The weak ones are the rare classes with no
characteristic wording: `grief` has six examples in the whole test split, and
`realization` and `relief` are distinctions a human annotator finds hard too.

Full per-class precision, recall and support: [`metrics.json`](metrics.json).

## What the training does

Standard fine-tune of `distilbert-base-uncased`, six epochs, AdamW at 5e-5,
batch size 16, with two decisions worth calling out.

**Multi-label collapsed to single-label.** GoEmotions annotates each comment with
any number of emotions. This keeps only the first and treats the task as 28-way
single-label classification. That is a simplification, and it puts a ceiling on
every number above — an example genuinely carrying two emotions can only ever be
counted half right. Doing it properly means a sigmoid head with per-class
thresholds, which is the obvious next version.

**Class-weighted loss.** Weights are inverse frequency, normalised. Without them
cross-entropy converges on a model that says `neutral`, scores 0.296, and looks
fine until you compute macro F1. This is the change that makes the rare classes
score at all.

## Running it

```
pip install -r requirements.txt

python train.py --epochs 6     # writes checkpoint.pt
python evaluate.py             # metrics on the test split, writes metrics.json
python predict.py "some text"  # or pipe lines on stdin
```

The checkpoint is **not in this repository**. DistilBERT's weights are 256 MB and
the AdamW optimiser state another 512 MB, both over GitHub's 100 MB file limit,
and the optimiser state is only good for resuming a run that finished long ago.
`train.py` reproduces the checkpoint in roughly fifteen minutes on a mid-range
GPU.

## Where this came from, and what was wrong with it

[`notebooks/exploration.ipynb`](notebooks/exploration.ipynb) is the original
working notebook, kept as the record of the run that produced the checkpoint. It
does not execute top to bottom, and the scripts here are the runnable version.
Three things were fixed on the way out of it:

**It was never evaluated.** The notebook printed a training loss and stopped. A
fine-tune with no held-out numbers is a claim, not a result — `evaluate.py` is
the missing half, and every number on this page comes from it.

**The training loop referenced an undefined name.** `loss_fn(logits, labels)`,
with `labels` never assigned inside the loop. It only ever ran because a stale
global was left over from an out-of-order cell execution.

**Every example was padded to 512 tokens.** GoEmotions comments have a median
length under twenty tokens, so most of the compute went into padding. Dynamic
padding to the longest sequence in each batch is mathematically identical — the
attention mask already hides padding from the model — and several times faster.

## Known limitations

- Single-label treatment of a multi-label dataset, as above.
- Trained and evaluated on Reddit comments; it will do worse on anything that
  does not read like one.
- `neutral` absorbs a lot. "why does nothing in this codebase make sense" comes
  back as `neutral 0.87` when `annoyance` is the better answer.
- No calibration work. The probabilities are softmax outputs and should not be
  read as confidence.
