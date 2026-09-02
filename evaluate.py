"""Measure the fine-tuned checkpoint on the GoEmotions test split.

The notebook that produced this checkpoint never evaluated it on held-out data -
it printed a training loss and stopped. This is the missing half.

Accuracy alone would be misleading here. The label distribution is dominated by
`neutral`, so a model that answered `neutral` every time would score respectably
while being useless. Macro F1 - which weights all 28 classes equally - is the
number that says whether the fine-tune actually learned the rare emotions, and
the majority-class baseline below is what any real number has to beat.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 28
NEUTRAL = 27
ROOT = Path(__file__).parent


def first_label(example):
    """GoEmotions is multi-label; the fine-tune treats it as single-label.

    Keeping only the first annotated label is the simplification the training run
    made, so evaluation has to make the same one or the numbers mean nothing. It
    also puts a ceiling on any score reported here: examples genuinely carrying
    two emotions can only ever be half right.
    """
    labels = example["labels"]
    example["label_id"] = labels[0] if labels else NEUTRAL
    return example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    raw = load_dataset("go_emotions", "simplified")[args.split]
    names = raw.features["labels"].feature.names
    raw = raw.map(first_label)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = raw.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=args.max_length),
        batched=True,
    )
    encoded = encoded.remove_columns([c for c in encoded.column_names if c not in
                                      ("input_ids", "attention_mask", "label_id")])
    encoded = encoded.rename_column("label_id", "labels")
    encoded.set_format("torch")

    from transformers import DataCollatorWithPadding

    loader = DataLoader(
        encoded,
        batch_size=args.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    predictions, truth = [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            truth.extend(labels.tolist())

    accuracy = accuracy_score(truth, predictions)
    macro_f1 = f1_score(truth, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(truth, predictions, average="weighted", zero_division=0)

    # The bar any fine-tune has to clear: always answer the most common class.
    majority = max(set(truth), key=truth.count)
    baseline_accuracy = accuracy_score(truth, [majority] * len(truth))
    baseline_macro = f1_score(
        truth, [majority] * len(truth), average="macro", zero_division=0
    )

    print(f"\nexamples          {len(truth)}")
    print(f"accuracy          {accuracy:.4f}   (majority baseline {baseline_accuracy:.4f})")
    print(f"macro F1          {macro_f1:.4f}   (majority baseline {baseline_macro:.4f})")
    print(f"weighted F1       {weighted_f1:.4f}")
    print("\n" + classification_report(
        truth, predictions, target_names=names, zero_division=0, digits=3
    ))

    report = classification_report(
        truth, predictions, target_names=names, zero_division=0, output_dict=True
    )
    (ROOT / "metrics.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "examples": len(truth),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "majority_baseline_accuracy": baseline_accuracy,
                "majority_baseline_macro_f1": baseline_macro,
                "per_class": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote metrics.json")


if __name__ == "__main__":
    main()
