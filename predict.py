"""Classify a sentence with the fine-tuned checkpoint.

    python predict.py "I can't believe this actually worked"
    python predict.py            # reads lines from stdin
"""

import argparse
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 28


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="text to classify")
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    names = load_dataset("go_emotions", "simplified")["train"].features["labels"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    lines = [" ".join(args.text)] if args.text else [ln.strip() for ln in sys.stdin]

    for line in filter(None, lines):
        inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=128)
        with torch.inference_mode():
            probabilities = model(**inputs).logits.softmax(dim=1)[0]
        top = torch.topk(probabilities, args.top)
        guesses = ", ".join(
            f"{names[i]} {p:.2f}" for p, i in zip(top.values, top.indices)
        )
        print(f"{line}\n  -> {guesses}\n")


if __name__ == "__main__":
    main()
