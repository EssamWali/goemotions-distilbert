"""Fine-tune DistilBERT on GoEmotions as a single-label 28-class problem.

    python train.py --epochs 6

Two things differ from the notebook this came from, both deliberate:

* The notebook padded every example to 512 tokens. GoEmotions is Reddit comments
  and the median is under twenty tokens, so that spent most of the compute on
  padding. Dynamic padding to the longest sequence in each batch is equivalent -
  the attention mask already makes padding invisible to the model - and much
  faster.
* The notebook's loop referenced an undefined `labels`, so as saved it could not
  run. The loss is taken against `batch["labels"]`.
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 28
NEUTRAL = 27
ROOT = Path(__file__).parent


def first_label(example):
    """GoEmotions is multi-label; this treats it as single-label by keeping the
    first annotation. It is a simplification, and it caps how well any model here
    can score - an example genuinely carrying two emotions can only be half right."""
    labels = example["labels"]
    example["labels"] = labels[0] if labels else NEUTRAL
    return example


def build(split, tokenizer, max_length):
    data = load_dataset("go_emotions", "simplified")[split].map(first_label)
    data = data.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=max_length),
        batched=True,
    )
    keep = ("input_ids", "attention_mask", "labels")
    data = data.remove_columns([c for c in data.column_names if c not in keep])
    data.set_format("torch")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--out", default="checkpoint.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data = build("train", tokenizer, args.max_length)
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    # GoEmotions is dominated by `neutral`; unweighted cross-entropy converges on a
    # model that says `neutral` and scores respectably on accuracy while learning
    # nothing about the rare emotions. Weighting by inverse frequency is what makes
    # macro F1 worth reporting.
    counts = Counter(int(x) for x in train_data["labels"])
    frequencies = torch.tensor(
        [counts.get(i, 1) for i in range(NUM_LABELS)], dtype=torch.float
    )
    weights = 1.0 / frequencies
    weights = (weights / weights.sum()).to(device)
    loss_fn = CrossEntropyLoss(weight=weights)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        loop = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"epoch {epoch + 1} mean loss {total / len(loader):.4f}")
        torch.save(model.state_dict(), ROOT / args.out)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
