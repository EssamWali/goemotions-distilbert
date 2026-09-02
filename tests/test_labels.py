"""The label simplification is the one piece of logic that has to agree between
training and evaluation. If they ever disagree the reported numbers are fiction,
so it is worth a test even in a project this small."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import NEUTRAL, first_label as eval_first_label  # noqa: E402
from train import first_label as train_first_label  # noqa: E402


def test_first_annotation_is_kept():
    assert train_first_label({"labels": [4, 11, 27]})["labels"] == 4
    assert eval_first_label({"labels": [4, 11, 27]})["label_id"] == 4


def test_empty_annotation_falls_back_to_neutral():
    assert train_first_label({"labels": []})["labels"] == NEUTRAL
    assert eval_first_label({"labels": []})["label_id"] == NEUTRAL


def test_training_and_evaluation_agree():
    for labels in ([], [0], [27], [3, 9], [12, 0, 5]):
        train_value = train_first_label({"labels": list(labels)})["labels"]
        eval_value = eval_first_label({"labels": list(labels)})["label_id"]
        assert train_value == eval_value, labels
