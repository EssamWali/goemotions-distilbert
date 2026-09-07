"""Draw the per-class F1 chart in the README from metrics.json.

    python scripts/plot_metrics.py            # writes docs/img/per-class-f1.png

Reads metrics.json at the repository root (written by evaluate.py) and saves a
horizontal bar chart of per-class F1 on the test split, sorted from strongest to
weakest, with each class's support printed at the end of its bar. Needs only
matplotlib, which is not in requirements.txt because nothing else here uses it:

    pip install matplotlib
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "metrics.json"
OUT = ROOT / "docs" / "img" / "per-class-f1.png"

SURFACE = "#fcfcfb"
BAR = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SKIP = {"accuracy", "macro avg", "weighted avg"}


def main():
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = [
        (name, row["f1-score"], int(row["support"]))
        for name, row in metrics["per_class"].items()
        if name not in SKIP
    ]
    rows.sort(key=lambda r: r[1])  # ascending, so the strongest ends up on top
    names = [r[0] for r in rows]
    f1 = [r[1] for r in rows]
    support = [r[2] for r in rows]
    macro = metrics["macro_f1"]

    fig, ax = plt.subplots(figsize=(8.5, 8.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y = range(len(rows))
    ax.barh(y, f1, height=0.56, color=BAR, zorder=3)
    ax.axvline(macro, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(
        macro + 0.008, len(rows) - 0.35, f"macro F1 {macro:.3f}",
        color=TEXT_SECONDARY, fontsize=8.5, va="center", ha="left",
    )

    for i, (value, n) in enumerate(zip(f1, support)):
        ax.text(value + 0.01, i, f"{value:.2f}", va="center", ha="left",
                fontsize=8.5, color=TEXT, zorder=4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        ax.text(1.0, i, f"n = {n:,}", va="center", ha="right",
                fontsize=8, color=TEXT_SECONDARY)

    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9, color=TEXT)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    ax.set_xlabel("F1 on the GoEmotions test split (5,427 examples)",
                  fontsize=9, color=TEXT_SECONDARY, labelpad=8)
    ax.set_title(
        "Per-class F1, DistilBERT fine-tune, 28 emotions",
        fontsize=11, color=TEXT, loc="left", pad=26,
    )
    ax.text(
        0, 1.018, "sorted by F1; n is the number of test examples in that class",
        transform=ax.transAxes, fontsize=8.5, color=TEXT_SECONDARY, va="bottom",
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
