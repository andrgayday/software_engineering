"""График качества готовой модели по сохранённому отчёту. Требует uv sync --group dev."""
import argparse
import json
from pathlib import Path
from sentiment_app.config import ROOT


def generate(report_path: Path, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(report_path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    matrix = report["confusion_matrix"]
    axes[0].imshow(matrix, cmap="Blues")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axes[0].text(j, i, str(value), ha="center", va="center",
                         color="white" if value > max(map(max, matrix)) / 2 else "black")
    axes[0].set(xticks=range(3), yticks=range(3), xticklabels=report["labels"],
                yticklabels=report["labels"], xlabel="Predicted", ylabel="Actual", title="Confusion matrix")
    bars = axes[1].bar(["Accuracy", "Macro-F1"], [report["accuracy"], report["macro_f1"]], color=["#304fb1", "#167348"])
    axes[1].bar_label(bars, fmt="%.3f")
    axes[1].set(ylim=(0, 1.15), title=f'Demonstration only: {report["sample_size"]} examples')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/pretrained.metrics.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/quality.svg")
    args = parser.parse_args()
    generate(args.report, args.output)
