"""Проверить готовую модель на небольшом демонстрационном наборе."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from sentiment_app.config import MODEL_NAME, MODEL_REVISION, ROOT
from sentiment_app.models.pretrained import load_pretrained, predict_batch

LABELS = ["negative", "neutral", "positive"]


def metrics(actual, predicted):
    if not actual or len(actual) != len(predicted):
        raise ValueError("Нужны непустые списки меток одинаковой длины")
    matrix = [[0] * 3 for _ in range(3)]
    for truth, guess in zip(actual, predicted):
        matrix[LABELS.index(truth)][LABELS.index(guess)] += 1
    f1s = []
    for i in range(3):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(3) if j != i)
        fn = sum(matrix[i][j] for j in range(3) if j != i)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return {"accuracy": sum(matrix[i][i] for i in range(3)) / len(actual),
            "macro_f1": sum(f1s) / 3, "confusion_matrix": matrix, "labels": LABELS}


def run(dataset: Path, output: Path, offline: bool):
    with dataset.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not row["text"].strip() or row["label"] not in LABELS for row in rows):
        raise ValueError("CSV должен содержать непустой text и метку negative/neutral/positive")
    torch.set_num_threads(2)
    tokenizer, model = load_pretrained(local_files_only=offline)
    predictions = []
    for start in range(0, len(rows), 16):
        predictions.extend(predict_batch([row["text"] for row in rows[start:start + 16]], tokenizer, model))
    report = {"model": MODEL_NAME, "revision": MODEL_REVISION,
              "dataset": dataset.name, "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
              "sample_size": len(rows),
              "limitation": "Небольшой набор примеров составлен вручную для демонстрации. "
                            "Метрики не являются оценкой качества на реальных пользовательских данных.",
              **metrics([row["label"] for row in rows], predictions),
              "examples": [{**row, "predicted": guess} for row, guess in zip(rows, predictions)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["sample_size", "accuracy", "macro_f1"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/examples.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/pretrained.metrics.json")
    parser.add_argument("--offline", action="store_true", help="Только локальный кеш Hugging Face")
    args = parser.parse_args()
    run(args.dataset, args.output, args.offline)
