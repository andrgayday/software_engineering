import pytest
from evaluation.run import metrics


def test_metrics_with_confusion():
    result = metrics(["negative", "neutral", "positive"], ["neutral", "neutral", "positive"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((0 + 2 / 3 + 1) / 3)
    assert result["confusion_matrix"] == [[0, 1, 0], [0, 1, 0], [0, 0, 1]]


@pytest.mark.parametrize("truth,predicted", [([], []), (["positive"], [])])
def test_invalid_inputs(truth, predicted):
    with pytest.raises(ValueError):
        metrics(truth, predicted)
