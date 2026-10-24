from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    # Без контекстного менеджера TestClient не запускает lifespan.
    tokenizer = Mock(return_value={"input_ids": torch.tensor([[1, 2]])})
    model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[0.0, 0.0, 0.0]])))
    model.config = SimpleNamespace(
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        max_position_embeddings=512,
    )
    app.state.tokenizer = tokenizer
    app.state.model = model
    client = TestClient(app)
    yield client
    client.close()
    del app.state.tokenizer
    del app.state.model


@pytest.mark.parametrize(
    ("logits", "expected"),
    [
        ([4.0, 0.0, 0.0], "negative"),
        ([0.0, 4.0, 0.0], "neutral"),
        ([0.0, 0.0, 4.0], "positive"),
    ],
)
def test_sentiment(client, logits, expected):
    app.state.model.return_value.logits = torch.tensor([logits])
    response = client.post("/sentiment", json={"text": "  Пример текста  "})
    assert response.status_code == 200
    assert response.json() == {"sentiment": expected}
    app.state.tokenizer.assert_called_once_with(
        "Пример текста", return_tensors="pt", truncation=True, max_length=512
    )


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": ""}, {"text": "   "}, {"text": 123}, {"text": "а" * 5001}],
)
def test_invalid_text(client, payload):
    response = client.post("/sentiment", json=payload)
    assert response.status_code == 422
    app.state.model.assert_not_called()


def test_invalid_model(client):
    assert client.post("/sentiment?model=unknown", json={"text": "Пример"}).status_code == 422


def test_local_interface(client):
    response = client.get("/")
    assert response.status_code == 200
    assert '<html lang="ru">' in response.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_openapi_contract(client):
    spec = client.get("/openapi.json").json()
    for path in ["/sentiment"]:
        assert "422" in spec["paths"][path]["post"]["responses"]
    assert spec["components"]["schemas"]["SentimentRequest"]["properties"]["text"]["maxLength"] == 5000


def test_custom_model_is_no_longer_available(client):
    assert client.post("/sentiment?model=custom", json={"text": "Пример"}).status_code == 422
    assert client.post("/compare", json={"text": "Пример"}).status_code == 404
