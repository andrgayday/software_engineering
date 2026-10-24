"""Обработка HTTP-запросов анализа тональности."""
from typing import Literal
from fastapi import APIRouter, Request
from sentiment_app.api.schemas import SentimentRequest, SentimentResponse
from sentiment_app.models.pretrained import predict

router = APIRouter()


@router.post("/sentiment", response_model=SentimentResponse, summary="Оценить тональность")
def sentiment(payload: SentimentRequest, request: Request,
              model: Literal["pretrained"] = "pretrained") -> SentimentResponse:
    # Параметр сохранён для прежних клиентов; custom теперь явно возвращает 422.
    return SentimentResponse(sentiment=predict(
        payload.text, request.app.state.tokenizer, request.app.state.model,
    ))
