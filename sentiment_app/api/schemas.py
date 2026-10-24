"""Входные данные и JSON-ответ API."""
from typing import Annotated, Literal
from pydantic import BaseModel, StringConstraints


class SentimentRequest(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]


class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
