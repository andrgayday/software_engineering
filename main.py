"""Точка входа: uv run uvicorn main:app --host 127.0.0.1 --port 8000."""
from sentiment_app.application import create_app

app = create_app()
