"""Создание FastAPI-приложения и управление жизненным циклом моделей."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sentiment_app.api.routes import router
from sentiment_app.config import ROOT
from sentiment_app.models.pretrained import load_pretrained


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tokenizer, app.state.model = load_pretrained()
    yield
    del app.state.tokenizer, app.state.model


def create_app() -> FastAPI:
    app = FastAPI(
        title="Анализ тональности русского текста", version="1.0.0",
        description="Готовая модель RuBERT-tiny определяет положительную, "
                    "нейтральную и отрицательную тональность русского текста.",
        lifespan=lifespan,
    )
    app.include_router(router)
    web = ROOT / "sentiment_app" / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web / "index.html")

    return app
