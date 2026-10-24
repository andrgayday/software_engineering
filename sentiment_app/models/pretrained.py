"""Загрузка и пакетный инференс готовой модели Hugging Face."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentiment_app.config import MODEL_NAME, MODEL_REVISION


def load_pretrained(*, local_files_only=False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, use_safetensors=True, local_files_only=local_files_only,
    ).eval()
    # Отдельная память устраняет наблюдавшийся SIGBUS при mmap safetensors на Mac.
    for parameter in model.parameters():
        parameter.data = parameter.data.clone()
    return tokenizer, model


def predict_batch(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=model.config.max_position_embeddings)
    with torch.inference_mode():
        indices = model(**inputs).logits.argmax(dim=-1).tolist()
    return [model.config.id2label[index] for index in indices]


def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=model.config.max_position_embeddings)
    with torch.inference_mode():
        index = model(**inputs).logits.argmax(dim=-1).item()
    return model.config.id2label[index]
