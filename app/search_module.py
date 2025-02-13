from app.redis_client import redis_client
from app.database import SessionLocal
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
from typing import cast

from app.models import Article

# Загружаем модель
# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("all-mpnet-base-v2")


# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


def find_answer(project_id: int, question: str) -> None | list[str]:
    articles = []
    db = get_db()
    cahed_data = redis_client.get(f"project_{project_id}")
    if cahed_data:
        print("From cache")
        articles = json.loads(cast(str, cahed_data))
    else:
        print("From db")
        db_articles = (
            db.query(Article)
            .filter(Article.project_id == project_id, Article.deleted_at.is_(None))
            .all()
        )
        articles = [
            {
                "id": cast(int, article.id),
                "name": str(article.name),
                "content": str(article.content),
            }
            for article in db_articles
        ]
        try:
            data_to_cache = json.dumps(articles)
            try:
                redis_client.setex(
                    name=f"project_{project_id}",
                    time=60 * 10,
                    value=data_to_cache,
                )
            except Exception as e:
                print(f"Redis error: {e}")
        except Exception as e:
            print(f"Error while serializing data: {e}")
    # Тексты для поиска
    texts = []
    data = []

    for article in articles:
        if "name" in article and "id" in article and "content" in article:
            print(article["name"])
            texts.append(article["name"])
            data.append(
                {
                    "id": article["id"],
                    "name": article["name"],
                    "content": article["content"],
                }
            )
        else:
            continue

    # Генерируем эмбеддинги для текстов
    text_embeddings = model.encode(texts, convert_to_tensor=True)

    # Генерируем эмбеддинг для вопроса
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Считаем схожесть
    similarities = util.cos_sim(question_embedding, text_embeddings)
    # Находим текст с максимальной схожестью
    if similarities.is_cuda:
        similarities = similarities.cpu()
    np_scores = similarities.numpy()
    print(np_scores)
    best_match_idx = np.argmax(np_scores)
    res = []
    for article in data:
        if texts[best_match_idx] == article["name"]:
            res.append(article["content"])
            continue
        words = question.split()
        positives = [True for word in words if word in article["name"]]
        if len(positives) > 0:
            res.append(article["content"])
            continue
    return res
