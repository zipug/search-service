from app.models import Article
from app.database import SessionLocal
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from typing import cast
import numpy as np
import torch
import torch.nn.functional as F
import annoy
import os

model_name = "ai-forever/sbert_large_nlu_ru"
# Загружаем модель
# model = SentenceTransformer("all-MiniLM-L6-v2")
# model = SentenceTransformer("all-mpnet-base-v2")
# model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
# model = AutoModel.from_pretrained(
#    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# )
# model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
# model = AutoModel.from_pretrained(model_name)
model = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
db = SessionLocal()
index = None
index_store = {}
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.4))


def get_data(project_id: int) -> list:
    articles = []
    db_articles = (
        db.query(Article)
        .filter(Article.project_id == project_id, Article.deleted_at.is_(None))
        .all()
    )
    articles = [
        {
            "id": cast(int, article.id),
            "name": str(article.name),
            "description": str(article.description),
            "content": str(article.content),
            "project_id": cast(int, article.project_id),
        }
        for article in db_articles
    ]
    return articles


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def find_answer(project_id: int, question: str) -> None | list[str]:
    global index, index_store
    articles = get_data(project_id)
    # Тексты для поиска
    texts = []
    changes = 0

    for article in articles:
        if "name" in article and "id" in article and "content" in article:
            if article["id"] in index_store:
                continue
            article_embedding = model.encode(
                f"{article['name']}. {article['description']}. {article['content']}",
                convert_to_numpy=True,
            )
            changes += 1
            index_store[article["id"]] = {
                "id": article["id"],
                "name": article["name"],
                "description": article["description"],
                "content": article["content"],
                "project_id": article["project_id"],
                "vector": article_embedding,
            }
    if changes > 0 or index is None:
        print(f"Building index with {len(index_store)} items")
        index = annoy.AnnoyIndex(1024, "angular")
        for id, val in index_store.items():
            index.add_item(id, val["vector"])
        index.build(10)

    question_embedding = model.encode(question, convert_to_numpy=True)
    print(f"Question embedding: {question_embedding.shape}")
    if index is None:
        index = annoy.AnnoyIndex(1024, "angular")
        for id, val in index_store.items():
            index.add_item(id, val["vector"])
        index.build(10)
    ids, distances = index.get_nns_by_vector(
        question_embedding, n=3, include_distances=True
    )
    print(f"Found {len(ids)} items with distances: {distances}")
    res = list()
    similarities = [1 - d / 2 for d in distances]
    for i, sim in zip(ids, similarities):
        print(f"ID: {i}, similarity: {sim:.4f}")
        if sim > SIMILARITY_THRESHOLD:
            res.append(
                {
                    "id": index_store[i]["id"],
                    "name": index_store[i]["name"],
                    "content": index_store[i]["content"],
                    "description": index_store[i]["description"],
                }
            )
    return res


def find_answer_1(project_id: int, question: str) -> None | list[str]:
    articles = get_data(project_id)
    # Тексты для поиска
    texts = []
    data = []

    for article in articles:
        if "name" in article and "id" in article and "content" in article:
            print(article["name"])
            texts.append(
                f"{article['name']} {article['description']} {article['content']}"
            )
            data.append(
                {
                    "id": article["id"],
                    "name": article["name"],
                    "description": article["description"],
                    "content": article["content"],
                    "project_id": article["project_id"],
                }
            )
        else:
            continue

    # Генерируем эмбеддинги для текстов
    encoded_texts = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        model_out = model(**encoded_texts)
    text_embeddings = mean_pooling(model_out, encoded_texts["attention_mask"])
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    # Генерируем эмбеддинг для вопроса
    encoded_question = tokenizer(
        question, return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        model_out = model(**encoded_question)
    question_embedding = mean_pooling(model_out, encoded_question["attention_mask"])
    question_embedding = F.normalize(question_embedding, p=2, dim=1)

    # Считаем схожесть
    similarities = util.cos_sim(question_embedding, text_embeddings)
    # Находим текст с максимальной схожестью
    if similarities.is_cuda:
        similarities = similarities.cpu()
    np_scores = similarities.numpy()
    print(np_scores)
    best_match_idx = np.argmax(np_scores)
    max_score = np.max(np_scores)
    res = []
    for article in data:
        if texts[best_match_idx] != article["name"]:
            if max_score > 0.3:
                res.append(
                    {
                        "id": article["id"],
                        "name": article["name"],
                        "content": article["content"],
                        "description": article["description"],
                    }
                )
            # words = question.split(" ")
            # positives = [
            #    True for word in words if word.lower() in article["name"].lower()
            # ]
            # if len(positives) > 0:
            #    res.append(
            #        {
            #            "id": article["id"],
            #            "name": article["name"],
            #            "content": article["content"],
            #            "description": article["description"],
            #        }
            #    )
            #    continue
            continue
        if texts[best_match_idx] == article["name"]:
            if max_score < 0.3:
                continue
            res.append(
                {
                    "id": article["id"],
                    "name": article["name"],
                    "content": article["content"],
                    "description": article["description"],
                }
            )
            continue
    return res
