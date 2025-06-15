from fastapi import APIRouter
from app.schemas import SearchResponse
from app.search_module import find_answer

router = APIRouter()


@router.get("/search", response_model=SearchResponse)
def search_article(id: int, query: str):
    answers = find_answer(project_id=id, question=query)
    return {"data": answers}
