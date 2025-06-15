from pydantic import BaseModel
from typing import Optional


class ArticleBase(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    content: str
    project_id: int


class ArticleResponse(ArticleBase):
    id: int

    # Enable ORM mode to allow returning SQLAlchemy models directly
    class Config:
        from_attributes = True


class DataDto(BaseModel):
    id: int
    name: str
    content: str


class SearchResponse(BaseModel):
    data: list[DataDto]


class ArticleSchema(BaseModel):
    id: int
    name: str
    content: str
