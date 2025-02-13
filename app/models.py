from sqlalchemy import Column, Integer, String, Time
from sqlalchemy.orm import relationship
from app.database import Base


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String)
    content = Column(String)
    project_id = Column(Integer, foreign_key=True, foreign_key_column="projects.id")
    created_at = Column(Time)
    updated_at = Column(Time)
    deleted_at = Column(Time)
