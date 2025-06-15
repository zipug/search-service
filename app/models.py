from sqlalchemy import Column, ForeignKey, Integer, String, Time
from sqlalchemy.orm import relationship
from app.database import Base


class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)


class Article(Base):
    __tablename__ = "articles"

    project = relationship("Project")

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String)
    content = Column(String)
    project_id = Column(Integer, ForeignKey("projects.id"))
    created_at = Column(Time)
    updated_at = Column(Time)
    deleted_at = Column(Time)
