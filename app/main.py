from fastapi import FastAPI
from app.database import Base, engine
from app.endpoints.search import router as search_router

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(search_router)
