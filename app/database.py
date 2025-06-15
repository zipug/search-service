import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_USER = os.getenv("POSTGRES_USER", "usr")
DATABASE_PASSWORD = os.getenv("POSTGRES_PASSWORD", "pwd")
DATABASE_HOST = os.getenv("POSTGRES_HOST", "localhost")
DATABASE_PORT = os.getenv("POSTGRES_PORT", "5439")
DATABASE_NAME = os.getenv("POSTGRES_DB_NAME", "db")

DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a SessionLocal class for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()
