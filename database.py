import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Database Connection Setup
# We define the database file location (SQLite)
DATABASE_URL = "sqlite:///./breed_history.db"

# Create the engine, allowing multiple threads for FastAPI concurrency
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create a session factory to manage database operations
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class that our tables will inherit from
Base = declarative_base()

# 2. Database Schema Definition
class PredictionRow(Base):
    __tablename__ = "breed_prediction"

    # Unique ID for each record
    id = Column(Integer, primary_key=True, index=True)
    
    # Data fields for the prediction
    filename = Column(String)
    breed = Column(String)
    confidence = Column(Float)
    
    # Automatically stores the time of the record creation
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Field for user feedback (optional, so nullable=True)
    user_correction = Column(String, nullable=True)

# Note: In your main app.py, you would use 'Base.metadata.create_all(bind=engine)'
# to actually create the file and the table if they don't exist yet.