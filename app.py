import io
import json  
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Depends
from PIL import Image
from torchvision import models, transforms
from database import SessionLocal, engine, Base, PredictionRow
from sqlalchemy.orm import Session
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Pet Breed Classifier")

# Allow frontend UI to connect to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables if they don't exist
Base.metadata.create_all(bind=engine)

# Get database session for each request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom head for the model
class CustomCnnHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, num_classes)    
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load breed names
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

# Rebuild model structure
model = models.resnet50(weights=None)
features_size = model.fc.in_features
model.fc = CustomCnnHead(in_features=features_size, num_classes= len(CLASS_NAMES))

# Load trained weights and set to evaluation mode
model.load_state_dict(torch.load('resnet50_pets_augmented.pth', map_location= device))
model = model.to(device)
model.eval()

# Image preprocessing pipeline
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

@app.get("/")
async def UI():
    return FileResponse('index.html')


# Set confidence threshold to filter out bad predictions
CONFIDENCE_THRESHOLD = 60.0

@app.post("/predict")
async def prediction(file: UploadFile = File(...), db: Session = Depends(get_db)):
    
    # Read uploaded file and convert to RGB
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Prepare image tensor and add batch dimension
    tensor = inference_transform(image).unsqueeze(0).to(device)

    # Run inference without calculating gradients
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence_score, predict = torch.max(probabilities, 1)
    
    predicted_breed_index = predict.item()
    final_confidence = round(confidence_score.item() * 100, 2)

    # Apply threshold guardrail
    if final_confidence < CONFIDENCE_THRESHOLD:
        predicted_breed = "Unknown / Not a recognized breed"
    else:
        predicted_breed = CLASS_NAMES[predicted_breed_index]

    # Save log entry to database
    new_log = PredictionRow(
        filename= file.filename,
        breed= predicted_breed,
        confidence= final_confidence
    )

    db.add(new_log)
    db.commit()

    return {
        "filename": file.filename,
        "breed": predicted_breed,
        "confidence": final_confidence
    }

@app.get("/analytics")
def get_analytics(db: Session = Depends(get_db)):
    
    # Get all prediction logs from database
    predictions = db.query(PredictionRow).all()
    if not predictions:
        return {"total_scans": 0}

    # Load into data frame for easy metrics calculation
    df = pd.DataFrame([{"breed": p.breed, "confidence": p.confidence} for p in predictions])
    
    total_scans = len(df)
    
    recognized_df = df[df["breed"] != "Unknown / Not a recognized breed"]
    avg_confidence_breed = round(recognized_df["confidence"].mean(), 2) if not recognized_df.empty else 0
    
    unrecognized_count = (df["breed"] == "Unknown / Not a recognized breed").sum()
    
    breed_dist ={str(k): int(v) for k, v in df["breed"].value_counts().items()}
    
    return {
        "total_scans": int(total_scans),
        "avg_confidence_recognized": float(avg_confidence_breed),
        "unrecognized_count": int(unrecognized_count),
        "breed_distribution": breed_dist 
    }

# Define what data we expect from the frontend feedback form
class FeedbackPayload(BaseModel):
    filename: str
    correction: str

@app.post("/feedback")
async def save_feedback(data: FeedbackPayload, db: Session = Depends(get_db)):
    
   # Find the latest logged record for this filename by sorting by ID descending
    record = db.query(PredictionRow).filter(PredictionRow.filename == data.filename).order_by(PredictionRow.id.desc()).first()

    if not record:
        return {"status": "error", "message": "Prediction not found in database"}
        
    record.user_correction = data.correction
    db.commit()
    
    return {"status": "success", "message": "Feedback saved"}