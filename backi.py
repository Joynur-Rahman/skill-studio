
from pathlib import Path

import joblib
from fastapi import FastAPI

# Find the project folder and saved model
MODEL_PATH = Path.home() / "Downloads" / "placement_model.joblib"

# Load the trained machine-learning model
model = joblib.load(MODEL_PATH)

# Create our backend application
app = FastAPI(title="Placement Prediction API")


# Check whether the backend is running
@app.get("/")
def home():
    return {"message": "Placement Prediction API is running!"}