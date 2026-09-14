from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import shutil
import tempfile
import os

from src.predict import predict


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allow requests from the React frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Image Classifier API is running"
    }


@app.post("/predict")
def predict_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False,
    ) as temp_file:

        shutil.copyfileobj(
            file.file,
            temp_file,
        )

        temp_path = temp_file.name

    try:
        results = predict(temp_path)

        return {
            "filename": file.filename,
            "predictions": results,
        }

    finally:
        os.unlink(temp_path)