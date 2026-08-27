import pickle
import re
import string
from pathlib import Path

import nltk
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
FRONTEND_DIR = BASE_DIR.parent / "frontend"


# Download stopwords only if they are not available
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


# Load tokenizer, configuration, labels, and trained GRU model
with open(MODEL_DIR / "tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open(MODEL_DIR / "config.pkl", "rb") as file:
    config = pickle.load(file)

with open(MODEL_DIR / "label_mapping.pkl", "rb") as file:
    label_mapping = pickle.load(file)

model = load_model(MODEL_DIR / "gru_model.keras")


app = FastAPI(title="Email Spam Detector API")


# Serve CSS and JavaScript files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class PredictionRequest(BaseModel):
    email_text: str = Field(..., min_length=1, max_length=10000)


def clean_text(text: str) -> str:
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove English stopwords
    words = [word for word in text.split() if word not in STOP_WORDS]
    text = " ".join(words)

    # Remove links
    text = re.sub(r"http\S+", "", text)

    return text.strip()


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "GRU Spam Detector"
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    cleaned_text = clean_text(payload.email_text)

    if not cleaned_text:
        raise HTTPException(
            status_code=422,
            detail="Please enter meaningful email text."
        )

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded_text = pad_sequences(
        sequence,
        maxlen=config["max_length"],
        padding="post"
    )

    spam_probability = float(
        model.predict(padded_text, verbose=0)[0][0]
    )

    prediction_id = 1 if spam_probability > 0.5 else 0

    confidence = (
        spam_probability * 100
        if prediction_id == 1
        else (1 - spam_probability) * 100
    )

    return {
        "prediction": label_mapping[prediction_id],
        "spam_probability": round(spam_probability, 4),
        "confidence": round(confidence, 2)
    }