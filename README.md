# Email Spam Detector

A machine-learning web application that classifies email content as **Spam** or **Not Spam** using a GRU (Gated Recurrent Unit) recurrent neural network.

## Features

- Classifies an email as Spam or Not Spam
- GRU-based deep learning model built with TensorFlow/Keras
- Text preprocessing with lowercase conversion, punctuation removal, stop-word removal, and tokenization
- FastAPI prediction API
- Responsive frontend built with HTML, CSS, and JavaScript
- Sample safe and spam emails for quick testing

## Tech Stack

- Python
- TensorFlow / Keras
- FastAPI
- NLTK
- HTML, CSS, JavaScript

## Project Structure

```text
Email-Spam/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── model/
│       ├── gru_model.keras
│       ├── tokenizer.pkl
│       ├── config.pkl
│       └── label_mapping.pkl
│
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Devanshs7586/Email-spam.git
cd Email-spam/backend
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI server

```bash
uvicorn main:app --reload
```

Open the application at:

```text
http://127.0.0.1:8000
```

## API Usage

### Endpoint

```text
POST /predict
```

### Request body

```json
{
  "email_text": "Congratulations! You have won a reward. Click here to claim it."
}
```

### Response example

```json
{
  "prediction": "Spam",
  "spam_probability": 0.9984,
  "confidence": 99.84
}
```

The website displays the final model classification as Spam or Not Spam. The probability is used internally by the model to make that binary decision.

## Render Deployment

This project can be deployed as two Render services:

1. **Backend:** Create a **Web Service** from the `backend` folder.
2. **Frontend:** Create a **Static Site** from the `frontend` folder.

For the backend service, use:

```text
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

When deploying the frontend separately, update `script.js` to call your deployed backend URL, for example:

```js
fetch("https://your-backend-name.onrender.com/predict", { ... })
```

Also enable CORS in the FastAPI backend so that the frontend can call the API.

## Important Note

This model is trained for binary classification. It identifies text patterns associated with spam and legitimate emails, but it cannot independently verify whether a sender, URL, attachment, or company is genuine. Always use caution with suspicious emails.
