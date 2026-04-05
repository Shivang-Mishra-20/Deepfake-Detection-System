# Deepfake Detection System

Production-ready deepfake image detection system built using EfficientNet and served via FastAPI with a modular, scalable backend architecture.

---

## 🚀 Overview

This project is an end-to-end AI system that detects whether an image is **real or deepfake**.
It integrates a trained deep learning model with a structured backend API for real-time inference.

Unlike basic ML demos, this project follows a **clean architecture approach**, separating concerns across training, inference, preprocessing, and API layers.

---

## 🧠 Model Details

* **Architecture:** EfficientNetB0 (Transfer Learning)
* **Input Size:** 224 × 224
* **Output:** Binary classification (Real / Fake)
* **Framework:** TensorFlow / Keras
* **Inference Mode:** Optimized (compile=False, safe_mode=False)

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* FastAPI
* Uvicorn
* NumPy / Pillow

---

## 🏗️ System Architecture

```id="arch1"
Client → FastAPI Routes → Service Layer → Preprocessing → Model → Prediction
```

---

## 📁 Project Structure

```id="arch2"
Deepfake-Detection-System/
│
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── routes/                 # API endpoints
│   │   └── predict.py
│   ├── schemas/                # Request/response schemas
│   │   └── request.py
│   ├── services/               # Core inference logic
│   │   └── inference.py
│   └── utils/                  # Image preprocessing
│       └── image_preprocess.py
│
├── configs/
│   └── config.yaml             # Central configuration
│
├── models/
│   ├── final_model.keras       # Trained model
│   └── class_indices.json      # Label mapping
│
├── model/                      # Training + evaluation code
│   ├── train.py
│   ├── evaluate.py
│   ├── architecture.py
│   └── explainability.py
│
├── data/
│   └── validation_checks.py
│
├── notebooks/
│   └── training.ipynb          # Model training workflow
│
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   └── run_api.sh
│
├── tests/
│   └── test_inference.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Clone repository

```id="qs1"
git clone https://github.com/Shivang-Mishra-20/Deepfake-Detection-System.git
cd Deepfake-Detection-System
```

### 2. Create virtual environment

```id="qs2"
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```id="qs3"
pip install -r requirements.txt
```

### 4. Run the API server

```id="qs4"
uvicorn app.main:app
```

### 5. Open interactive docs

```id="qs5"
http://127.0.0.1:8000/docs
```

---

## 📌 API Usage

### Endpoint

```id="api1"
POST /predict
```

### Request

Upload an image file:

```id="api2"
curl -X POST "http://127.0.0.1:8000/predict" \
-F "file=@image.jpg"
```

### Response

```id="api3"
{
  "prediction": "fake",
  "confidence": 0.87
}
```

---

## 🧪 Testing

Run inference tests:

```id="test1"
pytest tests/
```

---

## 📊 Dataset

* Source: *(Add dataset source — Kaggle / custom / etc.)*
* Classes: Real vs Fake
* Size: *(Add number of samples)*

---

## 🧠 Key Features

* Clean modular architecture (routes, services, utils)
* Config-driven pipeline (config.yaml)
* Scalable FastAPI backend
* End-to-end ML pipeline (training → inference)
* Test coverage for inference layer
* Docker support for deployment

---

## ⚠️ Limitations

* Performance depends on dataset quality
* Limited generalization to unseen deepfake types
* Currently supports images only (no video processing)

---

## 🚀 Future Improvements

* Video deepfake detection
* Larger dataset and improved generalization
* Model optimization (quantization / pruning)
* Cloud deployment (AWS / GCP)
* Frontend interface

---

## 🐳 Docker (Optional)

```id="docker1"
docker build -t deepfake-detector .
docker run -p 8000:8000 deepfake-detector
```

---

## 📬 Author

**Shivang Mishra**

---
