# Deepfake Detection System

## Overview
A production-oriented Deepfake Detection System built using EfficientNet, featuring a modular ML pipeline, FastAPI-based inference API, and explainability using Grad-CAM.

This project is designed to be built and run **completely free of cost** using local resources and open-source tools only (no paid APIs, no cloud dependency required).

---

## Features
- EfficientNet-based deepfake classifier
- Config-driven training pipeline
- FastAPI inference API
- Image preprocessing pipeline
- Model versioning
- Grad-CAM explainability
- Dataset validation utilities
- 100% free and local execution

---

## Project Structure

```

deepfake-detector/
├── app/                # API layer
├── model/              # Training & evaluation
├── data/               # Dataset & validation
├── configs/            # Configurations
├── models/             # Saved models
├── scripts/            # Run scripts
├── tests/              # Unit tests
├── README.md
└── requirements.txt

````

---

## Installation

```bash
git clone <your-repo-url>
cd deepfake-detector

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
````

---

## Dataset

You can use any free public dataset such as:

* FaceForensics++
* Celeb-DF
* DeepFake Detection Challenge (DFDC)

Place dataset inside:

```
data/raw/
```

---

## Training

```bash
python model/train.py
```

---

## Evaluation

```bash
python model/evaluate.py
```

---

## Run API

```bash
uvicorn app.main:app --reload
```

---

## API Endpoint

### POST /predict

Upload an image and receive prediction:

**Response:**

```json
{
  "prediction": "fake",
  "confidence": 0.92
}
```

---

## Explainability

Grad-CAM is used to highlight image regions influencing predictions, improving transparency and trust in model decisions.

---

## Tech Stack

* TensorFlow / Keras
* FastAPI
* OpenCV
* NumPy
* Python

---

## Cost Consideration

This project is intentionally designed to run at **zero cost**:

* No paid APIs
* No cloud GPU required (can train on CPU, slower but free)
* Fully open-source stack

---

## Future Improvements

* Real-time webcam detection
* Video-based deepfake detection
* Model optimization (ONNX / TensorRT)
* Optional cloud deployment (if scaling required)

---

## Author

Shivang Mishra

```

---

Next step is where your project actually becomes dangerous instead of decorative.

Say **“start config”**.
```
