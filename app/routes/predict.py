from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

from app.services.inference import predict

router = APIRouter()


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = predict(image, router.config)

    return result