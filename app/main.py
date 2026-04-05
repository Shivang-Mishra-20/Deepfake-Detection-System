from fastapi import FastAPI
import yaml

from app.routes import predict as predict_route
from app.services.inference import load_model_and_classes


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


app = FastAPI(title="Deepfake Detection API")

config = load_config()

# Attach config to router (important)
predict_route.router.config = config

# Load model at startup
@app.on_event("startup")
def startup_event():
    print("=== STARTUP STARTED ===")
    load_model_and_classes(config)
    print("=== STARTUP FINISHED ===")


# Register routes
app.include_router(predict_route.router)