import io
import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("runs/detect/Taiwan_Traffic_Project/weights/best.pt")

class ImagePayload(BaseModel):
    image: str

@app.post("/detect")
async def detect(payload: ImagePayload):
    img_bytes = base64.b64decode(payload.image)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    results = model(img_np, conf=0.5)

    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        boxes.append({
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "confidence": round(conf, 2),
            "label": label
        })

    return {
        "boxes": boxes,
        "img_width": int(img_np.shape[1]),
        "img_height": int(img_np.shape[0])
    }

@app.get("/")
async def root():
    return {"status": "server is running"}