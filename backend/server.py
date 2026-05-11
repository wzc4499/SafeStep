import base64
import json
from tkinter import Image
import cv2
import numpy as np
import heapq  # 新增：用於處理優先權佇列
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 匯入優化後的模組
import func

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    print("📱 手機端成功建立 WebSocket 連線")

LABEL_ZH = {
    'obstacle': '障礙物',
    'human': '行人',
    'motorcycle': '機車',
    'car': '轎車',
    'pickup truck': '小貨車',
    'truck': '卡車',
    'bus': '公車',
    'Traffic sign': '交通標誌',
}

class ImagePayload(BaseModel):
    image: str

def estimate_danger(x1, y1, x2, y2, img_w, img_h):
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    ratio = box_area / img_area

    if ratio > 0.25:
        return "高", ratio
    elif ratio > 0.08:
        return "中", ratio
    else:
        return "低", ratio

@app.post("/detect")
async def detect(payload: ImagePayload):
    img_bytes = base64.b64decode(payload.image)
    img = Image.open(io.BytesIO(img_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img_np = np.array(img)
    img_h, img_w = img_np.shape[:2]

    results = model(img_np, conf=0.35)

    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label_en = model.names[cls]
        label = LABEL_ZH.get(label_en, label_en)

        danger_level, area_ratio = estimate_danger(x1, y1, x2, y2, img_w, img_h)
        estimated_distance = round(1.0 / (area_ratio + 0.01) * 0.5, 1)
        estimated_distance = min(estimated_distance, 99.9)

        boxes.append({
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "confidence": round(conf, 2),
            "label": label,
            "danger": danger_level,
            "distance": estimated_distance,
        })

    danger_order = {"高": 0, "中": 1, "低": 2}
    boxes.sort(key=lambda b: danger_order[b["danger"]])

    return {
        "boxes": boxes,
        "img_width": img_w,
        "img_height": img_h
    }

@app.get("/")
async def root():
    return {"status": "server is running"}