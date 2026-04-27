import io
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import numpy as np
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("runs/detect/Taiwan_Traffic_Project/weights/best.pt")

# 使用 WebSocket 傳輸影像
@app.websocket("/ws/detect")

async def websocket_detect(websocket: WebSocket):
    # 1. 接受前端 (Expo Go) 的連線請求
    await websocket.accept()
    print("手機端成功建立 WebSocket 連線")

    try:
        # 2. 建立無限迴圈，持續接收影像串流
        while True:
            # 接收前端傳來的資料 (取代原本的 payload: ImagePayload)
            data = await websocket.receive_text()

            # 3. 防呆機制：過濾 Expo 可能自帶的 data:image/jpeg;base64, 前綴
            raw_base64 = data.split(",")[-1] if "," in data else data

            # 解碼與影像轉換 (沿用你原本的精準寫法)
            img_bytes = base64.b64decode(raw_base64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)

            # 4. YOLO 推論
            results = model(img_np, conf=0.5, verbose=False)

            # 解析 Bounding Boxes
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

            # 結果打包 JSON 字串並回傳手機
            response_data = {
                "boxes": boxes,
                "img_width": int(img_np.shape[1]),
                "img_height": int(img_np.shape[0])
            }
            await websocket.send_text(json.dumps(response_data))

    except WebSocketDisconnect:
        # 當手機端關閉 App 或斷網時，安全地捕捉斷線事件
        print("Error: 手機端連線已中斷")
    except Exception as e:
        print(f"warning: 處理影像時發生錯誤: {e}")

@app.get("/")
async def root():
    return {"status": "server is running"}

# 以下為 Http 協定
# ------------------------------------------------------------------------------------------------------
# class ImagePayload(BaseModel):
#     image: str

# @app.post("/detect")
# async def detect(payload: ImagePayload):
#     img_bytes = base64.b64decode(payload.image)
#     img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#     img_np = np.array(img)

#     results = model(img_np, conf=0.5)

#     boxes = []
#     for box in results[0].boxes:
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#         conf = float(box.conf[0])
#         cls = int(box.cls[0])
#         label = model.names[cls]
#         boxes.append({
#             "x1": x1, "y1": y1,
#             "x2": x2, "y2": y2,
#             "confidence": round(conf, 2),
#             "label": label
#         })

#     return {
#         "boxes": boxes,
#         "img_width": int(img_np.shape[1]),
#         "img_height": int(img_np.shape[0])
#     }

# @app.get("/")
# async def root():
#     return {"status": "server is running"}

