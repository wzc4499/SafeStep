import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

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

    # 為「這個連線」建立專屬的物件追蹤記憶體，避免多裝置互相干擾
    client_object_history = {}

    try:
        while True:
            data = await websocket.receive_text()
            raw_base64 = data.split(",")[-1] if "," in data else data

            # 效能優化：跳過 PIL，直接將 Base64 轉為 OpenCV BGR 格式
            img_bytes = base64.b64decode(raw_base64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img_np is None:
                continue

            img_height, img_width = img_np.shape[:2]

            # 1. 執行 YOLO 辨識
            _, detected_items = func.VisionProcessor.detect_objects(img_np)

            # 2. 進行危險評估與排序 (傳入該連線專屬的歷史記憶體)
            analyzed_items = func.RiskEvaluator.evaluate_frame_risk(
                detected_items,
                img_height,
                client_object_history
            )

            # 3. 整理回傳格式
            response_boxes = []
            for item in analyzed_items:
                x1, y1, x2, y2 = item["bbox"]
                label = func.VisionProcessor.get_label_name(item["class_id"])

                response_boxes.append({
                    "track_id": item["track_id"],
                    "label": label,
                    "confidence": round(item["confidence"], 2),
                    "risk_score": round(item["risk_score"], 2),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                })

            # 4. 回傳前端
            response_data = {
                "boxes": response_boxes,
                "img_width": img_width,
                "img_height": img_height
            }
            await websocket.send_text(json.dumps(response_data))

    except WebSocketDisconnect:
        print("⚠️ 手機端連線已中斷")
        # 由於 client_object_history 是區域變數，斷線後會自動被 Python 垃圾回收機制清掉，無須手動 clear

    except Exception as e:
        print(f"❌ 處理影像時發生錯誤: {e}")

@app.get("/")
async def root():
    return {"status": "server is running"}