import base64
import json
import cv2
import numpy as np
import heapq  # 新增：用於處理優先權佇列
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

            # 2. 進行危險評估與排序 (【更新】接收警報佇列並啟用行人模式)
            analyzed_items, alert_queue = func.RiskEvaluator.evaluate_frame_risk(
                detected_items,
                img_height,
                client_object_history,
                mode="pedestrian" # 指定為行人模式
            )

            # 3. 整理一般辨識框的回傳格式
            response_boxes = []
            for item in analyzed_items:
                x1, y1, x2, y2 = item["bbox"]
                label = func.VisionProcessor.get_label_name(item["class_id"])

                response_boxes.append({
                    "track_id": item["track_id"],
                    "label": label,
                    "confidence": round(item["confidence"], 2),
                    "risk_score": round(item["risk_score"], 2),
                    # 【更新】將距離與速度也傳給前端，方便 UI 顯示
                    "distance_pct": round(item.get("proximity_pct", 0), 1),
                    "speed": round(item["speed"], 1) if item.get("speed") is not None else None,
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                })

            # 4. 【新增】處理優先權佇列，生成警報訊號給前端
            frontend_alerts = []
            while alert_queue:
                # 依序取出最危險的物件 (Max-Heap)
                neg_risk, q_id, q_item = heapq.heappop(alert_queue)
                real_risk = -neg_risk
                q_label = func.VisionProcessor.get_label_name(q_item["class_id"])

                frontend_alerts.append({
                    "track_id": q_id,
                    "label": q_label,
                    "risk_score": round(real_risk, 2),
                    "warning_msg": f"高危險 {q_label} 逼近！"
                })

            # 5. 回傳前端
            response_data = {
                "boxes": response_boxes,
                "alerts": frontend_alerts, # 【更新】將警報陣列加入 Payload
                "img_width": img_width,
                "img_height": img_height
            }
            await websocket.send_text(json.dumps(response_data))

    except WebSocketDisconnect:
        print("⚠️ 手機端連線已中斷")
        # 由於 client_object_history 是區域變數，斷線後會自動被 Python 垃圾回收機制清掉

    except Exception as e:
        print(f"❌ 處理影像時發生錯誤: {e}")

@app.get("/")
async def root():
    return {"status": "server is running"}