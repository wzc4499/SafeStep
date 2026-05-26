import io
import base64
import requests
import numpy as np
import cv2
import heapq
import asyncio
from PIL import Image, ImageOps
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional

from func import VisionProcessor, RiskEvaluator
from traffic_labels import SIGN_CLASSES, SIGN_ZH

app = FastAPI()

# 設定 CORS (跨來源資源共用)，允許任意前端來源連線此 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # 確保在 Server 啟動時先載入 PyTorch 號誌分類模型，避免第一次請求等待過久
    VisionProcessor.initialize_classifier()

MAPILLARY_ACCESS_TOKEN = " "

# 中英文標籤對照表
LABEL_ZH = {
    'obstacle': '障礙物',
    'human': '行人',
    'motorcycle': '機車',
    'car': '轎車',
    'pickup truck': '小貨車',
    'truck': '卡車',
    'bus': '公車',
    'Traffic sign': '交通標誌',
    'traffic light': '紅綠燈',
}

# 定義前端傳來的資料結構，使用 Pydantic 來做資料驗證
class SystemPayload(BaseModel):
    mode: str
    image: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    heading: Optional[float] = None
    action: Optional[str] = None

# def fetch_mapillary_image(lat, lng, heading):
#     ... [廢案：曾用於從地圖 API 抓取街景圖的邏輯] ...

def get_sign_name(sign_id: int) -> str:
    """透過 ID 取得交通號誌的中文名稱"""
    try:
        if sign_id != -1 and sign_id < len(SIGN_CLASSES):
            folder_name = SIGN_CLASSES[sign_id]
            return SIGN_ZH.get(folder_name, folder_name)
        return ""
    except Exception:
        return ""

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # 接受 WebSocket 連線請求
    await websocket.accept()

    # 存放目前 WebSocket session 中物件追蹤歷史紀錄，以便計算連續幀相對速度
    session_tracking_history = {}

    try:
        while True:
            # 等待接收前端傳送的 JSON 格式訊息
            data = await websocket.receive_json()

            try:
                # 透過 Pydantic 將資料轉為物件並驗證格式
                payload = SystemPayload(**data)
            except ValidationError as e:
                await websocket.send_json({"status": "error", "message": "資料格式錯誤", "details": e.errors()})
                continue

            if payload.mode == "motorcycle":
                #  ------ 廢案 ---------------------------------------------------
                # 機車模式的邏輯目前暫時被註解停用
                pass

            elif payload.mode == "pedestrian":
                # 驗證必要參數
                if not payload.image:
                    await websocket.send_json({"status": "error", "message": "缺少影像參數"})
                    continue

                try:
                    # 將前端傳來的 Base64 字串轉碼回 OpenCV 可讀的影像矩陣 (NumPy Array)
                    img_bytes = base64.b64decode(payload.image)

                    # 這裡直接從記憶體 Buffer 讀取影像位元組，減少 I/O 時間消耗
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    img_h, img_w = img_cv2.shape[:2]
                except Exception as e:
                    await websocket.send_json({"status": "error", "message": f"影像處理錯誤: {str(e)}"})
                    continue

                # [並行處理]
                # 將 YOLO 模型推論與傳統 CV 斑馬線辨識丟入不同執行緒同時運算
                # 這樣可以避免其中一個運算阻塞(Blocking)另一個運算，提升每秒幀數(FPS)
                task_yolo = asyncio.to_thread(VisionProcessor.detect_objects, img_cv2.copy())
                task_lane = asyncio.to_thread(VisionProcessor.detect_lanes, img_cv2.copy())

                # 等待兩個影像處理任務同時完成，將結果解構
                (yolo_annotated, detected_items), (lane_annotated, lane_data) = await asyncio.gather(task_yolo, task_lane)

                # 風險評估 (依賴 YOLO 結果，已確認以距離與逼近速度為主)
                # 使用保存的 session_tracking_history 來持續比對同一物件
                analyzed_items, alert_queue = RiskEvaluator.evaluate_frame_risk(
                    detected_items, img_w, img_h, session_tracking_history, mode="pedestrian"
                )

                boxes = []
                # 以畫好斑馬線與人行道的 lane_annotated 作為最終影像的底圖
                final_image = lane_annotated.copy()

                # 遍歷經過風險評估後的物件，準備將資料組合送回前端
                for item in analyzed_items:
                    label_en = VisionProcessor.get_label_name(item["class_id"])

                    # 針對號誌進行二次辨識
                    light_status = -1
                    if label_en.lower() in ["traffic light", "pedestrian light"]:
                        try:
                            # 去判斷紅、綠、黃燈
                            light_status = VisionProcessor.traffic_lights(img_cv2, item["bbox"])
                        except Exception as e:
                            print(f"紅綠燈辨識失敗: {e}")

                    specific_sign_id = -1
                    specific_sign_name = ""
                    # 若為交通標誌，調用 PyTorch 模型確認詳細種類
                    if (label_en.lower() == "traffic sign"
                            and VisionProcessor.sign_classifier is not None
                            and VisionProcessor.sign_transforms is not None):
                        try:
                            specific_sign_id = VisionProcessor.classify_traffic_sign(img_cv2, item["bbox"])
                            specific_sign_name = get_sign_name(specific_sign_id)
                        except Exception as e:
                            print(f"號誌分類失敗: {e}")

                    # 將前端所需的所有資訊存入陣列
                    boxes.append({
                        "track_id": item["track_id"],
                        "x1": item["bbox"][0], "y1": item["bbox"][1],
                        "x2": item["bbox"][2], "y2": item["bbox"][3],
                        "confidence": round(item["confidence"], 2),
                        "label": LABEL_ZH.get(label_en, label_en),
                        "danger": item["danger_level"],
                        "danger_pct": item["danger_pct"],
                        "distance": item["distance"],
                        "area_ratio": item["area_ratio"],
                        "speed": item["speed"],
                        "light_status": light_status,
                        "specific_sign_id": specific_sign_id,
                        "specific_sign_name": specific_sign_name,
                    })

                    # 將 YOLO 偵測到的框，根據危險程度動態畫回 final_image
                    x1, y1, x2, y2 = map(int, item["bbox"])
                    danger = item.get("danger_level", "低")

                    # 以不同顏色標示物件危險程度
                    if danger == "高":
                        color = (0, 0, 255)   # 紅色 (高危險)
                    elif danger == "中":
                        color = (0, 165, 255) # 橘色 (中危險)
                    else:
                        color = (0, 255, 0)   # 綠色 (低危險)

                    cv2.rectangle(final_image, (x1, y1), (x2, y2), color, 2)
                    # (選用) 將中文標籤繪製於圖上：
                    # display_label = LABEL_ZH.get(label_en, label_en)
                    # cv2.putText(final_image, display_label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 將合成好的最終影像轉回 JPG 格式的 Base64，以便前端網頁渲染
                _, buffer = cv2.imencode('.jpg', final_image)
                processed_image_b64 = base64.b64encode(buffer).decode('utf-8')

                # 發送 WebSocket 訊息給客戶端
                await websocket.send_json({
                    "status": "success",
                    "mode": "pedestrian",
                    "boxes": boxes, # 所有物件的座標、特徵與危險評估資料
                    "img_width": img_w,
                    "img_height": img_h,
                    "lanes": lane_data, # 人行道與斑馬線的多邊形座標陣列
                    "processed_image": processed_image_b64, # 標註完畢的圖像
                })

    except WebSocketDisconnect:
        # 當客戶端 (App 或網頁) 斷線時，清空當下的追蹤紀錄
        print("Client 已經斷開連線，清理追蹤紀錄...")
        session_tracking_history.clear()

@app.get("/")
async def root():
    # 提供一個簡單的 HTTP 測試入口，確認伺服器有成功運行
    return {"status": "SafeStep Server is running! WebSocket endpoint is at /ws"}