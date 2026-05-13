import io
import base64
import requests
import numpy as np
import cv2
import heapq
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional

# 引入已經寫好的 func.py 模組
# 這裡貫徹了「關注點分離」：視覺辨識與數學運算都在 func.py 處理，server.py 只負責通訊與流程控制
from func import VisionProcessor, RiskEvaluator

app = FastAPI()

# 設定 CORS (跨來源資源共用)
# 這非常重要，它允許你的前端 (可能是 React, Vue 或純 HTML) 跨網域來存取這個 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 開放所有來源連線，正式上線時建議改為你的前端網址
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 系統與 API 設定
# Mapillary Access Token (用於抓取街景圖)
MAPILLARY_ACCESS_TOKEN = "YOUR_MAPILLARY_TOKEN_HERE"

# 翻譯字典：將 YOLO 辨識出的英文標籤轉為繁體中文，方便前端直接顯示
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

# 統一的請求 Payload 模型 (用於驗證 WebSocket 收到的 JSON 格式)
# 使用 Pydantic 可以自動檢查前端傳來的資料型態是否正確
class SystemPayload(BaseModel):
    mode: str  # 選項: "motorcycle" 或 "pedestrian"
    image: Optional[str] = None      # 行人模式專用：Base64 影像
    lat: Optional[float] = None      # 機車模式專用：緯度
    lng: Optional[float] = None      # 機車模式專用：經度
    heading: Optional[float] = None  # 機車模式專用：機車行進方位角 (0-360度)
    action: Optional[str] = None     # 機車模式專用：導航指令 (如 turn_left)

# ---- 輔助函式：Mapillary 影像抓取
def fetch_mapillary_image(lat, lng, heading):
    """
    根據 GPS 與方位角，從 Mapillary 抓取最近的前方街景圖片。
    這能讓系統在使用者抵達路口前，預先判斷交通號誌。
    """
    # 1. 建立一個包含該座標的微小搜尋框 (Bounding Box)，尋找附近的影像 ID
    search_url = f"https://graph.mapillary.com/images?access_token={MAPILLARY_ACCESS_TOKEN}&fields=id,compass_angle&bbox={lng-0.0005},{lat-0.0005},{lng+0.0005},{lat+0.0005}"
    try:
        response = requests.get(search_url, timeout=5)
        if response.status_code != 200: return None

        data = response.json().get('data', [])
        if not data: return None

        # 2. 篩選出「視角最接近機車前進方向」的圖片
        # 這是為了避免抓到路口「往回看」或「往側邊看」的街景圖
        best_image_id = None
        min_angle_diff = 360
        for img_data in data:
            angle = img_data.get('compass_angle', 0)
            diff = abs((angle - heading + 180) % 360 - 180)
            if diff < min_angle_diff and diff < 45: # 容忍 45 度的視角誤差
                min_angle_diff = diff
                best_image_id = img_data['id']

        if not best_image_id: return None

        # 3. 取得該圖片的 1024px 縮圖下載網址
        img_url_req = f"https://graph.mapillary.com/{best_image_id}?access_token={MAPILLARY_ACCESS_TOKEN}&fields=thumb_1024_url"
        res = requests.get(img_url_req, timeout=5)
        img_url = res.json().get('thumb_1024_url')

        # 4. 下載圖片，並轉換為 OpenCV 的格式 (BGR Numpy Array) 供 YOLO 使用
        if img_url:
            img_resp = requests.get(img_url, timeout=5)
            img_bytes = np.frombuffer(img_resp.content, np.uint8)
            img_np = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            return img_np

    except Exception as e:
        print(f"Mapillary 抓取失敗: {e}")
        return None
    return None

# ---- 核心 WebSocket 端點：雙向即時通訊
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # 接受前端的 WebSocket 連線請求
    await websocket.accept()

    # 建立「專屬於此連線」的歷史字典。
    # 這樣如果有 10 個使用者同時連線，他們算出的「逼近速度」才不會互相干擾 (狀態隔離)
    session_tracking_history = {}

    try:
        # 保持連線開啟，持續接收前端傳來的畫面或座標
        while True:
            # 接收前端的 JSON 資料
            data = await websocket.receive_json()

            # 透過 Pydantic 驗證資料是否符合我們定義的格式
            try:
                payload = SystemPayload(**data)
            except ValidationError as e:
                await websocket.send_json({"status": "error", "message": "資料格式錯誤", "details": e.errors()})
                continue

            # 機車模式 (導航 + Mapillary 街景判斷)
            if payload.mode == "motorcycle":
                # 檢查必填參數
                if None in [payload.lat, payload.lng, payload.heading, payload.action]:
                    await websocket.send_json({"status": "error", "message": "機車模式參數缺失"})
                    continue

                # Mapillary 抓圖
                street_img = fetch_mapillary_image(payload.lat, payload.lng, payload.heading)
                if street_img is None:
                    await websocket.send_json({"status": "success", "mode": "motorcycle", "message": "無街景資料", "signs_detected": [], "two_stage_warning": False})
                    continue

                # 呼叫 YOLO 模型偵測圖片中的所有物件
                _, detected_items = VisionProcessor.detect_objects(street_img)
                traffic_signs = []
                two_stage_warning = False

                # 篩選出交通號誌，並檢查是否為兩段式左轉牌
                for item in detected_items:
                    if item["class_id"] == 7: # 7 是 YOLO 的 Traffic sign 類別
                        traffic_signs.append({"bbox": item["bbox"], "confidence": item["confidence"]})

                        # 只有當導航要求「左轉」時，才需要特別去檢查號誌顏色
                        if "left" in payload.action.lower():
                            x1, y1, x2, y2 = map(int, item["bbox"])
                            # 確保裁切框沒有超出圖片邊界 (避免 OpenCV 報錯)
                            y1, y2 = max(0, y1), min(street_img.shape[0], y2)
                            x1, x2 = max(0, x1), min(street_img.shape[1], x2)
                            sign_roi = street_img[y1:y2, x1:x2]

                            if sign_roi.size > 0:
                                # 將影像從 BGR 轉為 HSV 色彩空間，更容易過濾特定顏色
                                hsv = cv2.cvtColor(sign_roi, cv2.COLOR_BGR2HSV)
                                # 設定台灣「兩段式左轉牌」常見的深藍色範圍
                                mask = cv2.inRange(hsv, np.array([100, 150, 50]), np.array([140, 255, 255]))

                                # 計算藍色像素佔整個號誌面積的比例，大於 30% 就視為兩段式左轉
                                if (cv2.countNonZero(mask) / (sign_roi.shape[0] * sign_roi.shape[1] + 1e-6)) > 0.3:
                                    two_stage_warning = True

                # 組合回傳訊息
                response_msg = "需兩段式左轉" if two_stage_warning else "可直接左轉"
                await websocket.send_json({
                    "status": "success", "mode": "motorcycle", "message": response_msg,
                    "signs_detected": traffic_signs, "two_stage_warning": two_stage_warning
                })


            # -- 行人模式 (Server 不做運算，完全依賴 func.py 回傳值)

            elif payload.mode == "pedestrian":
                if not payload.image:
                    await websocket.send_json({"status": "error", "message": "缺少影像參數"})
                    continue

                try:
                    # 影像前處理
                    # 1. 解碼 Base64 字串
                    img_bytes = base64.b64decode(payload.image)
                    img_pil = Image.open(io.BytesIO(img_bytes))

                    # 2. 自動修正手機照片的 EXIF 旋轉問題 (避免照片轉了 90 度導致辨識失敗)
                    img_pil = ImageOps.exif_transpose(img_pil)
                    img_pil = img_pil.convert("RGB")
                    img_np = np.array(img_pil)

                    # 3. 轉為 OpenCV 預設的 BGR 色彩空間
                    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    img_h, img_w = img_cv2.shape[:2]

                except Exception as e:
                    await websocket.send_json({"status": "error", "message": f"影像處理錯誤: {str(e)}"})
                    continue

                # 1. 呼叫 VisionProcessor 取得畫面中所有物件的邊界框與類別
                annotated_image, detected_items = VisionProcessor.detect_objects(img_cv2)

                # 2. 呼叫 RiskEvaluator 計算所有危險度 (包含面積、距離、速度與綜合風險)
                # 寬高傳給 func.py，讓 func.py 負責學計算
                analyzed_items, alert_queue = RiskEvaluator.evaluate_frame_risk(
                    detected_items,
                    img_w,
                    img_h,
                    session_tracking_history, # 傳入連線專屬的字典來追蹤跨幀速度
                    mode="pedestrian"
                )

                # 3. Server 組裝 JSON 回傳前端
                boxes = []
                for item in analyzed_items:
                    # 取得英文標籤
                    label_en = VisionProcessor.get_label_name(item["class_id"])

                    boxes.append({
                        "track_id": item["track_id"],
                        "x1": item["bbox"][0], "y1": item["bbox"][1],
                        "x2": item["bbox"][2], "y2": item["bbox"][3],
                        "confidence": round(item["confidence"], 2),
                        "label": LABEL_ZH.get(label_en, label_en),
                        # 以下數值皆由 func.py 計算好後直接拿來用，Server 不干涉邏輯
                        "danger": item["danger_level"],
                        "danger_pct": item["danger_pct"],
                        "distance": item["distance"],
                        "area_ratio": item["area_ratio"],
                        "speed": item["speed"],
                    })

                # 將結果打包發送給前端
                await websocket.send_json({
                    "status": "success",
                    "mode": "pedestrian",
                    "boxes": boxes,
                    "img_width": img_w,
                    "img_height": img_h
                })

    except WebSocketDisconnect:
        print("Client 已經斷開連線，清理追蹤紀錄...")
        session_tracking_history.clear()

@app.get("/")
async def root():
    return {"status": "SafeStep Server is running! WebSocket endpoint is at /ws"}