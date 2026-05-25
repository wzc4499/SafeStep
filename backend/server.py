import io
import base64
import requests
import numpy as np
import cv2
import heapq
from PIL import Image, ImageOps
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional

from func import VisionProcessor, RiskEvaluator
from traffic_labels import SIGN_CLASSES, SIGN_ZH

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    VisionProcessor.initialize_classifier()

MAPILLARY_ACCESS_TOKEN = " "

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

class SystemPayload(BaseModel):
    mode: str
    image: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    heading: Optional[float] = None
    action: Optional[str] = None

def fetch_mapillary_image(lat, lng, heading):
    search_url = f"https://graph.mapillary.com/images?access_token={MAPILLARY_ACCESS_TOKEN}&fields=id,compass_angle&bbox={lng-0.0005},{lat-0.0005},{lng+0.0005},{lat+0.0005}"
    try:
        response = requests.get(search_url, timeout=5)
        if response.status_code != 200: return None
        data = response.json().get('data', [])
        if not data: return None

        best_image_id = None
        min_angle_diff = 360
        for img_data in data:
            angle = img_data.get('compass_angle', 0)
            diff = abs((angle - heading + 180) % 360 - 180)
            if diff < min_angle_diff and diff < 45:
                min_angle_diff = diff
                best_image_id = img_data['id']

        if not best_image_id: return None

        img_url_req = f"https://graph.mapillary.com/{best_image_id}?access_token={MAPILLARY_ACCESS_TOKEN}&fields=thumb_1024_url"
        res = requests.get(img_url_req, timeout=5)
        img_url = res.json().get('thumb_1024_url')

        if img_url:
            img_resp = requests.get(img_url, timeout=5)
            img_bytes = np.frombuffer(img_resp.content, np.uint8)
            img_np = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            return img_np

    except Exception as e:
        print(f"Mapillary 抓取失敗: {e}")
        return None
    return None

def get_sign_name(sign_id: int) -> str:
    try:
        if sign_id != -1 and sign_id < len(SIGN_CLASSES):
            folder_name = SIGN_CLASSES[sign_id]
            return SIGN_ZH.get(folder_name, folder_name)
        return ""
    except Exception:
        return ""

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_tracking_history = {}

    try:
        while True:
            data = await websocket.receive_json()

            try:
                payload = SystemPayload(**data)
            except ValidationError as e:
                await websocket.send_json({"status": "error", "message": "資料格式錯誤", "details": e.errors()})
                continue

            if payload.mode == "motorcycle":
                if None in [payload.lat, payload.lng, payload.heading, payload.action]:
                    await websocket.send_json({"status": "error", "message": "機車模式參數缺失"})
                    continue

                street_img = fetch_mapillary_image(payload.lat, payload.lng, payload.heading)
                if street_img is None:
                    await websocket.send_json({
                        "status": "success", "mode": "motorcycle",
                        "message": "無街景資料", "signs_detected": [], "two_stage_warning": False
                    })
                    continue

                _, detected_items = VisionProcessor.detect_objects(street_img)
                traffic_signs = []
                two_stage_warning = False

                for item in detected_items:
                    if item["class_id"] == 7:
                        sign_id = -1
                        sign_name = ""
                        if VisionProcessor.sign_classifier is not None and VisionProcessor.sign_transforms is not None:
                            try:
                                sign_id = VisionProcessor.classify_traffic_sign(street_img, item["bbox"])
                                sign_name = get_sign_name(sign_id)
                            except Exception as e:
                                print(f"號誌分類失敗: {e}")

                        traffic_signs.append({
                            "bbox": item["bbox"],
                            "confidence": item["confidence"],
                            "specific_sign_id": sign_id,
                            "specific_sign_name": sign_name,
                        })

                        if "left" in payload.action.lower():
                            x1, y1, x2, y2 = map(int, item["bbox"])
                            y1, y2 = max(0, y1), min(street_img.shape[0], y2)
                            x1, x2 = max(0, x1), min(street_img.shape[1], x2)
                            sign_roi = street_img[y1:y2, x1:x2]

                            if sign_roi.size > 0:
                                hsv = cv2.cvtColor(sign_roi, cv2.COLOR_BGR2HSV)
                                mask = cv2.inRange(hsv, np.array([100, 150, 50]), np.array([140, 255, 255]))
                                if (cv2.countNonZero(mask) / (sign_roi.shape[0] * sign_roi.shape[1] + 1e-6)) > 0.3:
                                    two_stage_warning = True

                response_msg = "需兩段式左轉" if two_stage_warning else "可直接左轉"
                await websocket.send_json({
                    "status": "success", "mode": "motorcycle", "message": response_msg,
                    "signs_detected": traffic_signs, "two_stage_warning": two_stage_warning
                })

            elif payload.mode == "pedestrian":
                if not payload.image:
                    await websocket.send_json({"status": "error", "message": "缺少影像參數"})
                    continue

                try:
                    img_bytes = base64.b64decode(payload.image)
                    img_pil = Image.open(io.BytesIO(img_bytes))
                    img_pil = ImageOps.exif_transpose(img_pil)
                    img_pil = img_pil.convert("RGB")
                    img_np = np.array(img_pil)
                    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    img_h, img_w = img_cv2.shape[:2]
                except Exception as e:
                    await websocket.send_json({"status": "error", "message": f"影像處理錯誤: {str(e)}"})
                    continue

                annotated_image, detected_items = VisionProcessor.detect_objects(img_cv2)

                # 斑馬線偵測，同時取得座標
                lane_data = {"zebra": [], "sidewalk": [], "img_w": img_w, "img_h": img_h}
                try:
                    annotated_image, lane_data = VisionProcessor.detect_lanes(annotated_image)
                except Exception as e:
                    print(f"斑馬線/人行道辨識發生錯誤: {e}")

                analyzed_items, alert_queue = RiskEvaluator.evaluate_frame_risk(
                    detected_items, img_w, img_h, session_tracking_history, mode="pedestrian"
                )

                boxes = []
                for item in analyzed_items:
                    label_en = VisionProcessor.get_label_name(item["class_id"])

                    light_status = -1
                    if label_en.lower() in ["traffic light", "pedestrian light"]:
                        try:
                            light_status = VisionProcessor.traffic_lights(img_cv2, item["bbox"])
                        except Exception as e:
                            print(f"紅綠燈辨識失敗: {e}")

                    specific_sign_id = -1
                    specific_sign_name = ""
                    if (label_en.lower() == "traffic sign"
                            and VisionProcessor.sign_classifier is not None
                            and VisionProcessor.sign_transforms is not None):
                        try:
                            specific_sign_id = VisionProcessor.classify_traffic_sign(img_cv2, item["bbox"])
                            specific_sign_name = get_sign_name(specific_sign_id)
                        except Exception as e:
                            print(f"號誌分類失敗: {e}")

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

                _, buffer = cv2.imencode('.jpg', annotated_image)
                processed_image_b64 = base64.b64encode(buffer).decode('utf-8')

                await websocket.send_json({
                    "status": "success",
                    "mode": "pedestrian",
                    "boxes": boxes,
                    "img_width": img_w,
                    "img_height": img_h,
                    "lanes": lane_data,
                    "processed_image": processed_image_b64,
                })

    except WebSocketDisconnect:
        print("Client 已經斷開連線，清理追蹤紀錄...")
        session_tracking_history.clear()

@app.get("/")
async def root():
    return {"status": "SafeStep Server is running! WebSocket endpoint is at /ws"}