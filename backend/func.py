import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==========================================
# 1. 流程控制與危險評估 (輔助邏輯系統)
# ==========================================
class RiskEvaluator:
    # 基礎危險權重 (拉到類別層級，避免每幀重複建立)
    BASE_RISK_SCORES = {
        1: 10,  # 行人
        2: 8,   # 機車
        0: 7,   # 障礙物
        3: 5,   # 汽車
        4: 4,   # 小貨車
        5: 4,   # 大貨車
        6: 4,   # 公車
        7: 1    # 交通號誌
    }
    WEIGHT_DISTANCE = 0.6
    WEIGHT_SPEED = 0.4

    @staticmethod
    def calculate_relative_speed(track_id, current_y2, history_dict):
        """計算像素移動速度。history_dict 由外部 (server) 提供以支援多裝置。"""
        if track_id == -1 or track_id not in history_dict:
            history_dict[track_id] = current_y2
            return 0.0

        prev_y2 = history_dict[track_id]
        delta_y = current_y2 - prev_y2
        history_dict[track_id] = current_y2

        return delta_y / 10.0

    @staticmethod
    def evaluate_frame_risk(detected_items, image_height, history_dict):
        """處理單幀所有物件，計算危險度並回傳排序後的列表"""
        analyzed_items = []
        current_frame_ids = set()

        for item in detected_items:
            track_id = item["track_id"]
            cls_id = item["class_id"]
            y2 = item["bbox"][3]

            current_frame_ids.add(track_id)

            # 1. 取得基礎分數
            base_score = RiskEvaluator.BASE_RISK_SCORES.get(cls_id, 1)

            # 2. 距離危險度 (0~1)
            proximity = y2 / image_height

            # 3. 相對逼近速度
            approach_speed = RiskEvaluator.calculate_relative_speed(track_id, y2, history_dict)

            # 4. 綜合危險值
            speed_factor = 0.1 if approach_speed < 0 else (1.0 + approach_speed)
            total_risk = base_score * ((proximity * RiskEvaluator.WEIGHT_DISTANCE) + (speed_factor * RiskEvaluator.WEIGHT_SPEED))

            item["risk_score"] = total_risk
            analyzed_items.append(item)

        # 【記憶體維護】清理已離開畫面的物件
        obsolete_ids = set(history_dict.keys()) - current_frame_ids
        for obs_id in obsolete_ids:
            if obs_id != -1:
                del history_dict[obs_id]

        # 依危險分數由高到低排序 (使用內建 sort 效能優於 PriorityQueue)
        analyzed_items.sort(key=lambda x: x["risk_score"], reverse=True)

        return analyzed_items

# ==========================================
# 2. 影像辨識與視覺處理
# ==========================================
class VisionProcessor:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "Taiwan_Traffic_Project", "weights", "best.pt")

    yolo_model = YOLO(MODEL_PATH)

    @staticmethod
    def detect_objects(image):
        """執行物件追蹤與特徵提取"""
        results = VisionProcessor.yolo_model.track(
            image,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.3,
            verbose=False
        )

        annotated_image = results[0].plot()
        detected_items = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                track_id = int(box.id[0]) if box.id is not None else -1

                detected_items.append({
                    "track_id": track_id,
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": coords
                })

        return annotated_image, detected_items

    @staticmethod
    def get_label_name(class_id):
        """取得標籤名稱"""
        return VisionProcessor.yolo_model.names.get(class_id, f"Unknown-{class_id}")

    @staticmethod
    def detect_lanes(image):
        """車道線辨識 (維持原邏輯)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (int(width * 0.25), height),
            (int(width * 0.85), height),
            (int(width * 0.60), int(height * 0.4)),
            (int(width * 0.40), int(height * 0.4))
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=100)
        line_image = np.zeros_like(image)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1: continue
                slope = (y2 - y1) / (x2 - x1)
                if 0.5 < abs(slope) < 2.0:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 4)

        return cv2.addWeighted(image, 0.8, line_image, 1, 0)