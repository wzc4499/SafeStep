import cv2
import numpy as np
import os
from ultralytics import YOLO
import heapq

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy import VideoFileClip # 注意這裡從 moviepy 改用 moviepy.editor
from IPython.display import HTML

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 流程控制 (輔助邏輯系統) --------------------------------------------------------
class RiskEvaluator:
    # 基礎危險權重 (拉到類別層級，避免每幀重複建立)
    BASE_RISK_SCORES = {
        1: 20,  # 行人
        2: 15,  # 機車
        3: 15,  # 汽車
        4: 15,  # 小貨車
        5: 18,  # 大貨車
        6: 18,  # 公車
        0: 10,  # 障礙物
        7: 0    # 交通號誌
    }

    # 定義「會動的車輛」類別 ID (對應 YOLO 的 COCO 資料集)
    VEHICLE_CLASSES = {2, 3, 4, 5, 6} # 機車、汽車、小貨車、大貨車、公車

    # 警報閾值設定
    ALERT_THRESHOLD = 70.0

    @staticmethod
    # -- 計算物件物件距離 (相對逼近速度) --
    def calculate_relative_speed(track_id, current_y2, history_dict):
        """計算像素移動速度"""
        if track_id == -1 or track_id not in history_dict:
            history_dict[track_id] = current_y2
            return 0.0

        prev_y2 = history_dict[track_id]
        delta_y = current_y2 - prev_y2
        history_dict[track_id] = current_y2

        return delta_y / 10.0

    @staticmethod
    # -- 危險性評估 --
    def evaluate_frame_risk(detected_items, image_width, image_height, history_dict, mode="pedestrian"):
        """處理單幀所有物件，計算危險度並回傳排序後的列表與警報佇列"""
        analyzed_items = []
        current_frame_ids = set()
        alert_queue = [] # 優先權佇列

        for item in detected_items:
            track_id = item["track_id"]
            cls_id = item["class_id"]
            x1, y1, x2, y2 = item["bbox"]

            current_frame_ids.add(track_id)

            # --- 新增的面積比例與距離估算邏輯 (整合至此) ---
            box_area = (x2 - x1) * (y2 - y1)
            img_area = image_width * image_height
            area_ratio = box_area / img_area if img_area > 0 else 0
            danger_pct = round(area_ratio * 100, 1)

            if area_ratio > 0.25:
                danger_level = "高"
            elif area_ratio > 0.08:
                danger_level = "中"
            else:
                danger_level = "低"

            estimated_distance = round(1.0 / (area_ratio + 0.01) * 0.5, 1)
            estimated_distance = min(estimated_distance, 99.9)

            item["danger_level"] = danger_level
            item["area_ratio"] = round(area_ratio, 4)
            item["danger_pct"] = danger_pct
            item["distance"] = estimated_distance
            # ---------------------------------------------

            # 主指標：物理動態評估 (滿分 80 分)

            # [距離分數] 佔 50 分。越靠近畫面底部 (1.0)，分數越高。
            proximity = y2 / image_height
            distance_score = proximity * 50.0

            approach_speed = None
            speed_score = 0.0

            # 模式邏輯判斷：行人模式
            if mode == "pedestrian":
                if cls_id in RiskEvaluator.VEHICLE_CLASSES:
                    # 辨識出會動的機車、車子，額外計算速度
                    approach_speed = RiskEvaluator.calculate_relative_speed(track_id, y2, history_dict)

                    if approach_speed < 0:
                        # 物件正在遠離 (delta_y 為負)
                        speed_score = 0.0
                        distance_score *= 0.2  # 正在遠離的物件，其距離威脅度大幅降低 (打 2 折)
                    else:
                        speed_score = min(approach_speed * 15.0, 30.0)
                else:
                    # 非車輛物件 (如行人、障礙物)，不計算逼近速度
                    speed_score = 0.0
            #　模式邏輯 : 機車模式
            elif mode == "motorcycle":
                approach_speed = RiskEvaluator.calculate_relative_speed(track_id, y2, history_dict)
                if approach_speed < 0:
                    speed_score = 0.0
                    distance_score *= 0.2
                else:
                    speed_score = min(approach_speed * 15.0, 30.0)

            # 副指標：物件類別加成 (滿分 20 分)
            class_bonus = RiskEvaluator.BASE_RISK_SCORES.get(cls_id, 5) # 預設未知物件加 5 分

            # 計算總分 (滿分 100 分)
            total_risk = distance_score + speed_score + class_bonus

            item["risk_score"] = total_risk
            item["proximity_pct"] = proximity * 100
            item["speed"] = approach_speed
            analyzed_items.append(item)

            # 若高於警報閾值，推入優先權佇列 (Max-Heap 實作，以負值存入)
            if total_risk >= RiskEvaluator.ALERT_THRESHOLD:
                heapq.heappush(alert_queue, (-total_risk, track_id, item))

        # 【記憶體維護】清理已離開畫面的物件
        obsolete_ids = set(history_dict.keys()) - current_frame_ids
        for obs_id in obsolete_ids:
            if obs_id != -1:
                del history_dict[obs_id]

        # 依危險分數由高到低排序
        analyzed_items.sort(key=lambda x: x["risk_score"], reverse=True)

        return analyzed_items, alert_queue



# 影像辨識與視覺處理 ------------------------------------------------------------------------
class VisionProcessor:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "Taiwan_Traffic_Project", "weights", "best.pt")

    yolo_model = YOLO(MODEL_PATH)

    # === 建立類別變數來快取校正矩陣，避免每幀重複計算 ===
    _mtx = None
    _dist = None

    CLASSIFIER_PATH = os.path.join(BASE_DIR, "runs", "detect", "Taiwan_Traffic_Project", "weights", "best_mobilenet_v3_large.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sign_classifier = None
    sign_transforms = None
    NUM_CLASSES = 372
    @classmethod
    def initialize_classifier(cls):
        """初始化 PyTorch 號誌分類器，只載入一次"""
        if cls.sign_classifier is not None:
            return

        # 檢查權重檔是否存在
        if not os.path.exists(cls.CLASSIFIER_PATH):
            print(f"[警告] 找不到分類器權重檔: {cls.CLASSIFIER_PATH}")
            return

        try:
            # 建立模型結構
            cls.sign_classifier = models.mobilenet_v3_large(weights=None)
            num_ftrs = cls.sign_classifier.classifier[3].in_features
            cls.sign_classifier.classifier[3] = nn.Linear(num_ftrs, cls.NUM_CLASSES)

            # 載入權重並送入 GPU/CPU
            cls.sign_classifier.load_state_dict(torch.load(cls.CLASSIFIER_PATH, map_location=cls.device))
            cls.sign_classifier = cls.sign_classifier.to(cls.device)
            cls.sign_classifier.eval() # 設置為評估模式

            # 設定與訓練時相同的圖像預處理轉換
            cls.sign_transforms = transforms.Compose([
                transforms.Resize((224, 224)), # 強制將 YOLO 給的框框變形為 224x224，不切邊緣
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            print("PyTorch 交通號誌分類器載入成功！")
        except Exception as e:
            print(f"分類器載入失敗: {e}")

    @classmethod
    def initialize_camera_calibration(cls):
        """初始化相機校正，確保整個程式生命週期只執行一次"""
        if cls._mtx is not None:
            return # 已經校正過，直接返回

        print("正在進行相機校正，請稍候...")
        calib_path = os.path.join(cls.BASE_DIR, 'camera_cal', 'calibration*.jpg')
        images_path = glob.glob(calib_path)

        # 準備物體點 (object points)
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        objpoints = [] # 真實世界空間中的 3D 點
        imgpoints = [] # 影像平面上的 2D 點
        img_size = None

        if not images_path:
            print(f"[警告] 找不到校正影像於 {calib_path}，將略過畸變校正。請確認資料夾位置。")
            return

        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = (gray.shape[1], gray.shape[0])
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        if objpoints and img_size:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
            cls._mtx = mtx
            cls._dist = dist
            print("相機校正完成！")

    @staticmethod
    # 物件偵測
    def detect_objects(image):
        """執行物件追蹤與特徵提取"""
        results = VisionProcessor.yolo_model.track(
            image,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.3,
            verbose=False
        )

        annotated_image = image.copy()
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
    # 第 2 階段辨識：交通號誌細部分類
    def classify_traffic_sign(image, bbox):
        """
        將 YOLO 圈出的交通號誌丟入 PyTorch 分類器進行辨識
        """
        VisionProcessor.initialize_classifier()
        if VisionProcessor.sign_classifier is None:
            return -1

        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            return -1

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)

        input_tensor = VisionProcessor.sign_transforms(pil_img).unsqueeze(0).to(VisionProcessor.device)

        with torch.no_grad():
            outputs = VisionProcessor.sign_classifier(input_tensor)
            _, preds = torch.max(outputs, 1)

        return int(preds[0])

    @staticmethod
    # 斑馬線及人行道辨識
    def detect_lanes(image):
        # 1. 確保已進行相機校正 (利用外部快取，避免每幀重複計算導致效能崩潰)
        VisionProcessor.initialize_camera_calibration()

        img_shape = image.shape
        img_size = (img_shape[1], img_shape[0])
        h, w = img_shape[:2]

        def get_perspective_matrices(img_shape):
            h, w = img_shape[:2]
            is_landscape = w > h
            center_x = w * 0.5
            IS_DASHCAM_VIDEO = False
            if is_landscape:
                if IS_DASHCAM_VIDEO:
                    top_y, bottom_y = h * 0.62, h * 0.95
                    top_width, bottom_width = w * 0.15, w * 0.75
                else:
                    top_y, bottom_y = h * 0.30, h * 1.0
                    top_width, bottom_width = w * 0.20, w * 0.95
            else:
                top_y, bottom_y = h * 0.50, h * 1.0
                top_width, bottom_width = w * 0.35, w * 0.95
            src = np.float32([
                [center_x - bottom_width / 2, bottom_y],
                [center_x - top_width / 2, top_y],
                [center_x + top_width / 2, top_y],
                [center_x + bottom_width / 2, bottom_y]
            ])
            offset_x = w * 0.25 if is_landscape else w * 0.15
            dst = np.float32([
                [offset_x, h],
                [offset_x, 0],
                [w - offset_x, 0],
                [w - offset_x, h]
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            Minv = cv2.getPerspectiveTransform(dst, src)
            return M, Minv

        def detect_green_walkway(img):
            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            mask[0:int(h * 0.5), :] = 0
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(mask)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 2000:
                    cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
            return clean_mask

        def detect_zebra_crossing(undist_img, M, img_size):
            hsv = cv2.cvtColor(undist_img, cv2.COLOR_RGB2HSV)
            lower_white = np.array([0, 0, 215])
            upper_white = np.array([180, 50, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            warped = cv2.warpPerspective(white_mask, M, img_size)
            kernel_open = np.ones((3, 3), np.uint8)
            kernel_close = np.ones((15, 15), np.uint8)
            morph = cv2.morphologyEx(warped, cv2.MORPH_OPEN, kernel_open)
            morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_close)
            merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 35))
            merged = cv2.dilate(morph, merge_kernel, iterations=1)
            contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            final_mask = np.zeros_like(merged)
            valid_stripe_pts = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                if area < 1000:
                    continue
                roi = morph[y:y+h, x:x+w]
                stripe_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                temp_stripes = []
                stripe_count = 0
                for s in stripe_contours:
                    sx, sy, sw, sh = cv2.boundingRect(s)
                    s_area = cv2.contourArea(s)
                    if s_area > 250 and sw > sh * 2.5 and sw > 60:
                        stripe_count += 1
                        temp_stripes.append(s)
                if stripe_count >= 3:
                    for s in temp_stripes:
                        s_global = s.copy()
                        s_global[:, :, 0] += x
                        s_global[:, :, 1] += y
                        valid_stripe_pts.append(s_global)
            if len(valid_stripe_pts) > 0:
                all_pts = np.concatenate(valid_stripe_pts, axis=0)
                hull = cv2.convexHull(all_pts)
                cv2.drawContours(final_mask, [hull], -1, 255, -1)
            return final_mask

        M, Minv = get_perspective_matrices(img_shape)

        if VisionProcessor._mtx is not None and VisionProcessor._dist is not None:
            undist_img = cv2.undistort(image, VisionProcessor._mtx, VisionProcessor._dist, None, VisionProcessor._mtx)
        else:
            undist_img = image.copy()

        green_mask = detect_green_walkway(undist_img)
        zebra_warped_mask = detect_zebra_crossing(undist_img, M, img_size)

        zebra_real_mask = cv2.warpPerspective(zebra_warped_mask, Minv, img_size)

        overlay = np.zeros_like(undist_img)

        zebra_contours, _ = cv2.findContours(zebra_real_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 轉換座標格式供前端 SVG 使用
        zebra_coords = [cnt.reshape(-1, 2).tolist() for cnt in zebra_contours]
        green_coords = [cnt.reshape(-1, 2).tolist() for cnt in green_contours]

        return undist_img, {
            "zebra": zebra_coords,
            "sidewalk": green_coords,
            "img_w": img_size[0],
            "img_h": img_size[1],
        }

    @staticmethod
    #　行人紅綠燈及一車輛用紅綠燈燈號辨識
    def traffic_lights(image, bbox):
        """
        辨識紅綠燈燈號顏色 (適用於車輛紅綠燈與行人紅綠燈)
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return -1

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        _, bright_structure_mask = cv2.threshold(v_channel, 150, 255, cv2.THRESH_BINARY)

        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        lower_green = np.array([40, 70, 50])
        upper_green = np.array([90, 255, 255])

        lower_yellow = np.array([15, 70, 50])
        upper_yellow = np.array([35, 255, 255])

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red_base = cv2.bitwise_or(mask_red1, mask_red2)

        mask_green_base = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow_base = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask_red = cv2.bitwise_and(mask_red_base, bright_structure_mask)
        mask_green = cv2.bitwise_and(mask_green_base, bright_structure_mask)
        mask_yellow = cv2.bitwise_and(mask_yellow_base, bright_structure_mask)

        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        yellow_pixels = cv2.countNonZero(mask_yellow)

        threshold = 15
        max_pixels = max(red_pixels, green_pixels, yellow_pixels)

        if max_pixels < threshold:
            return -1
        if max_pixels == red_pixels:
            return 0
        elif max_pixels == green_pixels:
            return 1
        else:
            return 2


# ==========================================
# 測試 -----------------------------------------------------------------------------------------------
# ==========================================
if __name__ == "__main__":
    # # 1. 設定圖片路徑
    # image_path = r"C:\Users\USER\OneDrive\圖片\Screenshots\螢幕擷取畫面 2026-05-14 121643.png"

    # image_bgr = cv2.imread(image_path)
    # if image_bgr is None:
    #     print(f"[錯誤] 找不到圖片，請確認路徑: {image_path}")
    #     exit()

    # print("=" * 65)
    # print("開始測試影像辨識系統...")

    # # ==========================================
    # # 測試 1: 斑馬線與人行道辨識 (您剛剛修改的核心邏輯)
    # # ==========================================
    # print("-> 執行斑馬線/車道線辨識 (detect_lanes)...")

    # lane_result_image = VisionProcessor.detect_lanes(image_bgr.copy())

    # h_lane, w_lane = lane_result_image.shape[:2]
    # if h_lane > 800: # 防呆：避免高畫質圖片超出螢幕
    #     scale = 800 / h_lane
    #     lane_result_image = cv2.resize(lane_result_image, (int(w_lane * scale), int(h_lane * scale)))

    # cv2.imshow("Zebra Crossing & Lane Detection", lane_result_image)

    # # ==========================================
    # # 測試 2: YOLO 物件偵測與號誌辨識 (原本的測試)
    # # ==========================================
    # print("-> 執行 YOLO 物件偵測與 PyTorch 號誌分類...")
    # annotated_image, detected_items = VisionProcessor.detect_objects(image_bgr.copy())

    # sign_count = 0
    # if detected_items:
    #     for item in detected_items:
    #         class_name = VisionProcessor.get_label_name(item["class_id"])
    #         if class_name.lower() == "traffic sign":
    #             sign_count += 1
    #             box = item["bbox"]
    #             sign_id = VisionProcessor.classify_traffic_sign(image_bgr, box)
    #             box_ints = [int(c) for c in box]
    #             print(f"  > 🚦 發現號誌！座標 {box_ints}, PyTorch 分類 ID: {sign_id}")

    # if sign_count == 0:
    #     print("  > 畫面中未偵測到任何交通標誌。")

    # h_obj, w_obj = annotated_image.shape[:2]
    # if h_obj > 800:
    #     scale = 800 / h_obj
    #     annotated_image = cv2.resize(annotated_image, (int(w_obj * scale), int(h_obj * scale)))

    # cv2.imshow("YOLO Object Detection", annotated_image)

    # print("=" * 65)
    # print(">>> 圖片已顯示！(請注意可能有兩個視窗)")
    # print(">>> 請在影像視窗按下『任意鍵』以進入下一步或關閉。")

    # # 等待按鍵關閉所有視窗
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pass