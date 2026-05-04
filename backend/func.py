import cv2
import numpy as np
import os
from ultralytics import YOLO


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
    def evaluate_frame_risk(detected_items, image_height, history_dict):
        """處理單幀所有物件，計算危險度並回傳排序後的列表"""
        analyzed_items = []
        current_frame_ids = set()

        for item in detected_items:
            track_id = item["track_id"]
            cls_id = item["class_id"]
            y2 = item["bbox"][3]

            current_frame_ids.add(track_id)

            # 主指標：物理動態評估 (滿分 80 分)

            # [距離分數] 佔 50 分。越靠近畫面底部 (1.0)，分數越高。
            proximity = y2 / image_height
            distance_score = proximity * 50.0

            # [速度分數] 佔 30 分。
            approach_speed = RiskEvaluator.calculate_relative_speed(track_id, y2, history_dict)

            if approach_speed < 0:
                # 物件正在遠離 (delta_y 為負)
                speed_score = 0.0
                distance_score *= 0.2  # 正在遠離的物件，其距離威脅度大幅降低 (打 2 折)
            else:
                # 物件正在逼近或靜止，乘 15.0 作為放大係數，並限制最高只能加 30 分，避免極端數值
                speed_score = min(approach_speed * 15.0, 30.0)

            # 主指標總分
            kinematic_score = distance_score + speed_score

            # 副指標：物件類別加成 (滿分 20 分)
            class_bonus = RiskEvaluator.BASE_RISK_SCORES.get(cls_id, 5) # 預設未知物件加 5 分

            # 計算總分 (滿分 100 分)
            total_risk = kinematic_score + class_bonus

            item["risk_score"] = total_risk
            analyzed_items.append(item)

        # 【記憶體維護】清理已離開畫面的物件
        obsolete_ids = set(history_dict.keys()) - current_frame_ids
        for obs_id in obsolete_ids:
            if obs_id != -1:
                del history_dict[obs_id]

        # 依危險分數由高到低排序
        analyzed_items.sort(key=lambda x: x["risk_score"], reverse=True)

        return analyzed_items


# 影像辨識與視覺處理 ------------------------------------------------------------------------
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

# 測試 -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. 設定圖片路徑
    image_path = "C:/mypy/Intelligent_Assistance_Systems/dataset/images/train/t_88.png"  # 請換成你的圖片檔名
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"無法讀取圖片: {image_path}，請確認檔案是否存在與路徑是否正確。")
        exit()

    # 處理 4 通道 (BGRA) 圖片轉為 3 通道 (BGR)
    if frame.shape[-1] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    image_height = frame.shape[0]
    history_dict = {} # 單張圖片無法算速度，但為了相容函式還是提供空字典

    print("開始測試 YOLO 單張影像辨識...")
    print("=" * 65)

    # 2. 呼叫辨識
    annotated_image, detected_items = VisionProcessor.detect_objects(frame)
    analyzed_items = RiskEvaluator.evaluate_frame_risk(detected_items, image_height, history_dict)

    # 3. 輸出結果
    if analyzed_items:
        print(f"偵測到 {len(analyzed_items)} 個物件:")
        for item in analyzed_items:
            track_id = item["track_id"]
            class_name = VisionProcessor.get_label_name(item["class_id"])
            risk_score = item["risk_score"]
            proximity_percentage = (item["bbox"][3] / image_height) * 100

            print(f"  > ID: {track_id:2d} | 物件: {class_name:10} | 畫面距離: {proximity_percentage:5.1f}% | 危險指數: {risk_score:5.2f}")
        print("-" * 65)

    # 4. 顯示圖片
    cv2.imshow("Image Test", annotated_image)
    cv2.waitKey(0) # 加上 0 代表無限等待，直到你按任意鍵關閉視窗
    cv2.destroyAllWindows()


    # # 1. 設定影像來源 (可輸入測試影片的路徑 "test_video.mp4"，或輸入 0 使用本機攝影機)
    # video_source = "C:/Users/USER/OneDrive/Desktop/asf.png"
    # cap = cv2.VideoCapture(video_source)

    # if not cap.isOpened():
    #     print(f"無法開啟影像來源: {video_source}，請確認路徑或攝影機連接。")
    #     exit()

    # # 2. 初始化歷史字典，讓 RiskEvaluator 可以計算跨幀的相對速度
    # history_dict = {}

    # print("開始測試 YOLO 影像辨識與危險性評估系統...")
    # print("=" * 65)

    # frame_count = 0
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("影片播放結束或無法讀取畫面。")
    #         break

    #     # ==========================================
    #     # [新增] 檢查影像通道數，如果是 4 (BGRA)，就轉成 3 (BGR)
    #     # ==========================================
    #     if frame.shape[-1] == 4:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    #     frame_count += 1
    #     image_height = frame.shape[0]

    #     # 3. 呼叫 VisionProcessor 進行 YOLO 辨識與追蹤
    #     annotated_image, detected_items = VisionProcessor.detect_objects(frame)

    #     # 4. 呼叫 RiskEvaluator 計算危險指數，並取得排序後的列表
    #     analyzed_items = RiskEvaluator.evaluate_frame_risk(detected_items, image_height, history_dict)

    #     # 5. 在終端機輸出要求資訊
    #     if analyzed_items:
    #         print(f"[Frame {frame_count:04d}] 偵測到 {len(analyzed_items)} 個物件:")
    #         for item in analyzed_items:
    #             # 取得所需資訊
    #             track_id = item["track_id"]
    #             class_name = VisionProcessor.get_label_name(item["class_id"])
    #             risk_score = item["risk_score"]

    #             # 這裡的距離根據你的邏輯，是以 y2 在畫面中的比例來判斷 (越接近底部代表越近)
    #             y2 = item["bbox"][3]
    #             proximity_percentage = (y2 / image_height) * 100

    #             # 格式化輸出
    #             print(f"  > ID: {track_id:2d} | 物件: {class_name:10} | 畫面距離(接近度): {proximity_percentage:5.1f}% | 危險指數: {risk_score:5.2f}")
    #         print("-" * 65)

    #     # 6. 可視化展示 (將辨識與車道線畫在畫面上顯示)
    #     # 若需要結合車道線，可取消註解下方這行：
    #     # annotated_image = VisionProcessor.detect_lanes(annotated_image)

    #     cv2.imshow("Risk Evaluation Test", annotated_image)

    #     # 按下 'q' 鍵退出迴圈
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         print("使用者手動中斷測試。")
    #         break

    # # 釋放資源
    # cap.release()
    # cv2.destroyAllWindows()