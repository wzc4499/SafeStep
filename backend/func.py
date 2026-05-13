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
                #
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

    # === [修復] 建立類別變數來快取校正矩陣，避免每幀重複計算 ===
    _mtx = None
    _dist = None

    @classmethod
    def initialize_camera_calibration(cls):
        """初始化相機校正，確保整個程式生命週期只執行一次"""
        if cls._mtx is not None:
            return # 已經校正過，直接返回

        print("正在進行相機校正，請稍候...")
        # 將絕對路徑改為相對路徑，增加程式在不同電腦上的可移植性
        # 如果你必須使用原本的絕對路徑，請將 calib_path 替換為 r'C:\Users\USER\OneDrive\Desktop\programing_designer\SafeStep\backend\camera_cal\calibration*.jpg'
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
    # 車道辨識
    # 程式參考於 https://github.com/bob800530/CarND-Advanced-Lane-Lines.git
    # 修改 src 透視變換使其動態化以進行市區車道辨識
    def detect_lanes(image):
        # 確保相機已經校正 (只會真正執行一次)
        VisionProcessor.initialize_camera_calibration()

        # ---- 影像閾值處理 (來自 saturation.py)
        def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
            """計算 x 或 y 方向的 Sobel 絕對值，並套用閾值"""
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if orient == 'x':
                abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel))
            if orient == 'y':
                abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel))

            scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
            binary_output = np.zeros_like(scaled_sobel)
            binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
            return binary_output

        def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
            """計算梯度的強度 (Magnitude) 並套用閾值"""
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            gradmag = np.sqrt(sobelx**2 + sobely**2)

            scale_factor = np.max(gradmag) / 255
            gradmag = (gradmag / scale_factor).astype(np.uint8)
            binary_output = np.zeros_like(gradmag)
            binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
            return binary_output

        def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
            """計算梯度的方向 (Direction) 並套用閾值"""
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

            absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
            binary_output = np.zeros_like(absgraddir)
            binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
            return binary_output

        def hls_select(img, thresh=(0, 255)):
            """轉換為 HLS 色彩空間，並對 S 通道及 L 通道套用閾值過濾"""
            hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            L = hls[:,:,1]
            S = hls[:,:,2]
            binary_output = np.zeros_like(S)
            binary_output[(S > thresh[0]) & (S <= thresh[1]) & (L > 50)] = 1
            return binary_output

        def combine_threshs(grad_x, grad_y, mag_binary, dir_binary, col_binary):
            """將方向、強度、色彩等多種閾值結果合併為單一的二值化影像"""
            combined = np.zeros_like(dir_binary)
            combined[((grad_x == 1) & (grad_y == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (col_binary == 1)] = 1
            return combined


        # ---- 透視變換 (來自 perspective.py)

        # 定義透視變換的來源點 (src) 與目標點 (dst)
        def get_transform_matrices(img):
            """根據輸入影像的尺寸，動態計算並回傳透視變換矩陣 M 與反向矩陣 Minv"""
            height = img.shape[0]
            width = img.shape[1]

            # === 設定動態比例參數 ===
            # 這些比例是根據原本 1280x720 的最佳點位換算出來的
            bottom_y = height * 0.97      # 底部位於影像高度的 97% 處
            top_y = height * 0.64         # 頂部位於影像高度的 64% 處 (地平線下方)

            bottom_left_x = width * 0.22  # 左下角位於寬度的 22%
            bottom_right_x = width * 0.88 # 右下角位於寬度的 88%
            top_left_x = width * 0.46     # 左上角位於寬度的 46%
            top_right_x = width * 0.57    # 右上角位於寬度的 57%

            # 1. 動態建立 src (來源點)
            src = np.float32([
                [bottom_left_x, bottom_y],  # 左下
                [top_left_x, top_y],        # 左上
                [top_right_x, top_y],       # 右上
                [bottom_right_x, bottom_y]  # 右下
            ])

            # 2. 動態建立 dst (目標點，也就是鳥瞰圖展開後的矩形)
            offset_x = width * 0.2  # 左右兩側留白 20%
            dst = np.float32([
                [offset_x, height],          # 左下
                [offset_x, 0],               # 左上
                [width - offset_x, 0],       # 右上
                [width - offset_x, height]   # 右下
            ])

            # 3. 計算並回傳矩陣
            M = cv2.getPerspectiveTransform(src, dst)
            Minv = cv2.getPerspectiveTransform(dst, src)

            return M, Minv

        def perspective(img):
            """將影像轉換為鳥瞰圖"""
            M, _ = get_transform_matrices(img)
            return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

        def unperspective(img, original_img_shape):
            """將鳥瞰圖轉回原本的攝影機視角
            注意：因為轉回來的圖片需要配合原始尺寸，所以傳入原始影像的 shape
            """
            _, Minv = get_transform_matrices(np.zeros(original_img_shape)) # 建立一個假影像來取尺寸
            return cv2.warpPerspective(img, Minv, (original_img_shape[1], original_img_shape[0]))

        def draw_perspective_polygon(img):
            """(測試用) 在原圖上畫出用於透視變換的基準區域多邊形"""
            # 這邊為了相容性宣告一個假的 src
            height, width = img.shape[:2]
            src = np.float32([[width*0.22, height*0.97], [width*0.46, height*0.64], [width*0.57, height*0.64], [width*0.88, height*0.97]])
            pts = np.array([src[0], src[1], src[2], src[3]], np.int32)
            img_copy = np.copy(img)
            cv2.polylines(img_copy, [pts], True, (0, 255, 255), 3)
            return img_copy


        # ---- 尋找車道線與多項式擬合 (整合自 test.py)
        def find_lane_pixels(binary_warped):
            """使用滑動視窗法在二值化的鳥瞰圖中尋找車道線像素"""
            # 對影像下半部取直方圖
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

            midpoint = int(histogram.shape[0]//2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            nwindows = 9
            margin = 100
            minpix = 50
            window_height = int(binary_warped.shape[0]//nwindows)

            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            leftx_current = leftx_base
            rightx_current = rightx_base
            left_lane_inds = []
            right_lane_inds = []

            for window in range(nwindows):
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # 畫出滑動視窗 (視覺化用途)
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

                # 找出視窗內的非零像素索引
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                if len(good_left_inds) > minpix:
                    leftx_current = int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = int(np.mean(nonzerox[good_right_inds]))

            try:
                left_lane_inds = np.concatenate(left_lane_inds)
                right_lane_inds = np.concatenate(right_lane_inds)
            except ValueError:
                pass

            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            return leftx, lefty, rightx, righty, out_img

        def fit_polynomial(binary_warped):
            """為找到的車道線像素擬合二次多項式，並回傳標記影像與相關數據"""
            leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

            try:
                left_fit = np.polyfit(lefty, leftx, 2)
                right_fit = np.polyfit(righty, rightx, 2)
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            except TypeError:
                print('擬合線段失敗！')
                left_fit = [1, 1, 0]
                right_fit = [1, 1, 0]
                left_fitx = 1*ploty**2 + 1*ploty
                right_fitx = 1*ploty**2 + 1*ploty

            # 將找到的車道線像素分別上色為紅、藍
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

            line_img = np.zeros_like(out_img)
            window_img = np.zeros_like(out_img)
            line_img[lefty, leftx] = [255, 0, 0]
            line_img[righty, rightx] = [0, 0, 255]

            # 設定車道覆蓋區域的多邊形
            left_line_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            left_line_pts = np.hstack((left_line_window, right_line_window))
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))

            # (來自 test.py 的快速 Numpy 陣列賦值繪製法，可繪出黃色的擬合曲線)
            # 注意：使用 np.clip 避免超出陣列邊界報錯
            valid_left = np.clip(left_fitx, 0, binary_warped.shape[1]-1).astype(np.int64)
            valid_right = np.clip(right_fitx, 0, binary_warped.shape[1]-1).astype(np.int64)
            out_img[ploty.astype(np.int64), valid_left] = [255, 255, 0]
            out_img[ploty.astype(np.int64), valid_right] = [255, 255, 0]

            # 合併多邊形區域與車道像素
            region_img = cv2.addWeighted(line_img, 1, window_img, 0.3, 0)

            return region_img, left_fit, right_fit, left_fitx, right_fitx, ploty


        # ---- 計算曲率與中心偏移
        def add_curvature_and_data(img, left_fitx, right_fitx, ploty):
            """計算並將曲率半徑、車輛偏離中心的距離寫入影像中"""
            ym_per_pix = 25/720 # y 維度的每像素公尺數
            xm_per_pix = 3.7/800 # x 維度的每像素公尺數

            y_eval = np.max(ploty)

            # 將像素空間轉換為真實世界(公尺)空間重新擬合
            left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
            right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

            # 計算曲率半徑
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

            # 計算車輛偏移量 (假設攝影機裝在車子正中央)
            road_mid = img.shape[1] / 2
            car_mid = (right_fitx[-1] + left_fitx[-1]) / 2 # 影像最底部的車道中心點
            mid_dev = car_mid - road_mid
            mid_dev_meter = abs(mid_dev) * xm_per_pix

            # 準備顯示文字
            curvature_word = f'Radius of Curvature = {left_curverad:.2f} (m)'
            direction = 'right' if mid_dev > 0 else 'left'
            mid_dev_word = f'Vehicle is {mid_dev_meter:.2f} m {direction} of center'

            # 在圖片上繪製文字
            cv2.putText(img, curvature_word, (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(img, mid_dev_word, (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

            return img


        # ---- 主程式管線與執行
        def process_image(img):
            """處理單一幀影像的完整流程"""
            # 1. 校正影像畸變 (使用類別變數快取的 mtx, dist)
            if VisionProcessor._mtx is not None and VisionProcessor._dist is not None:
                undist = cv2.undistort(img, VisionProcessor._mtx, VisionProcessor._dist, None, VisionProcessor._mtx)
            else:
                undist = img.copy() # 如果沒找到校正圖檔，就直接使用原圖

            # 2. 獲取各項閾值並合併
            grad_x = abs_sobel_thresh(undist, orient='x', sobel_kernel=15, thresh=(30, 100))
            grad_y = abs_sobel_thresh(undist, orient='y', sobel_kernel=15, thresh=(30, 100))
            mag_binary = mag_thresh(undist, sobel_kernel=15, thresh=(50, 100))
            dir_binary = dir_threshold(undist, sobel_kernel=15, thresh=(0.7, 1.3))
            hls_binary = hls_select(undist, thresh=(170, 255))
            combined_binary = combine_threshs(grad_x, grad_y, mag_binary, dir_binary, hls_binary)

            # 3. 轉換視角至鳥瞰圖
            warped = perspective(combined_binary)

            # 4. 偵測車道線並產生涵蓋區域圖
            region_img, left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial(warped)

            # 5. 反向透視變換，將車道區域轉回原始視角
            region_real_img = unperspective(region_img, undist.shape).astype(np.uint8)

            # 6. 與原始無畸變影像疊加
            result_img = cv2.addWeighted(undist, 1, region_real_img, 0.5, 0)

            # 7. 加入數據分析文字
            final_img = add_curvature_and_data(result_img, left_fitx, right_fitx, ploty)

            return final_img

        # [修復] 將實際處理過後的圖像回傳
        return process_image(image)

# 測試 -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # =========================================================================
    # 以下為你原本註解掉的物件偵測/攝影機測試邏輯 (保留原樣)
    # 若需使用，請將註解取消即可。
    # =========================================================================

    # 因為計算速度需要連續幀，這裡改用您原本註解掉的連續影像/攝影機測試邏輯
    # 1. 設定影像來源 (可輸入測試影片路徑，或輸入 0 使用本機攝影機)
    video_source = "C:/Users/USER/OneDrive/Desktop/test_video.mp4" # 請改為您的測試影片或 0
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"無法開啟影像來源: {video_source}，請確認路徑或攝影機連接。")
        exit()

    # 2. 初始化歷史字典
    history_dict = {}

    print("開始測試 YOLO 影像辨識與危險性評估系統 【行人模式】...")
    print("=" * 65)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("影片播放結束或無法讀取畫面。")
            break

        # 檢查影像通道數，如果是 4 (BGRA)，就轉成 3 (BGR)
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frame_count += 1
        image_height = frame.shape[0]
        image_width = frame.shape[1]

        # 3. 呼叫 VisionProcessor 進行 YOLO 辨識與追蹤
        annotated_image, detected_items = VisionProcessor.detect_objects(frame)

        # 4. 呼叫 RiskEvaluator 計算危險指數，啟用「行人模式」
        analyzed_items, alert_queue = RiskEvaluator.evaluate_frame_risk(
            detected_items,
            image_width,
            image_height,
            history_dict,
            mode="pedestrian"
        )

        # 5. 在終端機輸出要求資訊
        if analyzed_items:
            print(f"\n[Frame {frame_count:04d}] 偵測到 {len(analyzed_items)} 個物件:")
            for item in analyzed_items:
                track_id = item["track_id"]
                class_name = VisionProcessor.get_label_name(item["class_id"])
                risk_score = item["risk_score"]
                dist_pct = item["proximity_pct"]

                # 速度顯示格式化
                if item["speed"] is not None:
                    speed_str = f"{item['speed']:+5.1f} px/f"
                else:
                    speed_str = "  N/A (非車輛)"

                print(f"  > ID: {track_id:2d} | 物件: {class_name:10} | 畫面距離: {dist_pct:5.1f}% | 速度: {speed_str} | 危險指數: {risk_score:5.2f}")

        # 6. 輸出優先權佇列警報 (模擬交給前端發出警報)
        if alert_queue:
            print("-" * 25 + " 🚨 警報發送佇列 " + "-" * 25)
            # 複製一份 queue 來 pop 輸出，確保順序從最危險開始
            temp_queue = alert_queue.copy()
            while temp_queue:
                neg_risk, q_id, q_item = heapq.heappop(temp_queue)
                real_risk = -neg_risk
                q_label = VisionProcessor.get_label_name(q_item["class_id"])
                print(f"  [!] 發送警報 -> 物件: {q_label} (ID: {q_id}) | 危險度高達: {real_risk:.2f} !!!")
            print("-" * 65)

        # 7. 可視化展示
        cv2.imshow("Risk Evaluation Test - Pedestrian Mode", annotated_image)

        # 按下 'q' 鍵退出迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("使用者手動中斷測試。")
            break

    # 釋放資源
    cap.release()
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
    #     image_width = frame.shape[1]

    #     # 3. 呼叫 VisionProcessor 進行 YOLO 辨識與追蹤
    #     annotated_image, detected_items = VisionProcessor.detect_objects(frame)

    #     # 4. 呼叫 RiskEvaluator 計算危險指數，並取得排序後的列表
    #     analyzed_items = RiskEvaluator.evaluate_frame_risk(detected_items, image_width, image_height, history_dict)

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