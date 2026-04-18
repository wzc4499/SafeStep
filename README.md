# SafeStep — 交通標誌辨識 App

即時辨識台灣交通標誌的手機應用程式。手機透過鏡頭拍攝畫面，送至電腦後端進行 YOLO 辨識，再將結果即時顯示在手機螢幕上。

專為輔助視障者或行人安全設計，以手機取代硬體眼鏡，輕便易用。

---

## 專案架構

```
SafeStep/
├── MyVisionApp/        # 前端 (React Native + Expo)
└── backend/            # 後端 (Python + FastAPI + YOLO)
    ├── server.py
    └── runs/
        └── detect/
            └── Taiwan_Traffic_Project/
                └── weights/
                    └── best.pt   ← 訓練好的模型（需另外索取）
```

---

## 環境需求

### 電腦端（後端）
- Python 3.11 或 3.12
- 以下 Python 套件（見安裝步驟）

### 手機端（前端）
- iOS 或 Android 手機皆可
- 安裝 **Expo Go**（App Store / Google Play 免費下載）

### 網路
- 手機與電腦必須連接**同一個 WiFi**

---

## 安裝步驟

### 1. Clone 專案

```bash
git clone https://github.com/wzc4499/SafeStep.git
cd SafeStep
```

### 2. 安裝前端套件

```bash
cd MyVisionApp
npm install
npx expo install expo-camera expo-location
```

### 3. 安裝後端套件

```bash
cd ../backend
pip install ultralytics fastapi uvicorn python-multipart pillow numpy
```

### 4. 放入模型檔案

將訓練好的 `best.pt` 放到以下路徑：

```
backend/runs/detect/Taiwan_Traffic_Project/weights/best.pt
```

> ⚠️ 模型檔案較大，不放在 GitHub 上，請向組員索取 `best.pt`。

### 5. 設定你的電腦 IP

在 `MyVisionApp/app/index.tsx` 找到這一行：

```typescript
const SERVER_URL = 'http://192.168.8.200:8000/detect';
```

將 `192.168.8.200` 改成**你自己電腦**的區網 IP。

查詢方式（Windows）：
```powershell
ipconfig
```
找 `IPv4 Address` 那一行的數字。

---

## 啟動方式

### 第一步：啟動後端（電腦）

```bash
cd backend
uvicorn server:app --host 0.0.0.0 --port 8000
```

看到以下訊息代表成功：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 第二步：啟動前端（電腦）

開**新的終端機視窗**：

```bash
cd MyVisionApp
npx expo start
```

### 第三步：手機連線

1. 手機打開 **Expo Go**
2. 掃描終端機出現的 **QR Code**
3. App 自動開啟

---

## 使用方式

1. 開啟 App 後點選「**授權相機**」
2. 點選「**開始辨識**」
3. 將鏡頭對準交通標誌
4. 畫面會自動標出綠色框框與標誌名稱及信心度
5. 點選「**停止辨識**」結束

---

## 模型資訊

| 項目 | 內容 |
|------|------|
| 模型架構 | YOLOv8 |
| 訓練資料 | 台灣交通標誌圖案 |
| 模型檔案 | `best.pt`（需另外索取） |
| 辨識門檻 | 信心度 50% 以上才顯示 |

---

## 常見問題

**Q：clone 下來有紅色波浪線或錯誤**
```bash
cd MyVisionApp
npm install
npx expo install expo-camera expo-location
```

**Q：手機顯示「連線錯誤，確認 WiFi」**
- 確認手機和電腦在同一個 WiFi
- 確認後端 server 有在跑
- 確認 `index.tsx` 裡的 IP 是你電腦的 IP

**Q：辨識框位置跑掉**
- 確認後端有正確回傳 `img_width` 和 `img_height`
- 重啟後端 server 再試一次

**Q：模型載入失敗**
- 確認 `best.pt` 放在正確路徑
- 確認 `ultralytics` 有正確安裝

---

## 技術堆疊

| 層級 | 技術 |
|------|------|
| 前端 | React Native、Expo、TypeScript |
| 後端 | Python、FastAPI、Uvicorn |
| 辨識 | YOLOv8（Ultralytics） |
| 通訊 | HTTP REST API、Base64 圖片傳輸 |
