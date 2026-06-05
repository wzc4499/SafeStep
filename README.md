# SafeStep — 交通標誌辨識 App

手機鏡頭即時辨識交通物件，顯示距離與危險值，協助行人安全導航。

---

## 環境需求

- Python 3.11 或 3.12
- Node.js 18+
- 手機安裝 **Expo Go**（免費，App Store / Google Play）
- 手機與電腦連接**同一個 WiFi**

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
npx expo install expo-camera expo-location expo-av
```

### 3. 安裝後端套件

```bash
cd ../backend
pip install ultralytics fastapi uvicorn python-multipart pillow numpy
```

### 4. 放入模型檔案

將 `best.pt` 放到：

```
backend/runs/detect/Taiwan_Traffic_Project/weights/best.pt
```

> ⚠️ 模型檔案不在 GitHub 上，請向組員索取。

### 5. 設定你的電腦 IP（每個人都要改！）

> ⚠️ 這步很重要，每台電腦 IP 都不一樣，直接用別人的 IP 會連不上。

**查詢你電腦的 IP（Windows）：**

```powershell
ipconfig
```

找 `IPv4 Address` 那行，例如 `192.168.1.100`。

**修改 `MyVisionApp/app/index.tsx` 第一行：**

```typescript
const SERVER_URL = 'http://你的IP:8000/detect';
```

例如：
```typescript
const SERVER_URL = 'http://192.168.1.100:8000/detect';
```

---

## 啟動方式

### 終端機一：後端

```bash
cd backend
uvicorn server:app --host 0.0.0.0 --port 8000
```

出現 `Uvicorn running on http://0.0.0.0:8000` 代表成功。

### 終端機二：前端

```bash
cd MyVisionApp
npx expo start
npx expo start --dev-client
```

用手機 Expo Go 掃 QR Code 開啟 App。

---

## 使用方式

1. 開啟 App → 授權相機
2. 點「開始辨識」
3. 鏡頭對準道路場景
4. 畫面自動標出物件、距離、危險值
5. 危險值 ≥ 70% 自動發出警報聲

---

## 常見問題

**Q：clone 下來有紅色波浪線**
```bash
cd MyVisionApp
npm install
npx expo install expo-camera expo-location expo-av
```

**Q：手機顯示「連線錯誤」**
- 確認手機和電腦在**同一個 WiFi**
- 確認 `index.tsx` 裡的 IP 是**你自己電腦**的 IP，不是別人的
- 確認後端 server 正在執行中

**Q：模型載入失敗**
- 確認 `best.pt` 放在正確路徑
- 重新安裝：`pip install ultralytics`
