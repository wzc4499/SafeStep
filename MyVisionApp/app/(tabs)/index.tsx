import { useRef, useState, useEffect } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Dimensions, ScrollView } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

// 設定後端伺服器的基礎網址。未來切換模式時，會在後方動態加上 /pedestrian 或 /motorcycle
const BASE_URL = 'http://192.168.***.***:8000';

// 取得當前設備螢幕的寬高，用於後續計算辨識框的繪製比例
const { width, height } = Dimensions.get('window');

// 定義 YOLO 辨識回傳的單一物件資料結構
type Box = {
  x1: number; y1: number;
  x2: number; y2: number;
  label: string;
  confidence: number;
  danger: string;
  distance: number;
};

// 定義後端 API 回傳的整體 JSON 資料結構
type DetectResponse = {
  boxes: Box[];
  img_width: number;
  img_height: number;
};

const DANGER_COLOR: Record<string, string> = {
  高: '#ff3333',
  中: '#ffaa00',
  低: '#00cc66',
};

export default function App() {
  // --- 狀態管理 (State) ---
  // 相機權限狀態
  const [permission, requestPermission] = useCameraPermissions();
  // 儲存當前畫面偵測到的所有物件框
  const [boxes, setBoxes] = useState<Box[]>([]);

  // 儲存後端回傳的原始影像尺寸，用於將辨識框精準映射到手機螢幕上
  const [imgSize, setImgSize] = useState({ w: 1772, h: 1080 });

  // 控制是否正在執行辨識的開關
  const [isDetecting, setIsDetecting] = useState(false);

  // 顯示當前系統狀態的提示文字
  const [status, setStatus] = useState('準備中...');


  // 核心狀態：紀錄目前選擇的模式。若為 null 代表尚未選擇（停留在選擇畫面）
  const [activeMode, setActiveMode] = useState<string | null>(null);

  // --- 參考物件 (Refs) ---
  // 綁定相機元件，用來呼叫拍照功能 (takePictureAsync)
  const cameraRef = useRef<CameraView>(null);

  // 儲存 setInterval 的 ID，方便後續隨時清除計時器。
  // 使用 useRef 而非 useState 是因為改變 Ref 不會觸發畫面重新渲染。
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // 用來控制 WebSocket 連線
  const wsRef = useRef<WebSocket | null>(null);

  // --- 生命週期 (Lifecycle) ---
  // 當元件卸載 (App 關閉) 時，確保清除計時器，避免記憶體流失或背景持續發送請求
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

// --- 核心功能：開始辨識 (改用 WebSocket) ---
  const startDetection = () => {
    if (!activeMode) return;

    setIsDetecting(true);
    setStatus(`連線中 (${activeMode === 'pedestrian' ? '行人' : '機車'}模式)...`);

    // 建立 WebSocket 連線 (注意：URL 必須是 ws:// 開頭，並對應 server.py 的路徑)
    // 請將 BASE_URL 的 http:// 替換為 ws://
    const wsUrl = `ws://192.168.***.***:8000/ws/detect/${activeMode}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    // 當連線成功開啟時
    ws.onopen = () => {
      setStatus(`辨識中...`);

      // 啟動定時器，持續發送畫面
      intervalRef.current = setInterval(async () => {
        // 若相機未準備好，或 WebSocket 斷線則不發送
        if (!cameraRef.current || ws.readyState !== WebSocket.OPEN) return;

        try {
          const photo = await cameraRef.current.takePictureAsync({
            base64: true,
            quality: 0.4,
            skipProcessing: true,
          });

          if (photo?.base64) {
             // 透過 WebSocket 傳送 Base64 字串
             ws.send(photo.base64);
          }
        } catch (e) {
          console.log("擷取畫面失敗:", e);
        }
      }, 1500); // <-- 注意：見下方效能優化建議
    };

    // 接收後端處理完的辨識與警報資料
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setBoxes(data.boxes || []);
      setImgSize({ w: data.img_width, h: data.img_height });
      setStatus(`偵測到 ${data.boxes?.length || 0} 個物件`);

      // 【警報觸發邏輯】若後端有傳來警報佇列，可在此觸發手機震動或音效
      if (data.alerts && data.alerts.length > 0) {
        console.warn(`🚨 警報: ${data.alerts[0].warning_msg}`);
        // 這裡可以加入震動 API: Vibration.vibrate(500);
      }
    };

    // 3. 發生錯誤時的處理
    ws.onerror = (e) => {
      console.log("WebSocket 錯誤: ", e);
      // 直接把錯誤訊息傳給 stopDetection
      stopDetection('連線失敗，請檢查 IP 與 Wi-Fi');
    };

    // 4. 連線關閉時的處理
    ws.onclose = (e) => {
      // 確保不是因為手動點擊停止而關閉時，才顯示斷線
      if (isDetecting) {
        stopDetection('與伺服器斷線');
      }
    };
  };

  // --- 核心功能：停止辨識 ---
const stopDetection = (customMessage = '已停止') => {
    setIsDetecting(false);
    setStatus(customMessage); // 顯示傳入的訊息，不再強制覆蓋成已停止

    if (intervalRef.current) clearInterval(intervalRef.current);

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setBoxes([]);
  };

  // --- 核心功能：退出當前模式 ---
  const exitMode = () => {
    stopDetection(); // 確保退出前先停止背景辨識程序
    setActiveMode(null); // 將狀態設為 null，畫面會自動透過條件渲染切換回「模式選擇」介面
    setStatus('準備中...');
  };

  // --- 畫面渲染 0：處理相機權限 ---
  if (!permission) return <View />; // 權限狀態還在載入中，顯示空白

  if (!permission.granted) {
    // 若尚未授權，顯示要求授權的畫面
    return (
      <View style={styles.container}>
        <Text style={styles.text}>需要相機權限</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonDarkText}>授權相機</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const cameraHeight = height * 0.65;

  return (
    <View style={styles.container}>
      {/* 相機區域 */}
      <View style={{ height: cameraHeight }}>
        <CameraView style={{ flex: 1 }} facing="back" ref={cameraRef}>
          {boxes.map((box, i) => {
            const color = DANGER_COLOR[box.danger] || '#00cc66';
            return (
              <View
                key={i}
                style={[styles.box, {
                  left: (box.x1 / imgSize.w) * width,
                  top: (box.y1 / imgSize.h) * cameraHeight,
                  width: ((box.x2 - box.x1) / imgSize.w) * width,
                  height: ((box.y2 - box.y1) / imgSize.h) * cameraHeight,
                  borderColor: color,
                  backgroundColor: `${color}22`,
                }]}
              >
                <Text style={[styles.boxLabel, { backgroundColor: color }]}>
                  {box.label}
                </Text>
              </View>
            );
          })}
        </CameraView>
      </View>

      {/* 狀態列 */}
      <View style={styles.statusBar}>
        <Text style={styles.statusText}>{status}</Text>
      </View>

      {/* 底部控制按鈕區 */}
      {/* 原本 : <View style={styles.controls}> */}
      <View style={(styles as any).controls}>
        {/* 開始/停止 辨識按鈕 */}
        <TouchableOpacity
          style={[styles.button, isDetecting && styles.buttonStop]} // 若正在辨識，套用紅色樣式
          onPress={() => isDetecting ? stopDetection() : startDetection()}
        >
          <Text style={styles.buttonDarkText}>
            {isDetecting ? '停止辨識' : '開始辨識'}
          </Text>
        </TouchableOpacity>

        {/* 退出模式按鈕 */}
        <TouchableOpacity style={styles.exitButton} onPress={exitMode}>
          <Text style={styles.exitButtonText}>退出模式</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// --- 樣式表 (CSS in JS) ---
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#111' },
  text: { color: '#fff', fontSize: 18, marginBottom: 20, textAlign: 'center' },

  // 選擇畫面專用樣式
  selectionContainer: {
    flex: 1,
    backgroundColor: '#111',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  titleText: { color: '#fff', fontSize: 24, fontWeight: 'bold', marginBottom: 40 },
  selectionBtn: {
    backgroundColor: '#333',
    paddingVertical: 20,
    paddingHorizontal: 40,
    borderRadius: 15,
    marginBottom: 20,
    width: '80%',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#555',
  },
  selectionBtnText: { color: '#00ff00', fontSize: 20, fontWeight: 'bold' },

  // 相機畫面專用樣式
  topBar: {
    backgroundColor: '#111',
    paddingTop: 50, // 避開手機頂部狀態列/瀏海
    paddingBottom: 15,
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  topBarText: { color: '#00ff00', fontSize: 16, fontWeight: '600' },
  statusBar: {
    backgroundColor: '#222',
    padding: 8,
    alignItems: 'center',
  },
  statusText: { color: '#fff', fontSize: 13 },
  listArea: {
    flex: 1,
    paddingHorizontal: 12,
    paddingTop: 6,
  },
  emptyText: {
    color: '#666',
    textAlign: 'center',
    marginTop: 12,
    fontSize: 13,
  },
  listItem: {
    alignItems: 'center',
    flexDirection: 'row', // 將按鈕橫向並排
    justifyContent: 'space-evenly', // 平均分配按鈕間距
  },

  // 按鈕樣式
  button: {
    backgroundColor: '#fff',
    paddingVertical: 14,
    paddingHorizontal: 30,
    borderRadius: 30,
  },
  buttonStop: { backgroundColor: '#ff4444' }, // 停止辨識時的紅色按鈕
  buttonDarkText: { fontSize: 16, fontWeight: '700', color: '#000' },
  exitButton: {
    backgroundColor: '#333',
    paddingVertical: 14,
    paddingHorizontal: 30,
    borderRadius: 30,
    borderWidth: 1,
    borderColor: '#666',
  },
  exitButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },

  // 辨識框樣式
  box: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00ff00', // 螢光綠外框
    backgroundColor: 'rgba(0,255,0,0.1)', // 半透明綠色填充
  },
  boxLabel: {
    color: '#000',
    fontSize: 11,
    fontWeight: '700',
    paddingHorizontal: 4,
    paddingVertical: 1,
    borderRadius: 3,
  },
});