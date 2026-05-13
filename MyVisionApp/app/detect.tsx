import { useRef, useState, useEffect, useCallback } from 'react';
import {
  StyleSheet, View, Text, TouchableOpacity, ScrollView
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Audio } from 'expo-av';
import * as Location from 'expo-location';
import { useLocalSearchParams, useRouter, Stack } from 'expo-router';

const WS_URL = 'ws://192.168.8.200:8000/ws'; 
const ALARM_THRESHOLD = 70;

type Box = {
  track_id: number;
  x1: number; y1: number;
  x2: number; y2: number;
  label: string;
  confidence: number;
  danger: string;
  danger_pct: number;
  distance: number;
  area_ratio: number;
  speed: number;
};

const DANGER_COLOR: Record<string, string> = {
  高: '#ff3333',
  中: '#ffaa00',
  低: '#00cc66',
};

export default function Detect() {
  const { mode } = useLocalSearchParams<{ mode: string }>();
  const router = useRouter();
  const isPedestrian = mode === 'pedestrian';

  const [permission, requestPermission] = useCameraPermissions();
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [imgSize, setImgSize] = useState({ w: 1080, h: 1920 });
  const [isDetecting, setIsDetecting] = useState(false);
  const [status, setStatus] = useState('準備中...');
  
  const cameraRef = useRef<CameraView>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isAlarmPlaying = useRef(false);
  const isCapturing = useRef(false);

  useEffect(() => {
    Audio.setAudioModeAsync({ playsInSilentModeIOS: true });
    
    const connectWS = () => {
      const ws = new WebSocket(WS_URL);
      ws.onopen = () => setStatus('伺服器連線成功');
      ws.onmessage = (e) => handleServerMessage(JSON.parse(e.data));
      ws.onerror = () => setStatus('連線錯誤，請檢查網路');
      ws.onclose = () => setStatus('連線已中斷');
      wsRef.current = ws;
    };

    connectWS();

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const handleServerMessage = (data: any) => {
    if (data.status === 'error') {
      setStatus(`錯誤: ${data.message}`);
      return;
    }

    // 只要後端有回傳 boxes，不論什麼模式都更新畫面
    if (data.boxes) {
      setBoxes(data.boxes);
      setImgSize({ w: data.img_width, h: data.img_height });
    }

    if (data.mode === 'pedestrian') {
      if (data.boxes?.some((b: Box) => b.danger_pct >= ALARM_THRESHOLD)) {
        playAlarm();
        setStatus('⚠️ 危險警報！請注意安全');
      } else {
        setStatus(`偵測中 - 發現 ${data.boxes?.length || 0} 個物件`);
      }
    } else if (data.mode === 'motorcycle') {
      if (data.two_stage_warning) {
        playAlarm();
        setStatus('🚨 注意：前方路口需兩段式左轉');
      } else {
        setStatus(`導航中：${data.message}`);
      }
    }
  };

  const playAlarm = useCallback(async () => {
    if (isAlarmPlaying.current) return;
    isAlarmPlaying.current = true;
    try {
      const { sound } = await Audio.Sound.createAsync(
        { uri: 'https://www.soundjay.com/buttons/beep-07a.mp3' },
        { shouldPlay: true }
      );
      sound.setOnPlaybackStatusUpdate((s) => {
        if (s.isLoaded && s.didJustFinish) {
          isAlarmPlaying.current = false;
          sound.unloadAsync();
        }
      });
    } catch {
      isAlarmPlaying.current = false;
    }
  }, []);

  const startDetection = async () => {
    if (!isPedestrian) {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setStatus('機車模式需要定位權限');
        return;
      }
    }

    setIsDetecting(true);
    setStatus('辨識啟動中...');

    intervalRef.current = setInterval(async () => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      try {
        if (!cameraRef.current || isCapturing.current) return;
        
        isCapturing.current = true;
        const photo = await cameraRef.current.takePictureAsync({
          base64: true,
          quality: 0.3, 
          skipProcessing: true,
        });
        const pureBase64 = photo?.base64 ? photo.base64.replace(/^data:image\/\w+;base64,/, "") : "";

        if (isPedestrian) {
          wsRef.current.send(JSON.stringify({
            mode: 'pedestrian',
            image: pureBase64
          }));
        } else {
          const loc = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
          wsRef.current.send(JSON.stringify({
            mode: 'motorcycle',
            image: pureBase64, // 把機車模式的影像也送出，讓後端未來可做擴充
            lat: loc.coords.latitude,
            lng: loc.coords.longitude,
            heading: loc.coords.heading || 0,
            action: 'turn_left'
          }));
        }
        isCapturing.current = false;
      } catch (err) {
        console.error('傳輸失敗', err);
        isCapturing.current = false;
      }
    }, isPedestrian ? 500 : 1500); 
  };

  const stopDetection = () => {
    setIsDetecting(false);
    setStatus('已停止辨識');
    if (intervalRef.current) clearInterval(intervalRef.current);
    setBoxes([]);
  };

  if (!permission) return <View />;
  if (!permission.granted) {
    return (
      <View style={styles.permContainer}>
        <Text style={styles.permText}>需要相機權限才能運作</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>開啟權限</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* 隱藏 Expo Router 預設的黑色標題列，解決雙層標題太擠的問題 */}
      <Stack.Screen options={{ headerShown: false }} />

      {/* 自訂的頂部列 */}
      <View style={[styles.topBar, { backgroundColor: isPedestrian ? '#1a73e8' : '#e84c1a' }]}>
        <TouchableOpacity onPress={() => { stopDetection(); router.back(); }}>
          <Text style={styles.backBtn}>← 返回</Text>
        </TouchableOpacity>
        <Text style={styles.modeLabel}>
          {isPedestrian ? '🚶 行人辨識模式' : '🏍️ 機車導航助手'}
        </Text>
        <View style={{ width: 48 }} />
      </View>

      {/* 相機區域：改用 flex 讓它自動延展，不再寫死高度 */}
      <View style={styles.cameraWrapper}>
        <CameraView style={{ flex: 1 }} facing="back" ref={cameraRef}>
          {boxes.map((box, i) => {
            const color = DANGER_COLOR[box.danger] || '#00cc66';
            return (
              <View
                key={`${box.track_id}-${i}`}
                style={[styles.box, {
                  // 改用百分比計算，不受螢幕長寬比影響
                  left: `${(box.x1 / imgSize.w) * 100}%`,
                  top: `${(box.y1 / imgSize.h) * 100}%`,
                  width: `${((box.x2 - box.x1) / imgSize.w) * 100}%`,
                  height: `${((box.y2 - box.y1) / imgSize.h) * 100}%`,
                  borderColor: color,
                }]}
              >
                <View style={[styles.boxInfo, { backgroundColor: color }]}>
                  <Text style={styles.boxLabel}>{box.label}</Text>
                  {/* 把危險度資訊加回來了 */}
                  <Text style={styles.boxDetail}>{box.distance}m｜危{box.danger_pct}%</Text>
                </View>
              </View>
            );
          })}
        </CameraView>
      </View>

      <View style={[styles.statusBar, (status.includes('⚠️') || status.includes('🚨')) && styles.statusDanger]}>
        <Text style={styles.statusText}>{status}</Text>
      </View>

      {/* 列表區域 */}
      <ScrollView style={styles.listArea}>
        {boxes.map((box, i) => {
          const color = DANGER_COLOR[box.danger] || '#00cc66';
          return (
            <View key={`${box.track_id}-${i}`} style={[styles.listItem, { borderLeftColor: color }]}>
              <View style={{ flex: 1 }}>
                <Text style={styles.listLabel}>{box.label} (ID: {box.track_id})</Text>
                <Text style={styles.listSub}>
                  距離: {box.distance}m | 速度: {box.speed?.toFixed(1) || 0} px/f
                </Text>
              </View>
              <View style={[styles.dangerBadge, { backgroundColor: color }]}>
                <Text style={styles.dangerText}>危險 {box.danger_pct}%</Text>
              </View>
            </View>
          );
        })}
      </ScrollView>

      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.button, isDetecting && styles.buttonStop]}
          onPress={isDetecting ? stopDetection : startDetection}
        >
          <Text style={styles.buttonText}>{isDetecting ? '停止' : '開始'}</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#111' },
  permContainer: { flex: 1, backgroundColor: '#111', justifyContent: 'center', alignItems: 'center' },
  permText: { color: '#fff', fontSize: 18, marginBottom: 20 },
  topBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingTop: 50, paddingBottom: 15 },
  backBtn: { color: '#fff', fontSize: 15 },
  modeLabel: { color: '#fff', fontSize: 16, fontWeight: '700' },
  cameraWrapper: { flex: 4, width: '100%', overflow: 'hidden' },
  statusBar: { backgroundColor: '#222', padding: 10, alignItems: 'center' },
  statusDanger: { backgroundColor: '#8b0000' },
  statusText: { color: '#fff', fontSize: 14, fontWeight: '600' },
  listArea: { flex: 2, padding: 12 },
  listItem: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#222', borderRadius: 8, borderLeftWidth: 4, padding: 12, marginBottom: 8 },
  listLabel: { color: '#fff', fontSize: 16, fontWeight: '600' },
  listSub: { color: '#aaa', fontSize: 13, marginTop: 4 },
  dangerBadge: { borderRadius: 6, paddingHorizontal: 8, paddingVertical: 4 },
  dangerText: { color: '#fff', fontSize: 12, fontWeight: '700' },
  controls: { padding: 15, alignItems: 'center', backgroundColor: '#111' },
  button: { backgroundColor: '#fff', paddingVertical: 15, paddingHorizontal: 60, borderRadius: 30 },
  buttonStop: { backgroundColor: '#ff4444' },
  buttonText: { fontSize: 18, fontWeight: '700', color: '#111' },
  box: { position: 'absolute', borderWidth: 2, borderRadius: 4 },
  boxInfo: { position: 'absolute', top: -22, left: -2, paddingHorizontal: 4, borderRadius: 2 },
  boxLabel: { color: '#fff', fontSize: 11, fontWeight: '700' },
  boxDetail: { color: '#fff', fontSize: 9 },
});