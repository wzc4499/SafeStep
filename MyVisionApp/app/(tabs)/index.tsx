import { useRef, useState, useEffect, useCallback } from 'react';
import {
  StyleSheet, View, Text, TouchableOpacity,
  Dimensions, ScrollView
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Audio } from 'expo-av';

const SERVER_URL = 'http://192.168.8.200:8000/detect';
const { width, height } = Dimensions.get('window');
const CAMERA_HEIGHT = height * 0.65;
const ALARM_THRESHOLD = 70; // 危險值 % 超過這個就發警報

type Box = {
  x1: number; y1: number;
  x2: number; y2: number;
  label: string;
  confidence: number;
  danger: string;
  danger_pct: number;
  distance: number;
  area_ratio: number;
};

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
  const [permission, requestPermission] = useCameraPermissions();
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [imgSize, setImgSize] = useState({ w: 1080, h: 1772 });
  const [isDetecting, setIsDetecting] = useState(false);
  const [status, setStatus] = useState('準備中...');
  const cameraRef = useRef<CameraView>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const prevRatios = useRef<Record<string, number>>({});
  const soundRef = useRef<Audio.Sound | null>(null);
  const isAlarmPlaying = useRef(false);

  // 初始化音效
  useEffect(() => {
    Audio.setAudioModeAsync({ playsInSilentModeIOS: true });
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      soundRef.current?.unloadAsync();
    };
  }, []);

  // 播放警報嗶聲
  const playAlarm = useCallback(async () => {
    if (isAlarmPlaying.current) return;
    isAlarmPlaying.current = true;
    try {
      const { sound } = await Audio.Sound.createAsync(
        { uri: 'https://www.soundjay.com/buttons/beep-07a.mp3' },
        { shouldPlay: true }
      );
      soundRef.current = sound;
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

  const startDetection = () => {
    setIsDetecting(true);
    setStatus('辨識中...');
    intervalRef.current = setInterval(async () => {
      try {
        if (!cameraRef.current) return;
        const photo = await cameraRef.current.takePictureAsync({
          base64: true,
          quality: 0.6,
          skipProcessing: true,
        });
        if (!photo?.base64) return;

        const res = await fetch(SERVER_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: photo.base64 }),
        });
        const data: DetectResponse = await res.json();
        const newBoxes = data.boxes || [];

        // 計算逼近速度（area_ratio 變化）
        const boxesWithSpeed = newBoxes.map((box) => {
          const key = box.label;
          const prev = prevRatios.current[key] ?? box.area_ratio;
          const speed = box.area_ratio - prev;
          prevRatios.current[key] = box.area_ratio;
          return { ...box, approach_speed: speed };
        });

        // 排序：危險值高 → 逼近速度快
        boxesWithSpeed.sort((a, b) => {
          const dangerDiff = b.danger_pct - a.danger_pct;
          if (Math.abs(dangerDiff) > 5) return dangerDiff;
          return (b.approach_speed ?? 0) - (a.approach_speed ?? 0);
        });

        setBoxes(boxesWithSpeed as Box[]);
        setImgSize({ w: data.img_width, h: data.img_height });

        // 檢查是否需要警報
        const needAlarm = boxesWithSpeed.some(b => b.danger_pct >= ALARM_THRESHOLD);
        if (needAlarm) {
          playAlarm();
          setStatus(`⚠️ 危險警報！請注意安全`);
        } else {
          setStatus(`偵測到 ${newBoxes.length} 個物件`);
        }
      } catch (e) {
        setStatus('連線錯誤，確認 WiFi');
      }
    }, 1500);
  };

  const stopDetection = () => {
    setIsDetecting(false);
    setStatus('已停止');
    if (intervalRef.current) clearInterval(intervalRef.current);
    setBoxes([]);
    prevRatios.current = {};
  };

  if (!permission) return <View />;

  if (!permission.granted) {
    return (
      <View style={styles.permContainer}>
        <Text style={styles.permText}>需要相機權限</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>授權相機</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* 相機區域 */}
      <View style={{ height: CAMERA_HEIGHT }}>
        <CameraView style={{ flex: 1 }} facing="back" ref={cameraRef}>
          {(boxes as any[]).map((box, i) => {
            const color = DANGER_COLOR[box.danger] || '#00cc66';
            const boxW = ((box.x2 - box.x1) / imgSize.w) * width;
            const boxH = ((box.y2 - box.y1) / imgSize.h) * CAMERA_HEIGHT;
            const boxL = (box.x1 / imgSize.w) * width;
            const boxT = (box.y1 / imgSize.h) * CAMERA_HEIGHT;

            return (
              <View
                key={i}
                style={[styles.box, {
                  left: boxL,
                  top: boxT,
                  width: boxW,
                  height: boxH,
                  borderColor: color,
                  backgroundColor: `${color}22`,
                }]}
              >
                {/* 框框內資訊 */}
                <View style={[styles.boxInfo, { backgroundColor: color }]}>
                  <Text style={styles.boxLabel}>{box.label}</Text>
                  <Text style={styles.boxDetail}>
                    {box.distance}m｜危{box.danger_pct}%
                  </Text>
                </View>
              </View>
            );
          })}
        </CameraView>
      </View>

      {/* 狀態列 */}
      <View style={[
        styles.statusBar,
        (boxes as any[]).some(b => b.danger_pct >= ALARM_THRESHOLD) && styles.statusDanger
      ]}>
        <Text style={styles.statusText}>{status}</Text>
      </View>

      {/* 物件清單 */}
      <ScrollView style={styles.listArea}>
        {boxes.length === 0 && isDetecting && (
          <Text style={styles.emptyText}>未偵測到物件</Text>
        )}
        {(boxes as any[]).map((box, i) => {
          const color = DANGER_COLOR[box.danger] || '#00cc66';
          const speedText = box.approach_speed > 0.005
            ? '🔴 快速逼近'
            : box.approach_speed > 0
              ? '🟡 緩慢靠近'
              : '🟢 穩定';

          return (
            <View key={i} style={[styles.listItem, { borderLeftColor: color }]}>
              <View style={{ flex: 1 }}>
                <Text style={styles.listLabel}>{box.label}</Text>
                <Text style={styles.listSub}>
                  信心度 {Math.round(box.confidence * 100)}%　距離約 {box.distance}m
                </Text>
                <Text style={styles.listSub}>{speedText}</Text>
              </View>
              <View style={[styles.dangerBadge, { backgroundColor: color }]}>
                <Text style={styles.dangerText}>危險 {box.danger_pct}%</Text>
              </View>
            </View>
          );
        })}
      </ScrollView>

      {/* 按鈕 */}
      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.button, isDetecting && styles.buttonStop]}
          onPress={isDetecting ? stopDetection : startDetection}
        >
          <Text style={styles.buttonText}>
            {isDetecting ? '停止辨識' : '開始辨識'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#111' },
  permContainer: {
    flex: 1, backgroundColor: '#111',
    justifyContent: 'center', alignItems: 'center'
  },
  permText: { color: '#fff', fontSize: 18, marginBottom: 20 },
  statusBar: { backgroundColor: '#222', padding: 8, alignItems: 'center' },
  statusDanger: { backgroundColor: '#8b0000' },
  statusText: { color: '#fff', fontSize: 13 },
  listArea: { flex: 1, paddingHorizontal: 12, paddingTop: 6 },
  emptyText: { color: '#666', textAlign: 'center', marginTop: 12, fontSize: 13 },
  listItem: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: '#222', borderRadius: 8,
    borderLeftWidth: 4, padding: 10, marginBottom: 6,
  },
  listLabel: { color: '#fff', fontSize: 15, fontWeight: '600' },
  listSub: { color: '#aaa', fontSize: 12, marginTop: 2 },
  dangerBadge: { borderRadius: 6, paddingHorizontal: 8, paddingVertical: 4 },
  dangerText: { color: '#fff', fontSize: 12, fontWeight: '700' },
  controls: { padding: 16, alignItems: 'center', backgroundColor: '#111' },
  button: {
    backgroundColor: '#fff', paddingVertical: 14,
    paddingHorizontal: 40, borderRadius: 30,
  },
  buttonStop: { backgroundColor: '#ff4444' },
  buttonText: { fontSize: 16, fontWeight: '600', color: '#111' },
  box: { position: 'absolute', borderWidth: 2, borderRadius: 4 },
  boxInfo: { borderRadius: 3, paddingHorizontal: 4, paddingVertical: 2 },
  boxLabel: { color: '#000', fontSize: 11, fontWeight: '700' },
  boxDetail: { color: '#000', fontSize: 10 },
});