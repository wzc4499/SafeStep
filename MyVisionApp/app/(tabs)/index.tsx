import { useRef, useState, useEffect } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Dimensions, ScrollView } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

const SERVER_URL = 'http://192.168.8.200:8000/detect';
const { width, height } = Dimensions.get('window');

type Box = {
  x1: number; y1: number;
  x2: number; y2: number;
  label: string;
  confidence: number;
  danger: string;
  distance: number;
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

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
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
        setBoxes(data.boxes || []);
        setImgSize({ w: data.img_width, h: data.img_height });

        const highDanger = data.boxes?.filter(b => b.danger === '高').length || 0;
        if (highDanger > 0) {
          setStatus(`⚠️ 危險！${highDanger} 個高危險物件`);
        } else {
          setStatus(`偵測到 ${data.boxes?.length || 0} 個物件`);
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
  };

  if (!permission) return <View />;

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>需要相機權限</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>授權相機</Text>
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

      {/* 物件清單 */}
      <ScrollView style={styles.listArea}>
        {boxes.length === 0 && isDetecting && (
          <Text style={styles.emptyText}>未偵測到物件</Text>
        )}
        {boxes.map((box, i) => {
          const color = DANGER_COLOR[box.danger] || '#00cc66';
          return (
            <View key={i} style={[styles.listItem, { borderLeftColor: color }]}>
              <View style={{ flex: 1 }}>
                <Text style={styles.listLabel}>{box.label}</Text>
                <Text style={styles.listSub}>
                  信心度 {Math.round(box.confidence * 100)}%　
                  距離約 {box.distance}m
                </Text>
              </View>
              <View style={[styles.dangerBadge, { backgroundColor: color }]}>
                <Text style={styles.dangerText}>危險值 {box.danger}</Text>
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
  text: { color: '#fff', fontSize: 18, marginBottom: 20, textAlign: 'center' },
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
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#222',
    borderRadius: 8,
    borderLeftWidth: 4,
    padding: 10,
    marginBottom: 6,
  },
  listLabel: { color: '#fff', fontSize: 15, fontWeight: '600' },
  listSub: { color: '#aaa', fontSize: 12, marginTop: 2 },
  dangerBadge: {
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  dangerText: { color: '#fff', fontSize: 12, fontWeight: '700' },
  controls: {
    padding: 16,
    alignItems: 'center',
    backgroundColor: '#111',
  },
  button: {
    backgroundColor: '#fff',
    paddingVertical: 14,
    paddingHorizontal: 40,
    borderRadius: 30,
  },
  buttonStop: { backgroundColor: '#ff4444' },
  buttonText: { fontSize: 16, fontWeight: '600', color: '#111' },
  box: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 4,
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