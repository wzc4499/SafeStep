import { useRef, useState, useEffect } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Dimensions } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

const SERVER_URL = 'http://192.168.8.200:8000/detect';
const { width, height } = Dimensions.get('window');

type Box = {
  x1: number; y1: number;
  x2: number; y2: number;
  label: string;
  confidence: number;
};

type DetectResponse = {
  boxes: Box[];
  img_width: number;
  img_height: number;
};

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [imgSize, setImgSize] = useState({ w: 1772, h: 1080 });
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
          quality: 0.4,
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
        setStatus(`偵測到 ${data.boxes?.length || 0} 個物件`);
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

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing="back" ref={cameraRef}>
        {boxes.map((box, i) => (
          <View
            key={i}
            style={[styles.box, {
              left: (box.x1 / imgSize.w) * width,
              top: (box.y1 / imgSize.h) * (height * 0.75),
              width: ((box.x2 - box.x1) / imgSize.w) * width,
              height: ((box.y2 - box.y1) / imgSize.h) * (height * 0.75),
            }]}
          >
            <Text style={styles.boxLabel}>
              {box.label} {Math.round(box.confidence * 100)}%
            </Text>
          </View>
        ))}
      </CameraView>

      <View style={styles.statusBar}>
        <Text style={styles.statusText}>{status}</Text>
      </View>

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
  container: { flex: 1, backgroundColor: '#000' },
  camera: { flex: 1 },
  text: { color: '#fff', fontSize: 18, marginBottom: 20, textAlign: 'center' },
  statusBar: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 10,
    alignItems: 'center',
  },
  statusText: { color: '#fff', fontSize: 14 },
  controls: {
    padding: 20,
    backgroundColor: '#000',
    alignItems: 'center',
  },
  button: {
    backgroundColor: '#fff',
    paddingVertical: 14,
    paddingHorizontal: 40,
    borderRadius: 30,
  },
  buttonStop: { backgroundColor: '#ff4444' },
  buttonText: { fontSize: 16, fontWeight: '600' },
  box: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00ff00',
    backgroundColor: 'rgba(0,255,0,0.1)',
  },
  boxLabel: {
    backgroundColor: '#00ff00',
    color: '#000',
    fontSize: 11,
    fontWeight: '600',
    paddingHorizontal: 4,
  },
});