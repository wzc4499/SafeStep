import { useRef, useState, useEffect, useCallback } from 'react';
import {
  StyleSheet, View, Text, TouchableOpacity, ScrollView, Vibration
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Audio } from 'expo-av';
import * as Location from 'expo-location';
import { useLocalSearchParams, useRouter, Stack } from 'expo-router';
import Svg, { Polygon } from 'react-native-svg';

const WS_URL = 'ws://192.168.222.3:8000/ws';
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
  light_status: number;
  specific_sign_name: string;
};

type LaneData = {
  zebra: number[][][];
  sidewalk: number[][][];
  img_w: number;
  img_h: number;
};

const DANGER_COLOR: Record<string, string> = {
  高: '#ff3333',
  中: '#ffaa00',
  低: '#00cc66',
};

const LIGHT_COLOR: Record<number, string> = {
  0: '#ff3333',
  1: '#00cc66',
  2: '#ffcc00',
};

const LIGHT_TEXT: Record<number, string> = {
  0: '🔴 停止！',
  1: '🟢 可通行',
  2: '🟡 注意！快速通過',
};

export default function Detect() {
  const { mode } = useLocalSearchParams<{ mode: string }>();
  const router = useRouter();
  const isPedestrian = mode === 'pedestrian';

  const [permission, requestPermission] = useCameraPermissions();
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [imgSize, setImgSize] = useState({ w: 1080, h: 1920 });
  const [lanes, setLanes] = useState<LaneData>({ zebra: [], sidewalk: [], img_w: 1080, img_h: 1920 });
  const [isDetecting, setIsDetecting] = useState(false);
  const [status, setStatus] = useState('準備中...');
  const [viewSize, setViewSize] = useState({ w: 0, h: 0 });

  const cameraRef = useRef<CameraView>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isAlarmPlaying = useRef(false);
  const isCapturing = useRef(false);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(WS_URL);
    ws.onopen = () => setStatus('伺服器連線成功');
    ws.onmessage = (e) => handleServerMessage(JSON.parse(e.data));
    ws.onerror = () => setStatus('連線錯誤，請檢查網路');
    ws.onclose = () => {
      setStatus('連線已中斷，重新連線中...');
      reconnectTimeout.current = setTimeout(connectWS, 3000);
    };
    wsRef.current = ws;
  }, []);

  useEffect(() => {
    Audio.setAudioModeAsync({ playsInSilentModeIOS: true, staysActiveInBackground: true });
    connectWS();
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const handleServerMessage = (data: any) => {
    if (data.status === 'error') { setStatus(`錯誤: ${data.message}`); return; }

    if (data.boxes) {
      setBoxes(data.boxes);
      setImgSize({ w: data.img_width, h: data.img_height });
    }
    if (data.lanes) {
      setLanes(data.lanes);
    }

    if (data.mode === 'pedestrian') {
      const lights = data.boxes?.filter((b: Box) => b.light_status >= 0) || [];
      const redLight = lights.some((b: Box) => b.light_status === 0);
      const yellowLight = lights.some((b: Box) => b.light_status === 2);
      const greenLight = lights.some((b: Box) => b.light_status === 1);

      if (data.boxes?.some((b: Box) => b.danger_pct >= ALARM_THRESHOLD)) {
        playAlarm();
        setStatus('⚠️ 危險警報！請注意安全');
      } else if (redLight) {
        setStatus('🔴 紅燈 — 請停止，不可通行');
      } else if (yellowLight) {
        setStatus('🟡 黃燈 — 注意！請快速通過');
      } else if (greenLight) {
        setStatus('🟢 綠燈 — 可安全通行');
      } else {
        setStatus(`偵測中 - 發現 ${data.boxes?.length || 0} 個物件`);
      }
    } else if (data.mode === 'motorcycle') {
      if (data.two_stage_warning) { playAlarm(); setStatus('🚨 注意：前方路口需兩段式左轉'); }
      else { setStatus(`導航中：${data.message}`); }
    }
  };

  const playAlarm = useCallback(async () => {
    if (isAlarmPlaying.current) return;
    isAlarmPlaying.current = true;
    Vibration.vibrate([0, 500, 200, 500]);
    try {
      await Audio.setAudioModeAsync({ playsInSilentModeIOS: true, staysActiveInBackground: true });
      const { sound } = await Audio.Sound.createAsync(
        require('../assets/beep.mp3'),
        { shouldPlay: true, volume: 1.0 }
      );
      sound.setOnPlaybackStatusUpdate((s) => {
        if (s.isLoaded && s.didJustFinish) { isAlarmPlaying.current = false; sound.unloadAsync(); }
      });
    } catch { isAlarmPlaying.current = false; }
  }, []);

  const startDetection = async () => {
    if (!isPedestrian) {
      const { status: locStatus } = await Location.requestForegroundPermissionsAsync();
      if (locStatus !== 'granted') { setStatus('機車模式需要定位權限'); return; }
    }
    setIsDetecting(true);
    setStatus('辨識啟動中...');

    intervalRef.current = setInterval(async () => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      try {
        if (!cameraRef.current || isCapturing.current) return;
        isCapturing.current = true;
        const photo = await cameraRef.current.takePictureAsync({
          base64: true, quality: 0.3, skipProcessing: true,
        });
        const pureBase64 = photo?.base64?.replace(/^data:image\/\w+;base64,/, '') || '';

        if (isPedestrian) {
          wsRef.current.send(JSON.stringify({ mode: 'pedestrian', image: pureBase64 }));
        } else {
          const loc = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
          wsRef.current.send(JSON.stringify({
            mode: 'motorcycle', image: pureBase64,
            lat: loc.coords.latitude, lng: loc.coords.longitude,
            heading: loc.coords.heading || 0, action: 'turn_left',
          }));
        }
        isCapturing.current = false;
      } catch (err) {
        console.error('傳輸失敗', err);
        isCapturing.current = false;
      }
    }, isPedestrian ? 300 : 1500);
  };

  const stopDetection = () => {
    setIsDetecting(false);
    setStatus('已停止辨識');
    if (intervalRef.current) clearInterval(intervalRef.current);
    setBoxes([]);
    setLanes({ zebra: [], sidewalk: [], img_w: 1080, img_h: 1920 });
  };

  const isTrafficSign = (box: Box) =>
    box.specific_sign_name && box.specific_sign_name !== '未知號誌' && box.specific_sign_name !== '';
  const isLightBox = (box: Box) => box.light_status >= 0;
  const getBoxLabel = (box: Box) => isLightBox(box) ? LIGHT_TEXT[box.light_status] : isTrafficSign(box) ? box.specific_sign_name : box.label;
  const getListLabel = (box: Box) => isLightBox(box) ? LIGHT_TEXT[box.light_status] : isTrafficSign(box) ? box.specific_sign_name : box.label;

  // 把後端座標轉換成前端百分比位置的 SVG points 字串
  const toSvgPoints = (contour: number[][], imgW: number, imgH: number, vw: number, vh: number) => {
    return contour.map(([x, y]) =>
      `${(x / imgW) * vw},${(y / imgH) * vh}`
    ).join(' ');
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
      <Stack.Screen options={{ headerShown: false }} />

      <View style={[styles.topBar, { backgroundColor: isPedestrian ? '#1a73e8' : '#e84c1a' }]}>
        <TouchableOpacity onPress={() => { stopDetection(); router.back(); }}>
          <Text style={styles.backBtn}>← 返回</Text>
        </TouchableOpacity>
        <Text style={styles.modeLabel}>
          {isPedestrian ? '🚶 行人辨識模式' : '🏍️ 機車導航助手'}
        </Text>
        <View style={{ width: 48 }} />
      </View>

      {/* 相機區域 */}
      <View
        style={styles.cameraWrapper}
        onLayout={(e) => {
          const { width, height } = e.nativeEvent.layout;
          setViewSize({ w: width, h: height });
        }}
      >
        <CameraView style={{ flex: 1 }} facing="back" ref={cameraRef}>

          {/* 斑馬線和人行道 SVG 疊加層 */}
          {viewSize.w > 0 && (
            <Svg
              style={StyleSheet.absoluteFill}
              width={viewSize.w}
              height={viewSize.h}
            >
              {/* 斑馬線：改為亮黃色、高不透明度 (0.85)、加粗邊框 (低視能友善) */}
              {lanes.zebra.map((contour, i) => (
                <Polygon
                  key={`zebra-${i}`}
                  points={toSvgPoints(contour, lanes.img_w, lanes.img_h, viewSize.w, viewSize.h)}
                  fill="rgba(255, 255, 0, 0.85)"
                  stroke="rgba(255, 255, 255, 1)"
                  strokeWidth="6"
                />
              ))}
              {/* 人行道：改為螢光綠、高不透明度 (0.85)、加粗邊框 (低視能友善) */}
              {lanes.sidewalk.map((contour, i) => (
                <Polygon
                  key={`sidewalk-${i}`}
                  points={toSvgPoints(contour, lanes.img_w, lanes.img_h, viewSize.w, viewSize.h)}
                  fill="rgba(100, 255, 100, 0.85)"
                  stroke="rgba(255, 255, 255, 1)"
                  strokeWidth="6"
                />
              ))}
            </Svg>
          )}

          {/* 物件框框 */}
          {boxes.map((box, i) => {
            const isSign = isTrafficSign(box);
            const isLight = isLightBox(box);
            const color = isLight
              ? (LIGHT_COLOR[box.light_status] || '#00cc66')
              : isSign ? '#FFA500'
              : (DANGER_COLOR[box.danger] || '#00cc66');

            return (
              <View
                key={`${box.track_id}-${i}`}
                style={[styles.box, {
                  left: `${(box.x1 / imgSize.w) * 100}%`,
                  top: `${(box.y1 / imgSize.h) * 100}%`,
                  width: `${((box.x2 - box.x1) / imgSize.w) * 100}%`,
                  height: `${((box.y2 - box.y1) / imgSize.h) * 100}%`,
                  borderColor: color,
                  backgroundColor: `${color}10`,
                }]}
              >
                <View style={[styles.boxInfo, { backgroundColor: `${color}cc` }]}>
                  <Text style={styles.boxLabel} numberOfLines={2}>{getBoxLabel(box)}</Text>
                  {!isSign && !isLight && (
                    <Text style={styles.boxDetail} numberOfLines={1}>
                      {box.distance}m｜危{box.danger_pct}%
                    </Text>
                  )}
                </View>
              </View>
            );
          })}
        </CameraView>
      </View>

      {/* 狀態列 */}
      <View style={[
        styles.statusBar,
        status.includes('⚠️') || status.includes('🚨') ? styles.statusDanger
          : status.includes('🔴') ? styles.statusRed
          : status.includes('🟡') ? styles.statusYellow
          : status.includes('🟢') ? styles.statusGreen
          : null
      ]}>
        <Text style={styles.statusText}>{status}</Text>
      </View>

      {/* 物件清單 */}
      <ScrollView style={styles.listArea}>
        {boxes.map((box, i) => {
          const isSign = isTrafficSign(box);
          const isLight = isLightBox(box);
          const color = isLight
            ? (LIGHT_COLOR[box.light_status] || '#00cc66')
            : isSign ? '#FFA500'
            : (DANGER_COLOR[box.danger] || '#00cc66');
          const speedText = (box.speed ?? 0) > 2 ? '🔴 快速逼近'
            : (box.speed ?? 0) > 0.5 ? '🟡 緩慢靠近' : '🟢 穩定';

          return (
            <View key={`${box.track_id}-${i}`} style={[styles.listItem, { borderLeftColor: color }]}>
              <View style={{ flex: 1 }}>
                <Text style={styles.listLabel}>{getListLabel(box)}</Text>
                {!isSign && !isLight && (
                  <>
                    <Text style={styles.listSub}>距離: {box.distance}m｜速度: {box.speed?.toFixed(1) || 0} px/f</Text>
                    <Text style={styles.listSub}>{speedText}</Text>
                  </>
                )}
              </View>
              {!isSign && !isLight && (
                <View style={[styles.dangerBadge, { backgroundColor: color }]}>
                  <Text style={styles.dangerText}>危險 {box.danger_pct}%</Text>
                </View>
              )}
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
  statusRed: { backgroundColor: '#8b0000' },
  statusYellow: { backgroundColor: '#7a5c00' },
  statusGreen: { backgroundColor: '#1a5c2e' },
  statusText: { color: '#fff', fontSize: 14, fontWeight: '600' },
  listArea: { flex: 2, padding: 12 },
  listItem: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#222', borderRadius: 8, borderLeftWidth: 4, padding: 12, marginBottom: 8 },
  listLabel: { color: '#fff', fontSize: 15, fontWeight: '600' },
  listSub: { color: '#aaa', fontSize: 12, marginTop: 3 },
  dangerBadge: { borderRadius: 6, paddingHorizontal: 8, paddingVertical: 4 },
  dangerText: { color: '#fff', fontSize: 12, fontWeight: '700' },
  controls: { padding: 15, alignItems: 'center', backgroundColor: '#111' },
  button: { backgroundColor: '#fff', paddingVertical: 15, paddingHorizontal: 60, borderRadius: 30 },
  buttonStop: { backgroundColor: '#ff4444' },
  buttonText: { fontSize: 18, fontWeight: '700', color: '#111' },
  box: { position: 'absolute', borderWidth: 1, borderRadius: 3 },
  boxInfo: { position: 'absolute', bottom: -46, left: -1, paddingHorizontal: 4, paddingVertical: 2, borderRadius: 3, minWidth: 80, maxWidth: 160 },
  boxLabel: { color: '#fff', fontSize: 10, fontWeight: '700' },
  boxDetail: { color: '#fff', fontSize: 9 },
});