import { BlurView } from 'expo-blur';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Haptics from 'expo-haptics';
import * as Location from 'expo-location';
import React, { useEffect, useRef, useState } from 'react';
import { Dimensions, StyleSheet, Switch, Text, TouchableOpacity, View } from 'react-native';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

interface DetectedObject {
  id: number;
  label: string;
  x: number; y: number; w: number; h: number;
  ttc: number;
}

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isTestMode, setIsTestMode] = useState(false);
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [isRecording, setIsRecording] = useState(true);
  const testInterval = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const blink = setInterval(() => setIsRecording(prev => !prev), 1000);
    
    (async () => {
      await requestPermission();
      // 雖然不顯示速度，但保留權限請求以供未來碰撞偵測使用 [cite: 6]
      await Location.requestForegroundPermissionsAsync();
    })();
    return () => clearInterval(blink);
  }, []);

  useEffect(() => {
    if (isTestMode) {
      testInterval.current = setInterval(() => {
        const fakeData: DetectedObject[] = [
          { id: 1, label: 'Car', x: 0.2, y: 0.4, w: 0.3, h: 0.2, ttc: Math.random() * 6 },
          { id: 2, label: 'Pedestrian', x: 0.7, y: 0.3, w: 0.1, h: 0.5, ttc: 8 }
        ];
        setDetectedObjects(fakeData);
        if (fakeData.some(o => o.ttc < 2)) Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      }, 1000);
    } else {
      if (testInterval.current) clearInterval(testInterval.current);
      setDetectedObjects([]);
    }
    return () => { if (testInterval.current) clearInterval(testInterval.current); };
  }, [isTestMode]);

  const getDisplayName = (label: string) => {
    if (label === 'Car') return '前方車輛';
    if (label === 'Pedestrian') return '行人';
    return label;
  };

  const getWarningStyle = (ttc: number) => {
    if (ttc < 2) return { color: '#FF3B30', text: '危險！即將碰撞', width: 4 };
    if (ttc < 5) return { color: '#FFCC00', text: `注意：${ttc.toFixed(1)}秒`, width: 3 };
    return { color: '#4CD964', text: '安全距離', width: 2 };
  };

  if (!permission?.granted) return <View style={styles.center}><TouchableOpacity onPress={requestPermission} style={styles.btn}><Text style={{color:'white'}}>啟用相機權限</Text></TouchableOpacity></View>;

  return (
    <View style={styles.container}>
      <CameraView style={StyleSheet.absoluteFillObject} facing="back">
        
        {detectedObjects.map((obj) => {
          const warning = getWarningStyle(obj.ttc);
          return (
            <View key={obj.id} style={[styles.boundingBox, {
              left: obj.x * SCREEN_WIDTH,
              top: obj.y * SCREEN_HEIGHT,
              width: obj.w * SCREEN_WIDTH,
              height: obj.h * SCREEN_HEIGHT,
              borderColor: warning.color,
              borderWidth: warning.width,
            }]}>
              <View style={[styles.labelTag, { backgroundColor: warning.color }]}>
                <Text style={styles.labelText}>{getDisplayName(obj.label)} | {warning.text}</Text>
              </View>
            </View>
          );
        })}

        <View style={styles.header}>
          <View style={styles.row}>
            <View style={[styles.dot, { opacity: isRecording ? 1 : 0 }]} />
            <Text style={styles.headerText}>道路監控中</Text>
          </View>
          {/* 時速顯示已移除 [cite: 1] */}
        </View>

        <BlurView intensity={60} tint="dark" style={styles.footer}>
          <View style={styles.row}>
            <Text style={{color: 'white', marginRight: 15, fontSize: 16, fontWeight: 'bold'}}>系統模擬環境</Text>
            <Switch value={isTestMode} onValueChange={setIsTestMode} trackColor={{ false: "#767577", true: "#4CD964" }} />
          </View>
          <Text style={styles.hint}>系統狀態：{isTestMode ? "接收模擬環境訊號中..." : "待命狀態，鏡頭運作正常"}</Text>
        </BlurView>

      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  btn: { backgroundColor: '#007AFF', padding: 15, borderRadius: 10 },
  header: { position: 'absolute', top: 50, left: 20, right: 20, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  headerText: { color: 'white', fontWeight: 'bold', fontSize: 16, textShadowColor: 'black', textShadowRadius: 2 },
  dot: { width: 10, height: 10, borderRadius: 5, backgroundColor: '#FF3B30', marginRight: 8 },
  row: { flexDirection: 'row', alignItems: 'center' },
  boundingBox: { position: 'absolute', borderRadius: 4 },
  labelTag: { position: 'absolute', top: -24, left: -2, paddingHorizontal: 8, paddingVertical: 4, borderRadius: 4 },
  labelText: { color: 'black', fontSize: 12, fontWeight: 'bold' },
  footer: { position: 'absolute', bottom: 40, left: 20, right: 20, padding: 20, borderRadius: 20, alignItems: 'center', overflow: 'hidden' },
  hint: { color: '#E0E0E0', fontSize: 12, marginTop: 8 }
});