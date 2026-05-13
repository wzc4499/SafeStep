import { StyleSheet, View, Text, TouchableOpacity, Dimensions } from 'react-native';
import { useRouter } from 'expo-router';

const { width } = Dimensions.get('window');

export default function Home() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>SafeStep</Text>
      <Text style={styles.subtitle}>請選擇使用模式</Text>

      <TouchableOpacity
        style={[styles.modeButton, { backgroundColor: '#1a73e8' }]}
        onPress={() => router.push({ pathname: '/detect', params: { mode: 'pedestrian' } })}
      >
        <Text style={styles.modeIcon}>🚶</Text>
        <Text style={styles.modeTitle}>行人模式</Text>
        <Text style={styles.modeDesc}>偵測周遭車輛、障礙物與交通標誌</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.modeButton, { backgroundColor: '#e84c1a' }]}
        onPress={() => router.push({ pathname: '/detect', params: { mode: 'motorcycle' } })}
      >
        <Text style={styles.modeIcon}>🏍️</Text>
        <Text style={styles.modeTitle}>機車模式</Text>
        <Text style={styles.modeDesc}>偵測前方車輛、行人與交通標誌</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  title: {
    fontSize: 36,
    fontWeight: '800',
    color: '#fff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#aaa',
    marginBottom: 48,
  },
  modeButton: {
    width: width - 48,
    borderRadius: 16,
    padding: 24,
    marginBottom: 20,
    alignItems: 'center',
  },
  modeIcon: { fontSize: 48, marginBottom: 8 },
  modeTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#fff',
    marginBottom: 6,
  },
  modeDesc: {
    fontSize: 13,
    color: 'rgba(255,255,255,0.8)',
    textAlign: 'center',
  },
});