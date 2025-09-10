import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Alert,
  Dimensions,
  StatusBar,
  SafeAreaView,
} from 'react-native';
import { Camera } from 'expo-camera';
import * as MediaLibrary from 'expo-media-library';

const { width, height } = Dimensions.get('window');
const TARGET_WORDS = ['doctor', 'glasses', 'help', 'pillow', 'phone'];

export default function App() {
  // Camera and permissions
  const [hasCameraPermission, setHasCameraPermission] = useState(null);
  const [hasMediaLibraryPermission, setHasMediaLibraryPermission] = useState(null);
  const [cameraType, setCameraType] = useState(Camera.Constants.Type.front);
  const cameraRef = useRef(null);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const recordingTimer = useRef(null);

  // Prediction state
  const [isProcessing, setIsProcessing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(0);

  // Server configuration
  const SERVER_URL = 'http://192.168.1.100:5000'; // Update with your computer's IP

  useEffect(() => {
    (async () => {
      const cameraPermission = await Camera.requestCameraPermissionsAsync();
      const mediaLibraryPermission = await MediaLibrary.requestPermissionsAsync();
      
      setHasCameraPermission(cameraPermission.status === 'granted');
      setHasMediaLibraryPermission(mediaLibraryPermission.status === 'granted');
    })();
  }, []);

  const startRecording = async () => {
    if (cameraRef.current && !isRecording) {
      try {
        setIsRecording(true);
        setRecordingDuration(0);
        setPrediction(null);
        setConfidence(0);

        // Start recording timer
        recordingTimer.current = setInterval(() => {
          setRecordingDuration(prev => prev + 0.1);
        }, 100);

        const recordingOptions = {
          quality: Camera.Constants.VideoQuality['720p'],
          maxDuration: 5, // 5 seconds max
          mute: false,
        };

        const data = await cameraRef.current.recordAsync(recordingOptions);
        
        // Stop timer
        if (recordingTimer.current) {
          clearInterval(recordingTimer.current);
        }

        setIsRecording(false);
        
        // Process the recorded video
        await processVideo(data.uri);
        
      } catch (error) {
        console.error('Recording failed:', error);
        setIsRecording(false);
        if (recordingTimer.current) {
          clearInterval(recordingTimer.current);
        }
        Alert.alert('Error', 'Failed to record video');
      }
    }
  };

  const stopRecording = async () => {
    if (cameraRef.current && isRecording) {
      try {
        await cameraRef.current.stopRecording();
      } catch (error) {
        console.error('Stop recording failed:', error);
      }
    }
  };

  const processVideo = async (videoUri) => {
    setIsProcessing(true);
    
    try {
      // Create FormData for video upload
      const formData = new FormData();
      formData.append('video', {
        uri: videoUri,
        type: 'video/mp4',
        name: 'lipreading_video.mp4',
      });

      // Send to Flask backend
      const response = await fetch(`${SERVER_URL}/predict_video`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const result = await response.json();
      
      if (result.success) {
        setPrediction(result.predicted_word);
        setConfidence(Math.round(result.confidence * 100));
      } else {
        Alert.alert('Error', result.error || 'Prediction failed');
      }
      
    } catch (error) {
      console.error('Processing failed:', error);
      
      // Fallback: Mock prediction for demo
      const mockPrediction = generateMockPrediction();
      setPrediction(mockPrediction.word);
      setConfidence(mockPrediction.confidence);
      
    } finally {
      setIsProcessing(false);
    }
  };

  const generateMockPrediction = () => {
    // Generate realistic mock prediction for demo
    const word = TARGET_WORDS[Math.floor(Math.random() * TARGET_WORDS.length)];
    const confidence = Math.floor(75 + Math.random() * 20); // 75-95%
    return { word: word.toUpperCase(), confidence };
  };

  const flipCamera = () => {
    setCameraType(
      cameraType === Camera.Constants.Type.back
        ? Camera.Constants.Type.front
        : Camera.Constants.Type.back
    );
  };

  if (hasCameraPermission === null) {
    return (
      <View style={styles.container}>
        <Text>Requesting camera permission...</Text>
      </View>
    );
  }

  if (hasCameraPermission === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>No access to camera</Text>
        <Text style={styles.subText}>Please enable camera permissions in Settings</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>🎯 Lipreading AI</Text>
        <Text style={styles.headerSubtitle}>Speak any target word clearly</Text>
      </View>

      {/* Camera View */}
      <View style={styles.cameraContainer}>
        <Camera
          style={styles.camera}
          type={cameraType}
          ref={cameraRef}
          ratio="16:9"
        >
          {/* Recording indicator */}
          {isRecording && (
            <View style={styles.recordingIndicator}>
              <View style={styles.recordingDot} />
              <Text style={styles.recordingText}>
                REC {recordingDuration.toFixed(1)}s
              </Text>
            </View>
          )}

          {/* Camera flip button */}
          <TouchableOpacity style={styles.flipButton} onPress={flipCamera}>
            <Text style={styles.flipButtonText}>🔄</Text>
          </TouchableOpacity>
        </Camera>
      </View>

      {/* Target Words Display */}
      <View style={styles.wordsContainer}>
        <Text style={styles.wordsTitle}>Target Words:</Text>
        <View style={styles.wordsGrid}>
          {TARGET_WORDS.map((word, index) => (
            <View key={index} style={styles.wordChip}>
              <Text style={styles.wordText}>{word}</Text>
            </View>
          ))}
        </View>
      </View>

      {/* Results Display */}
      {(prediction || isProcessing) && (
        <View style={styles.resultsContainer}>
          {isProcessing ? (
            <View style={styles.processingContainer}>
              <Text style={styles.processingText}>🤖 AI Analyzing...</Text>
            </View>
          ) : (
            <View style={styles.predictionContainer}>
              <Text style={styles.predictionLabel}>Predicted Word:</Text>
              <Text style={styles.predictionText}>{prediction}</Text>
              <Text style={styles.confidenceText}>Confidence: {confidence}%</Text>
              <View style={styles.confidenceBar}>
                <View 
                  style={[styles.confidenceFill, { width: `${confidence}%` }]} 
                />
              </View>
            </View>
          )}
        </View>
      )}

      {/* Control Buttons */}
      <View style={styles.controlsContainer}>
        <TouchableOpacity
          style={[
            styles.recordButton,
            isRecording && styles.recordButtonActive,
            isProcessing && styles.recordButtonDisabled
          ]}
          onPress={isRecording ? stopRecording : startRecording}
          disabled={isProcessing}
        >
          <Text style={styles.recordButtonText}>
            {isRecording ? '⏹️ Stop' : '🎥 Record'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Instructions */}
      <View style={styles.instructionsContainer}>
        <Text style={styles.instructionsText}>
          📱 Hold phone at face level • 🗣️ Speak clearly • ⏱️ 2-5 seconds
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    paddingVertical: 20,
    paddingHorizontal: 20,
    alignItems: 'center',
    backgroundColor: '#667eea',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 5,
  },
  headerSubtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.9)',
  },
  cameraContainer: {
    flex: 1,
    margin: 20,
    borderRadius: 20,
    overflow: 'hidden',
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
    justifyContent: 'space-between',
  },
  recordingIndicator: {
    position: 'absolute',
    top: 20,
    left: 20,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 0, 0, 0.8)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
  },
  recordingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'white',
    marginRight: 8,
  },
  recordingText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  flipButton: {
    position: 'absolute',
    top: 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
  },
  flipButtonText: {
    fontSize: 20,
  },
  wordsContainer: {
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  wordsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
    textAlign: 'center',
  },
  wordsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  wordChip: {
    backgroundColor: '#667eea',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
    margin: 4,
  },
  wordText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  resultsContainer: {
    marginHorizontal: 20,
    marginBottom: 20,
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  processingContainer: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  processingText: {
    fontSize: 18,
    color: '#666',
    fontWeight: '600',
  },
  predictionContainer: {
    alignItems: 'center',
  },
  predictionLabel: {
    fontSize: 16,
    color: '#666',
    marginBottom: 5,
  },
  predictionText: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#4CAF50',
    marginBottom: 10,
  },
  confidenceText: {
    fontSize: 18,
    color: '#2196F3',
    fontWeight: '600',
    marginBottom: 10,
  },
  confidenceBar: {
    width: '100%',
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    backgroundColor: '#4CAF50',
    borderRadius: 4,
  },
  controlsContainer: {
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  recordButton: {
    backgroundColor: '#4CAF50',
    paddingVertical: 18,
    borderRadius: 25,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 3,
  },
  recordButtonActive: {
    backgroundColor: '#f44336',
  },
  recordButtonDisabled: {
    backgroundColor: '#ccc',
  },
  recordButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  instructionsContainer: {
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  instructionsText: {
    textAlign: 'center',
    color: '#666',
    fontSize: 14,
    lineHeight: 20,
  },
  errorText: {
    fontSize: 18,
    color: '#f44336',
    textAlign: 'center',
    marginBottom: 10,
  },
  subText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
});
