import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Dimensions,
  SafeAreaView,
  StatusBar,
} from 'react-native';
import { Camera } from 'expo-camera';
import * as MediaLibrary from 'expo-media-library';
import * as Speech from 'expo-speech';
import Constants from 'expo-constants';

const { width, height } = Dimensions.get('window');

// Configuration - Update this with your computer's IP address
const API_URL = Constants.expoConfig?.extra?.apiUrl || 'http://192.168.1.100:5000';

// 4-class labels from the 75.9% model
const CLASS_LABELS = ['my_mouth_is_dry', 'i_need_to_move', 'doctor', 'pillow'];

export default function App() {
  // Camera and permissions
  const [hasCameraPermission, setHasCameraPermission] = useState(null);
  const [hasMediaLibraryPermission, setHasMediaLibraryPermission] = useState(null);
  const cameraRef = useRef(null);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordedVideo, setRecordedVideo] = useState(null);
  const [countdown, setCountdown] = useState(0);

  // Prediction state
  const [isProcessing, setIsProcessing] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [abstain, setAbstain] = useState(false);

  // UI state
  const [currentScreen, setCurrentScreen] = useState('recording'); // 'recording' or 'results'

  useEffect(() => {
    (async () => {
      const cameraPermission = await Camera.requestCameraPermissionsAsync();
      const mediaLibraryPermission = await MediaLibrary.requestPermissionsAsync();
      
      setHasCameraPermission(cameraPermission.status === 'granted');
      setHasMediaLibraryPermission(mediaLibraryPermission.status === 'granted');
    })();
  }, []);

  const getConfidenceBadgeStyle = (confidence) => {
    if (confidence >= 0.75) return styles.confidenceHigh;
    if (confidence >= 0.50) return styles.confidenceMedium;
    return styles.confidenceLow;
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 0.75) return 'High';
    if (confidence >= 0.50) return 'Medium';
    return 'Low';
  };

  const startRecording = async () => {
    if (!cameraRef.current) return;

    try {
      // Start countdown
      setCountdown(3);
      const countdownInterval = setInterval(() => {
        setCountdown((prev) => {
          if (prev <= 1) {
            clearInterval(countdownInterval);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      // Wait for countdown
      await new Promise(resolve => setTimeout(resolve, 3000));

      // Start recording
      setIsRecording(true);
      const video = await cameraRef.current.recordAsync({
        quality: Camera.Constants.VideoQuality['720p'],
        maxDuration: 3, // 3 seconds max
      });

      setRecordedVideo(video);
      setIsRecording(false);

      // Automatically process the video
      await processVideo(video.uri);

    } catch (error) {
      console.error('Recording error:', error);
      Alert.alert('Recording Error', 'Failed to record video. Please try again.');
      setIsRecording(false);
      setCountdown(0);
    }
  };

  const processVideo = async (videoUri) => {
    setIsProcessing(true);
    setPredictions(null);
    setAbstain(false);

    try {
      const formData = new FormData();
      formData.append('video', {
        uri: videoUri,
        type: 'video/mp4',
        name: 'lipreading_video.mp4',
      });

      console.log('Sending video to:', `${API_URL}/predict`);

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const result = await response.json();
      console.log('Server response:', result);

      if (result.success) {
        setPredictions(result.top2);
        setAbstain(result.abstain);
        setCurrentScreen('results');
      } else {
        Alert.alert('Prediction Error', result.error || 'Failed to analyze video');
      }

    } catch (error) {
      console.error('Processing error:', error);
      Alert.alert('Network Error', 'Could not connect to server. Please check your connection.');
    } finally {
      setIsProcessing(false);
    }
  };

  const speakPrediction = (text) => {
    // Convert class name to speakable text
    const speakableText = text.replace(/_/g, ' ');
    Speech.speak(speakableText, {
      language: 'en-US',
      pitch: 1.0,
      rate: 0.8,
    });
  };

  const handleConfirmPrediction = (prediction) => {
    speakPrediction(prediction.class);
    Alert.alert(
      'Confirmed',
      `Speaking: "${prediction.class.replace(/_/g, ' ')}"`,
      [{ text: 'OK' }]
    );
  };

  const handleManualSelection = (className) => {
    speakPrediction(className);
    Alert.alert(
      'Manual Selection',
      `Speaking: "${className.replace(/_/g, ' ')}"`,
      [{ text: 'OK' }]
    );
  };

  const resetToRecording = () => {
    setCurrentScreen('recording');
    setPredictions(null);
    setRecordedVideo(null);
    setAbstain(false);
    setCountdown(0);
  };

  if (hasCameraPermission === null || hasMediaLibraryPermission === null) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Requesting permissions...</Text>
      </View>
    );
  }

  if (hasCameraPermission === false || hasMediaLibraryPermission === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>Camera and media library permissions are required</Text>
        <TouchableOpacity style={styles.button} onPress={() => {
          Camera.requestCameraPermissionsAsync();
          MediaLibrary.requestPermissionsAsync();
        }}>
          <Text style={styles.buttonText}>Grant Permissions</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (currentScreen === 'results') {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" />
        
        <View style={styles.header}>
          <Text style={styles.title}>AI Prediction Results</Text>
          <Text style={styles.subtitle}>75.9% Validation Accuracy Model</Text>
        </View>

        <View style={styles.resultsContainer}>
          {abstain ? (
            <View style={styles.abstainContainer}>
              <Text style={styles.abstainTitle}>Uncertain Prediction</Text>
              <Text style={styles.abstainSubtitle}>Please select manually:</Text>
              
              <View style={styles.manualButtonsContainer}>
                {CLASS_LABELS.map((className) => (
                  <TouchableOpacity
                    key={className}
                    style={styles.manualButton}
                    onPress={() => handleManualSelection(className)}
                  >
                    <Text style={styles.manualButtonText}>
                      {className.replace(/_/g, ' ')}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          ) : (
            <View style={styles.predictionsContainer}>
              <Text style={styles.predictionsTitle}>Top Predictions:</Text>
              
              {predictions?.map((prediction, index) => (
                <View key={index} style={styles.predictionCard}>
                  <View style={styles.predictionHeader}>
                    <Text style={styles.predictionClass}>
                      {prediction.class.replace(/_/g, ' ')}
                    </Text>
                    <View style={[styles.confidenceBadge, getConfidenceBadgeStyle(prediction.confidence)]}>
                      <Text style={styles.confidenceText}>
                        {getConfidenceLabel(prediction.confidence)}
                      </Text>
                    </View>
                  </View>
                  
                  <Text style={styles.confidencePercentage}>
                    {(prediction.confidence * 100).toFixed(1)}% confidence
                  </Text>
                  
                  {index === 0 && (
                    <TouchableOpacity
                      style={styles.confirmButton}
                      onPress={() => handleConfirmPrediction(prediction)}
                    >
                      <Text style={styles.confirmButtonText}>
                        Tap to Confirm & Speak
                      </Text>
                    </TouchableOpacity>
                  )}
                </View>
              ))}
            </View>
          )}
        </View>

        <TouchableOpacity style={styles.tryAgainButton} onPress={resetToRecording}>
          <Text style={styles.tryAgainButtonText}>Try Again</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      
      <View style={styles.header}>
        <Text style={styles.title}>Lip-Reading AI Demo</Text>
        <Text style={styles.subtitle}>Record 1-3 seconds of lip movement</Text>
      </View>

      <View style={styles.cameraContainer}>
        <Camera
          ref={cameraRef}
          style={styles.camera}
          type={Camera.Constants.Type.front}
          ratio="16:9"
        />
        
        {countdown > 0 && (
          <View style={styles.countdownOverlay}>
            <Text style={styles.countdownText}>{countdown}</Text>
          </View>
        )}

        <View style={styles.lipGuide}>
          <Text style={styles.lipGuideText}>Position lips in center</Text>
        </View>
      </View>

      <View style={styles.controlsContainer}>
        {isProcessing ? (
          <View style={styles.processingContainer}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.processingText}>Analyzing lip movement...</Text>
          </View>
        ) : (
          <TouchableOpacity
            style={[styles.recordButton, isRecording && styles.recordingButton]}
            onPress={startRecording}
            disabled={isRecording || countdown > 0}
          >
            <Text style={styles.recordButtonText}>
              {countdown > 0 ? `Starting in ${countdown}...` : 
               isRecording ? 'Recording...' : 'Record'}
            </Text>
          </TouchableOpacity>
        )}

        <Text style={styles.instructionText}>
          Say one of: "my mouth is dry", "I need to move", "doctor", "pillow"
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    alignItems: 'center',
    paddingVertical: 20,
    paddingHorizontal: 20,
    backgroundColor: '#ffffff',
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#212529',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
    color: '#6c757d',
  },
  cameraContainer: {
    flex: 1,
    margin: 20,
    borderRadius: 15,
    overflow: 'hidden',
    backgroundColor: '#000',
    position: 'relative',
  },
  camera: {
    flex: 1,
  },
  countdownOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  countdownText: {
    fontSize: 72,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  lipGuide: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    padding: 10,
    borderRadius: 8,
    alignItems: 'center',
  },
  lipGuideText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '500',
  },
  controlsContainer: {
    padding: 20,
    backgroundColor: '#ffffff',
    alignItems: 'center',
  },
  recordButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 40,
    paddingVertical: 15,
    borderRadius: 25,
    marginBottom: 15,
    minWidth: 200,
    alignItems: 'center',
  },
  recordingButton: {
    backgroundColor: '#FF3B30',
  },
  recordButtonText: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '600',
  },
  instructionText: {
    fontSize: 14,
    color: '#6c757d',
    textAlign: 'center',
    lineHeight: 20,
  },
  processingContainer: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  processingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '500',
  },
  loadingText: {
    marginTop: 20,
    fontSize: 16,
    color: '#6c757d',
  },
  errorText: {
    fontSize: 16,
    color: '#FF3B30',
    textAlign: 'center',
    marginBottom: 20,
    paddingHorizontal: 20,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 30,
    paddingVertical: 12,
    borderRadius: 8,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  resultsContainer: {
    flex: 1,
    padding: 20,
  },
  predictionsContainer: {
    flex: 1,
  },
  predictionsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#212529',
    marginBottom: 20,
    textAlign: 'center',
  },
  predictionCard: {
    backgroundColor: '#ffffff',
    padding: 20,
    borderRadius: 12,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  predictionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  predictionClass: {
    fontSize: 18,
    fontWeight: '600',
    color: '#212529',
    flex: 1,
    textTransform: 'capitalize',
  },
  confidenceBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  confidenceHigh: {
    backgroundColor: '#28a745',
  },
  confidenceMedium: {
    backgroundColor: '#ffc107',
  },
  confidenceLow: {
    backgroundColor: '#dc3545',
  },
  confidenceText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '600',
  },
  confidencePercentage: {
    fontSize: 14,
    color: '#6c757d',
    marginBottom: 15,
  },
  confirmButton: {
    backgroundColor: '#28a745',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  confirmButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  abstainContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  abstainTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#FF3B30',
    marginBottom: 10,
    textAlign: 'center',
  },
  abstainSubtitle: {
    fontSize: 16,
    color: '#6c757d',
    marginBottom: 30,
    textAlign: 'center',
  },
  manualButtonsContainer: {
    width: '100%',
  },
  manualButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginBottom: 10,
    alignItems: 'center',
  },
  manualButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  tryAgainButton: {
    backgroundColor: '#6c757d',
    marginHorizontal: 20,
    marginBottom: 20,
    paddingVertical: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  tryAgainButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
});
