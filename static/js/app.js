/**
 * Lipreading App JavaScript
 * Handles camera access, video recording, and communication with the Flask backend
 */

class LipreadingApp {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.startBtn = document.getElementById('startBtn');
        this.recordBtn = document.getElementById('recordBtn');
        this.stopBtn = document.getElementById('stopBtn');
        
        this.status = document.getElementById('status');
        this.countdown = document.getElementById('countdown');
        this.results = document.getElementById('results');
        this.error = document.getElementById('error');
        
        this.stream = null;
        this.isRecording = false;
        this.recordedFrames = [];
        this.recordingInterval = null;
        this.countdownInterval = null;
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.recordBtn.addEventListener('click', () => this.startRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
    }
    
    async startCamera() {
        try {
            this.updateStatus('Requesting camera access...');
            
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user' // Front-facing camera
                },
                audio: false
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            
            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
            };
            
            this.startBtn.disabled = true;
            this.recordBtn.disabled = false;
            this.updateStatus('Camera ready! Click "Record Word" to start.');
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.showError('Could not access camera. Please ensure you have granted camera permissions.');
        }
    }
    
    startRecording() {
        if (this.isRecording) return;
        
        this.isRecording = true;
        this.recordedFrames = [];
        
        this.recordBtn.disabled = true;
        this.stopBtn.disabled = false;
        
        this.hideResults();
        this.hideError();
        
        // Start countdown
        this.startCountdown(3, () => {
            this.updateStatus('Recording... Speak your word clearly!');
            this.startFrameCapture();
            
            // Auto-stop after 3 seconds
            setTimeout(() => {
                if (this.isRecording) {
                    this.stopRecording();
                }
            }, 3000);
        });
    }
    
    startCountdown(seconds, callback) {
        let count = seconds;
        this.countdown.textContent = count;
        this.updateStatus('Get ready...');
        
        this.countdownInterval = setInterval(() => {
            count--;
            if (count > 0) {
                this.countdown.textContent = count;
            } else {
                this.countdown.textContent = '';
                clearInterval(this.countdownInterval);
                callback();
            }
        }, 1000);
    }
    
    startFrameCapture() {
        // Capture frames at 10 FPS
        this.recordingInterval = setInterval(() => {
            this.captureFrame();
        }, 100);
    }
    
    captureFrame() {
        if (!this.isRecording) return;
        
        // Draw current video frame to canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Convert canvas to base64 image
        const frameData = this.canvas.toDataURL('image/jpeg', 0.8);
        this.recordedFrames.push(frameData);
        
        // Limit to 30 frames maximum
        if (this.recordedFrames.length > 30) {
            this.recordedFrames.shift();
        }
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        
        if (this.recordingInterval) {
            clearInterval(this.recordingInterval);
            this.recordingInterval = null;
        }
        
        if (this.countdownInterval) {
            clearInterval(this.countdownInterval);
            this.countdownInterval = null;
        }
        
        this.countdown.textContent = '';
        this.recordBtn.disabled = false;
        this.stopBtn.disabled = true;
        
        if (this.recordedFrames.length > 0) {
            this.updateStatus('Processing... Please wait.');
            this.sendFramesForPrediction();
        } else {
            this.updateStatus('No frames captured. Please try again.');
        }
    }
    
    async sendFramesForPrediction() {
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    frames: this.recordedFrames
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showResults(result);
                this.updateStatus('Prediction complete! Record another word or try again.');
            } else {
                this.showError(result.error || 'Prediction failed');
                this.updateStatus('Prediction failed. Please try again.');
            }
            
        } catch (error) {
            console.error('Error sending frames:', error);
            this.showError('Network error. Please check your connection and try again.');
            this.updateStatus('Network error. Please try again.');
        }
    }

    showResults(result) {
        this.hideError();

        // Update predicted word and confidence
        document.getElementById('predictedWord').textContent = result.predicted_word;
        document.getElementById('confidence').textContent = `${(result.confidence * 100).toFixed(1)}%`;

        // Update probability bars
        const probabilitiesContainer = document.getElementById('probabilities');
        probabilitiesContainer.innerHTML = '';

        // Sort probabilities by value (highest first)
        const sortedProbs = Object.entries(result.all_probabilities)
            .sort(([,a], [,b]) => b - a);

        sortedProbs.forEach(([word, probability]) => {
            const barContainer = document.createElement('div');
            barContainer.className = 'probability-bar';

            const label = document.createElement('div');
            label.className = 'probability-label';
            label.innerHTML = `
                <span class="probability-word">${word}</span>
                <span class="probability-value">${(probability * 100).toFixed(1)}%</span>
            `;

            const fill = document.createElement('div');
            fill.className = 'probability-fill';

            const fillInner = document.createElement('div');
            fillInner.className = 'probability-fill-inner';
            fillInner.style.width = `${probability * 100}%`;

            fill.appendChild(fillInner);
            barContainer.appendChild(label);
            barContainer.appendChild(fill);
            probabilitiesContainer.appendChild(barContainer);
        });

        this.results.classList.remove('hidden');
    }

    showError(message) {
        this.hideResults();
        document.getElementById('errorMessage').textContent = message;
        this.error.classList.remove('hidden');
    }

    hideResults() {
        this.results.classList.add('hidden');
    }

    hideError() {
        this.error.classList.add('hidden');
    }

    updateStatus(message) {
        this.status.textContent = message;
    }

    // Cleanup method
    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }

        if (this.recordingInterval) {
            clearInterval(this.recordingInterval);
        }

        if (this.countdownInterval) {
            clearInterval(this.countdownInterval);
        }
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const app = new LipreadingApp();

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        app.cleanup();
    });

    // Handle visibility change (e.g., switching tabs)
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && app.isRecording) {
            app.stopRecording();
        }
    });
});

// Service Worker registration for PWA capabilities (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
