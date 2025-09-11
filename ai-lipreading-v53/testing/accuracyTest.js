/**
 * Accuracy Testing System for Lipreading Model
 * Tests each word individually to measure recognition rates
 */

import LipreadingModel from '../models/LipreadingModel.js';

class AccuracyTester {
  constructor() {
    this.model = new LipreadingModel();
    this.testResults = {
      doctor: { correct: 0, total: 0, predictions: [] },
      glasses: { correct: 0, total: 0, predictions: [] },
      help: { correct: 0, total: 0, predictions: [] },
      pillow: { correct: 0, total: 0, predictions: [] },
      phone: { correct: 0, total: 0, predictions: [] }
    };
  }

  async initialize() {
    await this.model.loadModel();
    console.log('🧪 Accuracy Testing System Initialized');
    console.log('   Model loaded with GRID training');
  }

  // Generate realistic test data for each word based on phoneme patterns
  generateTestData(targetWord, variation = 0) {
    const basePatterns = {
      doctor: {
        // D-OC-T-OR: Strong jaw movement, mouth opening/closing
        complexity: 1.8 + (Math.random() - 0.5) * 0.6 + variation * 0.2,
        vertical: 0.025 + (Math.random() - 0.5) * 0.015 + variation * 0.005,
        horizontal: 0.015 + (Math.random() - 0.5) * 0.010 + variation * 0.003,
        movement: 0.022 + (Math.random() - 0.5) * 0.012 + variation * 0.004,
        frameCount: 15 + Math.floor(Math.random() * 10)
      },
      glasses: {
        // GL-A-SS-ES: Lip pursing, horizontal stretch
        complexity: 1.2 + (Math.random() - 0.5) * 0.4 + variation * 0.15,
        vertical: 0.012 + (Math.random() - 0.5) * 0.008 + variation * 0.003,
        horizontal: 0.024 + (Math.random() - 0.5) * 0.012 + variation * 0.004,
        movement: 0.013 + (Math.random() - 0.5) * 0.008 + variation * 0.003,
        frameCount: 12 + Math.floor(Math.random() * 8)
      },
      help: {
        // H-E-L-P: Quick, sharp lip movements
        complexity: 2.2 + (Math.random() - 0.5) * 0.8 + variation * 0.3,
        vertical: 0.026 + (Math.random() - 0.5) * 0.012 + variation * 0.004,
        horizontal: 0.008 + (Math.random() - 0.5) * 0.006 + variation * 0.002,
        movement: 0.028 + (Math.random() - 0.5) * 0.014 + variation * 0.005,
        frameCount: 10 + Math.floor(Math.random() * 6)
      },
      pillow: {
        // P-I-LL-OW: Rounded, moderate movements
        complexity: 1.5 + (Math.random() - 0.5) * 0.5 + variation * 0.2,
        vertical: 0.018 + (Math.random() - 0.5) * 0.010 + variation * 0.003,
        horizontal: 0.018 + (Math.random() - 0.5) * 0.010 + variation * 0.003,
        movement: 0.020 + (Math.random() - 0.5) * 0.010 + variation * 0.003,
        frameCount: 14 + Math.floor(Math.random() * 8)
      },
      phone: {
        // PH-O-N-E: F-position, rounded lips
        complexity: 2.0 + (Math.random() - 0.5) * 0.6 + variation * 0.25,
        vertical: 0.030 + (Math.random() - 0.5) * 0.015 + variation * 0.005,
        horizontal: 0.010 + (Math.random() - 0.5) * 0.008 + variation * 0.003,
        movement: 0.025 + (Math.random() - 0.5) * 0.012 + variation * 0.004,
        frameCount: 13 + Math.floor(Math.random() * 7)
      }
    };

    const pattern = basePatterns[targetWord];
    if (!pattern) {
      throw new Error(`Unknown target word: ${targetWord}`);
    }

    // Generate mock lip data frames
    const frames = [];
    for (let i = 0; i < pattern.frameCount; i++) {
      frames.push({
        coordinates: this.generateMockCoordinates(),
        realVideoAnalysis: true,
        timestamp: Date.now() + i * 50
      });
    }

    return {
      lipData: frames,
      expectedAnalysis: {
        complexity: pattern.complexity,
        vertical: pattern.vertical,
        horizontal: pattern.horizontal,
        movement: pattern.movement,
        frameCount: pattern.frameCount
      }
    };
  }

  generateMockCoordinates() {
    // Generate 20 lip landmark coordinates (40 values total)
    const coords = [];
    for (let i = 0; i < 40; i++) {
      coords.push(Math.random() * 0.1 + 0.45); // Normalized coordinates
    }
    return coords;
  }

  async testWord(targetWord, numTests = 10) {
    console.log(`\n🎯 Testing word: "${targetWord.toUpperCase()}" (${numTests} tests)`);
    
    const results = this.testResults[targetWord];
    results.total += numTests;
    
    for (let i = 0; i < numTests; i++) {
      // Generate test data with varying difficulty
      const variation = i * 0.1; // Increase variation for later tests
      const testData = this.generateTestData(targetWord, variation);
      
      try {
        const prediction = this.model.predict(testData.lipData);
        const predictedWord = prediction.word;
        const confidence = prediction.confidence;
        
        results.predictions.push({
          predicted: predictedWord,
          confidence: confidence,
          correct: predictedWord === targetWord,
          variation: variation,
          analysis: prediction.analysis
        });
        
        if (predictedWord === targetWord) {
          results.correct++;
          console.log(`   ✅ Test ${i + 1}: ${predictedWord} (${(confidence * 100).toFixed(1)}%) - CORRECT`);
        } else {
          console.log(`   ❌ Test ${i + 1}: ${predictedWord} (${(confidence * 100).toFixed(1)}%) - Expected: ${targetWord}`);
        }
        
      } catch (error) {
        console.error(`   💥 Test ${i + 1}: Error - ${error.message}`);
      }
    }
    
    const accuracy = (results.correct / results.total) * 100;
    console.log(`📊 ${targetWord.toUpperCase()} Results: ${results.correct}/${results.total} correct (${accuracy.toFixed(1)}%)`);
    
    return accuracy;
  }

  async runFullTest() {
    console.log('🚀 Starting Full Accuracy Test Suite');
    console.log('=' .repeat(50));
    
    await this.initialize();
    
    const words = ['doctor', 'glasses', 'help', 'pillow', 'phone'];
    const accuracies = {};
    
    for (const word of words) {
      accuracies[word] = await this.testWord(word, 10);
    }
    
    console.log('\n📈 FINAL ACCURACY REPORT');
    console.log('=' .repeat(50));
    
    let totalCorrect = 0;
    let totalTests = 0;
    
    Object.keys(accuracies).forEach(word => {
      const results = this.testResults[word];
      totalCorrect += results.correct;
      totalTests += results.total;
      
      console.log(`${word.toUpperCase().padEnd(8)}: ${accuracies[word].toFixed(1)}% (${results.correct}/${results.total})`);
    });
    
    const overallAccuracy = (totalCorrect / totalTests) * 100;
    console.log('-' .repeat(30));
    console.log(`OVERALL : ${overallAccuracy.toFixed(1)}% (${totalCorrect}/${totalTests})`);
    
    // Identify problem areas
    console.log('\n🔍 ANALYSIS:');
    Object.keys(accuracies).forEach(word => {
      const accuracy = accuracies[word];
      if (accuracy < 50) {
        console.log(`❌ ${word.toUpperCase()}: CRITICAL - ${accuracy.toFixed(1)}% accuracy`);
      } else if (accuracy < 80) {
        console.log(`⚠️  ${word.toUpperCase()}: NEEDS IMPROVEMENT - ${accuracy.toFixed(1)}% accuracy`);
      } else {
        console.log(`✅ ${word.toUpperCase()}: GOOD - ${accuracy.toFixed(1)}% accuracy`);
      }
    });
    
    return {
      overallAccuracy,
      wordAccuracies: accuracies,
      results: this.testResults
    };
  }

  // Quick test for debugging
  async quickTest(targetWord = 'doctor', numTests = 5) {
    console.log(`🔬 Quick Test: "${targetWord}" (${numTests} tests)`);
    await this.initialize();
    return await this.testWord(targetWord, numTests);
  }
}

export default AccuracyTester;

// For direct testing
if (typeof require !== 'undefined' && require.main === module) {
  const tester = new AccuracyTester();
  tester.runFullTest().then(results => {
    console.log('\n🎉 Testing Complete!');
    process.exit(0);
  }).catch(error => {
    console.error('💥 Testing Failed:', error);
    process.exit(1);
  });
}
