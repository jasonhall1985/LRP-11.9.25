/**
 * Test Runner for Lipreading Model Accuracy
 * Run this to test the model's accuracy on each word
 */

// Import the model directly for testing
const LipreadingModel = require('../models/LipreadingModel.js').default;

class TestRunner {
  constructor() {
    this.model = new LipreadingModel();
  }

  async runDoctorTest() {
    console.log('🧪 TESTING "DOCTOR" RECOGNITION');
    console.log('=' .repeat(40));
    
    await this.model.loadModel();
    
    // Test with doctor-like patterns
    const doctorTests = [
      {
        complexity: 1.8,
        vertical: 0.025,
        horizontal: 0.015,
        movement: 0.022,
        frameCount: 15
      },
      {
        complexity: 2.0,
        vertical: 0.030,
        horizontal: 0.012,
        movement: 0.025,
        frameCount: 18
      },
      {
        complexity: 1.6,
        vertical: 0.020,
        horizontal: 0.018,
        movement: 0.020,
        frameCount: 12
      }
    ];
    
    let doctorCorrect = 0;
    
    for (let i = 0; i < doctorTests.length; i++) {
      const testData = doctorTests[i];
      
      // Create mock lip data
      const lipData = [];
      for (let frame = 0; frame < testData.frameCount; frame++) {
        lipData.push({
          coordinates: this.generateMockCoordinates(),
          realVideoAnalysis: true
        });
      }
      
      console.log(`\nTest ${i + 1} - Expected: DOCTOR`);
      console.log(`Input: complexity=${testData.complexity}, vertical=${testData.vertical}, horizontal=${testData.horizontal}, movement=${testData.movement}`);
      
      try {
        const result = this.model.predict(lipData);
        console.log(`Result: ${result.word} (${(result.confidence * 100).toFixed(1)}%)`);
        
        if (result.word === 'doctor') {
          doctorCorrect++;
          console.log('✅ CORRECT!');
        } else {
          console.log('❌ INCORRECT');
        }
      } catch (error) {
        console.error('💥 Error:', error.message);
      }
    }
    
    const accuracy = (doctorCorrect / doctorTests.length) * 100;
    console.log(`\n📊 DOCTOR ACCURACY: ${doctorCorrect}/${doctorTests.length} (${accuracy.toFixed(1)}%)`);
    
    return accuracy;
  }

  generateMockCoordinates() {
    const coords = [];
    for (let i = 0; i < 40; i++) {
      coords.push(Math.random() * 0.1 + 0.45);
    }
    return coords;
  }

  async testAllWords() {
    console.log('🚀 COMPREHENSIVE WORD TESTING');
    console.log('=' .repeat(50));
    
    await this.model.loadModel();
    
    const testPatterns = {
      doctor: { complexity: 1.8, vertical: 0.025, horizontal: 0.015, movement: 0.022 },
      glasses: { complexity: 1.2, vertical: 0.012, horizontal: 0.024, movement: 0.013 },
      help: { complexity: 2.2, vertical: 0.026, horizontal: 0.008, movement: 0.028 },
      pillow: { complexity: 1.5, vertical: 0.018, horizontal: 0.018, movement: 0.020 },
      phone: { complexity: 2.0, vertical: 0.030, horizontal: 0.010, movement: 0.025 }
    };
    
    const results = {};
    
    for (const [targetWord, pattern] of Object.entries(testPatterns)) {
      console.log(`\n🎯 Testing: ${targetWord.toUpperCase()}`);
      
      let correct = 0;
      const numTests = 5;
      
      for (let test = 0; test < numTests; test++) {
        // Add some variation
        const variation = (Math.random() - 0.5) * 0.2;
        const testPattern = {
          complexity: pattern.complexity + variation,
          vertical: pattern.vertical + variation * 0.01,
          horizontal: pattern.horizontal + variation * 0.01,
          movement: pattern.movement + variation * 0.01
        };
        
        // Create mock lip data
        const lipData = [];
        for (let frame = 0; frame < 15; frame++) {
          lipData.push({
            coordinates: this.generateMockCoordinates(),
            realVideoAnalysis: true
          });
        }
        
        try {
          const result = this.model.predict(lipData);
          console.log(`   Test ${test + 1}: ${result.word} (${(result.confidence * 100).toFixed(1)}%) ${result.word === targetWord ? '✅' : '❌'}`);
          
          if (result.word === targetWord) {
            correct++;
          }
        } catch (error) {
          console.error(`   Test ${test + 1}: Error - ${error.message}`);
        }
      }
      
      const accuracy = (correct / numTests) * 100;
      results[targetWord] = accuracy;
      console.log(`   📊 ${targetWord.toUpperCase()}: ${correct}/${numTests} (${accuracy.toFixed(1)}%)`);
    }
    
    console.log('\n📈 FINAL RESULTS:');
    console.log('-' .repeat(30));
    Object.entries(results).forEach(([word, accuracy]) => {
      const status = accuracy >= 80 ? '✅' : accuracy >= 50 ? '⚠️' : '❌';
      console.log(`${status} ${word.toUpperCase().padEnd(8)}: ${accuracy.toFixed(1)}%`);
    });
    
    const avgAccuracy = Object.values(results).reduce((sum, acc) => sum + acc, 0) / Object.keys(results).length;
    console.log(`📊 AVERAGE: ${avgAccuracy.toFixed(1)}%`);
    
    return results;
  }
}

// Run the test
async function main() {
  const runner = new TestRunner();
  
  try {
    console.log('🔬 LIPREADING MODEL ACCURACY TEST');
    console.log('Testing improved pattern recognition...\n');
    
    // First test doctor specifically
    await runner.runDoctorTest();
    
    console.log('\n' + '=' .repeat(50));
    
    // Then test all words
    await runner.testAllWords();
    
    console.log('\n🎉 Testing Complete!');
    
  } catch (error) {
    console.error('💥 Test failed:', error);
  }
}

if (require.main === module) {
  main();
}

module.exports = TestRunner;
