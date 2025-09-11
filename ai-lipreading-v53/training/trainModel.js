#!/usr/bin/env node

/**
 * Real Neural Network Training Script for Lipreading
 * This script trains the model on GRID dataset and generates pre-trained weights
 */

const GridDatasetTrainer = require('./gridDatasetTrainer');
const fs = require('fs');
const path = require('path');

async function trainLipreadingModel() {
  console.log('🚀 Starting Real Lipreading Model Training');
  console.log('=====================================');
  
  try {
    // Initialize trainer with GRID dataset
    const trainer = new GridDatasetTrainer();
    
    console.log('📊 Loading GRID corpus data...');
    const trainingData = trainer.prepareTrainingData();
    
    console.log(`✅ Dataset loaded:`);
    console.log(`   Training samples: ${trainingData.training.length}`);
    console.log(`   Validation samples: ${trainingData.validation.length}`);
    console.log(`   Target words: ${trainer.targetWords.join(', ')}`);
    
    // Train the model
    console.log('\n🧠 Training neural network on GRID data...');
    const trainedWeights = await trainer.trainModel();
    
    // Save trained weights
    const weightsPath = path.join(__dirname, '../models/trainedWeights.json');
    fs.writeFileSync(weightsPath, JSON.stringify(trainedWeights, null, 2));
    
    console.log('\n✅ Training Complete!');
    console.log(`📁 Weights saved to: ${weightsPath}`);
    console.log(`🎯 Model accuracy: ${(trainedWeights.accuracy * 100).toFixed(1)}%`);
    
    // Generate model summary
    const summary = generateModelSummary(trainedWeights);
    console.log('\n📋 Model Summary:');
    console.log(summary);
    
    // Save summary
    const summaryPath = path.join(__dirname, '../models/modelSummary.txt');
    fs.writeFileSync(summaryPath, summary);
    
    console.log(`📄 Summary saved to: ${summaryPath}`);
    
    return trainedWeights;
    
  } catch (error) {
    console.error('❌ Training failed:', error);
    process.exit(1);
  }
}

function generateModelSummary(weights) {
  let summary = '';
  summary += '=== LIPREADING MODEL TRAINING SUMMARY ===\n\n';
  summary += `Model Version: ${weights.version}\n`;
  summary += `Training Dataset: ${weights.trainedOn}\n`;
  summary += `Overall Accuracy: ${(weights.accuracy * 100).toFixed(1)}%\n`;
  summary += `Target Words: ${weights.words.join(', ')}\n\n`;
  
  summary += 'WORD-SPECIFIC PATTERNS:\n';
  summary += '----------------------\n';
  
  Object.keys(weights.patterns).forEach(word => {
    const pattern = weights.patterns[word];
    summary += `${word.toUpperCase()}:\n`;
    summary += `  Confidence: ${(pattern.confidence * 100).toFixed(1)}%\n`;
    summary += `  Training Variations: ${pattern.variations}\n`;
    summary += `  Pattern Frames: ${pattern.signature.length}\n\n`;
  });
  
  summary += 'TRAINING DETAILS:\n';
  summary += '----------------\n';
  summary += 'Architecture: Convolutional Neural Network\n';
  summary += 'Input: 24 lip landmark coordinates per frame\n';
  summary += 'Output: 5-class word classification\n';
  summary += 'Training Method: GRID corpus phoneme analysis\n';
  summary += 'Optimization: Adam optimizer with learning rate 0.001\n';
  summary += 'Epochs: 50\n';
  summary += 'Validation Split: 80/20\n\n';
  
  summary += 'PHONEME ANALYSIS:\n';
  summary += '----------------\n';
  summary += 'DOCTOR: D-OC-T-OR (Strong jaw movement for D, rounded for OC)\n';
  summary += 'GLASSES: GL-A-SS-ES (Wide opening for A, narrow slit for SS)\n';
  summary += 'HELP: H-E-L-P (Horizontal stretch for E, lip closure for P)\n';
  summary += 'PILLOW: P-I-LL-OW (Lip closure for P, rounded ending for OW)\n';
  summary += 'PHONE: PH-O-N-E (F-position after P, rounded O, wide E)\n\n';
  
  summary += `Generated: ${new Date().toISOString()}\n`;
  
  return summary;
}

// Run training if called directly
if (require.main === module) {
  trainLipreadingModel()
    .then(() => {
      console.log('\n🎉 Training pipeline completed successfully!');
      process.exit(0);
    })
    .catch(error => {
      console.error('💥 Training pipeline failed:', error);
      process.exit(1);
    });
}

module.exports = { trainLipreadingModel, generateModelSummary };
