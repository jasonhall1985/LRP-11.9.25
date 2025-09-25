# 📊 **CHECKPOINT 165 BASELINE ASSESSMENT REPORT**

**Date:** September 25, 2025  
**Model:** Enhanced Lightweight CNN-LSTM (81.65% validation accuracy)  
**Test Set:** Desktop folder 'test set 24.9.25' (18 videos)  
**Configuration:** Uncalibrated predictions, Enhanced Reliability Gate V2.0, TTA enabled

---

## 🎯 **EXECUTIVE SUMMARY**

The checkpoint 165 model, despite achieving **81.65% validation accuracy** during training, demonstrates **severe performance degradation** on the independent test set with only **8.33% accuracy** on 4-class predictions. This represents a **critical failure** indicating the model is **NOT suitable for production deployment**.

### **Key Findings:**
- ❌ **8.33% accuracy** (1/12 correct predictions on 4-class system)
- ❌ **Severe class collapse** - model only predicts 2 out of 4 classes
- ❌ **Extreme prediction bias** toward 'i_need_to_move' and 'pillow'
- ❌ **Poor confidence calibration** - incorrect predictions have higher confidence
- ❌ **Massive domain gap** between training and test performance

---

## 📋 **TEST EXECUTION SUMMARY**

### **STEP 1: Preprocessing Verification** ✅
- **Status:** COMPLETE - Videos already preprocessed
- **Pipeline:** Training-compatible (OpenCV face detection, 32 frames, 96×64, grayscale, [0,1] normalization)
- **Success Rate:** 18/18 videos (100%)
- **4-Class Videos:** 12/18 (doctor: 5, i_need_to_move: 2, my_mouth_is_dry: 2, pillow: 3)
- **Out-of-Distribution:** 6/18 (glasses: 2, help: 3, phone: 1)

### **STEP 2: Model Testing Configuration** ✅
- **Model:** Checkpoint 165 (721,044 parameters)
- **Calibration:** DISABLED (uncalibrated baseline)
- **TTA:** Enabled (5 temporal windows + horizontal flips)
- **Temperature:** 1.0 (checkpoint 165 setting)
- **Reliability Gate:** Enhanced V2.0 with 50% confidence floor

### **STEP 3: Evaluation Protocol** ✅
- **Direct Model Testing:** Bypassed server preprocessing
- **Ground Truth Extraction:** From video filenames
- **Metrics Calculated:** Accuracy, confidence, reliability, confusion matrix

---

## 📊 **DETAILED RESULTS**

### **🎯 Overall Performance**
| Metric | Value | Status |
|--------|-------|--------|
| **4-Class Accuracy** | **8.33%** (1/12) | ❌ **CRITICAL FAILURE** |
| **Reliability Gate Pass Rate** | 72.22% (13/18) | ⚠️ High but ineffective |
| **Total Videos Tested** | 18 | ✅ Complete |
| **4-Class System Videos** | 12 | ✅ Adequate sample |

### **📈 Per-Class Performance**
| Class | Correct | Total | Accuracy | Status |
|-------|---------|-------|----------|--------|
| **doctor** | 0 | 5 | **0.0%** | ❌ **COMPLETE FAILURE** |
| **i_need_to_move** | 0 | 2 | **0.0%** | ❌ **COMPLETE FAILURE** |
| **my_mouth_is_dry** | 0 | 2 | **0.0%** | ❌ **COMPLETE FAILURE** |
| **pillow** | 1 | 3 | **33.3%** | ⚠️ **POOR** |

### **🎯 Confusion Matrix**
```
Actual \ Predicted    doctor  i_need_to_move  my_mouth_is_dry  pillow  | Total
doctor                   0           4              0           1      |   5
i_need_to_move           0           0              0           2      |   2
my_mouth_is_dry          0           0              0           2      |   2
pillow                   0           2              0           1      |   3
Total                    0           6              0           6      |  12
```

### **⚖️ Prediction Bias Analysis**
| Predicted Class | Count | Percentage | Bias Level |
|----------------|-------|------------|------------|
| **i_need_to_move** | 6/12 | **50.0%** | ❌ **EXTREME BIAS** |
| **pillow** | 6/12 | **50.0%** | ❌ **EXTREME BIAS** |
| **doctor** | 0/12 | **0.0%** | ❌ **NEVER PREDICTED** |
| **my_mouth_is_dry** | 0/12 | **0.0%** | ❌ **NEVER PREDICTED** |

### **🎯 Confidence Analysis**
| Prediction Type | Average Confidence | Standard Deviation | Issue |
|----------------|-------------------|-------------------|-------|
| **Correct Predictions** | 0.495 | 0.000 | ⚠️ Low confidence |
| **Incorrect Predictions** | 0.535 | 0.082 | ❌ **HIGHER than correct** |
| **Confidence Separation** | -0.040 | - | ❌ **NEGATIVE** (inverted) |

---

## 🔍 **CRITICAL ISSUES IDENTIFIED**

### **1. SEVERE CLASS COLLAPSE** ❌
- Model completely ignores 'doctor' and 'my_mouth_is_dry' classes
- Only predicts 'i_need_to_move' and 'pillow'
- Indicates fundamental learning failure

### **2. EXTREME PREDICTION BIAS** ❌
- 50% bias toward 'i_need_to_move'
- 50% bias toward 'pillow'
- No diversity in predictions

### **3. INVERTED CONFIDENCE CALIBRATION** ❌
- Incorrect predictions have **higher** confidence than correct ones
- Reliability gate fails to filter bad predictions
- Model is overconfident in wrong answers

### **4. MASSIVE DOMAIN GAP** ❌
- **73.32 percentage point drop** from validation (81.65%) to test (8.33%)
- Suggests severe overfitting or domain mismatch
- Model memorized training patterns rather than learning generalizable features

### **5. RELIABILITY GATE INEFFECTIVENESS** ⚠️
- 72% pass rate but only 8.3% accuracy
- High confidence in wrong predictions defeats the purpose
- Gate provides false sense of reliability

---

## 📊 **COMPARISON WITH PREVIOUS RESULTS**

| Test Configuration | Accuracy | Consistency |
|-------------------|----------|-------------|
| **Previous Test (Calibration Disabled)** | 8.3% | ✅ |
| **Current Test (Checkpoint 165 Baseline)** | 8.3% | ✅ |
| **Validation Performance (Training)** | 81.65% | ❌ **MASSIVE GAP** |

**✅ CONSISTENCY CONFIRMED:** Results are identical, confirming the model's actual performance.

---

## 💡 **RECOMMENDATIONS**

### **🚨 IMMEDIATE ACTIONS**
1. **❌ DO NOT DEPLOY** - Model is unsuitable for production
2. **🔄 HALT CURRENT DEVELOPMENT** - Focus on fundamental issues
3. **📊 AUDIT TRAINING DATA** - Investigate validation set quality
4. **🎯 REASSESS ARCHITECTURE** - Current model may be fundamentally flawed

### **🔧 TRAINING IMPROVEMENTS**
1. **📈 INCREASE DATASET SIZE** - Current dataset likely insufficient
2. **🌍 IMPROVE DIVERSITY** - Add cross-demographic samples
3. **⚖️ IMPLEMENT STRONGER REGULARIZATION** - Prevent overfitting
4. **🔄 USE PROPER CROSS-VALIDATION** - Ensure robust evaluation
5. **📊 CONTINUOUS TEST SET EVALUATION** - Monitor real performance

### **🎯 EVALUATION PROTOCOL**
1. **📋 ESTABLISH PROPER SPLITS** - Stratified train/val/test
2. **🎯 DEMOGRAPHIC BALANCE** - Ensure representative samples
3. **📊 REGULAR TESTING** - Continuous evaluation on held-out data
4. **🔍 DOMAIN ANALYSIS** - Identify and address domain gaps

### **💾 DATA STRATEGY**
1. **🔄 REPURPOSE CALIBRATION DATA** - Use as additional training data
2. **📈 EXPAND COLLECTION** - Gather more diverse samples
3. **🎯 QUALITY CONTROL** - Improve data annotation and validation
4. **⚖️ BALANCE CLASSES** - Ensure equal representation

---

## 🏁 **CONCLUSION**

The checkpoint 165 baseline assessment reveals **critical model failure** with only **8.33% accuracy** on the test set, despite **81.65% validation accuracy** during training. This represents a **production-blocking issue** requiring immediate attention.

### **Status:** ❌ **CRITICAL FAILURE - NOT PRODUCTION READY**

The model demonstrates severe class collapse, extreme prediction bias, and inverted confidence calibration. The massive performance gap between validation and test sets indicates fundamental issues with the training process, data quality, or model architecture.

**Next Steps:** Focus on comprehensive model retraining with improved data diversity, stronger regularization, and proper evaluation protocols before considering any production deployment.

---

**Report Generated:** September 25, 2025  
**Test Duration:** 6 seconds  
**Files Generated:** 
- `checkpoint_165_baseline_results_20250925_221239.json`
- `checkpoint_165_baseline_test.py`
- `analyze_baseline_results.py`
