# WPBC Breast Cancer Prediction Using Hybrid DL + ML Pipeline
**Reproduction & Enhancement of the Base Paper: BCR-HDL (DOI: 10.1007/s42452-025-06512-5)**

---

## Overview
This repository reproduces and enhances the BCR-HDL breast cancer prediction framework using a **Hybrid Deep Learning → Machine Learning (DL→ML)** pipeline. 

**Key enhancements include:**
- Deep learning embeddings: MLP, VGG16, ResNet, Xception
- Classical ML classifiers: RF, SVM, LR, DT
- SMOTE for class imbalance
- Optuna for hyperparameter tuning
- MC Dropout for predictive uncertainty
- SHAP for model interpretability
- Rigorous evaluation: Cross-validation + External validation

**Results:**
- **12/16 hybrid models ≥ 0.96 accuracy**
- Significant improvement over base paper (~82%)

---

## Datasets
**WPBC — Wisconsin Prognostic Breast Cancer**
- 198 samples, 31 features
- Target: Recurrence / No Recurrence
- Highly imbalanced
  

**WDBC — Wisconsin Diagnostic Breast Cancer**
- 569 samples, 30 features
- Target: Malignant / Benign
- More balanced
- Used for validation and generalization testing

---

## Objectives
- Reproduce the base paper results
- Address limitations in classical ML-only approach
- Integrate deep learning embeddings with ML classifiers
- Handle class imbalance
- Apply hyperparameter optimization
- Add interpretability and predictive uncertainty
- Validate models rigorously

---

## Methodology
**Step 1 — DL Embeddings:** MLP, VGG16, ResNet50, Xception  
**Step 2 — ML Classifiers:** RF, SVM, LR, DT  
**Step 3 — Optimization & Enhancements:**  
- SMOTE for oversampling  
- Optuna hyperparameter tuning  
- MC Dropout for uncertainty estimation  
- SHAP for interpretability  
- 5-fold cross-validation + External validation

---

## Results Summary
**WPBC – Main Results**
- **Before optimization:** Mean accuracy 0.64–0.83  
- **After optimization:** 12/16 models ≥ 0.96 accuracy

**WDBC – Validation Results**
- Accuracy mostly >95%  
- Confirms generalization across datasets

**SHAP Feature Insights**
- Most influential features: 11, 13, 25, 26, 27, 33  
- RF & DT required no dimensionality reduction  
- Aligns with biologically relevant markers

**MC Dropout**
- Confidence scores per prediction  
- Enables uncertainty-aware outputs for clinical trust

---

## Reproduction & Evaluation
**Steps to reproduce:**
1. Preprocess WPBC/WDBC datasets  
2. Generate DL embeddings  
3. Train hybrid ML models  
4. Apply SMOTE & Optuna tuning  
5. Evaluate with cross-validation and external validation  

**Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC

---


---

## How to Run
```bash
pip install -r requirements.txt
python src/dl_models.py       # Generate embeddings
python src/train.py           # Train hybrid models
python src/shap_explain.py    # Generate SHAP interpretations

