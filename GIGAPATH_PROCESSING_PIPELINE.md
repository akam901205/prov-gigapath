# GigaPath Processing Pipeline - Detailed Documentation

## Overview
This document describes the complete step-by-step processing pipeline for pathology image analysis using Microsoft's GigaPath foundation model with dual classifier validation.

## Complete Processing Pipeline

### **Stage 1: Image Upload & Preprocessing**

#### Step 1: Image Upload
- **Frontend:** User uploads pathology image (PNG, JPG, TIFF)
- **Format:** Convert to base64 encoding
- **API Call:** POST to `/api/single-image-analysis`

#### Step 2: Image Preprocessing
- **Load Image:** Convert base64 → PIL Image → RGB format
- **Resize:** Standard preprocessing for GigaPath model
- **Tensor Creation:** Apply transforms (resize, center crop, normalize)

#### Step 3: GigaPath Foundation Model Loading
- **Model:** `prov-gigapath/prov-gigapath` (1.13B parameters)
- **Authentication:** Hugging Face token required for gated model
- **Device:** CUDA if available, CPU fallback
- **Loading:** On-demand loading with global caching

### **Stage 2: Feature Extraction**

#### Step 4: GigaPath Inference
- **Input:** Preprocessed image tensor (224x224x3)
- **Model:** GigaPath tile encoder
- **Inference:** `encoder(input_tensor)` with `torch.no_grad()`
- **Output:** 1536-dimensional feature vector

#### Step 5: Feature Normalization
- **L2 Normalization:** `l2_features = features / np.linalg.norm(features)`
- **Purpose:** Standardize feature magnitude for similarity calculations
- **Result:** Unit-length 1536-dimensional vector

### **Stage 3: BACH Classifier Loading**

#### Step 6: Load Pre-trained Models
- **File:** `/workspace/bach_logistic_model.pkl`
- **Contents:**
  - Logistic Regression model (One-vs-Rest)
  - SVM RBF model (C=1.0, gamma='scale')
  - Label encoder (normal/benign/insitu/invasive)
  - Cross-validation scores
  - ROC curve data

#### Step 7: Model Validation
- **Check:** Verify both models loaded successfully
- **Training Data:** 100 BACH samples (25 per class)
- **Performance:**
  - Logistic Regression: 94.0% ± 3.7% CV accuracy
  - SVM RBF: 92.0% ± 4.0% CV accuracy

### **Stage 4: Dual Classification**

#### Step 8: Logistic Regression Classification
- **Method:** `bach_classifier.predict(l2_features)`
- **Algorithm:** One-vs-Rest Logistic Regression
- **Input:** 1536-dim L2-normalized GigaPath features
- **Output:**
  ```python
  {
    "predicted_class": "normal",
    "confidence": 0.87,
    "probabilities": {
      "normal": 0.87,
      "benign": 0.08, 
      "insitu": 0.03,
      "invasive": 0.02
    }
  }
  ```

#### Step 9: SVM RBF Classification (Parallel)
- **Method:** `bach_classifier.predict_svm(l2_features)`
- **Algorithm:** Support Vector Machine with RBF kernel
- **Input:** Same 1536-dim L2-normalized GigaPath features
- **Output:**
  ```python
  {
    "predicted_class": "normal",
    "confidence": 0.82,
    "probabilities": {
      "normal": 0.82,
      "benign": 0.12,
      "insitu": 0.04, 
      "invasive": 0.02
    }
  }
  ```

### **Stage 5: Hierarchical Consensus Analysis**

#### Step 10: BreakHis 3-Method Analysis
- **Dataset:** BreakHis breast cancer images (malignant/benign)
- **Method 1 - Cosine Similarity:**
  - Calculate similarity with all BreakHis features
  - Find top matching sample → get its label
- **Method 2 - Pearson Correlation:**
  - Calculate linear correlation with all BreakHis features  
  - Find highest correlation → get its label
- **Method 3 - Spearman Correlation:**
  - Calculate rank-based correlation with all BreakHis features
  - Find highest correlation → get its label

#### Step 11: BreakHis Consensus
- **Input:** 3 top-match labels from different methods
- **Logic:** Majority vote of the 3 methods
- **Output:** `breakhis_consensus` = "malignant" or "benign"

#### Step 12: Filtered BACH Classification
- **Filtering Logic:**
  - If BreakHis consensus = "benign" → Only consider normal/benign BACH samples
  - If BreakHis consensus = "malignant" → Only consider invasive/insitu BACH samples
- **Method:** Cosine similarity with filtered BACH samples only
- **Output:** Most similar sample from relevant category

### **Stage 6: Confidence Calculation**

#### Step 13: Multi-Source Confidence
- **Source 1:** BACH classifier confidence (Logistic or SVM)
- **Source 2:** Filtered BACH similarity score
- **Source 3:** Method agreement score

#### Step 14: Agreement Bonus
- **Perfect Agreement (3/3 methods):** +25% confidence boost
- **Partial Agreement (2/3 methods):** +15% confidence boost  
- **No Agreement:** Use base confidence only

#### Step 15: Confidence Level Assignment
- **HIGH:** >75% final confidence
- **MODERATE:** 55-75% final confidence
- **LOW:** <55% final confidence

### **Stage 7: Dimensionality Reduction & Visualization**

#### Step 16: UMAP Projection
- **Algorithm:** UMAP with supervised learning
- **Parameters:**
  - `n_neighbors=15`
  - `min_dist=0.0` (tight clustering)
  - `metric='cosine'`
  - `target_metric='categorical'` (supervised)
- **Input:** L2-normalized features + labels
- **Output:** 2D coordinates for visualization

#### Step 17: t-SNE Projection (Optimized)
- **Algorithm:** t-SNE with enhanced parameters
- **Parameters:**
  - `perplexity=50` (increased for better structure)
  - `learning_rate=200.0` (higher for separation)
  - `early_exaggeration=24.0` (enhanced clustering)
  - `max_iter=1500` (more iterations)
- **Post-processing:** Supervised separation enhancement
- **Output:** 2D coordinates with improved cluster separation

#### Step 18: PCA Projection
- **Algorithm:** Principal Component Analysis
- **Components:** 2D projection
- **Enhancement:** Cluster center separation boost
- **Output:** 2D coordinates showing linear variance

### **Stage 8: Response Assembly**

#### Step 19: Domain-Invariant Analysis
- **Combined Dataset:** BreakHis + BACH embeddings
- **Coordinates:** UMAP, t-SNE, PCA projections
- **New Image Position:** Projected onto existing space
- **Closest Matches:** Top 10 most similar from combined dataset

#### Step 20: Dataset-Specific Analysis
- **BreakHis Analysis:** Projection onto BreakHis-only space
- **BACH Analysis:** Projection onto BACH-only space
- **Separate Visualizations:** Independent coordinate systems

#### Step 21: GigaPath Verdict Assembly
```json
"gigapath_verdict": {
  "logistic_regression": {
    "predicted_class": "normal",
    "confidence": 0.87,
    "probabilities": {...}
  },
  "svm_rbf": {
    "predicted_class": "normal", 
    "confidence": 0.82,
    "probabilities": {...}
  },
  "roc_plot_base64": "...",
  "model_info": {...},
  "feature_analysis": {...},
  "interpretation": {...},
  "risk_indicators": {...}
}
```

#### Step 22: Diagnostic Verdict Assembly
- **Method Predictions:** Individual method results
- **Hierarchical Details:** Step-by-step consensus process
- **Summary:** Final diagnostic recommendation
- **Confidence Metrics:** Agreement status and classification method

### **Stage 9: Frontend Visualization**

#### Step 23: Tab-Based Display
- **Tab 1:** Domain-Invariant (combined dataset visualization)
- **Tab 2:** BreakHis Analysis (malignant/benign focus)
- **Tab 3:** BACH Analysis (4-class subtype focus)
- **Tab 4:** GigaPath Verdict (classifier results)
- **Tab 5:** Diagnostic Verdict (final consensus)

#### Step 24: Interactive Visualization
- **UMAP/t-SNE/PCA plots** with cached + new image points
- **Color coding** by diagnostic labels
- **Hover information** with similarity scores
- **Closest match displays** with confidence metrics

## **Key Technical Details**

### **Model Training (One-time Setup)**
```python
# Training Data: 100 BACH samples (25 per class)
features, labels = load_bach_data()

# Logistic Regression
lr_model = LogisticRegression(multi_class='ovr', solver='liblinear')
lr_cv_scores = cross_val_score(lr_model, features, labels, cv=5)

# SVM RBF  
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_cv_scores = cross_val_score(svm_model, features, labels, cv=5)

# Save both models
pickle.dump({
  'model': lr_model,
  'svm_model': svm_model,
  'cv_scores': lr_cv_scores,
  'svm_cv_scores': svm_cv_scores
}, file)
```

### **Runtime Inference (Per Image)**
```python
# 1. Extract GigaPath features
gigapath_features = encoder(image)  # 1536-dim

# 2. Normalize
l2_features = features / np.linalg.norm(features)

# 3. Dual prediction
lr_result = classifier.predict(l2_features)
svm_result = classifier.predict_svm(l2_features)

# 4. Hierarchical consensus (independent)
consensus = hierarchical_analysis(cached_features, l2_features)
```

### **Data Flow Validation**
- **No hardcoded values** in classifier predictions
- **Real probabilities** from trained models
- **Actual confidence scores** from model outputs
- **Dynamic confidence levels** based on method agreement
- **Real similarity scores** from cosine distance calculations

This ensures every image gets **unique, data-driven predictions** rather than static fallback values.