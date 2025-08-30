#!/usr/bin/env python3
"""
Enhanced Prototype-Based Classifier with Correlation Features
Adds Pearson, Spearman, and Distance Correlation features
"""
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

def distance_correlation(x, y):
    """Calculate distance correlation between two vectors"""
    try:
        # Compute distance matrices
        n = len(x)
        if n < 2:
            return 0.0
            
        a = squareform(pdist(x.reshape(-1, 1)))
        b = squareform(pdist(y.reshape(-1, 1)))
        
        # Center the distance matrices
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
        
        # Calculate distance correlation
        dcov_xy = np.sqrt(np.mean(A * B))
        dcov_xx = np.sqrt(np.mean(A * A))
        dcov_yy = np.sqrt(np.mean(B * B))
        
        if dcov_xx > 0 and dcov_yy > 0:
            return dcov_xy / np.sqrt(dcov_xx * dcov_yy)
        else:
            return 0.0
    except:
        return 0.0

def create_enhanced_prototype_classifier():
    """Create enhanced classifier with correlation features"""
    
    print("🔍 ENHANCED PROTOTYPE CLASSIFIER WITH CORRELATIONS")
    print("=" * 65)
    
    # Load whitened cache
    with open("/workspace/embeddings_cache_PROTOTYPE_WHITENED.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features'])
    labels = cache['combined']['labels']
    datasets = cache['combined']['datasets']
    
    # Convert to binary labels
    y = np.array([1 if label in ['malignant', 'invasive', 'insitu'] else 0 for label in labels])
    
    print(f"📊 Total samples: {len(features)}")
    print(f"🏷️ Class distribution: Benign={np.sum(y==0)}, Malignant={np.sum(y==1)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compute prototypes from training data only
    train_benign_proto = np.mean(X_train[y_train == 0], axis=0)
    train_malignant_proto = np.mean(X_train[y_train == 1], axis=0)
    
    print(f"🎯 Computing enhanced features for {len(X_train)} training samples...")
    
    def compute_enhanced_features(X, benign_proto, malignant_proto):
        """Compute comprehensive prototype-based features"""
        n_samples = len(X)
        
        # Basic similarity features
        cosine_benign = X @ benign_proto
        cosine_malignant = X @ malignant_proto
        
        # Distance features
        eucl_benign = np.linalg.norm(X - benign_proto, axis=1)
        eucl_malignant = np.linalg.norm(X - malignant_proto, axis=1)
        
        # Derived features
        sim_diff = cosine_malignant - cosine_benign
        dist_diff = eucl_benign - eucl_malignant
        sim_ratio = cosine_malignant / (cosine_benign + 1e-8)
        dist_ratio = eucl_malignant / (eucl_benign + 1e-8)
        
        # Correlation features (computed per sample)
        pearson_benign = np.zeros(n_samples)
        pearson_malignant = np.zeros(n_samples)
        spearman_benign = np.zeros(n_samples)
        spearman_malignant = np.zeros(n_samples)
        dcor_benign = np.zeros(n_samples)
        dcor_malignant = np.zeros(n_samples)
        
        print(f"   Computing correlation features for {n_samples} samples...")
        
        for i in range(n_samples):
            # Pearson correlations
            try:
                pearson_benign[i], _ = pearsonr(X[i], benign_proto)
                pearson_malignant[i], _ = pearsonr(X[i], malignant_proto)
            except:
                pearson_benign[i] = 0.0
                pearson_malignant[i] = 0.0
            
            # Spearman correlations
            try:
                spearman_benign[i], _ = spearmanr(X[i], benign_proto)
                spearman_malignant[i], _ = spearmanr(X[i], malignant_proto)
            except:
                spearman_benign[i] = 0.0
                spearman_malignant[i] = 0.0
            
            # Distance correlations (simplified version)
            try:
                dcor_benign[i] = distance_correlation(X[i], benign_proto)
                dcor_malignant[i] = distance_correlation(X[i], malignant_proto)
            except:
                dcor_benign[i] = 0.0
                dcor_malignant[i] = 0.0
        
        # Handle NaN values
        pearson_benign = np.nan_to_num(pearson_benign, 0.0)
        pearson_malignant = np.nan_to_num(pearson_malignant, 0.0)
        spearman_benign = np.nan_to_num(spearman_benign, 0.0)
        spearman_malignant = np.nan_to_num(spearman_malignant, 0.0)
        dcor_benign = np.nan_to_num(dcor_benign, 0.0)
        dcor_malignant = np.nan_to_num(dcor_malignant, 0.0)
        
        # Correlation-derived features
        pearson_diff = pearson_malignant - pearson_benign
        spearman_diff = spearman_malignant - spearman_benign
        dcor_diff = dcor_malignant - dcor_benign
        
        # Combine all features
        enhanced_features = np.column_stack([
            # Basic similarity/distance
            cosine_benign, cosine_malignant, eucl_benign, eucl_malignant,
            # Derived features
            sim_diff, dist_diff, sim_ratio, dist_ratio,
            # Correlation features
            pearson_benign, pearson_malignant, spearman_benign, spearman_malignant,
            dcor_benign, dcor_malignant,
            # Correlation differences
            pearson_diff, spearman_diff, dcor_diff
        ])
        
        return enhanced_features
    
    # Compute enhanced features
    X_train_enhanced = compute_enhanced_features(X_train, train_benign_proto, train_malignant_proto)
    X_test_enhanced = compute_enhanced_features(X_test, train_benign_proto, train_malignant_proto)
    
    print(f"📏 Enhanced feature shape: {X_train_enhanced.shape}")
    
    # Feature names for interpretation
    feature_names = [
        'cosine_benign', 'cosine_malignant', 'eucl_benign', 'eucl_malignant',
        'sim_diff', 'dist_diff', 'sim_ratio', 'dist_ratio',
        'pearson_benign', 'pearson_malignant', 'spearman_benign', 'spearman_malignant',
        'dcor_benign', 'dcor_malignant',
        'pearson_diff', 'spearman_diff', 'dcor_diff'
    ]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enhanced)
    X_test_scaled = scaler.transform(X_test_enhanced)
    
    # Train XGBoost with correlation features
    print("\\n🌳 TRAINING ENHANCED XGBOOST...")
    xgb_enhanced = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        reg_alpha=0.2,
        reg_lambda=0.2
    )
    
    xgb_enhanced.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = xgb_enhanced.predict(X_train_scaled)
    train_proba = xgb_enhanced.predict_proba(X_train_scaled)[:, 1]
    test_pred = xgb_enhanced.predict(X_test_scaled)
    test_proba = xgb_enhanced.predict_proba(X_test_scaled)[:, 1]
    
    train_acc = accuracy_score(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_proba)
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print(f"\\n📊 ENHANCED XGBOOST RESULTS:")
    print(f"   Training: {train_acc:.4f} accuracy, {train_auc:.4f} AUC")
    print(f"   Test: {test_acc:.4f} accuracy, {test_auc:.4f} AUC")
    print(f"   Overfitting: {train_auc - test_auc:.4f} AUC difference")
    
    # Cross-validation
    cv_scores = cross_val_score(xgb_enhanced, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"   CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance analysis
    importances = xgb_enhanced.feature_importances_
    
    print(f"\\n🎯 TOP FEATURE IMPORTANCES:")
    top_indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(top_indices):
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Group importance by feature type
    correlation_importance = np.sum([importances[i] for i, name in enumerate(feature_names) if 'pearson' in name or 'spearman' in name or 'dcor' in name])
    similarity_importance = np.sum([importances[i] for i, name in enumerate(feature_names) if 'cosine' in name])
    distance_importance = np.sum([importances[i] for i, name in enumerate(feature_names) if 'eucl' in name])
    derived_importance = np.sum([importances[i] for i, name in enumerate(feature_names) if 'diff' in name or 'ratio' in name])
    
    print(f"\\n📊 IMPORTANCE BY FEATURE TYPE:")
    print(f"   Correlation features: {correlation_importance:.4f} ({correlation_importance*100:.1f}%)")
    print(f"   Distance features: {distance_importance:.4f} ({distance_importance*100:.1f}%)")
    print(f"   Derived features: {derived_importance:.4f} ({derived_importance*100:.1f}%)")
    print(f"   Similarity features: {similarity_importance:.4f} ({similarity_importance*100:.1f}%)")
    
    # Final comparison
    print(f"\\n🏆 PERFORMANCE COMPARISON:")
    print(f"   Original Cosine: 0.9254 AUC")
    print(f"   Prototype Logistic: 0.9532 AUC")
    print(f"   Prototype XGBoost: 0.9432 AUC")
    print(f"   Enhanced XGBoost: {test_auc:.4f} AUC ⭐")
    
    improvement = test_auc - 0.9254
    print(f"   Improvement: +{improvement:.4f} AUC points ({improvement*100:.1f}%)")
    
    # Save enhanced classifier
    enhanced_classifier = {
        'model': xgb_enhanced,
        'scaler': scaler,
        'prototypes': {
            'benign': train_benign_proto,
            'malignant': train_malignant_proto
        },
        'feature_names': feature_names,
        'feature_importances': dict(zip(feature_names, importances)),
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'cv_auc': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'correlation_importance': correlation_importance,
        'method': 'enhanced_prototype_xgboost_with_correlations'
    }
    
    with open('/workspace/enhanced_prototype_classifier.pkl', 'wb') as f:
        pickle.dump(enhanced_classifier, f)
    
    print(f"\\n✅ ENHANCED PROTOTYPE CLASSIFIER CREATED!")
    print(f"   Features: {len(feature_names)} (including Pearson, Spearman, dcor)")
    print(f"   Test AUC: {test_auc:.4f}")
    print(f"   Correlation features contribute: {correlation_importance*100:.1f}% importance")
    print(f"   Saved to: enhanced_prototype_classifier.pkl")
    
    return enhanced_classifier

if __name__ == '__main__':
    result = create_enhanced_prototype_classifier()
    print(f"\\nFinal Enhanced AUC: {result['test_auc']:.4f}")