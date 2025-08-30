#!/usr/bin/env python3
"""
Prototype-Based XGBoost Classifier
Uses prototype similarities as features for XGBoost
"""
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def create_prototype_xgboost_classifier():
    """Create XGBoost using prototype similarities as features"""
    
    print("🔍 PROTOTYPE-BASED XGBOOST CLASSIFIER")
    print("=" * 50)
    
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
    
    print(f"🎯 Prototypes computed from {len(X_train)} training samples")
    
    # Method 1: Simple prototype similarities
    print("\\n🌳 METHOD 1: XGBoost on Prototype Similarities")
    
    train_benign_sim = X_train @ train_benign_proto
    train_malignant_sim = X_train @ train_malignant_proto
    X_train_proto = np.column_stack([train_benign_sim, train_malignant_sim])
    
    test_benign_sim = X_test @ train_benign_proto
    test_malignant_sim = X_test @ train_malignant_proto
    X_test_proto = np.column_stack([test_benign_sim, test_malignant_sim])
    
    # Train XGBoost
    xgb_proto = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_proto.fit(X_train_proto, y_train)
    
    train_pred = xgb_proto.predict(X_train_proto)
    train_proba = xgb_proto.predict_proba(X_train_proto)[:, 1]
    test_pred = xgb_proto.predict(X_test_proto)
    test_proba = xgb_proto.predict_proba(X_test_proto)[:, 1]
    
    train_acc = accuracy_score(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_proba)
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print(f"📊 METHOD 1 RESULTS:")
    print(f"   Training: {train_acc:.4f} accuracy, {train_auc:.4f} AUC")
    print(f"   Test: {test_acc:.4f} accuracy, {test_auc:.4f} AUC")
    print(f"   Overfitting: {train_auc - test_auc:.4f} AUC difference")
    
    # Method 2: Extended features with distances
    print("\\n🌳 METHOD 2: XGBoost on Extended Prototype Features")
    
    # Create extended features
    train_eucl_benign = np.linalg.norm(X_train - train_benign_proto, axis=1)
    train_eucl_malignant = np.linalg.norm(X_train - train_malignant_proto, axis=1)
    test_eucl_benign = np.linalg.norm(X_test - train_benign_proto, axis=1)
    test_eucl_malignant = np.linalg.norm(X_test - train_malignant_proto, axis=1)
    
    # Additional features: differences and ratios
    train_sim_diff = train_malignant_sim - train_benign_sim
    train_dist_diff = train_eucl_benign - train_eucl_malignant
    train_sim_ratio = train_malignant_sim / (train_benign_sim + 1e-8)
    
    test_sim_diff = test_malignant_sim - test_benign_sim
    test_dist_diff = test_eucl_benign - test_eucl_malignant
    test_sim_ratio = test_malignant_sim / (test_benign_sim + 1e-8)
    
    X_train_extended = np.column_stack([
        train_benign_sim, train_malignant_sim,
        train_eucl_benign, train_eucl_malignant,
        train_sim_diff, train_dist_diff, train_sim_ratio
    ])
    X_test_extended = np.column_stack([
        test_benign_sim, test_malignant_sim,
        test_eucl_benign, test_eucl_malignant,
        test_sim_diff, test_dist_diff, test_sim_ratio
    ])
    
    print(f"📏 Extended feature shape: {X_train_extended.shape}")
    
    # Train XGBoost with more features
    xgb_extended = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric='logloss',
        reg_alpha=0.1,
        reg_lambda=0.1
    )
    
    xgb_extended.fit(X_train_extended, y_train)
    
    train_pred_ext = xgb_extended.predict(X_train_extended)
    train_proba_ext = xgb_extended.predict_proba(X_train_extended)[:, 1]
    test_pred_ext = xgb_extended.predict(X_test_extended)
    test_proba_ext = xgb_extended.predict_proba(X_test_extended)[:, 1]
    
    train_acc_ext = accuracy_score(y_train, train_pred_ext)
    train_auc_ext = roc_auc_score(y_train, train_proba_ext)
    test_acc_ext = accuracy_score(y_test, test_pred_ext)
    test_auc_ext = roc_auc_score(y_test, test_proba_ext)
    
    print(f"📊 METHOD 2 RESULTS:")
    print(f"   Training: {train_acc_ext:.4f} accuracy, {train_auc_ext:.4f} AUC")
    print(f"   Test: {test_acc_ext:.4f} accuracy, {test_auc_ext:.4f} AUC")
    print(f"   Overfitting: {train_auc_ext - test_auc_ext:.4f} AUC difference")
    
    # Method 3: Prototype + Top discriminative features
    print("\\n🌳 METHOD 3: XGBoost Hybrid (Prototype + Raw Features)")
    
    # Find most discriminative features
    benign_mean = np.mean(X_train[y_train == 0], axis=0)
    malignant_mean = np.mean(X_train[y_train == 1], axis=0)
    feature_importance = np.abs(malignant_mean - benign_mean)
    top_features = np.argsort(feature_importance)[-100:]  # Top 100 discriminative features
    
    # Combine prototype features with raw features
    X_train_hybrid = np.column_stack([
        X_train_extended,  # All prototype features
        X_train[:, top_features]  # Top 100 raw features
    ])
    X_test_hybrid = np.column_stack([
        X_test_extended,
        X_test[:, top_features]
    ])
    
    print(f"📏 Hybrid feature shape: {X_train_hybrid.shape}")
    
    # Train XGBoost with hybrid features
    xgb_hybrid = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        eval_metric='logloss',
        reg_alpha=0.3,
        reg_lambda=0.3
    )
    
    xgb_hybrid.fit(X_train_hybrid, y_train)
    
    test_pred_hybrid = xgb_hybrid.predict(X_test_hybrid)
    test_proba_hybrid = xgb_hybrid.predict_proba(X_test_hybrid)[:, 1]
    
    test_acc_hybrid = accuracy_score(y_test, test_pred_hybrid)
    test_auc_hybrid = roc_auc_score(y_test, test_proba_hybrid)
    
    print(f"📊 METHOD 3 RESULTS:")
    print(f"   Test: {test_acc_hybrid:.4f} accuracy, {test_auc_hybrid:.4f} AUC")
    
    # Cross-validation on best method
    best_model = xgb_extended if test_auc_ext >= max(test_auc, test_auc_hybrid) else (xgb_hybrid if test_auc_hybrid > test_auc else xgb_proto)
    best_X_train = X_train_extended if test_auc_ext >= max(test_auc, test_auc_hybrid) else (X_train_hybrid if test_auc_hybrid > test_auc else X_train_proto)
    best_auc = max(test_auc, test_auc_ext, test_auc_hybrid)
    best_method = 'extended' if test_auc_ext >= max(test_auc, test_auc_hybrid) else ('hybrid' if test_auc_hybrid > test_auc else 'similarities')
    
    cv_scores = cross_val_score(best_model, best_X_train, y_train, cv=5, scoring='roc_auc')
    print(f"\\n🔄 CROSS-VALIDATION (best method):")
    print(f"   CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance for best model
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        print(f"\\n🎯 FEATURE IMPORTANCES (top features):")
        feature_names = ['cosine_benign', 'cosine_malignant', 'eucl_benign', 'eucl_malignant', 'sim_diff', 'dist_diff', 'sim_ratio']
        if len(importances) > len(feature_names):
            feature_names += [f'raw_feat_{i}' for i in range(len(feature_names), len(importances))]
        
        top_indices = np.argsort(importances)[::-1][:5]
        for idx in top_indices:
            if idx < len(feature_names):
                print(f"   {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Summary comparison
    print(f"\\n🏆 COMPARISON SUMMARY:")
    print(f"   Current Cosine: 0.9254 AUC (nearest centroid)")
    print(f"   XGBoost Method 1: {test_auc:.4f} AUC (similarities only)")
    print(f"   XGBoost Method 2: {test_auc_ext:.4f} AUC (extended features)")
    print(f"   XGBoost Method 3: {test_auc_hybrid:.4f} AUC (hybrid features)")
    
    # Save the best XGBoost model
    prototype_xgb_classifier = {
        'model': best_model,
        'feature_method': best_method,
        'prototypes': {
            'benign': train_benign_proto,
            'malignant': train_malignant_proto
        },
        'top_features': top_features if best_method == 'hybrid' else None,
        'test_accuracy': test_acc_ext if best_method == 'extended' else (test_acc_hybrid if best_method == 'hybrid' else test_acc),
        'test_auc': best_auc,
        'cv_auc': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_importances': dict(zip(feature_names[:len(importances)], importances)) if hasattr(best_model, 'feature_importances_') else None
    }
    
    # Save the prototype-based XGBoost classifier
    with open('/workspace/prototype_xgboost_classifier.pkl', 'wb') as f:
        pickle.dump(prototype_xgb_classifier, f)
    
    print(f"\\n✅ PROTOTYPE-BASED XGBOOST CREATED!")
    print(f"   Best method: {best_method}")
    print(f"   Test AUC: {best_auc:.4f}")
    print(f"   CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"   Saved to: prototype_xgboost_classifier.pkl")
    
    return prototype_xgb_classifier

if __name__ == '__main__':
    result = create_prototype_xgboost_classifier()
    print(f"\\nFinal XGBoost AUC: {result['test_auc']:.4f}")