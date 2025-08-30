#!/usr/bin/env python3
"""
Prototype-Based Logistic Regression Classifier
Uses prototype similarities as features for logistic regression
"""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def create_prototype_logistic_classifier():
    """Create logistic regression using prototype similarities as features"""
    
    print("🔍 PROTOTYPE-BASED LOGISTIC REGRESSION")
    print("=" * 55)
    
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
    
    # Method 1: Use cosine similarities to prototypes as features
    print("\n🎯 METHOD 1: Cosine Similarities as Features")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compute prototypes from training data only
    train_benign_proto = np.mean(X_train[y_train == 0], axis=0)
    train_malignant_proto = np.mean(X_train[y_train == 1], axis=0)
    
    # Create prototype-based features for training set
    train_benign_sim = X_train @ train_benign_proto
    train_malignant_sim = X_train @ train_malignant_proto
    X_train_proto = np.column_stack([train_benign_sim, train_malignant_sim])
    
    # Create prototype-based features for test set
    test_benign_sim = X_test @ train_benign_proto
    test_malignant_sim = X_test @ train_malignant_proto
    X_test_proto = np.column_stack([test_benign_sim, test_malignant_sim])
    
    print(f"📏 Prototype feature shape: {X_train_proto.shape}")
    
    # Train logistic regression on prototype similarities
    lr_proto = LogisticRegression(random_state=42, max_iter=1000)
    lr_proto.fit(X_train_proto, y_train)
    
    # Evaluate
    train_pred = lr_proto.predict(X_train_proto)
    train_proba = lr_proto.predict_proba(X_train_proto)[:, 1]
    test_pred = lr_proto.predict(X_test_proto)
    test_proba = lr_proto.predict_proba(X_test_proto)[:, 1]
    
    train_acc = accuracy_score(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_proba)
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print(f"\\n📊 METHOD 1 RESULTS:")
    print(f"   Training: {train_acc:.4f} accuracy, {train_auc:.4f} AUC")
    print(f"   Test: {test_acc:.4f} accuracy, {test_auc:.4f} AUC")
    print(f"   Overfitting: {train_auc - test_auc:.4f} AUC difference")
    
    # Method 2: Use prototype distances + raw features
    print("\\n🎯 METHOD 2: Prototype Distances + Original Features")
    
    # Create extended features: [cosine_benign, cosine_malignant, euclidean_benign, euclidean_malignant]
    train_eucl_benign = np.linalg.norm(X_train - train_benign_proto, axis=1)
    train_eucl_malignant = np.linalg.norm(X_train - train_malignant_proto, axis=1)
    test_eucl_benign = np.linalg.norm(X_test - train_benign_proto, axis=1)
    test_eucl_malignant = np.linalg.norm(X_test - train_malignant_proto, axis=1)
    
    X_train_extended = np.column_stack([
        train_benign_sim, train_malignant_sim, 
        train_eucl_benign, train_eucl_malignant
    ])
    X_test_extended = np.column_stack([
        test_benign_sim, test_malignant_sim,
        test_eucl_benign, test_eucl_malignant
    ])
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_extended)
    X_test_scaled = scaler.transform(X_test_extended)
    
    # Train logistic regression
    lr_extended = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    lr_extended.fit(X_train_scaled, y_train)
    
    train_pred_ext = lr_extended.predict(X_train_scaled)
    train_proba_ext = lr_extended.predict_proba(X_train_scaled)[:, 1]
    test_pred_ext = lr_extended.predict(X_test_scaled)
    test_proba_ext = lr_extended.predict_proba(X_test_scaled)[:, 1]
    
    train_acc_ext = accuracy_score(y_train, train_pred_ext)
    train_auc_ext = roc_auc_score(y_train, train_proba_ext)
    test_acc_ext = accuracy_score(y_test, test_pred_ext)
    test_auc_ext = roc_auc_score(y_test, test_proba_ext)
    
    print(f"📊 METHOD 2 RESULTS:")
    print(f"   Training: {train_acc_ext:.4f} accuracy, {train_auc_ext:.4f} AUC")
    print(f"   Test: {test_acc_ext:.4f} accuracy, {test_auc_ext:.4f} AUC")
    print(f"   Overfitting: {train_auc_ext - test_auc_ext:.4f} AUC difference")
    
    # Cross-validation
    cv_scores = cross_val_score(lr_extended, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"   CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Method 3: Hybrid approach with top features
    print("\\n🎯 METHOD 3: Prototype + Top Differential Features")
    
    # Find most discriminative features
    benign_mean = np.mean(X_train[y_train == 0], axis=0)
    malignant_mean = np.mean(X_train[y_train == 1], axis=0)
    feature_importance = np.abs(malignant_mean - benign_mean)
    top_features = np.argsort(feature_importance)[-50:]  # Top 50 discriminative features
    
    # Combine prototype similarities with top features
    X_train_hybrid = np.column_stack([
        X_train_proto,  # Prototype similarities
        X_train[:, top_features]  # Top 50 raw features
    ])
    X_test_hybrid = np.column_stack([
        X_test_proto,
        X_test[:, top_features] 
    ])
    
    # Scale and train
    scaler_hybrid = StandardScaler()
    X_train_hybrid_scaled = scaler_hybrid.fit_transform(X_train_hybrid)
    X_test_hybrid_scaled = scaler_hybrid.transform(X_test_hybrid)
    
    lr_hybrid = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
    lr_hybrid.fit(X_train_hybrid_scaled, y_train)
    
    test_pred_hybrid = lr_hybrid.predict(X_test_hybrid_scaled)
    test_proba_hybrid = lr_hybrid.predict_proba(X_test_hybrid_scaled)[:, 1]
    
    test_acc_hybrid = accuracy_score(y_test, test_pred_hybrid)
    test_auc_hybrid = roc_auc_score(y_test, test_proba_hybrid)
    
    print(f"📊 METHOD 3 RESULTS:")
    print(f"   Test: {test_acc_hybrid:.4f} accuracy, {test_auc_hybrid:.4f} AUC")
    
    # Summary comparison
    print(f"\\n🏆 COMPARISON SUMMARY:")
    print(f"   Current (Cosine): {0.9254:.4f} AUC (nearest centroid)")
    print(f"   Method 1 (LR Proto): {test_auc:.4f} AUC (logistic on similarities)")
    print(f"   Method 2 (LR Extended): {test_auc_ext:.4f} AUC (logistic on distances)")
    print(f"   Method 3 (LR Hybrid): {test_auc_hybrid:.4f} AUC (logistic on proto+features)")
    
    # Save the best logistic model
    best_model = lr_extended if test_auc_ext >= max(test_auc, test_auc_hybrid) else (lr_hybrid if test_auc_hybrid > test_auc else lr_proto)
    best_scaler = scaler if test_auc_ext >= max(test_auc, test_auc_hybrid) else (scaler_hybrid if test_auc_hybrid > test_auc else None)
    best_auc = max(test_auc, test_auc_ext, test_auc_hybrid)
    
    prototype_lr_classifier = {
        'model': best_model,
        'scaler': best_scaler,
        'prototypes': {
            'benign': train_benign_proto,
            'malignant': train_malignant_proto
        },
        'feature_method': 'extended' if test_auc_ext >= max(test_auc, test_auc_hybrid) else ('hybrid' if test_auc_hybrid > test_auc else 'similarities'),
        'test_accuracy': test_acc_ext if test_auc_ext >= max(test_auc, test_auc_hybrid) else (test_acc_hybrid if test_auc_hybrid > test_auc else test_acc),
        'test_auc': best_auc,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    # Save the prototype-based logistic classifier
    with open('/workspace/prototype_logistic_classifier.pkl', 'wb') as f:
        pickle.dump(prototype_lr_classifier, f)
    
    print(f"\\n✅ PROTOTYPE-BASED LOGISTIC REGRESSION CREATED!")
    print(f"   Best method: {prototype_lr_classifier['feature_method']}")
    print(f"   Test AUC: {best_auc:.4f}")
    print(f"   Saved to: prototype_logistic_classifier.pkl")
    
    return prototype_lr_classifier

if __name__ == '__main__':
    create_prototype_logistic_classifier()