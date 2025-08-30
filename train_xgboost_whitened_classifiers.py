#!/usr/bin/env python3
"""
Train XGBoost classifiers on whitened embeddings for complete tiered prediction system
"""
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import xgboost as xgb
from collections import Counter

def train_xgboost_classifiers():
    """Train XGBoost on both BreakHis and BACH whitened embeddings"""
    
    print("🚀 Loading whitened prototype cache...")
    with open("/workspace/embeddings_cache_PROTOTYPE_WHITENED.pkl", 'rb') as f:
        cache = pickle.load(f)
    
    features = np.array(cache['combined']['features'])
    labels = cache['combined']['labels'] 
    datasets = cache['combined']['datasets']
    filenames = cache['combined']['filenames']
    
    print(f"✅ Loaded {len(features)} whitened samples")
    print(f"   Feature shape: {features.shape}")
    print(f"   Datasets: {Counter(datasets)}")
    print(f"   Labels: {Counter(labels)}")
    
    # ===== BREAKHIS XGBOOST =====
    print("\n🔬 Training BreakHis XGBoost classifier...")
    
    # Filter BreakHis data
    breakhis_mask = np.array([ds == 'breakhis' for ds in datasets])
    breakhis_features = features[breakhis_mask]
    breakhis_labels = np.array([labels[i] for i in range(len(labels)) if breakhis_mask[i]])
    
    # Convert labels to binary (0=benign, 1=malignant)
    breakhis_y = np.array([1 if label == 'malignant' else 0 for label in breakhis_labels])
    
    print(f"   BreakHis samples: {len(breakhis_features)}")
    print(f"   Class distribution: {Counter(breakhis_y)}")
    
    # Train/test split
    X_train_bh, X_test_bh, y_train_bh, y_test_bh = train_test_split(
        breakhis_features, breakhis_y, test_size=0.2, random_state=42, stratify=breakhis_y
    )
    
    # Train XGBoost
    breakhis_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    breakhis_xgb.fit(X_train_bh, y_train_bh)
    
    # Evaluate
    y_pred_bh = breakhis_xgb.predict(X_test_bh)
    y_proba_bh = breakhis_xgb.predict_proba(X_test_bh)[:, 1]
    
    bh_accuracy = accuracy_score(y_test_bh, y_pred_bh)
    bh_auc = roc_auc_score(y_test_bh, y_proba_bh)
    
    print(f"✅ BreakHis XGBoost trained!")
    print(f"   Accuracy: {bh_accuracy:.4f}")
    print(f"   AUC: {bh_auc:.4f}")
    
    # Cross-validation
    cv_scores_bh = cross_val_score(breakhis_xgb, breakhis_features, breakhis_y, cv=5, scoring='accuracy')
    print(f"   CV Accuracy: {cv_scores_bh.mean():.4f} (+/- {cv_scores_bh.std() * 2:.4f})")
    
    # ===== BACH XGBOOST =====
    print("\n🔬 Training BACH XGBoost classifier...")
    
    # Filter BACH data  
    bach_mask = np.array([ds == 'bach' for ds in datasets])
    bach_features = features[bach_mask]
    bach_labels = np.array([labels[i] for i in range(len(labels)) if bach_mask[i]])
    
    # Convert BACH labels to binary (normal/benign = 0, invasive/insitu = 1)
    bach_y = np.array([1 if label in ['invasive', 'insitu'] else 0 for label in bach_labels])
    
    print(f"   BACH samples: {len(bach_features)}")
    print(f"   Class distribution: {Counter(bach_y)}")
    
    # Train/test split
    X_train_bach, X_test_bach, y_train_bach, y_test_bach = train_test_split(
        bach_features, bach_y, test_size=0.2, random_state=42, stratify=bach_y
    )
    
    # Train XGBoost
    bach_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6, 
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    bach_xgb.fit(X_train_bach, y_train_bach)
    
    # Evaluate
    y_pred_bach = bach_xgb.predict(X_test_bach)
    y_proba_bach = bach_xgb.predict_proba(X_test_bach)[:, 1]
    
    bach_accuracy = accuracy_score(y_test_bach, y_pred_bach)
    bach_auc = roc_auc_score(y_test_bach, y_proba_bach)
    
    print(f"✅ BACH XGBoost trained!")
    print(f"   Accuracy: {bach_accuracy:.4f}")
    print(f"   AUC: {bach_auc:.4f}")
    
    # Cross-validation
    cv_scores_bach = cross_val_score(bach_xgb, bach_features, bach_y, cv=5, scoring='accuracy')
    print(f"   CV Accuracy: {cv_scores_bach.mean():.4f} (+/- {cv_scores_bach.std() * 2:.4f})")
    
    # ===== UPDATE CLASSIFIER FILES =====
    print("\n💾 Updating classifier pickle files...")
    
    # Load and update BreakHis classifiers
    with open("/workspace/breakhis_classifiers_whitened.pkl", 'rb') as f:
        breakhis_classifiers = pickle.load(f)
    
    breakhis_classifiers['xgboost'] = {
        'model': breakhis_xgb,
        'accuracy': bh_accuracy,
        'auc': bh_auc,
        'cv_accuracy': cv_scores_bh.mean(),
        'cv_std': cv_scores_bh.std(),
        'training_samples': len(X_train_bh),
        'test_samples': len(X_test_bh)
    }
    
    with open("/workspace/breakhis_classifiers_whitened.pkl", 'wb') as f:
        pickle.dump(breakhis_classifiers, f)
    
    print("✅ Updated breakhis_classifiers_whitened.pkl with XGBoost")
    
    # Load and update BACH classifiers
    with open("/workspace/bach_classifiers_whitened.pkl", 'rb') as f:
        bach_classifiers = pickle.load(f)
    
    bach_classifiers['xgboost'] = {
        'model': bach_xgb,
        'accuracy': bach_accuracy,
        'auc': bach_auc,
        'cv_accuracy': cv_scores_bach.mean(),
        'cv_std': cv_scores_bach.std(),
        'training_samples': len(X_train_bach),
        'test_samples': len(X_test_bach)
    }
    
    with open("/workspace/bach_classifiers_whitened.pkl", 'wb') as f:
        pickle.dump(bach_classifiers, f)
    
    print("✅ Updated bach_classifiers_whitened.pkl with XGBoost")
    
    # ===== SUMMARY =====
    print("\n🎯 TRAINING COMPLETE!")
    print("=" * 50)
    print("BreakHis Classifiers Available:")
    for name, classifier in breakhis_classifiers.items():
        acc = classifier.get('accuracy', 'N/A')
        print(f"  • {name}: {acc:.4f}" if isinstance(acc, float) else f"  • {name}: {acc}")
    
    print("\nBACH Classifiers Available:")  
    for name, classifier in bach_classifiers.items():
        acc = classifier.get('accuracy', 'N/A')
        print(f"  • {name}: {acc:.4f}" if isinstance(acc, float) else f"  • {name}: {acc}")
    
    print("\n✅ Tiered prediction system now has complete XGBoost support!")
    print("   All classifiers trained on whitened embeddings for consistency")
    
    return {
        'breakhis_xgb_accuracy': bh_accuracy,
        'breakhis_xgb_auc': bh_auc,
        'bach_xgb_accuracy': bach_accuracy,
        'bach_xgb_auc': bach_auc,
        'status': 'complete'
    }

if __name__ == "__main__":
    results = train_xgboost_classifiers()
    print(f"\nFinal results: {results}")