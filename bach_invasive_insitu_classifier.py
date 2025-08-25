"""
BACH Invasive vs InSitu Binary Classifier
Specialized binary classification for malignant tissue subtypes
Uses same embedding cache and train/validation/test methodology
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class BACHInvasiveInsituClassifier:
    def __init__(self, cache_file='/workspace/embeddings_cache_4_CLUSTERS_FIXED_TSNE.pkl'):
        self.cache_file = cache_file
        self.lr_model = None
        self.svm_model = None
        self.xgb_model = None
        self.label_encoder = None
        self.test_scores_lr = None
        self.test_scores_svm = None
        self.test_scores_xgb = None
        self.test_roc_data_lr = None
        self.test_roc_data_svm = None
        self.test_roc_data_xgb = None
        self.data_splits = None
        self.class_names = ['insitu', 'invasive']
        
    def load_bach_invasive_insitu_data(self):
        """Load only invasive and insitu samples from BACH dataset"""
        print("Loading BACH invasive vs insitu data from cache...")
        
        with open(self.cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        # Extract BACH data only
        combined_data = cache['combined']
        bach_indices = [i for i, ds in enumerate(combined_data['datasets']) if ds == 'bach']
        
        if not bach_indices:
            raise ValueError("No BACH data found in cache")
        
        # Filter for only invasive and insitu samples
        invasive_insitu_indices = []
        for idx in bach_indices:
            label = combined_data['labels'][idx]
            if label in ['invasive', 'insitu']:
                invasive_insitu_indices.append(idx)
        
        if not invasive_insitu_indices:
            raise ValueError("No invasive/insitu samples found in BACH data")
        
        # Get filtered features and labels
        features = np.array([combined_data['features'][i] for i in invasive_insitu_indices])
        labels = [combined_data['labels'][i] for i in invasive_insitu_indices]
        filenames = [combined_data['filenames'][i] for i in invasive_insitu_indices]
        
        print(f"Loaded BACH invasive vs insitu data: {len(features)} samples")
        print(f"Classes: {set(labels)}")
        print(f"Class distribution: {[(label, labels.count(label)) for label in set(labels)]}")
        
        return features, labels, filenames
    
    def train_classifiers(self):
        """Train all three classifiers for invasive vs insitu"""
        features, labels, filenames = self.load_bach_invasive_insitu_data()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        
        print(f"Total samples: {len(features)}")
        print(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Train/validation/test splits (80%/10%/10% due to smaller dataset)
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )
        
        self.data_splits = {
            'train_size': len(X_train),
            'val_size': len(X_val), 
            'test_size': len(X_test),
            'train_distribution': [(cls, sum(y_train == i)) for i, cls in enumerate(self.label_encoder.classes_)],
            'val_distribution': [(cls, sum(y_val == i)) for i, cls in enumerate(self.label_encoder.classes_)],
            'test_distribution': [(cls, sum(y_test == i)) for i, cls in enumerate(self.label_encoder.classes_)]
        }
        
        print(f"\n📊 BACH INVASIVE vs INSITU SPLITS:")
        print(f"  Train: {len(X_train)} samples - {self.data_splits['train_distribution']}")
        print(f"  Validation: {len(X_val)} samples - {self.data_splits['val_distribution']}")
        print(f"  Test: {len(X_test)} samples - {self.data_splits['test_distribution']}")
        
        # Train all three classifiers
        self._train_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test)
        self._train_svm(X_train, X_val, X_test, y_train, y_val, y_test)
        if XGBOOST_AVAILABLE:
            self._train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)
        
        return True
    
    def _train_logistic_regression(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train Logistic Regression"""
        print("\n🔥 Training Invasive vs InSitu Logistic Regression...")
        self.lr_model = LogisticRegression(C=1.0, solver='liblinear', random_state=42, max_iter=1000)
        self.lr_model.fit(X_train, y_train)
        
        # Test performance
        test_pred_proba = self.lr_model.predict_proba(X_test)
        test_pred = self.lr_model.predict(X_test)
        test_accuracy = (test_pred == y_test).mean()
        
        print(f"  🎯 TEST Accuracy: {test_accuracy:.3f}")
        
        # Generate ROC data
        self._generate_roc_data(y_test, test_pred_proba, "lr")
        
        self.test_scores_lr = {
            'accuracy': test_accuracy,
            'predictions': test_pred,
            'probabilities': test_pred_proba
        }
    
    def _train_svm(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train SVM RBF"""
        print("\n🔥 Training Invasive vs InSitu SVM RBF...")
        self.svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        self.svm_model.fit(X_train, y_train)
        
        # Test performance
        test_pred_proba = self.svm_model.predict_proba(X_test)
        test_pred = self.svm_model.predict(X_test)
        test_accuracy = (test_pred == y_test).mean()
        
        print(f"  🎯 TEST Accuracy: {test_accuracy:.3f}")
        
        # Generate ROC data
        self._generate_roc_data(y_test, test_pred_proba, "svm")
        
        self.test_scores_svm = {
            'accuracy': test_accuracy,
            'predictions': test_pred,
            'probabilities': test_pred_proba
        }
    
    def _train_xgboost(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train XGBoost"""
        print("\n🔥 Training Invasive vs InSitu XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        self.xgb_model.fit(X_train, y_train)
        
        # Test performance
        test_pred_proba = self.xgb_model.predict_proba(X_test)
        test_pred = self.xgb_model.predict(X_test)
        test_accuracy = (test_pred == y_test).mean()
        
        print(f"  🎯 TEST Accuracy: {test_accuracy:.3f}")
        
        # Generate ROC data
        self._generate_roc_data(y_test, test_pred_proba, "xgb")
        
        self.test_scores_xgb = {
            'accuracy': test_accuracy,
            'predictions': test_pred,
            'probabilities': test_pred_proba
        }
    
    def _generate_roc_data(self, y_true, y_pred_proba, model_type):
        """Generate ROC data for binary classification"""
        # Binary classification - use positive class probabilities  
        y_scores = y_pred_proba[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        roc_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'roc_auc': float(roc_auc),
            'thresholds': thresholds.tolist(),
            'class_names': self.label_encoder.classes_.tolist(),
            'evaluation_type': 'TEST_SET',
            'task': 'Invasive vs InSitu'
        }
        
        if model_type == "lr":
            self.test_roc_data_lr = roc_data
        elif model_type == "svm":
            self.test_roc_data_svm = roc_data
        elif model_type == "xgb":
            self.test_roc_data_xgb = roc_data
        
        print(f"  ROC AUC: {roc_auc:.3f}")
    
    def predict_lr(self, features):
        """Predict using Logistic Regression"""
        if not self.lr_model:
            return None
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        probabilities = self.lr_model.predict_proba(features)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            },
            'algorithm': 'Logistic Regression',
            'task': 'Invasive vs InSitu'
        }
    
    def predict_svm(self, features):
        """Predict using SVM"""
        if not self.svm_model:
            return None
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        probabilities = self.svm_model.predict_proba(features)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            },
            'algorithm': 'SVM RBF',
            'task': 'Invasive vs InSitu'
        }
    
    def predict_xgb(self, features):
        """Predict using XGBoost"""
        if not XGBOOST_AVAILABLE or not self.xgb_model:
            return None
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        probabilities = self.xgb_model.predict_proba(features)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            },
            'algorithm': 'XGBoost',
            'task': 'Invasive vs InSitu'
        }
    
    def save_model(self, model_path='/workspace/bach_invasive_insitu_model.pkl'):
        """Save trained models"""
        model_data = {
            'lr_model': self.lr_model,
            'svm_model': self.svm_model,
            'xgb_model': self.xgb_model if XGBOOST_AVAILABLE else None,
            'label_encoder': self.label_encoder,
            'test_scores_lr': self.test_scores_lr,
            'test_scores_svm': self.test_scores_svm,
            'test_scores_xgb': self.test_scores_xgb,
            'test_roc_data_lr': self.test_roc_data_lr,
            'test_roc_data_svm': self.test_roc_data_svm,
            'test_roc_data_xgb': self.test_roc_data_xgb,
            'data_splits': self.data_splits,
            'class_names': self.class_names,
            'task': 'invasive_vs_insitu'
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Invasive vs InSitu model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_path='/workspace/bach_invasive_insitu_model.pkl'):
        """Load pre-trained models"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.lr_model = model_data['lr_model']
            self.svm_model = model_data['svm_model']
            self.xgb_model = model_data.get('xgb_model', None)
            self.label_encoder = model_data['label_encoder']
            self.test_scores_lr = model_data.get('test_scores_lr', None)
            self.test_scores_svm = model_data.get('test_scores_svm', None)
            self.test_scores_xgb = model_data.get('test_scores_xgb', None)
            self.test_roc_data_lr = model_data.get('test_roc_data_lr', None)
            self.test_roc_data_svm = model_data.get('test_roc_data_svm', None)
            self.test_roc_data_xgb = model_data.get('test_roc_data_xgb', None)
            self.data_splits = model_data.get('data_splits', None)
            self.class_names = model_data['class_names']
            
            print(f"Invasive vs InSitu model loaded from: {model_path}")
            if self.test_scores_lr:
                print(f"LR Test Accuracy: {self.test_scores_lr['accuracy']:.3f}")
            if self.test_scores_svm:
                print(f"SVM Test Accuracy: {self.test_scores_svm['accuracy']:.3f}")
            if self.test_scores_xgb and XGBOOST_AVAILABLE:
                print(f"XGBoost Test Accuracy: {self.test_scores_xgb['accuracy']:.3f}")
            return True
            
        except FileNotFoundError:
            print(f"Invasive vs InSitu model file not found: {model_path}")
            return False
        except Exception as e:
            print(f"Error loading Invasive vs InSitu model: {e}")
            return False

def train_invasive_insitu_classifier():
    """Train the invasive vs insitu classifier"""
    classifier = BACHInvasiveInsituClassifier()
    
    # Train all classifiers
    success = classifier.train_classifiers()
    
    if success:
        # Save the models
        model_path = classifier.save_model()
        
        print(f"\n✅ BACH Invasive vs InSitu Classifier Training Complete!")
        print(f"📊 Model saved: {model_path}")
        
        # Print final performance summary
        if classifier.test_scores_lr:
            print(f"🎯 LR Test Accuracy: {classifier.test_scores_lr['accuracy']:.3f}")
        if classifier.test_scores_svm:
            print(f"🎯 SVM Test Accuracy: {classifier.test_scores_svm['accuracy']:.3f}")
        if classifier.test_scores_xgb:
            print(f"🎯 XGBoost Test Accuracy: {classifier.test_scores_xgb['accuracy']:.3f}")
    
    return classifier

if __name__ == "__main__":
    classifier = train_invasive_insitu_classifier()