"""
BreakHis Binary Classifier (Malignant vs Non-Malignant)
Trains Logistic Regression and SVM RBF classifiers on GigaPath features
Uses proper train/validation/test splits with honest evaluation
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

class BreakHisBinaryClassifier:
    def __init__(self, cache_file='/workspace/embeddings_cache_REAL_GIGAPATH.pkl'):
        self.cache_file = cache_file
        self.lr_model = None
        self.svm_model = None
        self.label_encoder = None
        self.cv_scores_lr = None
        self.cv_scores_svm = None
        self.test_scores_lr = None
        self.test_scores_svm = None
        self.test_roc_data_lr = None
        self.test_roc_data_svm = None
        self.data_splits = None
        self.class_names = ['benign', 'malignant']  # Binary classification
        
    def load_breakhis_data(self):
        """Load BreakHis dataset features and labels from cache"""
        print("Loading BreakHis data from cache...")
        
        with open(self.cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        # Extract BreakHis data only
        combined_data = cache['combined']
        breakhis_indices = [i for i, ds in enumerate(combined_data['datasets']) if ds == 'breakhis']
        
        if not breakhis_indices:
            raise ValueError("No BreakHis data found in cache")
        
        # Get BreakHis features and labels
        features = np.array([combined_data['features'][i] for i in breakhis_indices])
        labels = [combined_data['labels'][i] for i in breakhis_indices]
        filenames = [combined_data['filenames'][i] for i in breakhis_indices]
        
        # Convert to binary labels (malignant vs non-malignant)
        binary_labels = ['malignant' if label == 'malignant' else 'benign' for label in labels]
        
        print(f"Loaded BreakHis data: {len(features)} samples")
        print(f"Original classes: {set(labels)}")
        print(f"Binary classes: {set(binary_labels)}")
        print(f"Binary distribution: {[(label, binary_labels.count(label)) for label in set(binary_labels)]}")
        
        return features, binary_labels, filenames
    
    def train_classifiers(self):
        """Train both classifiers with proper train/validation/test splits"""
        features, labels, filenames = self.load_breakhis_data()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        
        print(f"Total samples: {len(features)}")
        print(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        print(f"Class distribution: {[(cls, sum(y_encoded == i)) for i, cls in enumerate(self.label_encoder.classes_)]}")
        
        # PROPER TRAIN/VALIDATION/TEST SPLITS
        # First split: 80% train+val, 20% test (stratified)
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        
        # Second split: 75% train, 25% validation from remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )
        
        # Store split information
        self.data_splits = {
            'train_size': len(X_train),
            'val_size': len(X_val), 
            'test_size': len(X_test),
            'train_distribution': [(cls, sum(y_train == i)) for i, cls in enumerate(self.label_encoder.classes_)],
            'val_distribution': [(cls, sum(y_val == i)) for i, cls in enumerate(self.label_encoder.classes_)],
            'test_distribution': [(cls, sum(y_test == i)) for i, cls in enumerate(self.label_encoder.classes_)]
        }
        
        print(f"\\n📊 BREAKHIS BINARY DATA SPLITS:")
        print(f"  Train: {len(X_train)} samples - {self.data_splits['train_distribution']}")
        print(f"  Validation: {len(X_val)} samples - {self.data_splits['val_distribution']}")
        print(f"  Test: {len(X_test)} samples - {self.data_splits['test_distribution']}")
        
        # LOGISTIC REGRESSION TRAINING
        print("\\n🔥 Training Binary Logistic Regression...")
        self.lr_model = LogisticRegression(
            C=1.0,
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        
        # Train on training set only
        self.lr_model.fit(X_train, y_train)
        
        # Validation performance
        val_pred_proba_lr = self.lr_model.predict_proba(X_val)
        val_pred_lr = self.lr_model.predict(X_val)
        val_accuracy_lr = (val_pred_lr == y_val).mean()
        
        # TEST PERFORMANCE (held-out)
        test_pred_proba_lr = self.lr_model.predict_proba(X_test)
        test_pred_lr = self.lr_model.predict(X_test)
        test_accuracy_lr = (test_pred_lr == y_test).mean()
        
        print(f"  Validation Accuracy: {val_accuracy_lr:.3f}")
        print(f"  🎯 TEST Accuracy: {test_accuracy_lr:.3f}")
        
        # Generate TEST ROC curves for Logistic Regression
        self.generate_test_roc_curves(y_test, test_pred_proba_lr, "Logistic Regression", "lr")
        
        # SVM RBF TRAINING
        print("\\n🔥 Training Binary SVM RBF...")
        self.svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # Train on training set only
        self.svm_model.fit(X_train, y_train)
        
        # Validation performance
        svm_val_pred_proba = self.svm_model.predict_proba(X_val)
        svm_val_pred = self.svm_model.predict(X_val)
        svm_val_accuracy = (svm_val_pred == y_val).mean()
        
        # TEST PERFORMANCE (held-out)
        svm_test_pred_proba = self.svm_model.predict_proba(X_test)
        svm_test_pred = self.svm_model.predict(X_test)
        svm_test_accuracy = (svm_test_pred == y_test).mean()
        
        print(f"  Validation Accuracy: {svm_val_accuracy:.3f}")
        print(f"  🎯 TEST Accuracy: {svm_test_accuracy:.3f}")
        
        # Generate TEST ROC curves for SVM
        self.generate_test_roc_curves(y_test, svm_test_pred_proba, "SVM RBF", "svm")
        
        # Store test scores (what we actually report)
        self.test_scores_lr = {
            'accuracy': test_accuracy_lr, 
            'predictions': test_pred_lr, 
            'probabilities': test_pred_proba_lr
        }
        self.test_scores_svm = {
            'accuracy': svm_test_accuracy, 
            'predictions': svm_test_pred, 
            'probabilities': svm_test_pred_proba
        }
        
        # Print HONEST test performance
        print(f"\\n🎯 HONEST BREAKHIS TEST PERFORMANCE:")
        print(f"  Logistic Regression Test Accuracy: {test_accuracy_lr:.3f}")
        print(f"  SVM RBF Test Accuracy: {svm_test_accuracy:.3f}")
        
        print("\\n📈 Test Set Classification Reports:")
        print("\\nLogistic Regression (Test Set):")
        print(classification_report(y_test, test_pred_lr, target_names=self.label_encoder.classes_))
        
        print("\\nSVM RBF (Test Set):")
        print(classification_report(y_test, svm_test_pred, target_names=self.label_encoder.classes_))
        
        return self.lr_model, self.svm_model
    
    def generate_test_roc_curves(self, y_true, y_pred_proba, classifier_name, model_type):
        """Generate ROC curves from TEST SET predictions (honest evaluation)"""
        # For binary classification, use class 1 (malignant) probabilities
        y_scores = y_pred_proba[:, 1]  # Probabilities for malignant class
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Store TEST ROC data
        roc_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'roc_auc': float(roc_auc),
            'thresholds': thresholds.tolist(),
            'class_names': self.label_encoder.classes_.tolist(),
            'classifier': classifier_name,
            'evaluation_type': 'TEST_SET',
            'dataset': 'BreakHis'
        }
        
        if model_type == "lr":
            self.test_roc_data_lr = roc_data
        else:
            self.test_roc_data_svm = roc_data
        
        print(f"{classifier_name} TEST ROC AUC: {roc_auc:.3f}")
        
        return roc_data
    
    def plot_roc_curves(self, save_path=None, return_base64=False):
        """Generate combined ROC curve plots for both classifiers"""
        if not self.test_roc_data_lr or not self.test_roc_data_svm:
            raise ValueError("Test ROC data not available. Train classifiers first.")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot Logistic Regression ROC curve
        ax.plot(
            self.test_roc_data_lr['fpr'], 
            self.test_roc_data_lr['tpr'],
            color='blue',
            lw=2,
            label=f'Logistic Regression (AUC = {self.test_roc_data_lr["roc_auc"]:.3f})'
        )
        
        # Plot SVM ROC curve
        ax.plot(
            self.test_roc_data_svm['fpr'], 
            self.test_roc_data_svm['tpr'],
            color='green',
            lw=2,
            label=f'SVM RBF (AUC = {self.test_roc_data_svm["roc_auc"]:.3f})'
        )
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax.set_title('BreakHis Binary Classification ROC Curves\\n(Malignant vs Non-Malignant)\\nTest Set Performance', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"BreakHis ROC plot saved to: {save_path}")
        
        if return_base64:
            # Convert plot to base64 for frontend
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return plot_base64
        
        return fig
    
    def predict_lr(self, features):
        """Predict using Logistic Regression"""
        if self.lr_model is None:
            raise ValueError("Logistic Regression model not trained. Call train_classifiers() first.")
        
        # Ensure features are 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get probabilities for each class
        probabilities = self.lr_model.predict_proba(features)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Create result dictionary
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            },
            'algorithm': 'Logistic Regression'
        }
        
        return result
    
    def predict_svm(self, features):
        """Predict using SVM RBF"""
        if self.svm_model is None:
            raise ValueError("SVM model not trained. Call train_classifiers() first.")
        
        # Ensure features are 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get probabilities for each class
        probabilities = self.svm_model.predict_proba(features)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Create result dictionary
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            },
            'algorithm': 'SVM RBF'
        }
        
        return result
    
    def save_model(self, model_path='/workspace/breakhis_binary_model.pkl'):
        """Save trained models and data"""
        model_data = {
            'lr_model': self.lr_model,
            'svm_model': self.svm_model,
            'label_encoder': self.label_encoder,
            'cv_scores_lr': self.cv_scores_lr,
            'cv_scores_svm': self.cv_scores_svm,
            'test_scores_lr': self.test_scores_lr,
            'test_scores_svm': self.test_scores_svm,
            'test_roc_data_lr': self.test_roc_data_lr,
            'test_roc_data_svm': self.test_roc_data_svm,
            'data_splits': self.data_splits,
            'class_names': self.class_names
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"BreakHis binary model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_path='/workspace/breakhis_binary_model.pkl'):
        """Load pre-trained models"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.lr_model = model_data['lr_model']
            self.svm_model = model_data['svm_model']
            self.label_encoder = model_data['label_encoder']
            self.cv_scores_lr = model_data.get('cv_scores_lr', None)
            self.cv_scores_svm = model_data.get('cv_scores_svm', None)
            self.test_scores_lr = model_data.get('test_scores_lr', None)
            self.test_scores_svm = model_data.get('test_scores_svm', None)
            self.test_roc_data_lr = model_data.get('test_roc_data_lr', None)
            self.test_roc_data_svm = model_data.get('test_roc_data_svm', None)
            self.data_splits = model_data.get('data_splits', None)
            self.class_names = model_data['class_names']
            
            print(f"BreakHis binary model loaded from: {model_path}")
            if self.test_scores_lr and self.test_scores_svm:
                print(f"LR Test Accuracy: {self.test_scores_lr['accuracy']:.3f}")
                print(f"SVM Test Accuracy: {self.test_scores_svm['accuracy']:.3f}")
            return True
            
        except FileNotFoundError:
            print(f"BreakHis model file not found: {model_path}")
            return False
        except Exception as e:
            print(f"Error loading BreakHis model: {e}")
            return False

def train_breakhis_binary_classifier():
    """Main function to train BreakHis binary classifiers"""
    classifier = BreakHisBinaryClassifier()
    
    # Train both classifiers
    lr_model, svm_model = classifier.train_classifiers()
    
    # Save the models
    model_path = classifier.save_model()
    
    # Generate and save ROC plot
    roc_plot_path = '/workspace/breakhis_binary_roc_plot.png'
    classifier.plot_roc_curves(save_path=roc_plot_path)
    
    print(f"\\n✅ BreakHis Binary Classifier Training Complete!")
    print(f"📊 Model saved: {model_path}")
    print(f"📈 ROC plot saved: {roc_plot_path}")
    
    return classifier

if __name__ == "__main__":
    # Train the binary classifiers
    classifier = train_breakhis_binary_classifier()
    
    # Test with some sample predictions
    print("\\n🔍 Testing binary classifiers...")
    features, labels, filenames = classifier.load_breakhis_data()
    
    # Test on first 3 samples
    for i in range(3):
        lr_result = classifier.predict_lr(features[i])
        svm_result = classifier.predict_svm(features[i])
        actual = labels[i]
        print(f"Sample {i+1}: LR={lr_result['predicted_class']} ({lr_result['confidence']:.3f}), SVM={svm_result['predicted_class']} ({svm_result['confidence']:.3f}), Actual={actual}")