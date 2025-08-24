"""
Classifier Manager - Handles all ML model predictions safely
Provides isolated prediction functions with robust error handling
"""

import traceback
from bach_logistic_classifier import BACHLogisticClassifier
from breakhis_binary_classifier import BreakHisBinaryClassifier

class ClassifierManager:
    def __init__(self):
        self.bach_classifier = None
        self.breakhis_classifier = None
        self._bach_loaded = False
        self._breakhis_loaded = False
    
    def load_bach_classifier(self):
        """Load BACH classifier on demand"""
        if not self._bach_loaded:
            try:
                print("🔥 Loading BACH classifier...")
                self.bach_classifier = BACHLogisticClassifier()
                success = self.bach_classifier.load_model()
                self._bach_loaded = success
                print(f"🔥 BACH classifier loaded: {success}")
                if success:
                    print(f"🔥 BACH classes: {self.bach_classifier.class_names}")
                    print(f"🔥 BACH models available: LR={self.bach_classifier.model is not None}, SVM={self.bach_classifier.svm_model is not None}, XGB={hasattr(self.bach_classifier, 'xgb_model') and self.bach_classifier.xgb_model is not None}")
            except Exception as e:
                print(f"❌ BACH classifier loading failed: {e}")
                self._bach_loaded = False
        return self._bach_loaded
    
    def load_breakhis_classifier(self):
        """Load BreakHis classifier on demand"""
        if not self._breakhis_loaded:
            try:
                print("🔥 Loading BreakHis classifier...")
                self.breakhis_classifier = BreakHisBinaryClassifier()
                success = self.breakhis_classifier.load_model()
                self._breakhis_loaded = success
                print(f"🔥 BreakHis classifier loaded: {success}")
                if success:
                    print(f"🔥 BreakHis classes: {self.breakhis_classifier.class_names}")
                    print(f"🔥 BreakHis models available: LR={self.breakhis_classifier.lr_model is not None}, SVM={self.breakhis_classifier.svm_model is not None}, XGB={hasattr(self.breakhis_classifier, 'xgb_model') and self.breakhis_classifier.xgb_model is not None}")
            except Exception as e:
                print(f"❌ BreakHis classifier loading failed: {e}")
                self._breakhis_loaded = False
        return self._breakhis_loaded
    
    # BACH Prediction Methods
    def predict_bach_lr(self, features):
        """Safely predict with BACH Logistic Regression"""
        try:
            if not self.load_bach_classifier() or not self.bach_classifier.model:
                return None
            result = self.bach_classifier.predict(features)
            print(f"🔥 BACH LR: {result['predicted_class']} ({result['confidence']:.3f})")
            return result
        except Exception as e:
            print(f"❌ BACH LR prediction failed: {e}")
            return None
    
    def predict_bach_svm(self, features):
        """Safely predict with BACH SVM"""
        try:
            if not self.load_bach_classifier() or not self.bach_classifier.svm_model:
                return None
            result = self.bach_classifier.predict_svm(features)
            print(f"🔥 BACH SVM: {result['predicted_class']} ({result['confidence']:.3f})")
            return result
        except Exception as e:
            print(f"❌ BACH SVM prediction failed: {e}")
            return None
    
    def predict_bach_xgb(self, features):
        """Safely predict with BACH XGBoost"""
        try:
            if not self.load_bach_classifier() or not hasattr(self.bach_classifier, 'xgb_model') or not self.bach_classifier.xgb_model:
                return None
            result = self.bach_classifier.predict_xgb(features)
            print(f"🔥 BACH XGBoost: {result['predicted_class']} ({result['confidence']:.3f})")
            return result
        except Exception as e:
            print(f"❌ BACH XGBoost prediction failed: {e}")
            return None
    
    # BreakHis Prediction Methods
    def predict_breakhis_lr(self, features):
        """Safely predict with BreakHis Logistic Regression"""
        try:
            if not self.load_breakhis_classifier() or not self.breakhis_classifier.lr_model:
                return None
            result = self.breakhis_classifier.predict_lr(features)
            print(f"🔥 BreakHis LR: {result['predicted_class']} ({result['confidence']:.3f})")
            return result
        except Exception as e:
            print(f"❌ BreakHis LR prediction failed: {e}")
            return None
    
    def predict_breakhis_svm(self, features):
        """Safely predict with BreakHis SVM"""
        try:
            if not self.load_breakhis_classifier() or not self.breakhis_classifier.svm_model:
                return None
            result = self.breakhis_classifier.predict_svm(features)
            print(f"🔥 BreakHis SVM: {result['predicted_class']} ({result['confidence']:.3f})")
            return result
        except Exception as e:
            print(f"❌ BreakHis SVM prediction failed: {e}")
            return None
    
    def predict_breakhis_xgb(self, features):
        """Safely predict with BreakHis XGBoost"""
        try:
            if not self.load_breakhis_classifier() or not hasattr(self.breakhis_classifier, 'xgb_model') or not self.breakhis_classifier.xgb_model:
                return None
            result = self.breakhis_classifier.predict_xgb(features)
            print(f"🔥 BreakHis XGBoost: {result['predicted_class']} ({result['confidence']:.3f})")
            return result
        except Exception as e:
            print(f"❌ BreakHis XGBoost prediction failed: {e}")
            return None
    
    # Utility Methods
    def generate_roc_plot(self, classifier_type='bach'):
        """Generate ROC plot safely"""
        try:
            classifier = self.bach_classifier if classifier_type == 'bach' else self.breakhis_classifier
            if classifier and hasattr(classifier, 'plot_roc_curves'):
                return classifier.plot_roc_curves(return_base64=True)
        except Exception as e:
            print(f"⚠️ ROC plot failed for {classifier_type}: {e}")
        return None
    
    def get_model_info(self, classifier_type='bach'):
        """Get model performance info safely"""
        try:
            classifier = self.bach_classifier if classifier_type == 'bach' else self.breakhis_classifier
            if not classifier:
                return {"status": f"{classifier_type} not loaded"}
                
            info = {
                "classifier": classifier_type.upper(),
                "classes": getattr(classifier, 'class_names', []),
                "evaluation_type": "HELD_OUT_TEST_SET"
            }
            
            # Add test accuracies
            if hasattr(classifier, 'test_scores') and classifier.test_scores:
                info["test_accuracy_lr"] = float(classifier.test_scores['accuracy'])
            if hasattr(classifier, 'svm_test_scores') and classifier.svm_test_scores:
                info["test_accuracy_svm"] = float(classifier.svm_test_scores['accuracy'])
            if hasattr(classifier, 'xgb_test_scores') and classifier.xgb_test_scores:
                info["test_accuracy_xgb"] = float(classifier.xgb_test_scores['accuracy'])
                
            # Add ROC AUCs
            if hasattr(classifier, 'test_roc_data') and classifier.test_roc_data:
                info["test_roc_auc_lr"] = float(classifier.test_roc_data['roc_auc'].get('micro', 0))
            if hasattr(classifier, 'svm_test_roc_data') and classifier.svm_test_roc_data:
                info["test_roc_auc_svm"] = float(classifier.svm_test_roc_data['roc_auc'].get('micro', 0))
            if hasattr(classifier, 'xgb_test_roc_data') and classifier.xgb_test_roc_data:
                info["test_roc_auc_xgb"] = float(classifier.xgb_test_roc_data['roc_auc'].get('micro', 0))
                
            if hasattr(classifier, 'data_splits') and classifier.data_splits:
                info["data_splits"] = classifier.data_splits
                
            return info
        except Exception as e:
            print(f"⚠️ Model info failed for {classifier_type}: {e}")
            return {"status": f"{classifier_type} info failed", "error": str(e)}

# Global classifier manager instance
classifier_manager = ClassifierManager()