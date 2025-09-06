"""
2_classical_approach.py - Classical signal processing approach
Purpose: MFCC feature extraction + traditional ML classifiers (SVM, Random Forest, Neural Network)
"""

import os
import warnings
warnings.filterwarnings("ignore", message="Trying to estimate tuning from empty frequency set")

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from tqdm import tqdm

from utils import (
    load_audio_file,
    extract_classical_features,
    plot_confusion_matrix,
    PerformanceTracker,
    KEYWORD,
    FEATURE_DIM_CLASSICAL  # (49 from your utils)
)
from data_preparation import load_data_splits


class ClassicalKeywordSpotter:
    def __init__(self, keyword: str = KEYWORD):
        self.keyword = keyword
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_dim = None
        self.performance_tracker = PerformanceTracker()

    # -----------------------------
    # Feature Extraction
    # -----------------------------
    def extract_features_from_files(self, file_paths, labels):
        print("🎵 Extracting classical features (MFCC + spectral)...")
        features = []
        valid_labels = []

        missing = load_fail = feat_fail = 0
        lengths = []
        log = []
        MAX_LOG = 10

        for fp, lab in tqdm(zip(file_paths, labels), total=len(file_paths), desc="Processing files"):
            if not os.path.exists(fp):
                missing += 1
                if len(log) < MAX_LOG:
                    log.append(f"[MISSING] {fp}")
                continue

            audio, sr = load_audio_file(fp)
            if audio is None:
                load_fail += 1
                if len(log) < MAX_LOG:
                    log.append(f"[LOAD_FAIL] {fp}")
                continue

            feat = extract_classical_features(audio, sr, enforce_length=True)
            if feat is None:
                feat_fail += 1
                if len(log) < MAX_LOG:
                    log.append(f"[FEAT_FAIL] {fp}")
                continue

            features.append(feat)
            valid_labels.append(lab)
            lengths.append(feat.shape[0])

        print("🔎 Extraction summary:")
        print(f"  Total listed files: {len(file_paths)}")
        print(f"  Successful feature vectors: {len(features)}")
        print(f"  Missing paths: {missing}")
        print(f"  Load failures: {load_fail}")
        print(f"  Feature failures: {feat_fail}")
        if log:
            print("  Sample issues:")
            for line in log:
                print("    ", line)

        if len(features) == 0:
            print("❌ No usable features extracted.")
            return None, None

        features = np.stack(features, axis=0)
        labels_arr = np.array(valid_labels, dtype=np.int64)
        self.feature_dim = features.shape[1]

        # Sanity assertion (should match utils FEATURE_DIM_CLASSICAL)
        if self.feature_dim != FEATURE_DIM_CLASSICAL:
            print(f"⚠️ Warning: extracted dim {self.feature_dim} != FEATURE_DIM_CLASSICAL {FEATURE_DIM_CLASSICAL}")

        print(f"✅ Final feature matrix shape: {features.shape}")
        print(f"📊 Positives: {labels_arr.sum()} | Negatives: {len(labels_arr) - labels_arr.sum()}")
        return features, labels_arr

    # -----------------------------
    # Individual Model Trainers
    # -----------------------------
    def train_svm_classifier(self, X_train, y_train, X_val, y_val):
        print("🤖 Training SVM classifier...")
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01],
            'kernel': ['rbf']
        }
        svm = SVC(probability=True, random_state=42, class_weight='balanced')
        gs = GridSearchCV(svm, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        val_probs = best.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        acc = (val_preds == y_val).mean()
        auc = roc_auc_score(y_val, val_probs)
        cm = confusion_matrix(y_val, val_preds)
        fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) else 0
        print(f"SVM val acc={acc:.3f} auc={auc:.3f} fpr={fpr:.3f}")
        return {
            'model': best,
            'accuracy': acc,
            'auc': auc,
            'false_positive_rate': fpr,
            'val_predictions': val_preds,
            'val_probabilities': val_probs,
            'val_labels': y_val
        }

    def train_random_forest_classifier(self, X_train, y_train, X_val, y_val):
        print("🌳 Training Random Forest classifier...")
        param_grid = {
            'n_estimators': [150, 300],
            'max_depth': [20, None],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        gs = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        val_probs = best.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        acc = (val_preds == y_val).mean()
        auc = roc_auc_score(y_val, val_probs)
        cm = confusion_matrix(y_val, val_preds)
        fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) else 0
        print(f"RF val acc={acc:.3f} auc={auc:.3f} fpr={fpr:.3f}")
        return {
            'model': best,
            'accuracy': acc,
            'auc': auc,
            'false_positive_rate': fpr,
            'val_predictions': val_preds,
            'val_probabilities': val_probs,
            'val_labels': y_val,
            'feature_importance': best.feature_importances_
        }

    def train_neural_network_classifier(self, X_train, y_train, X_val, y_val):
        print("🧠 Training Neural Network classifier...")
        param_grid = {
            'hidden_layer_sizes': [(128, 64), (256, 128)],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.0001, 0.001]
        }
        nn = MLPClassifier(
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15
        )
        gs = GridSearchCV(nn, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        val_probs = best.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        acc = (val_preds == y_val).mean()
        auc = roc_auc_score(y_val, val_probs)
        cm = confusion_matrix(y_val, val_preds)
        fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) else 0
        print(f"NN val acc={acc:.3f} auc={auc:.3f} fpr={fpr:.3f}")
        return {
            'model': best,
            'accuracy': acc,
            'auc': auc,
            'false_positive_rate': fpr,
            'val_predictions': val_preds,
            'val_probabilities': val_probs,
            'val_labels': y_val
        }

    # -----------------------------
    # Orchestration
    # -----------------------------
    def train_all_models(self, data_splits):
        print("🚀 Starting classical approach training...\n")
        train_data = data_splits['train']
        val_data = data_splits['validation']

        X_train, y_train = self.extract_features_from_files(train_data['files'], train_data['labels'])
        if X_train is None:
            raise RuntimeError("No training features extracted.")

        X_val, y_val = self.extract_features_from_files(val_data['files'], val_data['labels'])
        if X_val is None:
            raise RuntimeError("No validation features extracted.")

        print("📏 Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.models['SVM'] = self.train_svm_classifier(X_train_scaled, y_train, X_val_scaled, y_val)
        self.models['Random Forest'] = self.train_random_forest_classifier(X_train_scaled, y_train, X_val_scaled, y_val)
        self.models['Neural Network'] = self.train_neural_network_classifier(X_train_scaled, y_train, X_val_scaled, y_val)

        self.select_best_model()

        for model_name, metrics in self.models.items():
            self.performance_tracker.add_experiment(
                f"classical_{model_name.lower().replace(' ', '_')}",
                {
                    'accuracy': metrics.get('accuracy', 0.0),
                    'auc': metrics.get('auc', 0.0),
                    'false_positive_rate': metrics.get('false_positive_rate', 0.0),
                    # Use None when test metrics are not present to keep JSON strict
                    'test_accuracy': metrics.get('test_accuracy', None),
                    'test_auc': metrics.get('test_auc', None),
                }
            )
        return self.models

    def select_best_model(self):
        best_score = -1
        best_name = None
        print("\n📊 MODEL COMPARISON:")
        print("-" * 48)
        for name, m in self.models.items():
            score = m['auc'] * (1 - m['false_positive_rate'])
            print(f"{name}: acc={m['accuracy']:.3f} auc={m['auc']:.3f} "
                  f"fpr={m['false_positive_rate']:.3f} score={score:.3f}")
            if score > best_score:
                best_score = score
                best_name = name
        self.best_model = best_name
        print(f"🏆 Best classical model: {best_name} (score={best_score:.3f})")

    def evaluate_on_test_set(self, data_splits):
        if not self.best_model:
            print("❌ No best model selected!")
            return None
        print(f"\n🧪 Evaluating {self.best_model} on test set...")
        test_data = data_splits['test']
        X_test, y_test = self.extract_features_from_files(test_data['files'], test_data['labels'])
        if X_test is None:
            raise RuntimeError("No test features extracted.")
        X_test_scaled = self.scaler.transform(X_test)
        model = self.models[self.best_model]['model']
        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = (preds == y_test).mean()
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) else 0
        print(f"Test acc={acc:.3f} auc={auc:.3f} fpr={fpr:.3f}")
        print("\nDetailed classification report:")
        print(classification_report(y_test, preds, target_names=['Not Keyword','Keyword']))
        self.models[self.best_model].update({
            'test_accuracy': acc,
            'test_auc': auc,
            'test_fpr': fpr,
            'test_predictions': preds,
            'test_probabilities': probs,
            'test_labels': y_test
        })
        return {
            'accuracy': acc,
            'auc': auc,
            'false_positive_rate': fpr,
            'predictions': preds,
            'probabilities': probs,
            'labels': y_test
        }

    # -----------------------------
    # Visualization 
    # -----------------------------
    def create_visualizations(self):
        print("📈 Creating visualizations...")
        if not self.models:
            print("No models to visualize.")
            return
        os.makedirs("results", exist_ok=True)

        model_names = list(self.models.keys())
        accuracies = [self.models[m]['accuracy'] for m in model_names]
        aucs = [self.models[m]['auc'] for m in model_names]
        fprs = [self.models[m]['false_positive_rate'] for m in model_names]

        # Composite figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(8, 6), constrained_layout=True
        )

        # Accuracy
        ax1.bar(model_names, accuracies, color='skyblue')
        ax1.set_title('Accuracy', fontsize=10)
        ax1.set_ylim(0, 1)

        # AUC
        ax2.bar(model_names, aucs, color='lightgreen')
        ax2.set_title('AUC', fontsize=10)
        ax2.set_ylim(0, 1)

        # FPR
        ax3.bar(model_names, fprs, color='salmon')
        ax3.set_title('False Positive Rate', fontsize=10)
        ax3.set_ylim(0, max(fprs) * 1.2 if fprs else 0.2)

        # ROC Curves
        for name in model_names:
            data = self.models[name]
            if 'val_labels' in data and 'val_probabilities' in data:
                fpr_vals, tpr_vals, _ = roc_curve(data['val_labels'], data['val_probabilities'])
                ax4.plot(fpr_vals, tpr_vals, label=f"{name} ({data['auc']:.2f})", linewidth=1)
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
        ax4.set_title("ROC Curves", fontsize=10)
        ax4.legend(fontsize=8, frameon=False)
        ax4.grid(alpha=0.25)

        fig.savefig("results/classical_models_comparison.png", dpi=140)
        # Non-blocking show
        plt.show(block=False)
        plt.pause(0.25)
        plt.close(fig)

        # Confusion matrix for best model
        if self.best_model and 'test_labels' in self.models[self.best_model]:
            best = self.models[self.best_model]
            fig_cm = plot_confusion_matrix(
                best['test_labels'],
                best['test_predictions'],
                f"Confusion Matrix - {self.best_model} (Test)"
            )
            fig_cm.set_size_inches(5, 4)
            fig_cm.savefig("results/confusion_matrix.png", dpi=140, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(0.25)
            plt.close(fig_cm)

        # Feature importance (Random Forest)
        if 'Random Forest' in self.models and 'feature_importance' in self.models['Random Forest']:
            imp = self.models['Random Forest']['feature_importance']
            idx = np.argsort(imp)[::-1][:15]
            fig_imp, ax_imp = plt.subplots(figsize=(7, 4.5))
            ax_imp.bar(range(len(idx)), imp[idx], color='mediumpurple')
            ax_imp.set_title("Random Forest Top 15 Features", fontsize=11)
            ax_imp.set_xlabel("Feature Index")
            ax_imp.set_ylabel("Importance")
            ax_imp.set_xticks(range(len(idx)))
            ax_imp.set_xticklabels([f"F{i}" for i in idx], rotation=45, fontsize=8)
            fig_imp.tight_layout()
            fig_imp.savefig("results/feature_importance.png", dpi=140)
            plt.show(block=False)
            plt.pause(0.25)
            plt.close(fig_imp)

        print("✅ Visualizations saved.")

    # -----------------------------
    # Persistence & Inference
    # -----------------------------
    def save_model(self, filename: str = "models/classical_keyword_spotter.pkl"):
        if not self.best_model:
            print("❌ No model to save.")
            return
        os.makedirs("models", exist_ok=True)
        data = {
            'model': self.models[self.best_model]['model'],
            'scaler': self.scaler,
            'keyword': self.keyword,
            'model_type': self.best_model,
            'feature_dim': self.feature_dim,
            'approach': 'classical',
            'performance_metrics': {
                name: {k: v for k, v in d.items()
                       if k in ['accuracy', 'auc', 'false_positive_rate']}
                for name, d in self.models.items()
            }
        }
        joblib.dump(data, filename)
        print(f"💾 Model saved: {filename}")
        return filename

    def load_model(self, filename: str = "models/classical_keyword_spotter.pkl"):
        try:
            data = joblib.load(filename)
            name = data['model_type']
            self.models[name] = {
                'model': data['model'],
                'accuracy': data['performance_metrics'][name]['accuracy'],
                'auc': data['performance_metrics'][name]['auc'],
                'false_positive_rate': data['performance_metrics'][name]['false_positive_rate']
            }
            self.scaler = data['scaler']
            self.keyword = data['keyword']
            self.best_model = name
            self.feature_dim = data['feature_dim']
            print(f"✅ Loaded model: {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def predict_audio_file(self, file_path: str, confidence_threshold: float = 0.7):
        if not self.best_model:
            print("❌ No trained model.")
            return None
        audio, sr = load_audio_file(file_path)
        if audio is None:
            return None
        feat = extract_classical_features(audio, sr)
        if feat is None:
            return None
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        model = self.models[self.best_model]['model']
        prob = model.predict_proba(feat_scaled)[0, 1]
        pred = int(prob >= confidence_threshold)
        return {
            'prediction': pred,
            'probability': float(prob),
            'detected': bool(pred),
            'confidence_level': 'High' if prob > 0.8 else 'Medium' if prob > 0.6 else 'Low',
            'model_type': f"Classical_{self.best_model}",
            'approach': 'classical'
        }


def main():
    print("=== CLASSICAL KEYWORD SPOTTER TRAINING ===")
    data_splits = load_data_splits()

    spotter = ClassicalKeywordSpotter(keyword="yes")
    spotter.train_all_models(data_splits)
    test_results = spotter.evaluate_on_test_set(data_splits)

    # Save before plotting (so a plotting failure never loses the model)
    model_path = spotter.save_model()
    spotter.performance_tracker.save_to_json("results/classical_performance_tracking.json")

    # Visualizations (non-blocking)
    try:
        spotter.create_visualizations()
    except Exception as e:
        print(f"⚠️ Visualization error (continuing): {e}")

    print("\n🎉 Training complete.")
    if spotter.best_model:
        bm = spotter.best_model
        if 'test_accuracy' in spotter.models[bm]:
            print(f"🏆 Best model: {bm}")
            print(f"📊 Test accuracy: {spotter.models[bm]['test_accuracy']:.3f}")
            print(f"📊 Test AUC: {spotter.models[bm]['test_auc']:.3f}")
    print(f"💾 Saved model: {model_path}")

    return spotter, test_results


if __name__ == "__main__":
    main()