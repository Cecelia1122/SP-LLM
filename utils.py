"""
utils.py - Shared utility functions for the keyword spotter project
Purpose: Common functions used across different modules
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json
from typing import List, Tuple, Dict, Any

def create_directories():
    """Create necessary project directories"""
    directories = ['data', 'models', 'results', 'src']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Project directories created")

def load_audio_file(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        # Normalize audio length to 1 second
        target_length = target_sr
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            
        # Normalize amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def extract_classical_features(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract classical signal processing features (MFCC + additional)
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        Feature vector
    """
    try:
        features = []
        
        # 1. MFCCs (core speech features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        features.extend(np.mean(mfccs, axis=1))  # Mean
        features.extend(np.std(mfccs, axis=1))   # Std deviation
        
        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
        
        # 3. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # 4. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        
        # 5. Energy features
        rms = librosa.feature.rms(y=audio)
        features.extend([np.mean(rms), np.std(rms)])
        
        # 6. Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(tempo)
        
        return np.array(features)
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, 
                         class_names: List[str] = None) -> plt.Figure:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        class_names: Class names for labels
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['Not Keyword', 'Keyword']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # Add performance metrics
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
    ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_roc_curves(models_data: Dict[str, Dict], title: str = "ROC Curves Comparison") -> plt.Figure:
    """
    Plot ROC curves for multiple models
    
    Args:
        models_data: Dictionary with model results
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, data in models_data.items():
        if 'test_labels' in data and 'test_probabilities' in data:
            fpr, tpr, _ = roc_curve(data['test_labels'], data['test_probabilities'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def save_results_summary(results: Dict[str, Any], filename: str = "results/analysis_report.txt"):
    """
    Save detailed results summary to text file
    
    Args:
        results: Results dictionary
        filename: Output filename
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write("=== KEYWORD SPOTTER ANALYSIS REPORT ===\n\n")
        
        # Classical models results
        if 'classical_models' in results:
            f.write("CLASSICAL APPROACH RESULTS:\n")
            f.write("-" * 40 + "\n")
            for model_name, metrics in results['classical_models'].items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Accuracy: {metrics.get('accuracy', 0):.3f}\n")
                f.write(f"  AUC: {metrics.get('auc', 0):.3f}\n")
                f.write(f"  False Positive Rate: {metrics.get('false_positive_rate', 0):.3f}\n")
        
        # Speech LLM results
        if 'speech_llm' in results:
            f.write(f"\n\nSPEECH LLM APPROACH RESULTS:\n")
            f.write("-" * 40 + "\n")
            metrics = results['speech_llm']
            f.write(f"  Test Accuracy: {metrics.get('test_accuracy', 0):.3f}\n")
            f.write(f"  Test Loss: {metrics.get('test_loss', 0):.3f}\n")
            f.write(f"  Training Time: {metrics.get('training_time', 0):.1f} minutes\n")
        
        # Comparison
        if 'comparison' in results:
            f.write(f"\n\nCOMPARISON SUMMARY:\n")
            f.write("-" * 40 + "\n")
            comp = results['comparison']
            f.write(f"Best Classical Model: {comp.get('best_classical', 'N/A')}\n")
            f.write(f"Best Classical Accuracy: {comp.get('best_classical_accuracy', 0):.3f}\n")
            f.write(f"Speech LLM Accuracy: {comp.get('speech_llm_accuracy', 0):.3f}\n")
            f.write(f"Winner: {comp.get('winner', 'N/A')}\n")
            f.write(f"Performance Difference: {comp.get('performance_difference', 0):.3f}\n")
        
        # Dataset info
        if 'dataset_info' in results:
            f.write(f"\n\nDATASET INFORMATION:\n")
            f.write("-" * 40 + "\n")
            dataset = results['dataset_info']
            f.write(f"Total Samples: {dataset.get('total_samples', 0)}\n")
            f.write(f"Positive Samples: {dataset.get('positive_samples', 0)}\n")
            f.write(f"Negative Samples: {dataset.get('negative_samples', 0)}\n")
            f.write(f"Train/Test Split: {dataset.get('train_split', 0.8):.1%}/{dataset.get('test_split', 0.2):.1%}\n")

def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', 
                      decimals: int = 1, length: int = 50, fill: str = '█'):
    """
    Print progress bar
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        decimals: Number of decimals in percent complete
        length: Character length of bar
        fill: Bar fill character
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def validate_audio_file(file_path: str) -> bool:
    """
    Validate if file is a valid audio file
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if valid audio file, False otherwise
    """
    try:
        audio, sr = librosa.load(file_path, duration=0.1)  # Load just 0.1 seconds for validation
        return len(audio) > 0 and sr > 0
    except:
        return False

class PerformanceTracker:
    """Track and store performance metrics across experiments"""
    
    def __init__(self):
        self.experiments = {}
    
    def add_experiment(self, name: str, metrics: Dict[str, float]):
        """Add experiment results"""
        self.experiments[name] = {
            'metrics': metrics,
            'timestamp': np.datetime64('now')
        }
    
    def get_best_model(self, metric: str = 'accuracy'):
        """Get best performing model based on specified metric"""
        if not self.experiments:
            return None
        
        best_score = -1
        best_model = None
        
        for name, data in self.experiments.items():
            if metric in data['metrics']:
                score = data['metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model = name
        
        return best_model, best_score
    
    def save_to_json(self, filename: str = "results/performance_tracking.json"):
        """Save performance tracking to JSON file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert numpy types to standard Python types for JSON serialization
        json_data = {}
        for name, data in self.experiments.items():
            json_data[name] = {
                'metrics': {k: float(v) for k, v in data['metrics'].items()},
                'timestamp': str(data['timestamp'])
            }
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)

# Audio processing utilities
def add_noise_to_audio(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """Add Gaussian noise to audio signal"""
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

def apply_time_shift(audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
    """Apply random time shift to audio signal"""
    shift = np.random.randint(-int(len(audio) * shift_max), int(len(audio) * shift_max))
    if shift > 0:
        return np.pad(audio, (shift, 0), mode='constant')[:len(audio)]
    else:
        return np.pad(audio, (0, -shift), mode='constant')[-len(audio):]

def apply_pitch_shift(audio: np.ndarray, sr: int = 16000, n_steps: float = None) -> np.ndarray:
    """Apply pitch shifting to audio signal"""
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)  # Random pitch shift
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

# Constants
KEYWORD = "yes"
SAMPLE_RATE = 16000
FEATURE_DIM_CLASSICAL = 52  # Expected feature dimension for classical approach
CONFIDENCE_THRESHOLD = 0.7
MODEL_NAMES = {
    'classical': ['SVM', 'Random Forest', 'Neural Network'],
    'speech_llm': 'Wav2Vec2'
}

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test directory creation
    create_directories()
    
    # Test performance tracker
    tracker = PerformanceTracker()
    tracker.add_experiment("test_model", {"accuracy": 0.85, "auc": 0.90})
    best_model, best_score = tracker.get_best_model()
    print(f"Best model: {best_model} with score: {best_score}")
    
    print("✅ Utility functions tested successfully!")