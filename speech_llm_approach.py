"""
3_speech_llm_approach.py
Fine‑tune Wav2Vec2 (facebook/wav2vec2-base) for keyword spotting (binary classification).

Optimized for CPU (Intel macOS) to reduce training time and suppress common warnings.

Key speed strategies applied:
- Freeze feature extractor + most encoder layers (train only last few + classifier).
- Disable gradient checkpointing (big slowdown on CPU).
- Smaller number of epochs with early stopping.
- Optional in‑memory (precomputed) feature caching (disabled by default).
- Thread control to avoid oversubscription.

If you still need faster runs, enable PRECOMPUTE_FEATURES = True below (requires enough RAM).
"""

import os
import time
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio  # Ensures native extension is loadable early
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

from utils import load_audio_file, PerformanceTracker, KEYWORD, SAMPLE_RATE
from data_preparation import load_data_splits

# --------------------------------------------------
# CONFIGURABLE FLAGS
# --------------------------------------------------
FREEZE_ENCODER_LAYERS = 10          # Freeze first N encoder layers (0..11 for wav2vec2-base)
FREEZE_FEATURE_EXTRACTOR = True
GRADIENT_CHECKPOINTING = False      # Disable on CPU for speed
EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 5e-5                # Slightly higher since few params train
EARLY_STOPPING_PATIENCE = 2
PRECOMPUTE_FEATURES = False         # Set True to precompute processor outputs (faster epoch loops)
MAX_LENGTH = 16000                  # Keep at 1 second (16 kHz * 1s). Shorten if all clips <1s (e.g. 12000)
NUM_WORKERS = 0                     # >0 can help if IO heavy; for CPU-bound WAV2VEC2 forward pass it's often not helpful
SEED = 42
SUPPRESS_WARNINGS = True

# --------------------------------------------------
# WARNING SUPPRESSION (optional)
# --------------------------------------------------
if SUPPRESS_WARNINGS:
    warnings.filterwarnings(
        "ignore",
        message=r"Passing `gradient_checkpointing` to a config initialization is deprecated",
        category=UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`evaluation_strategy` is deprecated",
        category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=r"torch.utils.checkpoint: please pass in use_reentrant",
        category=UserWarning
    )

# Avoid tokenizer parallel warning noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Thread control (tune if needed)
try:
    torch.set_num_threads(min(8, max(2, os.cpu_count() or 4)))
except Exception:
    pass
print(f"[Perf] torch.get_num_threads() = {torch.get_num_threads()}")


# --------------------------------------------------
# Dataset Definitions
# --------------------------------------------------
class SpeechCommandsDataset(Dataset):
    """
    Standard on-the-fly dataset (processor called inside __getitem__).
    Slower per step, but minimal memory footprint.
    """
    def __init__(self, file_paths, labels, processor, max_length=MAX_LENGTH):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def _prepare_audio(self, audio):
        if len(audio) > self.max_length:
            audio = audio[: self.max_length]
        elif len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode="constant")
        return audio

    def __getitem__(self, idx):
        audio, sr = load_audio_file(self.file_paths[idx], target_sr=SAMPLE_RATE)
        if audio is None:
            audio = np.zeros(self.max_length, dtype=np.float32)
        audio = self._prepare_audio(audio)

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class PrecomputedDataset(Dataset):
    """
    Faster iteration: processor already applied to all items.
    """
    def __init__(self, input_tensors: torch.Tensor, labels: torch.Tensor):
        self.inputs = input_tensors
        self.labels = labels

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return {
            "input_values": self.inputs[idx],
            "labels": self.labels[idx]
        }


def precompute_features(file_paths, labels, processor, max_length=MAX_LENGTH, desc="Precompute"):
    tensors = []
    lbls = []
    for fp, lab in tqdm(list(zip(file_paths, labels)), total=len(file_paths), desc=desc):
        audio, sr = load_audio_file(fp, target_sr=SAMPLE_RATE)
        if audio is None:
            audio = np.zeros(max_length, dtype=np.float32)
        if len(audio) > max_length:
            audio = audio[: max_length]
        elif len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)), mode="constant")
        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=True
        )
        tensors.append(inputs.input_values.squeeze(0))
        lbls.append(lab)
    return torch.stack(tensors), torch.tensor(lbls, dtype=torch.long)


# --------------------------------------------------
# Keyword Spotter
# --------------------------------------------------
class SpeechLLMKeywordSpotter:
    def __init__(self, keyword: str = KEYWORD, model_name: str = "facebook/wav2vec2-base"):
        self.keyword = keyword
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"[Device] {self.device}")
        print(f"[Keyword] {self.keyword}")
        print(f"[Model] {self.model_name}")

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = None
        self.performance_tracker = PerformanceTracker()
        self.training_history = {}
        self.test_results = {}

    # ----- Utilities -----
    def _print_trainable(self):
        total = 0
        trainable = 0
        for p in self.model.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        print(f"[Params] Trainable {trainable/1e6:.2f}M / Total {total/1e6:.2f}M "
              f"({100*trainable/total:.1f}%)")

    def set_seed(self, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ----- Model Init -----
    def initialize_model(
        self,
        num_labels: int = 2,
        freeze_feature_extractor: bool = True,
        freeze_encoder_layers: int = 0,
        gradient_checkpointing: bool = False
    ):
        print(f"[Init] Loading base model (labels={num_labels})")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        if freeze_feature_extractor:
            for p in self.model.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False
            print("[Freeze] Feature extractor")

        if freeze_encoder_layers > 0:
            frozen_params = 0
            for name, param in self.model.named_parameters():
                if name.startswith("wav2vec2.encoder.layers."):
                    layer_idx = int(name.split(".")[3])
                    if layer_idx < freeze_encoder_layers:
                        param.requires_grad = False
                        frozen_params += param.numel()
            print(f"[Freeze] Encoder layers < {freeze_encoder_layers} (params frozen: {frozen_params/1e6:.2f}M)")

        # Ensure gradient checkpoint setting
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("[GC] Gradient checkpointing ENABLED (slower on CPU).")
        else:
            self.model.gradient_checkpointing_disable()
            print("[GC] Gradient checkpointing DISABLED.")

        self.model.to(self.device)
        self._print_trainable()
        print("[Init] Model ready.")

    # ----- Metrics -----
    def compute_metrics(self, eval_pred):
        if isinstance(eval_pred, tuple):
            logits, labels = eval_pred
        else:
            logits = eval_pred.predictions
            labels = eval_pred.label_ids

        if isinstance(logits, tuple):
            logits = logits[0]

        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    # ----- Dataset Building -----
    def _build_datasets(self, splits, precompute: bool = False):
        if not precompute:
            train_ds = SpeechCommandsDataset(
                splits["train"]["files"], splits["train"]["labels"], self.processor
            )
            val_ds = SpeechCommandsDataset(
                splits["validation"]["files"], splits["validation"]["labels"], self.processor
            )
            test_ds = SpeechCommandsDataset(
                splits["test"]["files"], splits["test"]["labels"], self.processor
            )
        else:
            print("[Precompute] Building in‑memory tensors (train)...")
            tr_inputs, tr_labels = precompute_features(
                splits["train"]["files"], splits["train"]["labels"], self.processor
            )
            print("[Precompute] Validation...")
            va_inputs, va_labels = precompute_features(
                splits["validation"]["files"], splits["validation"]["labels"], self.processor, desc="Precompute (val)"
            )
            print("[Precompute] Test...")
            te_inputs, te_labels = precompute_features(
                splits["test"]["files"], splits["test"]["labels"], self.processor, desc="Precompute (test)"
            )
            train_ds = PrecomputedDataset(tr_inputs, tr_labels)
            val_ds = PrecomputedDataset(va_inputs, va_labels)
            test_ds = PrecomputedDataset(te_inputs, te_labels)

        print(f"[Data] Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)} (precompute={precompute})")
        return train_ds, val_ds, test_ds

    # ----- Training -----
    def train_model(
        self,
        data_splits,
        num_epochs: int = 4,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        output_dir: str = "models/speech_llm_model",
        freeze_feature_extractor: bool = True,
        freeze_encoder_layers: int = 10,
        evaluation_strategy: str = "epoch",  # keep for 4.41.2 compatibility
        early_stopping_patience: int = 2,
        gradient_checkpointing: bool = False,
        precompute: bool = False,
        seed: int = 42
    ):
        print("\n=== 🚀 Starting Wav2Vec2 Fine-Tuning (CPU-Optimized) ===\n")
        self.set_seed(seed)
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        self.initialize_model(
            num_labels=2,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_encoder_layers=freeze_encoder_layers,
            gradient_checkpointing=gradient_checkpointing
        )

        train_ds, val_ds, test_ds = self._build_datasets(data_splits, precompute=precompute)

        warmup_steps = 0  # On CPU small datasets, warmup often unnecessary

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=2000,        # Very infrequent to reduce overhead
            disable_tqdm=False,
            evaluation_strategy=evaluation_strategy,
            save_strategy=evaluation_strategy,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            lr_scheduler_type="linear",
            report_to=[],              # Avoid external logging dependencies
            gradient_checkpointing=gradient_checkpointing,
            fp16=False,
            dataloader_num_workers=NUM_WORKERS
        )

        callbacks = []
        if early_stopping_patience and early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )

        print("[Train] Loop starting...")
        train_result = trainer.train()
        total_time = time.time() - start_time
        print(f"[Train] Finished in {total_time/60:.2f} min")

        self.training_history = {
            "train_loss": train_result.training_loss,
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "time_seconds": total_time,
            "freeze_feature_extractor": freeze_feature_extractor,
            "freeze_encoder_layers": freeze_encoder_layers,
            "gradient_checkpointing": gradient_checkpointing,
            "precompute": precompute
        }

        print("[Eval] Validation metrics...")
        val_metrics = trainer.evaluate()

        print("[Eval] Test metrics...")
        test_output = trainer.predict(test_ds)
        logits = test_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        preds = np.argmax(logits, axis=-1)
        labels = test_output.label_ids
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

        test_accuracy = accuracy_score(labels, preds)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        test_auc = roc_auc_score(labels, probs)

        print("\n=== Test Results ===")
        print(f"Accuracy : {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall   : {test_recall:.4f}")
        print(f"F1       : {test_f1:.4f}")
        print(f"AUC      : {test_auc:.4f}")

        self.test_results = {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "auc": test_auc,
            "predictions": preds,
            "probabilities": probs,
            "labels": labels,
            "training_time_seconds": total_time,
        }

        self.performance_tracker.add_experiment(
            "wav2vec2_keyword_spotter_cpu_optimized",
            {
                "test_accuracy": test_accuracy,
                "test_f1": test_f1,
                "test_auc": test_auc,
                "training_time_minutes": total_time / 60,
            },
        )

        print("[Save] Model & processor")
        trainer.save_model()
        self.processor.save_pretrained(output_dir)

        return {
            "trainer": trainer,
            "validation_results": val_metrics,
            "test_results": self.test_results,
            "training_history": self.training_history,
        }

    # ----- Load -----
    def load_model(self, model_dir: str = "models/speech_llm_model"):
        try:
            print(f"[Load] {model_dir}")
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
            self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
            self.model.to(self.device).eval()
            print("[Load] Success")
            return True
        except Exception as e:
            print(f"[Load][Error] {e}")
            return False

    # ----- Predict -----
    def predict_audio_file(self, file_path: str, confidence_threshold: float = 0.7):
        if self.model is None:
            print("[Predict] Model not loaded.")
            return None
        audio, sr = load_audio_file(file_path, target_sr=SAMPLE_RATE)
        if audio is None:
            return None
        if len(audio) > MAX_LENGTH:
            audio = audio[:MAX_LENGTH]
        elif len(audio) < MAX_LENGTH:
            audio = np.pad(audio, (0, MAX_LENGTH - len(audio)), mode="constant")

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            max_length=MAX_LENGTH,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        p1 = probs[0, 1].item()
        pred = int(p1 >= confidence_threshold)
        return {
            "prediction": pred,
            "probability": p1,
            "detected": bool(pred),
            "confidence_level": "High" if p1 > 0.8 else "Medium" if p1 > 0.6 else "Low",
            "model_type": "Wav2Vec2ForSequenceClassification",
            "approach": "speech_llm",
        }

    # ----- Visualization -----
    def create_training_visualizations(self):
        if not self.test_results:
            print("[Viz] No test results.")
            return
        os.makedirs("results", exist_ok=True)

        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        vals = [
            self.test_results["accuracy"],
            self.test_results["precision"],
            self.test_results["recall"],
            self.test_results["f1"],
            self.test_results["auc"],
        ]
        bars = ax1.bar(metrics, vals, color="tomato", alpha=0.85)
        ax1.set_ylim(0, 1)
        ax1.set_title("Test Performance")
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        cm = confusion_matrix(self.test_results["labels"], self.test_results["predictions"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_title("Confusion Matrix")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")

        fpr, tpr, _ = roc_curve(self.test_results["labels"], self.test_results["probabilities"])
        ax3.plot(fpr, tpr, label=f"AUC {self.test_results['auc']:.3f}", color="crimson")
        ax3.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax3.set_title("ROC Curve")
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.legend()

        minutes = self.test_results["training_time_seconds"] / 60
        ax4.bar(["Training Time"], [minutes], color="orange")
        ax4.text(0, minutes + 0.2, f"{minutes:.1f} m", ha="center", va="bottom")
        ax4.set_ylabel("Minutes")
        ax4.set_title("Training Duration")

        plt.tight_layout()
        out = "results/speech_llm_performance.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[Viz] Saved {out}")

    # ----- Benchmark -----
    def benchmark_inference_speed(self, test_files, num_samples: int = 40):
        if self.model is None:
            print("[Benchmark] Model not loaded.")
            return None
        np.random.seed(SEED)
        subset = np.random.choice(test_files, min(num_samples, len(test_files)), replace=False)
        start = time.time()
        processed = 0
        for fp in tqdm(subset, desc="Benchmark"):
            r = self.predict_audio_file(fp)
            if r:
                processed += 1
        elapsed = time.time() - start
        if processed == 0:
            print("[Benchmark] No files processed.")
            return None
        avg = elapsed / processed
        sps = processed / elapsed
        print(f"[Benchmark] {processed} files in {elapsed:.2f}s | {avg*1000:.1f} ms/file | {sps:.2f} files/s")
        return {
            "total_time": elapsed,
            "avg_time_per_sample": avg,
            "samples_per_second": sps,
            "num_samples": processed
        }

    # ----- Save Info -----
    def save_model_info(self, filename: str = "models/speech_llm_info.pkl"):
        if not self.test_results:
            print("[SaveInfo] No test results.")
            return None
        info = {
            "keyword": self.keyword,
            "model_name": self.model_name,
            "approach": "speech_llm",
            "test_results": self.test_results,
            "training_history": self.training_history,
            "device": str(self.device),
        }
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        joblib.dump(info, filename)
        print(f"[SaveInfo] {filename}")
        return filename


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("=== SPEECH LLM KEYWORD SPOTTER (CPU Optimized) ===")
    if torch.cuda.is_available():
        print(f"[Env] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[Env] CPU mode")

    print("[Data] Loading splits...")
    splits = load_data_splits()

    spotter = SpeechLLMKeywordSpotter(keyword="yes")

    results = spotter.train_model(
        splits,
        num_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        freeze_feature_extractor=FREEZE_FEATURE_EXTRACTOR,
        freeze_encoder_layers=FREEZE_ENCODER_LAYERS,
        evaluation_strategy="epoch",
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        precompute=PRECOMPUTE_FEATURES,
        seed=SEED
    )

    spotter.create_training_visualizations()

    print("[Benchmark] Inference speed...")
    bench = spotter.benchmark_inference_speed(
        splits["test"]["files"],
        num_samples=40
    )

    spotter.save_model_info()
    spotter.performance_tracker.save_to_json("results/speech_llm_performance_tracking.json")

    print("\n=== ✅ Completed ===")
    print(f"Accuracy: {results['test_results']['accuracy']:.4f}")
    print(f"AUC     : {results['test_results']['auc']:.4f}")
    print(f"Train   : {results['test_results']['training_time_seconds']/60:.2f} min")
    if bench:
        print(f"Infer   : {bench['avg_time_per_sample']*1000:.1f} ms/sample")

    return spotter, results


if __name__ == "__main__":
    main()