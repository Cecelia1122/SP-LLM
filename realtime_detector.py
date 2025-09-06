"""
4_real_time_detector.py - Real-time keyword detection
Purpose: Live microphone detection using both classical and Speech LLM models

"""

import os
import sys
import pyaudio
import numpy as np
import threading
import queue
import time
import torch
import select
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any

from utils import extract_classical_features, SAMPLE_RATE, KEYWORD
from classical_approach import ClassicalKeywordSpotter
from speech_llm_approach import SpeechLLMKeywordSpotter


def _resolve_audio_path(file_path: str) -> Path:
    """
    Resolve a user-provided audio file path robustly:
    - Expands ~
    - If path starts with '/data/...', map to project 'data/...'
    - If still not found, try to locate by filename under 'data/'
    """
    p = Path(file_path).expanduser()
    if p.exists():
        return p

    # Map accidental absolute '/data/...' to local 'data/...'
    try:
        if str(p).startswith(os.sep + "data" + os.sep):
            local = Path("data") / p.relative_to(os.sep + "data")
            if local.exists():
                return local
    except Exception:
        pass

    # Try relative to project root explicitly
    local_rel = Path("data") / p.name
    if local_rel.exists():
        return local_rel

    # Fallback: search under data/ for the filename (first match)
    data_root = Path("data")
    if data_root.exists():
        for candidate in data_root.rglob(p.name):
            if candidate.is_file():
                return candidate

    # Return original (likely non-existent) to trigger a clear error message upstream
    return p


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Zero-mean and peak-normalize audio to [-1, 1]."""
    if audio.size == 0:
        return audio
    audio = audio.astype(np.float32)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if np.max(np.abs(audio)) > 0 else 1.0
    return audio / peak


def _select_peak_energy_window(audio: np.ndarray, sr: int, window_sec: float = 1.0, hop_sec: float = 0.05) -> np.ndarray:
    """
    Select the 1.0s window with the highest energy from a longer recording.
    If the recording is shorter than window length, pad to exact length.
    """
    target_len = int(sr * window_sec)
    if len(audio) <= target_len:
        # Pad or trim to exact length
        if len(audio) < target_len:
            pad = np.zeros(target_len - len(audio), dtype=np.float32)
            return np.concatenate([audio, pad])
        return audio[:target_len]

    hop = max(1, int(sr * hop_sec))
    best_energy = -1.0
    best_start = 0
    # Slide over the recording and find the max-energy window
    for start in range(0, len(audio) - target_len + 1, hop):
        segment = audio[start:start + target_len]
        energy = float(np.dot(segment, segment))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return audio[best_start:best_start + target_len]


class RealTimeKeywordDetector:
    def __init__(self, 
                 classical_model_path: str = "models/classical_keyword_spotter.pkl",
                 speech_llm_model_path: str = "models/speech_llm_model",
                 confidence_threshold_classical: float = 0.60,
                 confidence_threshold_llm: float = 0.70,
                 detection_cooldown: float = 1.5,
                 warmup_seconds: float = 1.5,
                 min_rms_dbfs: Optional[float] = None):
        """
        Initialize real-time keyword detector with both approaches
        
        Args:
            classical_model_path: Path to classical model
            speech_llm_model_path: Path to Speech LLM model
            confidence_threshold_classical: Threshold for classical predictions
            confidence_threshold_llm: Threshold for Speech LLM predictions
            detection_cooldown: Seconds between detections
            warmup_seconds: Ignore detections during this initial period (prevents startup blip)
            min_rms_dbfs: If provided, fixed energy gate in dBFS. If None, auto-calibrate from ambient noise.
        """
        
        self.threshold_classical = float(confidence_threshold_classical)
        self.threshold_llm = float(confidence_threshold_llm)
        self.detection_cooldown = float(detection_cooldown)
        self.WARMUP_SECONDS = float(warmup_seconds)
        self.user_min_rms_dbfs = min_rms_dbfs
        self.MIN_RMS_DBFS = float(min_rms_dbfs) if min_rms_dbfs is not None else None
        self.warmup_end_time = 0.0
        
        # Audio parameters
        self.CHUNK_SIZE = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = SAMPLE_RATE
        self.RECORD_SECONDS = 1.0  # continuous mode uses 1.0s sliding window
        
        # Buffer management
        self.audio_buffer = deque(maxlen=int(self.RATE * self.RECORD_SECONDS))
        self.audio_queue = queue.Queue(maxsize=10)
        
        # Detection tracking
        self.last_detection_time = 0
        self.is_running = False
        self.processing_thread = None
        
        # Load models
        self.classical_spotter = None
        self.speech_llm_spotter = None
        self.active_model = "classical"  # Default
        
        self._load_models(classical_model_path, speech_llm_model_path)

    @staticmethod
    def _rms_dbfs(x: np.ndarray) -> float:
        """
        Compute RMS in dBFS for raw int16-scale float array (±32768).
        Returns -120.0 for silence/near-zero.
        """
        if x.size == 0:
            return -120.0
        # Ensure float64 for stability
        x = x.astype(np.float64, copy=False)
        rms = np.sqrt(np.mean(x * x))
        if rms <= 1e-12:
            return -120.0
        return float(20.0 * np.log10(rms / 32768.0))

    def _calibrate_noise(self, stream, seconds: float = 1.5) -> float:
        """
        Sample ambient noise for 'seconds', compute baseline RMS dBFS over 1.0s windows,
        and set a dynamic energy gate slightly above the baseline.
        """
        print(f"🔧 Calibrating ambient noise for {seconds:.1f}s...")
        samples_needed = int(self.RATE * seconds)
        bytes_per_sample = 2
        collected = 0
        chunks = []

        # Discard a short preroll to flush old buffers
        preroll = int(self.RATE * 0.2)
        discarded = 0
        while discarded < preroll:
            data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
            discarded += len(data) // bytes_per_sample

        while collected < samples_needed:
            data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
            chunks.append(data)
            collected += len(data) // bytes_per_sample

        buf = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32)
        # Compute dbFS over 1s windows
        win = int(self.RATE * 1.0)
        if len(buf) < win:
            rms_db = self._rms_dbfs(buf)
            baseline = rms_db
        else:
            hop = win // 2
            vals = []
            for start in range(0, max(1, len(buf) - win + 1), hop):
                vals.append(self._rms_dbfs(buf[start:start + win]))
            baseline = float(np.median(vals)) if vals else self._rms_dbfs(buf)

        # Set gate a bit above baseline; clamp to a safe range
        gate = baseline + 6.0  # +6 dB above noise
        gate = max(min(gate, -35.0), -60.0)  # clamp [-60, -35]
        print(f"📏 Ambient baseline: {baseline:.1f} dBFS -> energy gate: {gate:.1f} dBFS")
        self.MIN_RMS_DBFS = gate
        return gate
        
    def _load_models(self, classical_path: str, speech_llm_path: str):
        """Load both classical and Speech LLM models"""
        print("🔧 Loading models...")
        
        # Load classical model
        try:
            self.classical_spotter = ClassicalKeywordSpotter()
            if self.classical_spotter.load_model(classical_path):
                print("✅ Classical model loaded successfully")
            else:
                print("⚠️ Classical model failed to load")
                self.classical_spotter = None
        except Exception as e:
            print(f"❌ Classical model loading error: {e}")
            self.classical_spotter = None
        
        # Load Speech LLM model
        try:
            self.speech_llm_spotter = SpeechLLMKeywordSpotter()
            if self.speech_llm_spotter.load_model(speech_llm_path):
                print("✅ Speech LLM model loaded successfully")
            else:
                print("⚠️ Speech LLM model failed to load")
                self.speech_llm_spotter = None
        except Exception as e:
            print(f"❌ Speech LLM model loading error: {e}")
            self.speech_llm_spotter = None
        
        # Determine available models
        available_models = []
        if self.classical_spotter:
            available_models.append("classical")
        if self.speech_llm_spotter:
            available_models.append("speech_llm")
        
        if not available_models:
            raise RuntimeError("❌ No models loaded successfully!")
        
        print(f"🎯 Available models: {', '.join(available_models)}")
        
        # Set default active model
        if "classical" in available_models:
            self.active_model = "classical"
        else:
            self.active_model = available_models[0]
        
        print(f"🔄 Active model: {self.active_model}")
    
    def switch_model(self, model_type: str):
        """Switch between classical and Speech LLM models"""
        target = model_type.strip().lower()
        if target in ("classical", "c"):
            if self.classical_spotter:
                self.active_model = "classical"
                print("🔄 Switched to Classical model")
            else:
                print("❌ Classical model not available")
        elif target in ("speech_llm", "llm", "speech", "s"):
            if self.speech_llm_spotter:
                self.active_model = "speech_llm"
                print("🔄 Switched to Speech LLM model")
            else:
                print("❌ Speech LLM model not available")
        elif target in ("m", "toggle"):
            if self.active_model == "classical" and self.speech_llm_spotter:
                self.active_model = "speech_llm"
                print("🔄 Toggled to Speech LLM model")
            elif self.active_model == "speech_llm" and self.classical_spotter:
                self.active_model = "classical"
                print("🔄 Toggled to Classical model")
            else:
                print("❌ No alternative model available to toggle")
        else:
            print(f"❌ Unknown model command: {model_type}")
        print(f"🤖 Active model now: {self.active_model}")
    
    def detect_keyword_from_audio(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect keyword from audio data using active model"""
        try:
            if self.active_model == "classical" and self.classical_spotter:
                # Extract classical features
                features = extract_classical_features(audio_data, self.RATE)
                if features is None:
                    return None
                
                # Scale and predict
                features_scaled = self.classical_spotter.scaler.transform(features.reshape(1, -1))
                model = self.classical_spotter.models[self.classical_spotter.best_model]['model']
                
                probability = float(model.predict_proba(features_scaled)[0][1])
                prediction = 1 if probability > self.threshold_classical else 0
                
                return {
                    'prediction': prediction,
                    'probability': probability,
                    'detected': prediction == 1,
                    'model_type': f'Classical_{self.classical_spotter.best_model}',
                    'approach': 'classical'
                }
            
            elif self.active_model == "speech_llm" and self.speech_llm_spotter:
                # Process with Speech LLM (expects ~1.0s, 16k samples)
                inputs = self.speech_llm_spotter.processor(
                    audio_data, 
                    sampling_rate=self.RATE, 
                    return_tensors="pt", 
                    padding=True,
                    max_length=16000,
                    truncation=True
                )
                
                # Move to device
                inputs = {k: v.to(self.speech_llm_spotter.device) for k, v in inputs.items()}
                
                # Predict
                with torch.no_grad():
                    outputs = self.speech_llm_spotter.model(**inputs)
                    predictions = torch.softmax(outputs.logits, dim=-1)
                
                probability = float(predictions[0][1].cpu().item())
                prediction = 1 if probability > self.threshold_llm else 0
                
                return {
                    'prediction': prediction,
                    'probability': probability,
                    'detected': prediction == 1,
                    'model_type': 'Speech_LLM_Wav2Vec2',
                    'approach': 'speech_llm'
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return None
    
    def process_audio_data(self, audio_data_raw: np.ndarray):
        """Process audio data and detect keyword with warm-up and energy gating"""

        # 1) Warm-up: ignore detections for the first WARMUP_SECONDS
        now = time.time()
        if now < self.warmup_end_time:
            return

        # 2) Pre-normalization energy gate (compute on raw int16-scale floats)
        if self.MIN_RMS_DBFS is not None:
            # Skip below-gate windows
            rms_dbfs = self._rms_dbfs(audio_data_raw)
            if rms_dbfs < self.MIN_RMS_DBFS:
                return

        # 3) Normalize before feature extraction / model inference
        audio_data = _normalize_audio(audio_data_raw)

        # 4) Detect keyword
        result = self.detect_keyword_from_audio(audio_data)
        
        if result and result['detected']:
            current_time = time.time()
            
            # Check cooldown period
            if current_time - self.last_detection_time > self.detection_cooldown:
                confidence_emoji = "🎯" if result['probability'] > 0.8 else "🎲" if result['probability'] > 0.6 else "🤔"
                
                print(f"{confidence_emoji} KEYWORD '{KEYWORD.upper()}' DETECTED!")
                print(f"   Model: {result['model_type']}")
                print(f"   Confidence: {result['probability']*100:.1f}%")
                print(f"   Approach: {result['approach']}")
                print()
                
                self.last_detection_time = current_time
    
    def audio_processing_worker(self):
        """Worker thread for processing audio data"""
        while self.is_running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                if audio_data is not None:
                    self.process_audio_data(audio_data)
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Worker thread error: {e}")
    
    def start_detection(self):
        """Start real-time keyword detection"""
        print("🎤 Starting Real-time Keyword Detection")
        print(f"📢 Say '{KEYWORD.upper()}' into your microphone")
        print(f"🤖 Active model: {self.active_model}")
        print("⌨️  Commands:")
        print("   - Type 'c' (or 'classical') + Enter to switch to Classical")
        print("   - Type 's' (or 'llm'/'speech') + Enter to switch to Speech LLM")
        print("   - Type 'm' (or 'toggle') + Enter to toggle models")
        print("   - Type 'q' + Enter to quit")
        print("   - Press Ctrl+C to force quit")
        print()
        
        p = pyaudio.PyAudio()
        stream = None
        
        try:
            # Find best input device
            default_device = p.get_default_input_device_info()
            print(f"🎙️ Using audio device: {default_device['name']}")
            
            # Create audio stream
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE,
                input_device_index=default_device['index']
            )
            
            # Set warm-up end time
            self.warmup_end_time = time.time() + self.WARMUP_SECONDS

            # If user didn't supply a fixed gate, calibrate ambient noise
            if self.user_min_rms_dbfs is None:
                try:
                    self._calibrate_noise(stream, seconds=self.WARMUP_SECONDS)
                except Exception as e:
                    # Fallback: disable gate on calibration failure
                    print(f"⚠️ Noise calibration failed: {e} (disabling energy gate)")
                    self.MIN_RMS_DBFS = None

            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(target=self.audio_processing_worker, daemon=True)
            self.processing_thread.start()
            
            print("✅ Detection started! Listening...")

            # Non-blocking command reader loop (in its own thread)
            def command_input():
                print("⌨️  Enter commands anytime (c/s/m/q):")
                while self.is_running:
                    try:
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
                        if rlist:
                            cmd = sys.stdin.readline().strip()
                            if not cmd:
                                continue
                            if cmd.lower() in ('q', 'quit', 'exit'):
                                print("🛑 Quit command received")
                                self.is_running = False
                                break
                            else:
                                self.switch_model(cmd.lower())
                    except Exception:
                        # Don't kill the app if input fails
                        time.sleep(0.2)

            input_thread = threading.Thread(target=command_input, daemon=True)
            input_thread.start()
            
            # Main audio capture loop
            while self.is_running:
                try:
                    # Read audio data
                    data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    
                    # Add to buffer
                    self.audio_buffer.extend(audio_chunk)
                    
                    # Process when buffer is full (1.0s)
                    if len(self.audio_buffer) >= int(self.RATE * self.RECORD_SECONDS):
                        audio_window = np.array(list(self.audio_buffer))
                        
                        # Add to processing queue if not full
                        if not self.audio_queue.full():
                            self.audio_queue.put(audio_window.copy())
                        
                        # Clear part of buffer (50% overlap)
                        for _ in range(len(self.audio_buffer) // 2):
                            self.audio_buffer.popleft()

                    # Yield a tiny bit so input thread can run smoothly
                    time.sleep(0.003)
                
                except OSError as e:
                    if hasattr(e, "errno") and e.errno == -9981:  # Input overflow (macOS)
                        print("⚠️ Audio buffer overflow - continuing...")
                        continue
                    else:
                        raise e
                        
                except KeyboardInterrupt:
                    print("\n🛑 Keyboard interrupt - stopping detection...")
                    break
                    
        except Exception as e:
            print(f"❌ Detection error: {e}")
            
        finally:
            # Cleanup
            print("🧹 Cleaning up resources...")
            self.is_running = False
            
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
            
            try:
                p.terminate()
            except:
                pass
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            
            print("✅ Cleanup complete!")


class SimpleDetector:
    """Simplified detector for testing and demonstration (Mode 2 & 3)"""
    
    def __init__(self, 
                 classical_model_path: str = "models/classical_keyword_spotter.pkl",
                 speech_llm_model_path: str = "models/speech_llm_model",
                 record_seconds: float = 2.5,
                 threshold: float = 0.7):
        
        # Load models
        self.classical_spotter = ClassicalKeywordSpotter()
        self.classical_loaded = self.classical_spotter.load_model(classical_model_path)
        
        self.speech_llm_spotter = SpeechLLMKeywordSpotter()
        self.speech_llm_loaded = self.speech_llm_spotter.load_model(speech_llm_model_path)
        
        self.CHUNK_SIZE = 1024
        self.RATE = SAMPLE_RATE
        self.RECORD_SECONDS = float(record_seconds)
        self.THRESH = float(threshold)

    def _record_exact_seconds(self, stream, seconds: float) -> np.ndarray:
        """
        Record exactly 'seconds' of audio using precise sample counting.
        Includes a small pre-roll discard to avoid stale buffered audio.
        """
        target_samples = int(self.RATE * seconds)
        bytes_per_sample = 2  # int16
        samples_collected = 0
        chunks = []

        # Discard small pre-roll (~150ms) to avoid buffered stale data
        pre_discard_samples = int(self.RATE * 0.15)
        discarded = 0
        while discarded < pre_discard_samples:
            data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
            discarded += len(data) // bytes_per_sample

        while samples_collected < target_samples:
            data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
            chunks.append(data)
            samples_collected += len(data) // bytes_per_sample

        audio = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32)

        # Trim or pad to exact length
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            pad = np.zeros(target_samples - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])

        return audio
    
    def test_both_models(self, audio_file_path: str):
        """Test both models on an audio file"""
        resolved = _resolve_audio_path(audio_file_path)
        if not resolved.exists():
            print(f"❌ File not found: {audio_file_path}")
            hint = ""
            if str(audio_file_path).startswith("/data/"):
                hint = "\n💡 Tip: Use 'data/...' (relative path) instead of '/data/...'."
            print(f"Please provide a valid path.{hint}")
            return {}

        print(f"🧪 Testing both models on: {resolved}")
        
        results = {}
        
        # Test classical model
        if self.classical_loaded:
            result = self.classical_spotter.predict_audio_file(str(resolved))
            results['classical'] = result
            if result:
                print(f"🔍 Classical: {result['probability']:.3f} confidence - {'✅ DETECTED' if result['detected'] else '❌ NOT DETECTED'}")
        
        # Test Speech LLM model
        if self.speech_llm_loaded:
            result = self.speech_llm_spotter.predict_audio_file(str(resolved))
            results['speech_llm'] = result
            if result:
                print(f"🤖 Speech LLM: {result['probability']:.3f} confidence - {'✅ DETECTED' if result['detected'] else '❌ NOT DETECTED'}")
        
        return results
    
    def record_and_test(self):
        """Record audio and test with both models (uses peak 1.0s window)"""
        print("🎤 Simple Detection Mode")
        print("📢 Press Enter to record and test, or 'quit' to exit")
        
        p = pyaudio.PyAudio()
        stream = None
        
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            while True:
                user_input = input("\n⏯️ Press Enter to start 3-2-1 countdown and record "
                                   f"({self.RECORD_SECONDS:.1f}s), or type 'quit': ").strip()
                if user_input.lower() == 'quit':
                    break
                
                # Countdown so you have time to prepare
                print("⏱️ Recording in:")
                for t in [3, 2, 1]:
                    print(f"  {t}...")
                    time.sleep(1.0)
                print("🎙️ Recording... Speak now!")
                
                # Record exact duration
                audio_data = self._record_exact_seconds(stream, self.RECORD_SECONDS)
                
                print("✅ Recording complete, processing...")
                
                # Normalize and select best 1.0s window
                audio_data = _normalize_audio(audio_data)
                one_sec = _select_peak_energy_window(audio_data, self.RATE, window_sec=1.0, hop_sec=0.05)
                
                # Test classical model
                if self.classical_loaded:
                    try:
                        features = extract_classical_features(one_sec, self.RATE)
                        if features is not None:
                            features_scaled = self.classical_spotter.scaler.transform(features.reshape(1, -1))
                            model = self.classical_spotter.models[self.classical_spotter.best_model]['model']
                            prob_classical = float(model.predict_proba(features_scaled)[0][1])
                            
                            print(f"🔍 Classical Model: {prob_classical:.3f} confidence - {'✅ DETECTED' if prob_classical > self.THRESH else '❌ NOT DETECTED'}")
                    except Exception as e:
                        print(f"❌ Classical model error: {e}")
                
                # Test Speech LLM model
                if self.speech_llm_loaded:
                    try:
                        inputs = self.speech_llm_spotter.processor(
                            one_sec, 
                            sampling_rate=self.RATE, 
                            return_tensors="pt", 
                            padding=True,
                            max_length=16000,
                            truncation=True
                        )
                        
                        inputs = {k: v.to(self.speech_llm_spotter.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = self.speech_llm_spotter.model(**inputs)
                            predictions = torch.softmax(outputs.logits, dim=-1)
                        
                        prob_speech_llm = float(predictions[0][1].cpu().item())
                        print(f"🤖 Speech LLM: {prob_speech_llm:.3f} confidence - {'✅ DETECTED' if prob_speech_llm > self.THRESH else '❌ NOT DETECTED'}")
                        
                    except Exception as e:
                        print(f"❌ Speech LLM error: {e}")
        
        finally:
            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
            finally:
                p.terminate()


def main():
    """Main function for real-time detection"""
    print("=== REAL-TIME KEYWORD DETECTION ===\n")
    
    print("Choose detection mode:")
    print("1. Full real-time detection (continuous listening)")
    print("2. Simple manual detection (record on demand)")
    print("3. Test specific audio file")
    
    choice = input("Enter choice (1-3): ").strip()
    
    try:
        if choice == "1":
            # Full real-time detection
            detector = RealTimeKeywordDetector(
                confidence_threshold_classical=0.60,
                confidence_threshold_llm=0.70,
                detection_cooldown=1.5,
                warmup_seconds=1.5,
                min_rms_dbfs=None  # auto-calibrate; set to -120.0 to disable energy gate
            )
            detector.start_detection()
            
        elif choice == "2":
            # Simple manual detection
            detector = SimpleDetector()
            detector.record_and_test()
            
        elif choice == "3":
            # Test specific file
            file_path = input("Enter audio file path: ").strip()
            if not file_path:
                print("❌ No file path provided")
                return
            
            detector = SimpleDetector()
            detector.test_both_models(file_path)
            
        else:
            print("❌ Invalid choice")
            
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("💡 Make sure to train the models first using:")
        print("   python 2_classical_approach.py")
        print("   python 3_speech_llm_approach.py")
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")


if __name__ == "__main__":
    main()