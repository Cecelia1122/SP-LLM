"""
4_real_time_detector.py - Real-time keyword detection
Purpose: Live microphone detection using both classical and Speech LLM models
"""

import pyaudio
import numpy as np
import threading
import queue
import time
import joblib
import torch
from collections import deque
from typing import Optional, Dict, Any

from utils import extract_classical_features, SAMPLE_RATE, KEYWORD
from classical_approach import ClassicalKeywordSpotter
from speech_llm_approach import SpeechLLMKeywordSpotter

class RealTimeKeywordDetector:
    def __init__(self, 
                 classical_model_path: str = "models/classical_keyword_spotter.pkl",
                 speech_llm_model_path: str = "models/speech_llm_model",
                 confidence_threshold: float = 0.7,
                 detection_cooldown: float = 2.0):
        """
        Initialize real-time keyword detector with both approaches
        
        Args:
            classical_model_path: Path to classical model
            speech_llm_model_path: Path to Speech LLM model
            confidence_threshold: Minimum confidence for detection
            detection_cooldown: Seconds between detections
        """
        
        self.confidence_threshold = confidence_threshold
        self.detection_cooldown = detection_cooldown
        
        # Audio parameters
        self.CHUNK_SIZE = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = SAMPLE_RATE
        self.RECORD_SECONDS = 1.0
        
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
        if model_type == "classical" and self.classical_spotter:
            self.active_model = "classical"
            print("🔄 Switched to Classical model")
        elif model_type == "speech_llm" and self.speech_llm_spotter:
            self.active_model = "speech_llm"
            print("🔄 Switched to Speech LLM model")
        else:
            print(f"❌ Model '{model_type}' not available")
    
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
                
                probability = model.predict_proba(features_scaled)[0][1]
                prediction = 1 if probability > self.confidence_threshold else 0
                
                return {
                    'prediction': prediction,
                    'probability': probability,
                    'detected': prediction == 1,
                    'model_type': f'Classical_{self.classical_spotter.best_model}',
                    'approach': 'classical'
                }
            
            elif self.active_model == "speech_llm" and self.speech_llm_spotter:
                # Process with Speech LLM
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
                
                probability = predictions[0][1].cpu().item()
                prediction = 1 if probability > self.confidence_threshold else 0
                
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
    
    def process_audio_data(self, audio_data: np.ndarray):
        """Process audio data and detect keyword"""
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Detect keyword
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
        print("   - Press 'c' + Enter to switch to Classical model")
        print("   - Press 's' + Enter to switch to Speech LLM model")
        print("   - Press 'q' + Enter to quit")
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
            
            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(target=self.audio_processing_worker, daemon=True)
            self.processing_thread.start()
            
            print("✅ Detection started! Listening...")
            
            # Start command input thread
            def command_input():
                while self.is_running:
                    try:
                        cmd = input().strip().lower()
                        if cmd == 'q':
                            print("🛑 Quit command received")
                            self.is_running = False
                            break
                        elif cmd == 'c':
                            self.switch_model("classical")
                        elif cmd == 's':
                            self.switch_model("speech_llm")
                    except:
                        break
            
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
                    
                    # Process when buffer is full
                    if len(self.audio_buffer) >= int(self.RATE * self.RECORD_SECONDS):
                        audio_window = np.array(list(self.audio_buffer))
                        
                        # Add to processing queue if not full
                        if not self.audio_queue.full():
                            self.audio_queue.put(audio_window.copy())
                        
                        # Clear part of buffer (50% overlap)
                        for _ in range(len(self.audio_buffer) // 2):
                            self.audio_buffer.popleft()
                
                except OSError as e:
                    if e.errno == -9981:  # Input overflow
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
    """Simplified detector for testing and demonstration"""
    
    def __init__(self, 
                 classical_model_path: str = "models/classical_keyword_spotter.pkl",
                 speech_llm_model_path: str = "models/speech_llm_model"):
        
        # Load models
        self.classical_spotter = ClassicalKeywordSpotter()
        self.classical_loaded = self.classical_spotter.load_model(classical_model_path)
        
        self.speech_llm_spotter = SpeechLLMKeywordSpotter()
        self.speech_llm_loaded = self.speech_llm_spotter.load_model(speech_llm_model_path)
        
        self.CHUNK_SIZE = 4096
        self.RATE = SAMPLE_RATE
        self.RECORD_SECONDS = 2.0
    
    def test_both_models(self, audio_file_path: str):
        """Test both models on an audio file"""
        print(f"🧪 Testing both models on: {audio_file_path}")
        
        results = {}
        
        # Test classical model
        if self.classical_loaded:
            result = self.classical_spotter.predict_audio_file(audio_file_path)
            results['classical'] = result
            if result:
                print(f"🔍 Classical: {result['probability']:.3f} confidence - {'✅ DETECTED' if result['detected'] else '❌ NOT DETECTED'}")
        
        # Test Speech LLM model
        if self.speech_llm_loaded:
            result = self.speech_llm_spotter.predict_audio_file(audio_file_path)
            results['speech_llm'] = result
            if result:
                print(f"🤖 Speech LLM: {result['probability']:.3f} confidence - {'✅ DETECTED' if result['detected'] else '❌ NOT DETECTED'}")
        
        return results
    
    def record_and_test(self):
        """Record audio and test with both models"""
        print("🎤 Simple Detection Mode")
        print("📢 Press Enter to record and test, or 'quit' to exit")
        
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            while True:
                user_input = input("\n⏯️ Press Enter to record (or 'quit'): ").strip()
                if user_input.lower() == 'quit':
                    break
                
                print("🎙️ Recording... speak now!")
                
                # Record audio
                frames = []
                for _ in range(int(self.RATE / self.CHUNK_SIZE * self.RECORD_SECONDS)):
                    data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
                
                print("✅ Recording complete, processing...")
                
                # Convert to audio array
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
                
                # Normalize
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Test classical model
                if self.classical_loaded:
                    features = extract_classical_features(audio_data, self.RATE)
                    if features is not None:
                        features_scaled = self.classical_spotter.scaler.transform(features.reshape(1, -1))
                        model = self.classical_spotter.models[self.classical_spotter.best_model]['model']
                        prob_classical = model.predict_proba(features_scaled)[0][1]
                        
                        print(f"🔍 Classical Model: {prob_classical:.3f} confidence - {'✅ DETECTED' if prob_classical > 0.7 else '❌ NOT DETECTED'}")
                
                # Test Speech LLM model
                if self.speech_llm_loaded:
                    try:
                        inputs = self.speech_llm_spotter.processor(
                            audio_data, 
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
                        
                        prob_speech_llm = predictions[0][1].cpu().item()
                        print(f"🤖 Speech LLM: {prob_speech_llm:.3f} confidence - {'✅ DETECTED' if prob_speech_llm > 0.7 else '❌ NOT DETECTED'}")
                        
                    except Exception as e:
                        print(f"❌ Speech LLM error: {e}")
        
        finally:
            stream.stop_stream()
            stream.close()
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
                confidence_threshold=0.7,
                detection_cooldown=2.0
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