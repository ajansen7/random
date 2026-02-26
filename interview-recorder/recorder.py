import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import time
import sys
from datetime import datetime
import pyperclip
from pynput import keyboard

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

# Configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1
MODEL_SIZE = "turbo"  # Optimized for fast English transcription

# Recording State
class State:
    RECORDING = 1
    PAUSED = 2
    STOPPED = 3

current_state = State.RECORDING

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def format_size(num_bytes):
    """Formats bytes into a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"

def on_press(key):
    global current_state
    try:
        if key == keyboard.Key.space:
            if current_state == State.RECORDING:
                current_state = State.PAUSED
            elif current_state == State.PAUSED:
                current_state = State.RECORDING
        elif key == keyboard.Key.enter:
            current_state = State.STOPPED
    except AttributeError:
        pass

def record_audio():
    """Records audio from the microphone with real-time feedback."""
    global current_state
    audio_data = []
    start_time = time.time()
    total_paused_time = 0
    pause_start = 0
    
    def callback(indata, frames, time_info, status):
        if status:
            print(f"\nStatus: {status}", file=sys.stderr)
        if current_state == State.RECORDING:
            audio_data.append(indata.copy())

    print("\n" + "="*45)
    print("🎙️  INTERVIEW RECORDER")
    print("---------------------------------------------")
    print("⌨️  [Space]  : Pause / Resume")
    print("⌨️  [Enter]  : Stop & Transcribe")
    print("="*45)
    print("🔴 RECORDING STARTED")

    current_state = State.RECORDING
    
    # Start the non-blocking keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
            while current_state != State.STOPPED:
                if current_state == State.PAUSED:
                    if pause_start == 0:
                        pause_start = time.time()
                    
                    sys.stdout.write(f"\r⏸️  PAUSED  | [Space] to Resume   ")
                    sys.stdout.flush()
                else:
                    if pause_start != 0:
                        total_paused_time += (time.time() - pause_start)
                        pause_start = 0
                    
                    elapsed = time.time() - start_time - total_paused_time
                    estimated_bytes = elapsed * SAMPLE_RATE * 2 
                    sys.stdout.write(f"\r⏱️  Time: {elapsed:.1f}s | 📦 Size: {format_size(estimated_bytes)} | [Space] Pause | [Enter] Stop   ")
                    sys.stdout.flush()
                
                time.sleep(0.1)
    finally:
        listener.stop()

    print("\n" + "="*45)
    print("🛑 RECORDING STOPPED")
    print("="*45)
    
    if not audio_data:
        return None
        
    return np.concatenate(audio_data, axis=0)

def transcribe_audio(filename):
    """Transcribes the given audio file using Whisper."""
    print("\n📦 Loading Whisper model (turbo)... this may take a few seconds...")
    import whisper 
    
    model = whisper.load_model(MODEL_SIZE)
    
    print("🧠 Transcribing response...")
    start_time = time.time()
    result = model.transcribe(filename)
    duration = time.time() - start_time
    
    print(f"✅ Finished in {duration:.2f}s")
    return result["text"].strip()

def main():
    try:
        # 1. Start Recording
        audio = record_audio()
        if audio is None:
            print("No audio recorded.")
            return

        # 2. Generate timestamped filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_filename = os.path.join(OUTPUT_DIR, f"interview_{timestamp}.wav")
        txt_filename = os.path.join(OUTPUT_DIR, f"interview_{timestamp}.txt")
        
        # 3. Save audio file
        print(f"💾 Saving audio to {wav_filename}...")
        wav.write(wav_filename, SAMPLE_RATE, audio)
        
        # 4. Transcribe
        transcription = transcribe_audio(wav_filename)
        
        # 5. Output results
        print("\n--- 📝 TRANSCRIPTION ---")
        print(transcription)
        print("------------------------\n")
        
        # 6. Save transcription to file
        with open(txt_filename, "w") as f:
            f.write(transcription)
        
        print(f"📁 Files saved in: {OUTPUT_DIR}")
        print(f"   - Audio: {os.path.basename(wav_filename)}")
        print(f"   - Text:  {os.path.basename(txt_filename)}")
        
        # 7. Clipboard Copy
        print("\n📋 Copy transcription to clipboard? [Y/n]")
        
        # Clear input buffer to prevent skipping the prompt
        try:
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass
            
        choice = input().lower().strip()
        
        if choice in ('n', 'no'):
            print("⏭️  Skipped clipboard copy.")
        else:
            try:
                pyperclip.copy(transcription)
                print("✨ Copied to clipboard!")
            except Exception as e:
                print(f"❌ Failed to copy to clipboard: {e}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
