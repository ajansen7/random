# 🎙️ Interview Recorder

A high-fidelity audio recording and transcription tool powered by OpenAI's Whisper (Turbo model). Designed for rapid interviews, this script provides real-time feedback and automatic clipboard integration.

## ✨ Features

- **Real-time Feedback**: Tracks recording duration and estimated file size.
- **Interactive Controls**: Pause, Resume, and Stop with simple keyboard shortcuts.
- **Fast Transcription**: Uses the Whisper `turbo` model for high-speed, accurate English transcription.
- **Smart Clipboard**: Automatically prompts to copy the transcription to your clipboard (defaults to "Yes").
- **Timestamped Storage**: Saves both `.wav` audio and `.txt` transcriptions in an `outputs/` folder.

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `[Space]` | Pause / Resume Recording |
| `[Enter]` | Stop Recording & Start Transcription |

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) (required by Whisper)
- PortAudio (required by sounddevice)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd interview-recorder
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Usage

Simply run the script:

```bash
python recorder.py
```

Follow the on-screen instructions. Once you stop the recording, the script will:
1. Save the audio to `outputs/interview_YYYYMMDD_HHMMSS.wav`.
2. Transcribe the audio using the Whisper Turbo model.
3. Save the text to `outputs/interview_YYYYMMDD_HHMMSS.txt`.
4. Ask if you want to copy the result to your clipboard (just hit `Enter` for "Yes").

## 📦 Requirements

- `openai-whisper`
- `sounddevice`
- `numpy`
- `scipy`
- `pyperclip`
- `pynput`

## ⚖️ License

MIT
