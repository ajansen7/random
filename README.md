# Random Scripts Repository

A collection of various scripts, experiments, and miscellaneous code.

## Projects

### 🧮 Aspens Calculator (`aspens-calculator/`)
- **Purpose**: Computer vision scripts likely for tile or board analysis.
- **Key Tools**: `opencv-python`, `numpy`.
- **Contents**: 
  - Several versions of a calculator (`calc_v*.py`)
  - `training.py` and `training-v2.py` for model training/data collection.
  - `tile-finder.py` for image analysis.

### 🦕 Dino Gen (`dino-gen/`)
- **Purpose**: QR code stitching and image processing, including a Streamlit app.
- **Key Tools**: `streamlit`, `opencv-python`, `pyzbar`.
- **Contents**:
  - `dino_app.py`: Streamlit-based application.
  - `qr_auto_stitch.py`, `qr_manual.py`, `qr_sticter.py`: Tools for QR code processing.

### 🍎 New Fruit (`new-fruit/`)
- **Purpose**: A static web page showcasing fruit/meat hybrids.
- **Key Tools**: HTML, CSS, assets (images and audio).
- **Contents**:
  - `index.html`: The main page.
  - Various image assets like `apple_steak.png` and `banana_sausage.png`.

## Setup

### Python Dependencies
To install the necessary Python libraries:

```bash
pip install -r requirements.txt
```

Note: `pyzbar` may require additional system-level libraries (e.g., `libzbar-dev` on Linux or `zbar` via Homebrew on macOS).

## Contributing
Feel free to add your own random scripts or improve existing ones!
