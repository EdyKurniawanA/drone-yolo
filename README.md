# Real-time YOLO Drone Monitoring

Monitoring drone's camera with YOLO inference in real time.

## Overview
- **Goal**: Real-time YOLO detection with GPS sync and logging.
- **OS**: Windows 10/11
- **Python**: 3.10+
- **Run style**: Anaconda environment; start GUI via `python app/gui.py`.
- **Models/weights**: Stored via Git LFS in this repo.

## Features
- Real-time YOLO inference on drone camera streams.
- GPS time sync and NMEA parsing.
- CSV logging for detections and system stats.
- Quick tests to validate prerequisites before full run.

## Prerequisites
- CUDA + cuDNN (optional, for GPU acceleration)
- PyTorch (CPU or CUDA build)
- Ultralytics/YOLO
- OpenCV
- FFmpeg (and RTMP source if streaming)
- GPS device or NMEA source

## Setup
1) Install Anaconda and create an env (Python 3.10+):
```bash
conda create -n drone-yolo python=3.10 -y
conda activate drone-yolo
```

2) Install core dependencies (pick CPU or CUDA):
```bash
# PyTorch - choose the right index-url for your GPU/driver
# CPU example:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Common libs
pip install ultralytics opencv-python pyqt5 ffmpeg-python pynmea2 psutil pandas
```

3) Ensure FFmpeg is available in PATH (or provide full path in your config).

## Quick Hardware/Software Checks
Run these to validate your environment before the full app:
- Phase 1 (`quick_test_phase1`):
  - `check-cpu.py`, `check-gpu.py`, `opencv-webcam.py`, `check-pyqt5.py`, `check-nmea.py`, `udp-listener.py`, `gps-test.py`
- Phase 2 (`quick_test_phase2`):
  - `rtmp-yolo-inference.py`, `yolo-inference.py`, `open-rtmp.py`, `phase2_harness.py`, `threaded-test-harness.py`, `gps-socket.py`
- Phase 3 (`quick_test_phase3`):
  - `phase3-harness.py`, `logger.py`, `sys-stats.py`, `gps-sync.py`, `class_counter.py`

Example:
```bash
conda activate drone-yolo
python quick_test_phase1/check-cpu.py
python quick_test_phase1/check-gpu.py
python quick_test_phase1/check-pyqt5.py
python quick_test_phase1/opencv-webcam.py
```

## Running the GUI
```bash
conda activate drone-yolo
python app/gui.py
```

## Models (Git LFS)
- YOLO weights (`*.pt`) are tracked by Git LFS and pulled automatically on clone.
- If you add new large/binary files, track them:
```bash
git lfs track "*.mp4"
git add .gitattributes
```

## Logs
- CSV logs in `logs/` (ignored from Git except where explicitly saved).

## Roadmap
- Add screenshots and demo near project end.
- Optional: packaged releases.

## License
None (all rights reserved) for now. Consider MIT/Apache-2.0 if you want others to use/contribute.
