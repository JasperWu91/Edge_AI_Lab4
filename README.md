# Edge AI Lab4 - YOLOv8 + Ollama Report Generator
This project demonstrates an **Edge AI application** that integrates the **Ultralytics YOLO model** with the **LLM model `llama3.2:1B`** to perform real-time object detection and generate **automated summaries**. It supports input from images, folders, videos, and USB cameras, and captures detection results every 10 seconds over a 30-second window for summarization using Ollama.

---

## Features

- Real-time object detection using YOLOv11.
- Supports multiple input sources: image file, folder, video file, USB webcam, or PiCamera.
- Records detection results at 10s, 20s, and 30s into a JSON file.
- Automatically generates a short 2–3 sentence summary using Ollama LLM (`llama3.2`).

## Requirements
- Python 3.8 or later
- Required packages:  ultralytics opencv-python ollama numpy

```bash
pip install ultralytics opencv-python ollama numpy
```
##Basic Command
```bash
python detect_report.py \
  --model runs/detect/train/weights/best.pt \
  --source usb0 \
  --thresh 0.4 \
  --resolution 640x480 \
  --record
```

