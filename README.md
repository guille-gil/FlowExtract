# Decomposed Parsing from Industrial Troubleshooting Guides

A modular pipeline for extracting procedural knowledge from industrial troubleshooting diagrams using YOLO detection, classification, OCR, arrow detection, and LLM reasoning.

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Train YOLO Detector

**For small datasets (40-50 images):**

Label your images with bounding boxes (3 classes: observation, decision, action) using [Roboflow](https://roboflow.com/) or [LabelImg](https://github.com/heartexlabs/labelImg).

Train with aggressive augmentation:
```bash
yolo detect train \
    data=yolo_data.yaml \
    model=yolov8s.pt \
    epochs=250 \
    imgsz=640 \
    batch=8 \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4 \
    degrees=10 \
    translate=0.1 \
    scale=0.5 \
    flipud=0.5 \
    fliplr=0.5 \
    mosaic=1.0 \
    mixup=0.1

cp runs/detect/train/weights/best.pt data/models/yolo_detector.pt
```

**Note:** YOLOv8s (not YOLOv8n) works better with small datasets. Built-in augmentation effectively multiplies your dataset.

### 3. Configure LLM (for MacBook Air)

Edit `configs/pipeline_config.yaml`:

**Option 1: Smaller model (recommended)**
```yaml
reasoning:
  model_name: "Qwen/Qwen2.5-3B-Instruct"  # Faster, lower memory
```

**Option 2: Quantized 7B model**
```yaml
reasoning:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  load_in_4bit: true  # Reduces memory to ~4-5GB
```

### 4. Train Classifier (Optional)
```bash
python -m src.train --data-dir element_crops --epochs 50
```

### 5. Run Pipeline
```bash
python -m src.main
```

## Project Structure

```
├── src/                      # All source code
│   ├── detection/           # YOLO element detection
│   ├── classification/      # Element type classification
│   ├── ocr/                # Text extraction
│   ├── arrows/             # Arrow detection
│   ├── reasoning/          # LLM reasoning
│   ├── evaluation/         # Evaluation metrics
│   ├── utils/              # Utilities
│   ├── main.py             # Main pipeline
│   ├── train.py            # Train classifier
│   └── evaluate.py         # Evaluate results
├── data/
│   ├── input/
│   │   ├── tocaps/         # Input images
│   │   └── annotations/    # Ground truth
│   ├── intermediate/       # Stage outputs
│   └── models/             # Trained weights
├── docs/                   # Documentation
├── configs/                # Configuration
└── notebooks/              # Exploration
```

## Usage

**Run full pipeline:**
```bash
python -m src.main
```

**Run specific stage:**
```bash
python -m src.main --stage 1  # Detection only
```

**Train classifier:**
```bash
python -m src.train --data-dir element_crops --epochs 50
```

**Evaluate:**
```bash
python -m src.evaluate \
    --predictions-dir data/intermediate/reasoning \
    --ground-truth-dir data/input/annotations
```

## Pipeline Stages

1. **Detection** (YOLO) → Bounding boxes
2. **Classification** (CNN) → Element types  
3. **OCR** (PaddleOCR/EasyOCR) → Text content
4. **Arrows** (Hough) → Connection graph
5. **Reasoning** (LLM) → Entities & relations

## Research Setup Notes

**Small Dataset (40-50 images):**
- Use YOLOv8s (not YOLOv8n) for better learning
- YOLO's built-in augmentation is sufficient
- Expected detection accuracy: 70-80%
- Suitable for proof-of-concept research

**MacBook Air Constraints:**
- Use Qwen2.5-3B or 4-bit quantized 7B
- Expect 30-60 seconds per page for LLM reasoning
- 16GB RAM recommended (8GB minimum with quantization)

## Configuration

Edit `configs/pipeline_config.yaml` to customize:
- YOLO model path and thresholds
- LLM model size and quantization
- OCR engine (PaddleOCR/EasyOCR)
- All pipeline parameters

See `docs/` for detailed documentation.

## Requirements

- Python 3.8+
- PyTorch, Ultralytics (YOLO)
- PaddleOCR or EasyOCR
- Transformers (Qwen2.5)

See `requirements.txt` for full list.
