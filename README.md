# Decomposed Parsing from Industrial Troubleshooting Guides

A modular pipeline for extracting procedural knowledge from industrial troubleshooting diagrams using YOLO detection, classification, OCR, arrow detection, and LLM reasoning.

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Train YOLO Detector

**For small datasets (40-50 images):**

Label your images with bounding boxes (4 classes: observation, decision, action, arrowhead) using [Roboflow](https://roboflow.com/) or [LabelImg](https://github.com/heartexlabs/labelImg).

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
```bash
python -m src.main
```

## Project Structure

```
├── src/                    # Source code
│   ├── main.py            # Main pipeline script
│   ├── evaluate.py        # Evaluation script
│   ├── pipeline/          # All 3 pipeline stages
│   │   ├── stage1_detector.py      # YOLO detection (6 classes)
│   │   ├── stage2_ocr.py           # Text extraction
│   │   └── stage3_connections.py   # Directed graph derivation
│   ├── evaluation/        # Evaluation metrics
│   └── utils/             # Utilities
├── data/
│   ├── input/
│   │   ├── tocaps/        # Input TOCAP images
│   │   └── yolo_annotations/  # YOLO training data (6 classes)
│   │       └── yolo_data.yaml  # YOLO config
│   ├── intermediate/      # Stage outputs
│   │   ├── detection/     # Stage 1 output
│   │   ├── ocr/          # Stage 2 output
│   │   └── arrows/       # Stage 3 output (directed graphs)
│   └── models/           # Trained weights
│       └── yolo_detector.pt
├── configs/              # Configuration
│   └── pipeline_config.yaml
└── docs/                 # Documentation
```
└── notebooks/              # Exploration

## Usage

**Run full pipeline:**
```bash
python -m src.main
```

**Run specific stage:**
```bash
python -m src.main --stage 1  # Detection only
```

**Evaluate:**
```bash
python -m src.evaluate \
    --predictions-dir data/intermediate/reasoning \
    --ground-truth-dir data/input/annotations
```

## Pipeline Stages

1. **Detection** (YOLO) → Detect & classify 6 element types:
   - process, decision, document, terminator, connector, arrowhead
2. **OCR** (PaddleOCR/EasyOCR) → Extract text content from elements
3. **Connection Derivation** → Build directed graph:
   - Line tracing from arrowhead blunt end to source box(es)
   - Junction detection for merged lines
   - Proximity matching for target detection (pointy end)
   - **Direction-based label assignment** for ja/nee labels (optional)
   - Output: Directed graph with nodes (elements) and edges (leads_to relations)

## Output Format

The pipeline produces a **directed graph** for each TOCAP diagram:
- **Nodes**: Detected elements (process, decision, document, terminator, connector)
- **Edges**: "leads_to" relations derived from arrow connections
  - Optional "label" field for ja/nee annotations (when present on arrows)
- **Format**: JSON with nodes and edges arrays

**Example:**
```json
{
  "tocap_001.png": {
    "nodes": {
      "box_1": {"type": "terminator", "text": "Start Tocap 26"},
      "box_2": {"type": "decision", "text": "Voldoet de kap aan..."},
      "box_3": {"type": "document", "text": "A1) Wissel product"}
    },
    "edges": [
      {"source": "box_1", "target": "box_2", "label": null},
      {"source": "box_2", "target": "box_3", "label": "nee"},
      {"source": "box_2", "target": "box_4", "label": "ja"}
    ]
  }
}
```
## Configuration

Edit `configs/pipeline_config.yaml` to customize:
- YOLO model path and thresholds
- Detection confidence and IoU thresholds
- OCR engine selection (PaddleOCR or EasyOCR)
- Connection derivation parameters
- Label assignment radii and scoring
.

## Requirements

- Python 3.8+
- PyTorch, Ultralytics (YOLO)
- PaddleOCR or EasyOCR
- Transformers (Qwen2.5)

See `requirements.txt` for full list.
