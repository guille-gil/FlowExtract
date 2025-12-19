# Decomposed Parsing from Industrial Troubleshooting Diagrams

A modular pipeline for extracting procedural knowledge from industrial troubleshooting diagrams using computer vision and OCR.

## Authors

- Guillermo Gil de Avalle Bellido (University of Groningen)
- Laura Maruster (University of Groningen)
- Christos Emmanouilidis (University of Groningen)

## Performance

Evaluated on 7 test images:

| Component | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| **Node Detection** | 98.4% | 99.2% | 98.8% |
| **Edge Detection** | 85.5% | 54.6% | 66.7% |
| **Node Type Classification** | - | - | 97.6% |
| **Edge Labels (ja/nee)** | - | - | 73.8% |
| **OCR Text Match** | - | - | 99.2% |

### Per-Class Detection

| Class | Detection Rate |
|-------|---------------|
| Decision | 100% |
| Document | 97.5% |
| Process | 100% |
| Connector | 100% |
| Terminator | 100% |
| Arrowhead | 73% |

> **Note:** Arrowhead detection (73%) is the main bottleneck limiting edge recall.

## Pipeline Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Stage 1       │     │   Stage 2       │     │   Stage 3       │
│   Detection     │────▶│   OCR           │────▶│   Connections   │
│   (YOLOv8)      │     │   (EasyOCR)     │     │   (Line Trace)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   6 element types         Text + Labels          Directed Graph
```

### Stage 1: Element Detection
- **Model:** YOLOv8s trained on 25 images
- **Classes:** arrowhead, decision, document, process, connector, terminator
- **Output:** Bounding boxes with class labels

### Stage 2: OCR Text Extraction
- **Engine:** EasyOCR (Dutch)
- **Function:** Extract text from detected boxes + identify ja/nee decision labels
- **Output:** Text content per node

### Stage 3: Connection Derivation
- **Method:** Line tracing from arrowheads to boxes
- **Direction:** Determined by arrowhead orientation (pointy end → target)
- **Output:** Directed graph with nodes and edges

## Installation

```bash
# Clone repository
git clone https://github.com/username/Decomposed-Parsing-from-Industrial-Troubleshooting-Guides.git
cd Decomposed-Parsing-from-Industrial-Troubleshooting-Guides

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run full pipeline
python -m src.main

# Run individual stages
python -m src.main --stage 1   # Detection only
python -m src.main --stage 2   # OCR only
python -m src.main --stage 3   # Connections only

# Evaluate results
python -m src.evaluate                     # Print metrics
python -m src.evaluate --charts            # Generate charts
python -m src.evaluate --output out.json   # Save to JSON
```

## Project Structure

```
├── src/
│   ├── main.py                    # Pipeline entry point
│   ├── evaluate.py                # Evaluation metrics + charts
│   └── pipeline/
│       ├── stage1_detector.py     # YOLOv8 detection
│       ├── stage2_ocr.py          # EasyOCR text extraction
│       └── stage3_connections.py  # Line tracing + graph construction
├── scripts/
│   └── train_yolo.py              # YOLO training script
├── configs/
│   └── pipeline_config.yaml       # Configuration
├── data/
│   ├── input/
│   │   ├── images/                # Train/val/test images
│   │   ├── labels/                # YOLO annotations
│   │   └── final_annotations/     # Ground truth JSON
│   ├── intermediate/              # Stage outputs
│   └── output/                    # Final predictions
└── runs/detect/train/weights/     # Trained model
```

## Output Format

The pipeline produces a directed graph for each input image:

```json
{
  "graph": {
    "nodes": [
      {"id": 0, "type": "decision", "text": "Is X correct?", "bbox": [100, 200, 150, 80]}
    ],
    "edges": [
      {"source": 0, "target": 1, "type": "leads_to", "label": "nee"}
    ]
  },
  "num_nodes": 15,
  "num_edges": 14
}
```

## Dataset

- **Total:** 35 industrial troubleshooting diagrams
- **Split:** 25 train / 3 validation / 7 test
- **Types:** Troubleshooting diagrams

## Training

To train YOLO on your own data:

```bash
python scripts/train_yolo.py --epochs 100
```

The trained model will be saved to `runs/detect/train/weights/best.pt`.


See `requirements.txt` for full dependencies.

## Citation

...

## License

This work is licensed under [CC BY-NC 4.0](LICENSE) - Creative Commons Attribution-NonCommercial 4.0 International.

You may share and adapt this work for non-commercial purposes with attribution.
