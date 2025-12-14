"""
Decomposed Parsing Pipeline

A modular pipeline for extracting procedural knowledge from industrial 
troubleshooting diagrams using YOLO detection, classification, OCR, 
arrow detection, and LLM reasoning.
"""

# Stage 1: Element Detection
from .detection import ElementDetector, YOLODetector

# Stage 2: Element Classification  
from .classification import ElementClassifier, ElementDataset

# Stage 3: OCR
from .ocr import OCREngine

# Stage 4: Arrow Detection
from .arrows import ArrowDetector

# Stage 5: LLM Reasoning
from .reasoning import LLMReasoner

# Evaluation
from .evaluation import Evaluator

# Utilities
from .utils import (
    load_config,
    save_json,
    load_json,
    get_image_paths,
    ensure_dir,
    draw_bounding_boxes,
    draw_element_types,
    draw_arrows
)

__version__ = '0.1.0'

__all__ = [
    # Detection
    'ElementDetector',
    'YOLODetector',
    # Classification
    'ElementClassifier',
    'ElementDataset',
    # OCR
    'OCREngine',
    # Arrows
    'ArrowDetector',
    # Reasoning
    'LLMReasoner',
    # Evaluation
    'Evaluator',
    # Utils
    'load_config',
    'save_json',
    'load_json',
    'get_image_paths',
    'ensure_dir',
    'draw_bounding_boxes',
    'draw_element_types',
    'draw_arrows',
]
