"""
Decomposed Parsing from Industrial Troubleshooting Guides

A pipeline for extracting procedural knowledge from industrial
troubleshooting diagrams using YOLO detection, OCR,
and connection derivation to build directed graphs.
"""

from .pipeline import ElementDetector, OCREngine, ArrowDetector
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
