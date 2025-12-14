#!/usr/bin/env python3
"""
Main Pipeline Script

Runs the complete decomposed parsing pipeline end-to-end or stage-by-stage.
"""

import argparse
import os
from pathlib import Path
from typing import Dict

from utils.io_utils import load_config, get_image_paths
from detection.detector import ElementDetector
from classification.classifier import ElementClassifier
from ocr.ocr_engine import OCREngine
from arrows.arrow_detector import ArrowDetector
from reasoning.llm_reasoner import LLMReasoner
from evaluation.evaluator import Evaluator


def run_stage_1(config: Dict, input_dir: str, output_dir: str):
    """Run Stage 1: Element Detection."""
    print("\n" + "="*60)
    print("STAGE 1: ELEMENT DETECTION")
    print("="*60)
    
    detector = ElementDetector(config)
    results = detector.process_directory(input_dir, output_dir)
    
    print(f"\nDetection complete. Processed {len(results)} images.")
    return results


def run_stage_2(config: Dict, input_dir: str, detection_dir: str, output_dir: str):
    """Run Stage 2: Element Classification."""
    print("\n" + "="*60)
    print("STAGE 2: ELEMENT CLASSIFICATION")
    print("="*60)
    
    classifier = ElementClassifier(config)
    
    # Get all detection JSON files
    detection_files = list(Path(detection_dir).glob('*_detection.json'))
    
    results = []
    for detection_file in detection_files:
        # Find corresponding image
        image_name = detection_file.stem.replace('_detection', '')
        image_paths = get_image_paths(input_dir)
        image_path = None
        
        for img_path in image_paths:
            if Path(img_path).stem == image_name:
                image_path = img_path
                break
        
        if image_path is None:
            print(f"Warning: Could not find image for {detection_file}")
            continue
        
        result = classifier.process_detections(image_path, str(detection_file), output_dir)
        results.append(result)
        print(f"✓ Classified {image_name}")
    
    print(f"\nClassification complete. Processed {len(results)} images.")
    return results


def run_stage_3(config: Dict, input_dir: str, classification_dir: str, output_dir: str):
    """Run Stage 3: OCR."""
    print("\n" + "="*60)
    print("STAGE 3: OCR TEXT EXTRACTION")
    print("="*60)
    
    ocr_engine = OCREngine(config)
    
    # Get all classification JSON files
    classification_files = list(Path(classification_dir).glob('*_classification.json'))
    
    results = []
    for classification_file in classification_files:
        # Find corresponding image
        image_name = classification_file.stem.replace('_classification', '')
        image_paths = get_image_paths(input_dir)
        image_path = None
        
        for img_path in image_paths:
            if Path(img_path).stem == image_name:
                image_path = img_path
                break
        
        if image_path is None:
            print(f"Warning: Could not find image for {classification_file}")
            continue
        
        result = ocr_engine.process_elements(image_path, str(classification_file), output_dir)
        results.append(result)
        print(f"✓ Extracted text from {image_name}")
    
    print(f"\nOCR complete. Processed {len(results)} images.")
    return results


def run_stage_4(config: Dict, input_dir: str, ocr_dir: str, output_dir: str):
    """Run Stage 4: Arrow Detection."""
    print("\n" + "="*60)
    print("STAGE 4: ARROW DETECTION")
    print("="*60)
    
    arrow_detector = ArrowDetector(config)
    ocr_engine = OCREngine(config)  # For label extraction
    
    # Get all OCR JSON files
    ocr_files = list(Path(ocr_dir).glob('*_ocr.json'))
    
    results = []
    for ocr_file in ocr_files:
        # Find corresponding image
        image_name = ocr_file.stem.replace('_ocr', '')
        image_paths = get_image_paths(input_dir)
        image_path = None
        
        for img_path in image_paths:
            if Path(img_path).stem == image_name:
                image_path = img_path
                break
        
        if image_path is None:
            print(f"Warning: Could not find image for {ocr_file}")
            continue
        
        result = arrow_detector.detect_arrows(image_path, str(ocr_file), output_dir, ocr_engine)
        results.append(result)
        print(f"✓ Detected {result['num_arrows']} arrows in {image_name}")
    
    print(f"\nArrow detection complete. Processed {len(results)} images.")
    return results


def run_stage_5(config: Dict, arrows_dir: str, output_dir: str):
    """Run Stage 5: LLM Reasoning."""
    print("\n" + "="*60)
    print("STAGE 5: LLM REASONING")
    print("="*60)
    
    reasoner = LLMReasoner(config)
    
    # Get all arrow JSON files
    arrow_files = list(Path(arrows_dir).glob('*_arrows.json'))
    
    results = []
    for arrow_file in arrow_files:
        image_name = arrow_file.stem.replace('_arrows', '')
        
        result = reasoner.reason(str(arrow_file), output_dir)
        results.append(result)
        print(f"✓ Reasoned over {image_name}: {len(result['entities'])} entities, {len(result['relations'])} relations")
    
    print(f"\nLLM reasoning complete. Processed {len(results)} images.")
    return results


def main():
    parser = argparse.ArgumentParser(description='Decomposed Parsing Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['all', '1', '2', '3', '4', '5'],
        default='all',
        help='Which stage to run (default: all)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set paths
    input_dir = args.input_dir or config['paths']['input_dir']
    intermediate_dir = config['paths']['intermediate_dir']
    output_dir = args.output_dir or config['paths']['output_dir']
    
    # Stage output directories
    detection_dir = os.path.join(intermediate_dir, 'detection')
    classification_dir = os.path.join(intermediate_dir, 'classification')
    ocr_dir = os.path.join(intermediate_dir, 'ocr')
    arrows_dir = os.path.join(intermediate_dir, 'arrows')
    reasoning_dir = os.path.join(intermediate_dir, 'reasoning')
    
    print("="*60)
    print("DECOMPOSED PARSING PIPELINE")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Intermediate directory: {intermediate_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Running stage(s): {args.stage}")
    
    # Run stages
    if args.stage == 'all' or args.stage == '1':
        run_stage_1(config, input_dir, detection_dir)
    
    if args.stage == 'all' or args.stage == '2':
        run_stage_2(config, input_dir, detection_dir, classification_dir)
    
    if args.stage == 'all' or args.stage == '3':
        run_stage_3(config, input_dir, classification_dir, ocr_dir)
    
    if args.stage == 'all' or args.stage == '4':
        run_stage_4(config, input_dir, ocr_dir, arrows_dir)
    
    if args.stage == 'all' or args.stage == '5':
        run_stage_5(config, arrows_dir, reasoning_dir)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    
    if args.stage == 'all':
        print(f"\nFinal results saved to: {reasoning_dir}")
    else:
        print(f"\nStage {args.stage} results saved to intermediate directory")


if __name__ == '__main__':
    main()
