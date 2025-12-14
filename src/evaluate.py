#!/usr/bin/env python3
"""
Evaluation script for comparing predictions against ground truth.

Usage:
    python -m src.evaluate --predictions-dir data/intermediate/reasoning --ground-truth-dir data/input/annotations
"""

import argparse
import os
from pathlib import Path
import json

from utils.io_utils import load_config, save_json
from evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--predictions-dir',
        type=str,
        required=True,
        help='Directory containing prediction JSON files'
    )
    parser.add_argument(
        '--ground-truth-dir',
        type=str,
        required=True,
        help='Directory containing ground truth annotation JSON files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/evaluation_results.json',
        help='Path to save evaluation results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Get prediction files
    pred_files = list(Path(args.predictions_dir).glob('*_reasoning.json'))
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Evaluate each file
    all_results = []
    
    for pred_file in pred_files:
        image_name = pred_file.stem.replace('_reasoning', '')
        
        # Find corresponding ground truth file
        gt_file = Path(args.ground_truth_dir) / f"{image_name}.json"
        
        if not gt_file.exists():
            print(f"Warning: No ground truth found for {image_name}")
            continue
        
        # Evaluate
        results = evaluator.evaluate(str(pred_file), str(gt_file))
        results['image_name'] = image_name
        all_results.append(results)
        
        print(f"\n{image_name}:")
        print(f"  Entities - P: {results['entities']['precision']:.3f}, "
              f"R: {results['entities']['recall']:.3f}, "
              f"F1: {results['entities']['f1']:.3f}")
        print(f"  Relations - P: {results['relations']['precision']:.3f}, "
              f"R: {results['relations']['recall']:.3f}, "
              f"F1: {results['relations']['f1']:.3f}")
    
    # Compute average scores
    if all_results:
        avg_entity_f1 = sum(r['entities']['f1'] for r in all_results) / len(all_results)
        avg_relation_f1 = sum(r['relations']['f1'] for r in all_results) / len(all_results)
        
        print("\n" + "="*60)
        print("OVERALL RESULTS")
        print("="*60)
        print(f"Average Entity F1: {avg_entity_f1:.3f}")
        print(f"Average Relation F1: {avg_relation_f1:.3f}")
        
        summary = {
            'num_images': len(all_results),
            'average_entity_f1': avg_entity_f1,
            'average_relation_f1': avg_relation_f1,
            'per_image_results': all_results
        }
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_json(summary, args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
