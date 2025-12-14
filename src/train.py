#!/usr/bin/env python3
"""
Training script for element classifier.

Usage:
    python -m src.train --data-dir data/training/element_crops --epochs 50
"""

import argparse
from utils.io_utils import load_config
from classification.classifier import ElementClassifier


def main():
    parser = argparse.ArgumentParser(description='Train element classifier')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory with training data (subdirectories: observation, decision, action)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize classifier
    classifier = ElementClassifier(config)
    
    # Train
    print(f"Training classifier on data from: {args.data_dir}")
    classifier.train(
        train_data_dir=args.data_dir,
        val_split=args.val_split,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    print(f"\nTraining complete. Model saved to: {classifier.model_path}")


if __name__ == '__main__':
    main()
