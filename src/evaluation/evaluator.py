"""
Evaluation Module

Computes entity and relation F1 scores against ground truth annotations.
"""

import numpy as np
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import re

from ..utils.io_utils import load_json


class Evaluator:
    """Evaluator for entity and relation extraction."""
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        eval_config = config.get('evaluation', {})
        
        # Entity matching parameters
        entity_config = eval_config.get('entity_matching', {})
        self.similarity_threshold = entity_config.get('similarity_threshold', 0.85)
        self.normalization = entity_config.get('normalization', True)
        
        # Relation matching parameters
        relation_config = eval_config.get('relation_matching', {})
        self.require_exact_match = relation_config.get('require_exact_match', True)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not self.normalization:
            return text
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Strip
        text = text.strip()
        
        return text
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute text similarity using sequence matching.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize texts
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)
        
        # Compute similarity
        similarity = SequenceMatcher(None, norm_text1, norm_text2).ratio()
        
        return similarity
    
    def match_entities(
        self,
        predicted_entities: List[Dict],
        ground_truth_entities: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match predicted entities to ground truth entities.
        
        Args:
            predicted_entities: List of predicted entity dictionaries
            ground_truth_entities: List of ground truth entity dictionaries
            
        Returns:
            Tuple of (matches, unmatched_predicted, unmatched_ground_truth)
            where matches is a list of (pred_idx, gt_idx) tuples
        """
        matches = []
        matched_pred = set()
        matched_gt = set()
        
        # Try to match each predicted entity
        for pred_idx, pred_entity in enumerate(predicted_entities):
            pred_text = pred_entity.get('text', '')
            
            best_match_idx = None
            best_similarity = 0.0
            
            for gt_idx, gt_entity in enumerate(ground_truth_entities):
                if gt_idx in matched_gt:
                    continue
                
                gt_text = gt_entity.get('text', '')
                similarity = self.text_similarity(pred_text, gt_text)
                
                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = gt_idx
            
            if best_match_idx is not None:
                matches.append((pred_idx, best_match_idx))
                matched_pred.add(pred_idx)
                matched_gt.add(best_match_idx)
        
        # Find unmatched
        unmatched_pred = [i for i in range(len(predicted_entities)) if i not in matched_pred]
        unmatched_gt = [i for i in range(len(ground_truth_entities)) if i not in matched_gt]
        
        return matches, unmatched_pred, unmatched_gt
    
    def evaluate_entities(
        self,
        predicted_entities: List[Dict],
        ground_truth_entities: List[Dict]
    ) -> Dict:
        """
        Evaluate entity extraction.
        
        Args:
            predicted_entities: List of predicted entities
            ground_truth_entities: List of ground truth entities
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        matches, unmatched_pred, unmatched_gt = self.match_entities(
            predicted_entities,
            ground_truth_entities
        )
        
        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'num_predicted': len(predicted_entities),
            'num_ground_truth': len(ground_truth_entities)
        }
    
    def match_relations(
        self,
        predicted_relations: List[Dict],
        ground_truth_relations: List[Dict],
        entity_matches: List[Tuple[int, int]],
        predicted_entities: List[Dict],
        ground_truth_entities: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match predicted relations to ground truth relations.
        
        Args:
            predicted_relations: List of predicted relations
            ground_truth_relations: List of ground truth relations
            entity_matches: List of (pred_idx, gt_idx) entity matches
            predicted_entities: List of predicted entities
            ground_truth_entities: List of ground truth entities
            
        Returns:
            Tuple of (matches, unmatched_predicted, unmatched_ground_truth)
        """
        # Create entity ID mapping
        entity_id_map = {}
        for pred_idx, gt_idx in entity_matches:
            pred_id = predicted_entities[pred_idx].get('id')
            gt_id = ground_truth_entities[gt_idx].get('id')
            entity_id_map[pred_id] = gt_id
        
        matches = []
        matched_pred = set()
        matched_gt = set()
        
        # Try to match each predicted relation
        for pred_idx, pred_rel in enumerate(predicted_relations):
            pred_source = pred_rel.get('source_entity_id')
            pred_target = pred_rel.get('target_entity_id')
            
            # Map to ground truth entity IDs
            gt_source = entity_id_map.get(pred_source)
            gt_target = entity_id_map.get(pred_target)
            
            if gt_source is None or gt_target is None:
                continue
            
            # Find matching ground truth relation
            for gt_idx, gt_rel in enumerate(ground_truth_relations):
                if gt_idx in matched_gt:
                    continue
                
                if (gt_rel.get('source_entity_id') == gt_source and
                    gt_rel.get('target_entity_id') == gt_target):
                    
                    matches.append((pred_idx, gt_idx))
                    matched_pred.add(pred_idx)
                    matched_gt.add(gt_idx)
                    break
        
        unmatched_pred = [i for i in range(len(predicted_relations)) if i not in matched_pred]
        unmatched_gt = [i for i in range(len(ground_truth_relations)) if i not in matched_gt]
        
        return matches, unmatched_pred, unmatched_gt
    
    def evaluate_relations(
        self,
        predicted_relations: List[Dict],
        ground_truth_relations: List[Dict],
        entity_matches: List[Tuple[int, int]],
        predicted_entities: List[Dict],
        ground_truth_entities: List[Dict]
    ) -> Dict:
        """
        Evaluate relation extraction.
        
        Args:
            predicted_relations: List of predicted relations
            ground_truth_relations: List of ground truth relations
            entity_matches: List of entity matches
            predicted_entities: List of predicted entities
            ground_truth_entities: List of ground truth entities
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        matches, unmatched_pred, unmatched_gt = self.match_relations(
            predicted_relations,
            ground_truth_relations,
            entity_matches,
            predicted_entities,
            ground_truth_entities
        )
        
        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'num_predicted': len(predicted_relations),
            'num_ground_truth': len(ground_truth_relations)
        }
    
    def evaluate(
        self,
        prediction_path: str,
        ground_truth_path: str
    ) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            prediction_path: Path to prediction JSON
            ground_truth_path: Path to ground truth JSON
            
        Returns:
            Evaluation results dictionary
        """
        # Load data
        predictions = load_json(prediction_path)
        ground_truth = load_json(ground_truth_path)
        
        pred_entities = predictions.get('entities', [])
        gt_entities = ground_truth.get('entities', [])
        pred_relations = predictions.get('relations', [])
        gt_relations = ground_truth.get('relations', [])
        
        # Evaluate entities
        entity_results = self.evaluate_entities(pred_entities, gt_entities)
        
        # Get entity matches for relation evaluation
        entity_matches, _, _ = self.match_entities(pred_entities, gt_entities)
        
        # Evaluate relations
        relation_results = self.evaluate_relations(
            pred_relations,
            gt_relations,
            entity_matches,
            pred_entities,
            gt_entities
        )
        
        return {
            'entities': entity_results,
            'relations': relation_results
        }
