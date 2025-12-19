#!/usr/bin/env python3
"""
Pipeline Evaluation Script

Comprehensive evaluation of the decomposed parsing pipeline.
Generates metrics per stage, per image, and overall.

Usage:
    python -m src.evaluate                          # Full evaluation
    python -m src.evaluate --output results.json    # Save to file
    python -m src.evaluate --charts                 # Generate charts
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

# YOLO class mapping
YOLO_CLASSES = {
    0: 'arrowhead', 1: 'connector', 2: 'decision', 
    3: 'document', 4: 'process', 5: 'terminator'
}


def text_similarity(a: str, b: str) -> float:
    """Calculate text similarity between two strings."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate precision, recall, F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


# ==============================================================================
# STAGE 1: DETECTION EVALUATION
# ==============================================================================

def evaluate_stage1(det_dir: Path, gt_dir: Path) -> Dict:
    """Evaluate Stage 1 element detection."""
    
    results = {'per_image': [], 'per_class': defaultdict(lambda: {'detected': 0, 'gt': 0})}
    
    for gt_file in sorted(gt_dir.glob("*.json")):
        image_name = gt_file.stem
        det_file = det_dir / f"{image_name}_detection.json"
        
        if not det_file.exists():
            continue
            
        gt = json.load(open(gt_file))
        det = json.load(open(det_file))
        image_key = list(gt.keys())[0]
        
        # Count detections by class
        det_counts = Counter(YOLO_CLASSES.get(e.get('class_id'), 'unknown') 
                            for e in det['elements'])
        
        # Count GT by type
        gt_counts = Counter(n['type'] for n in gt[image_key]['nodes'].values())
        gt_counts['arrowhead'] = len(gt[image_key]['edges'])  # Approximate
        
        # Per-class accumulation
        for cls_name in set(list(det_counts.keys()) + list(gt_counts.keys())):
            results['per_class'][cls_name]['detected'] += det_counts.get(cls_name, 0)
            results['per_class'][cls_name]['gt'] += gt_counts.get(cls_name, 0)
        
        # Per-image summary
        total_det = len([e for e in det['elements'] if e.get('class_id') != 0])  # Exclude arrowheads for node count
        total_gt = len(gt[image_key]['nodes'])
        
        results['per_image'].append({
            'image': image_name,
            'detected': total_det,
            'gt': total_gt,
            'arrowheads_det': det_counts.get('arrowhead', 0),
            'arrowheads_gt': gt_counts.get('arrowhead', 0)
        })
    
    return results


# ==============================================================================
# STAGE 2: OCR EVALUATION
# ==============================================================================

def evaluate_stage2(ocr_dir: Path, gt_dir: Path) -> Dict:
    """Evaluate Stage 2 OCR text extraction."""
    
    results = {'per_image': [], 'total_matches': 0, 'total_elements': 0, 'total_gt_nodes': 0}
    
    for gt_file in sorted(gt_dir.glob("*.json")):
        image_name = gt_file.stem
        ocr_file = ocr_dir / f"{image_name}_ocr.json"
        
        if not ocr_file.exists():
            continue
        
        gt = json.load(open(gt_file))
        ocr = json.load(open(ocr_file))
        image_key = list(gt.keys())[0]
        
        gt_nodes = gt[image_key]['nodes']
        ocr_elements = [e for e in ocr['elements'] if e.get('text', '').strip()]
        
        # Count text matches
        matches = 0
        matched_gt = set()
        for ocr_elem in ocr_elements:
            ocr_text = ocr_elem.get('text', '').strip()
            for gt_id, gt_node in gt_nodes.items():
                if gt_id not in matched_gt and text_similarity(ocr_text, gt_node['text']) > 0.7:
                    matches += 1
                    matched_gt.add(gt_id)
                    break
        
        # Elements with non-empty text
        elements_with_text = len(ocr_elements)
        total_elements = len([e for e in ocr['elements'] if e.get('class_id') != 0])
        
        results['per_image'].append({
            'image': image_name,
            'elements_with_text': elements_with_text,
            'total_elements': total_elements,
            'text_extract_rate': elements_with_text / total_elements if total_elements > 0 else 0,
            'gt_matches': matches,
            'gt_nodes': len(gt_nodes),
            'match_rate': matches / len(gt_nodes) if gt_nodes else 0
        })
        
        results['total_matches'] += matches
        results['total_elements'] += elements_with_text
        results['total_gt_nodes'] += len(gt_nodes)
    
    return results


# ==============================================================================
# STAGE 3: CONNECTION EVALUATION
# ==============================================================================

def evaluate_stage3(arrows_dir: Path, gt_dir: Path) -> Dict:
    """Evaluate Stage 3 connection derivation."""
    
    results = {
        'per_image': [],
        'node_totals': {'tp': 0, 'fp': 0, 'fn': 0, 'type_correct': 0, 'type_total': 0},
        'edge_totals': {'tp': 0, 'fp': 0, 'fn': 0, 'label_correct': 0, 'label_total': 0},
        'per_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'per_label': defaultdict(lambda: {'correct': 0, 'predicted': 0, 'gt': 0})
    }
    
    for gt_file in sorted(gt_dir.glob("*.json")):
        image_name = gt_file.stem
        arrows_file = arrows_dir / f"{image_name}_arrows.json"
        
        if not arrows_file.exists():
            continue
        
        pred = json.load(open(arrows_file))
        gt = json.load(open(gt_file))
        image_key = list(gt.keys())[0]
        
        gt_nodes = gt[image_key]['nodes']
        gt_edges = gt[image_key]['edges']
        pred_nodes = pred['graph']['nodes']
        pred_edges = pred['graph']['edges']
        
        # --- NODE MATCHING ---
        matched_gt = set()
        node_matches = []
        
        for pred_node in pred_nodes:
            best_match = None
            best_sim = 0
            for gt_id, gt_node in gt_nodes.items():
                if gt_id in matched_gt:
                    continue
                sim = text_similarity(pred_node['text'], gt_node['text'])
                if sim > best_sim and sim > 0.7:
                    best_sim = sim
                    best_match = (gt_id, gt_node)
            
            if best_match:
                matched_gt.add(best_match[0])
                node_matches.append((pred_node['id'], best_match[0], 
                                   pred_node.get('type', 'unknown'), 
                                   best_match[1].get('type', 'unknown')))
        
        node_tp = len(node_matches)
        node_fp = len(pred_nodes) - node_tp
        node_fn = len(gt_nodes) - node_tp
        
        # Type accuracy
        type_correct = sum(1 for _, _, pt, gt in node_matches if pt == gt)
        
        results['node_totals']['tp'] += node_tp
        results['node_totals']['fp'] += node_fp
        results['node_totals']['fn'] += node_fn
        results['node_totals']['type_correct'] += type_correct
        results['node_totals']['type_total'] += len(node_matches)
        
        # Per-type breakdown
        for _, _, pred_type, gt_type in node_matches:
            results['per_type'][gt_type]['total'] += 1
            if pred_type == gt_type:
                results['per_type'][gt_type]['correct'] += 1
        
        # --- EDGE MATCHING ---
        pred_to_gt = {pm[0]: pm[1] for pm in node_matches}
        
        pred_edges_gt = set()
        pred_labels = {}
        for edge in pred_edges:
            src_gt = pred_to_gt.get(edge['source'])
            tgt_gt = pred_to_gt.get(edge['target'])
            if src_gt and tgt_gt:
                pred_edges_gt.add((src_gt, tgt_gt))
                if edge.get('label'):
                    pred_labels[(src_gt, tgt_gt)] = edge['label']
        
        gt_edges_set = {(e['source'], e['target']) for e in gt_edges}
        gt_labels = {(e['source'], e['target']): e.get('label') for e in gt_edges}
        
        edge_tp_set = pred_edges_gt & gt_edges_set
        edge_tp = len(edge_tp_set)
        edge_fp = len(pred_edges_gt - gt_edges_set)
        edge_fn = len(gt_edges_set - pred_edges_gt)
        
        # Label accuracy (case-insensitive comparison for ja/Ja, nee/Nee)
        def normalize_label(label):
            return label.lower() if label else None
        
        label_correct = sum(1 for e in edge_tp_set if normalize_label(pred_labels.get(e)) == normalize_label(gt_labels.get(e)))
        
        results['edge_totals']['tp'] += edge_tp
        results['edge_totals']['fp'] += edge_fp
        results['edge_totals']['fn'] += edge_fn
        results['edge_totals']['label_correct'] += label_correct
        results['edge_totals']['label_total'] += len(edge_tp_set)
        
        # Per-label breakdown (normalized to lowercase)
        for edge in pred_edges_gt:
            label = normalize_label(pred_labels.get(edge))
            if label:
                results['per_label'][label]['predicted'] += 1
                
        for edge in gt_edges_set:
            label = normalize_label(gt_labels.get(edge))
            if label:
                results['per_label'][label]['gt'] += 1
                
        for edge in edge_tp_set:
            pred_label = normalize_label(pred_labels.get(edge))
            gt_label = normalize_label(gt_labels.get(edge))
            if pred_label and pred_label == gt_label:
                results['per_label'][pred_label]['correct'] += 1
        
        # Per-image summary
        node_f1 = calculate_metrics(node_tp, node_fp, node_fn)['f1']
        edge_f1 = calculate_metrics(edge_tp, edge_fp, edge_fn)['f1']
        
        results['per_image'].append({
            'image': image_name,
            'nodes': calculate_metrics(node_tp, node_fp, node_fn),
            'edges': calculate_metrics(edge_tp, edge_fp, edge_fn),
            'type_acc': type_correct / len(node_matches) if node_matches else 0,
            'label_acc': label_correct / len(edge_tp_set) if edge_tp_set else 0
        })
    
    return results


# ==============================================================================
# PRINT RESULTS
# ==============================================================================

def print_results(stage1: Dict, stage2: Dict, stage3: Dict):
    """Print formatted evaluation results."""
    
    print("\n" + "=" * 75)
    print("                    PIPELINE EVALUATION REPORT")
    print("=" * 75)
    
    # ---------- STAGE 1 ----------
    print("\n" + "─" * 75)
    print("STAGE 1: ELEMENT DETECTION (YOLO)")
    print("─" * 75)
    
    print("\n  Per-Class Detection:")
    print("  " + "─" * 50)
    print("  {:^15} {:^12} {:^12} {:^12}".format("Class", "Detected", "Ground Truth", "Rate"))
    print("  " + "─" * 50)
    
    for cls_name in ['decision', 'document', 'process', 'connector', 'terminator', 'arrowhead']:
        data = stage1['per_class'].get(cls_name, {'detected': 0, 'gt': 0})
        rate = data['detected'] / data['gt'] if data['gt'] > 0 else 0
        status = "✓" if rate >= 0.9 else "⚠" if rate >= 0.7 else "✗"
        print("  {:^15} {:^12} {:^12} {:^10.1%} {}".format(
            cls_name, data['detected'], data['gt'], rate, status
        ))
    print("  " + "─" * 50)
    
    # ---------- STAGE 2 ----------
    print("\n" + "─" * 75)
    print("STAGE 2: OCR TEXT EXTRACTION")
    print("─" * 75)
    
    total_rate = stage2['total_matches'] / stage2['total_gt_nodes'] if stage2['total_gt_nodes'] > 0 else 0
    
    print(f"\n  Overall Text Match Rate: {total_rate:.1%}")
    print(f"  Matched: {stage2['total_matches']}/{stage2['total_gt_nodes']} nodes")
    
    print("\n  Per-Image OCR Performance:")
    print("  " + "─" * 55)
    print("  {:^30} {:^12} {:^12}".format("Image", "Text Rate", "Match Rate"))
    print("  " + "─" * 55)
    for r in stage2['per_image']:
        print("  {:^30} {:^12.1%} {:^12.1%}".format(
            r['image'][:28], r['text_extract_rate'], r['match_rate']
        ))
    print("  " + "─" * 55)
    
    # ---------- STAGE 3 ----------
    print("\n" + "─" * 75)
    print("STAGE 3: CONNECTION DERIVATION")
    print("─" * 75)
    
    # Overall metrics
    node_metrics = calculate_metrics(
        stage3['node_totals']['tp'],
        stage3['node_totals']['fp'],
        stage3['node_totals']['fn']
    )
    edge_metrics = calculate_metrics(
        stage3['edge_totals']['tp'],
        stage3['edge_totals']['fp'],
        stage3['edge_totals']['fn']
    )
    type_acc = stage3['node_totals']['type_correct'] / stage3['node_totals']['type_total'] if stage3['node_totals']['type_total'] > 0 else 0
    label_acc = stage3['edge_totals']['label_correct'] / stage3['edge_totals']['label_total'] if stage3['edge_totals']['label_total'] > 0 else 0
    
    print("\n  ┌" + "─" * 58 + "┐")
    print("  │{:^58}│".format("OVERALL GRAPH CONSTRUCTION"))
    print("  ├" + "─" * 18 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┤")
    print("  │{:^18}│{:^12}│{:^12}│{:^12}│".format("Metric", "Precision", "Recall", "F1"))
    print("  ├" + "─" * 18 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┤")
    print("  │{:^18}│{:^12.1%}│{:^12.1%}│{:^12.1%}│".format(
        "Node Detection", node_metrics['precision'], node_metrics['recall'], node_metrics['f1']
    ))
    print("  │{:^18}│{:^12}│{:^12}│{:^12.1%}│".format("Node Types", "-", "-", type_acc))
    print("  │{:^18}│{:^12.1%}│{:^12.1%}│{:^12.1%}│".format(
        "Edge Detection", edge_metrics['precision'], edge_metrics['recall'], edge_metrics['f1']
    ))
    print("  │{:^18}│{:^12}│{:^12}│{:^12.1%}│".format("Edge Labels", "-", "-", label_acc))
    print("  └" + "─" * 18 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┘")
    
    # Per-type accuracy
    print("\n  Node Type Classification Accuracy:")
    print("  " + "─" * 45)
    print("  {:^15} {:^12} {:^12}".format("Type", "Correct", "Accuracy"))
    print("  " + "─" * 45)
    for type_name in sorted(stage3['per_type'].keys()):
        data = stage3['per_type'][type_name]
        acc = data['correct'] / data['total'] if data['total'] > 0 else 0
        print("  {:^15} {:>5}/{:<5} {:^12.1%}".format(type_name, data['correct'], data['total'], acc))
    print("  " + "─" * 45)
    
    # Label accuracy
    print("\n  Edge Label Accuracy:")
    print("  " + "─" * 50)
    print("  {:^10} {:^12} {:^12} {:^12}".format("Label", "Predicted", "GT", "Correct"))
    print("  " + "─" * 50)
    for label in sorted(stage3['per_label'].keys()):
        data = stage3['per_label'][label]
        print("  {:^10} {:^12} {:^12} {:^12}".format(
            label, data['predicted'], data['gt'], data['correct']
        ))
    print("  " + "─" * 50)
    
    # Per-image breakdown
    print("\n  Per-Image Performance:")
    print("  " + "─" * 70)
    print("  {:^32} {:>10} {:>10} {:>10} {:>8}".format("Image", "Node F1", "Edge F1", "Type Acc", "Labels"))
    print("  " + "─" * 70)
    for r in sorted(stage3['per_image'], key=lambda x: x['edges']['f1'], reverse=True):
        print("  {:^32} {:>10.1%} {:>10.1%} {:>10.1%} {:>8.1%}".format(
            r['image'][:30], r['nodes']['f1'], r['edges']['f1'], r['type_acc'], r['label_acc']
        ))
    print("  " + "─" * 70)
    
    # Detailed counts
    print("\n  Counts:")
    print(f"    Nodes: {stage3['node_totals']['tp']} correct, {stage3['node_totals']['fp']} FP, {stage3['node_totals']['fn']} FN")
    print(f"    Edges: {stage3['edge_totals']['tp']} correct, {stage3['edge_totals']['fp']} FP, {stage3['edge_totals']['fn']} FN")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 75)
    print("                         SUMMARY")
    print("=" * 75)
    
    print(f"""
  Stage 1 (Detection):
    - Node detection: ~{sum(stage1['per_class'][c]['detected'] for c in ['decision','document','process','connector','terminator'])}/{sum(stage1['per_class'][c]['gt'] for c in ['decision','document','process','connector','terminator'])} elements
    - Arrowhead detection: {stage1['per_class']['arrowhead']['detected']}/{stage1['per_class']['arrowhead']['gt']} ({stage1['per_class']['arrowhead']['detected']/stage1['per_class']['arrowhead']['gt']*100:.0f}%)

  Stage 2 (OCR):
    - Text extraction: {total_rate:.0%} match rate

  Stage 3 (Connections):
    - Node F1: {node_metrics['f1']:.1%}
    - Edge F1: {edge_metrics['f1']:.1%}
    - Type accuracy: {type_acc:.1%}
    - Label accuracy: {label_acc:.1%}

  Bottleneck: Arrowhead detection ({stage1['per_class']['arrowhead']['detected']}/{stage1['per_class']['arrowhead']['gt']}) limits edge recall
""")


def generate_charts(stage1: Dict, stage2: Dict, stage3: Dict, output_dir: str):
    """Generate publication-quality evaluation charts."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Skipping charts.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Professional style settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })
    
    # Color palettes
    COLORS = {
        'primary': '#2563eb',      # Blue
        'secondary': '#7c3aed',    # Purple
        'success': '#059669',      # Green
        'warning': '#d97706',      # Orange
        'danger': '#dc2626',       # Red
        'gray': '#6b7280',
        'light': '#f3f4f6',
    }
    
    PALETTE = ['#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626', '#0891b2']
    
    # Calculate metrics
    node_f1 = calculate_metrics(stage3['node_totals']['tp'], stage3['node_totals']['fp'], stage3['node_totals']['fn'])['f1']
    edge_f1 = calculate_metrics(stage3['edge_totals']['tp'], stage3['edge_totals']['fp'], stage3['edge_totals']['fn'])['f1']
    edge_p = calculate_metrics(stage3['edge_totals']['tp'], stage3['edge_totals']['fp'], stage3['edge_totals']['fn'])['precision']
    edge_r = calculate_metrics(stage3['edge_totals']['tp'], stage3['edge_totals']['fp'], stage3['edge_totals']['fn'])['recall']
    type_acc = stage3['node_totals']['type_correct'] / stage3['node_totals']['type_total'] if stage3['node_totals']['type_total'] > 0 else 0
    label_acc = stage3['edge_totals']['label_correct'] / stage3['edge_totals']['label_total'] if stage3['edge_totals']['label_total'] > 0 else 0
    ocr_rate = stage2['total_matches'] / stage2['total_gt_nodes'] if stage2['total_gt_nodes'] > 0 else 0
    
    # =========================================================================
    # CHART 1: Pipeline Overview (Dashboard Style)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pipeline Evaluation Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # 1a: Stage Performance Overview (Horizontal Bar)
    ax1 = axes[0, 0]
    metrics = ['Node F1', 'Type Acc', 'OCR Match', 'Edge F1', 'Label Acc']
    values = [node_f1, type_acc, ocr_rate, edge_f1, label_acc]
    colors = [COLORS['success'] if v >= 0.9 else COLORS['warning'] if v >= 0.7 else COLORS['danger'] for v in values]
    
    y_pos = np.arange(len(metrics))
    bars = ax1.barh(y_pos, values, color=colors, height=0.6, edgecolor='white', linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metrics)
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel('Score')
    ax1.set_title('Overall Performance Metrics')
    ax1.axvline(x=0.9, color=COLORS['gray'], linestyle=':', alpha=0.7, label='90% threshold')
    ax1.axvline(x=0.7, color=COLORS['gray'], linestyle='--', alpha=0.5, label='70% threshold')
    
    for bar, val in zip(bars, values):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.1%}', va='center', fontweight='bold', fontsize=11)
    
    # 1b: Detection Rate by Class
    ax2 = axes[0, 1]
    classes = ['decision', 'document', 'process', 'connector', 'terminator', 'arrowhead']
    rates = [stage1['per_class'][c]['detected'] / stage1['per_class'][c]['gt'] 
             if stage1['per_class'][c]['gt'] > 0 else 0 for c in classes]
    colors = [COLORS['success'] if r >= 0.9 else COLORS['warning'] if r >= 0.7 else COLORS['danger'] for r in rates]
    
    x_pos = np.arange(len(classes))
    bars = ax2.bar(x_pos, rates, color=colors, width=0.7, edgecolor='white', linewidth=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([c.title() for c in classes], rotation=30, ha='right')
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Detection Rate')
    ax2.set_title('Stage 1: Detection Rate by Class')
    ax2.axhline(y=1.0, color=COLORS['gray'], linestyle='-', alpha=0.3)
    
    for bar, val in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=10, fontweight='bold')
    
    # 1c: Edge Detection P/R/F1
    ax3 = axes[1, 0]
    edge_metrics_names = ['Precision', 'Recall', 'F1']
    edge_metrics_vals = [edge_p, edge_r, edge_f1]
    
    x_pos = np.arange(len(edge_metrics_names))
    bars = ax3.bar(x_pos, edge_metrics_vals, color=[COLORS['primary'], COLORS['secondary'], COLORS['success']], 
                   width=0.6, edgecolor='white', linewidth=1)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(edge_metrics_names)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Score')
    ax3.set_title('Stage 3: Edge Detection Metrics')
    
    for bar, val in zip(bars, edge_metrics_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', fontsize=12, fontweight='bold')
    
    # 1d: Per-Image Edge F1 (Sorted)
    ax4 = axes[1, 1]
    sorted_images = sorted(stage3['per_image'], key=lambda x: x['edges']['f1'], reverse=True)
    names = [r['image'].replace('_page_', '\n').split('-')[-1][:12] for r in sorted_images]
    edge_f1s = [r['edges']['f1'] for r in sorted_images]
    
    x_pos = np.arange(len(names))
    colors = [COLORS['success'] if v >= 0.9 else COLORS['warning'] if v >= 0.7 else COLORS['primary'] for v in edge_f1s]
    bars = ax4.bar(x_pos, edge_f1s, color=colors, width=0.7, edgecolor='white', linewidth=1)
    ax4.axhline(y=edge_f1, color=COLORS['danger'], linestyle='--', linewidth=2, label=f'Average: {edge_f1:.1%}')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, fontsize=9)
    ax4.set_ylim(0, 1)
    ax4.set_ylabel('Edge F1')
    ax4.set_title('Stage 3: Edge F1 by Image')
    ax4.legend(loc='lower right')
    
    for bar, val in zip(bars, edge_f1s):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'pipeline_dashboard.png'), dpi=200, 
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # CHART 2: Detailed Metrics (For Paper)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Detailed Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # 2a: Node Type Classification
    ax1 = axes[0]
    types = list(stage3['per_type'].keys())
    accs = [stage3['per_type'][t]['correct'] / stage3['per_type'][t]['total'] 
            if stage3['per_type'][t]['total'] > 0 else 0 for t in types]
    totals = [stage3['per_type'][t]['total'] for t in types]
    
    x_pos = np.arange(len(types))
    bars = ax1.bar(x_pos, accs, color=PALETTE[:len(types)], width=0.6, edgecolor='white', linewidth=1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([t.title() for t in types], rotation=30, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title('Node Type Accuracy')
    
    for bar, val, total in zip(bars, accs, totals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}\n(n={total})', ha='center', fontsize=9)
    
    # 2b: Label Detection (Stacked)
    ax2 = axes[1]
    labels = list(stage3['per_label'].keys())
    if labels:
        predicted = [stage3['per_label'][l]['predicted'] for l in labels]
        gt = [stage3['per_label'][l]['gt'] for l in labels]
        correct = [stage3['per_label'][l]['correct'] for l in labels]
        
        x_pos = np.arange(len(labels))
        width = 0.25
        
        ax2.bar(x_pos - width, gt, width, label='Ground Truth', color=COLORS['gray'], edgecolor='white')
        ax2.bar(x_pos, predicted, width, label='Predicted', color=COLORS['primary'], edgecolor='white')
        ax2.bar(x_pos + width, correct, width, label='Correct', color=COLORS['success'], edgecolor='white')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([l.upper() for l in labels])
        ax2.set_ylabel('Count')
        ax2.set_title('Edge Label Detection')
        ax2.legend()
    
    # 2c: Detection vs GT Counts
    ax3 = axes[2]
    categories = ['Nodes', 'Edges']
    detected = [stage3['node_totals']['tp'], stage3['edge_totals']['tp']]
    gt_counts = [stage3['node_totals']['tp'] + stage3['node_totals']['fn'],
                 stage3['edge_totals']['tp'] + stage3['edge_totals']['fn']]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, gt_counts, width, label='Ground Truth', color=COLORS['gray'], edgecolor='white')
    bars2 = ax3.bar(x_pos + width/2, detected, width, label='Correctly Detected', color=COLORS['success'], edgecolor='white')
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories)
    ax3.set_ylabel('Count')
    ax3.set_title('Detection Coverage')
    ax3.legend()
    
    for bar, val in zip(bars1, gt_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha='center', fontsize=10)
    for bar, val in zip(bars2, detected):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_metrics.png'), dpi=200,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # CHART 3: Stage Breakdown (Pipeline Flow)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create grouped bar chart for all stages
    stage_names = ['Stage 1:\nDetection', 'Stage 2:\nOCR', 'Stage 3:\nNodes', 'Stage 3:\nEdges']
    
    # Calculate stage metrics
    node_det_rate = sum(stage1['per_class'][c]['detected'] for c in ['decision','document','process','connector','terminator']) / \
                    sum(stage1['per_class'][c]['gt'] for c in ['decision','document','process','connector','terminator'])
    arrow_det_rate = stage1['per_class']['arrowhead']['detected'] / stage1['per_class']['arrowhead']['gt']
    
    main_metrics = [node_det_rate, ocr_rate, node_f1, edge_f1]
    secondary_metrics = [arrow_det_rate, None, type_acc, label_acc]
    
    x_pos = np.arange(len(stage_names))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, main_metrics, width, 
                   label='Primary Metric', color=COLORS['primary'], edgecolor='white', linewidth=1)
    
    # Only plot secondary where it exists
    secondary_vals = [v if v is not None else 0 for v in secondary_metrics]
    secondary_colors = [COLORS['secondary'] if v is not None else 'none' for v in secondary_metrics]
    bars2 = ax.bar(x_pos + width/2, secondary_vals, width,
                   label='Secondary Metric', color=COLORS['secondary'], edgecolor='white', linewidth=1)
    # Hide the OCR bar (no secondary metric)
    bars2[1].set_alpha(0)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stage_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Pipeline Performance by Stage', fontsize=14, fontweight='bold')
    
    # Add threshold lines
    ax.axhline(y=0.9, color=COLORS['success'], linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.7, color=COLORS['warning'], linestyle='--', alpha=0.5, linewidth=1)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['primary'], label='Primary (Det Rate / F1)'),
        Patch(facecolor=COLORS['secondary'], label='Secondary (Arrows / Type / Labels)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Annotations
    annotations = [
        ('Node Det', 'Arrow Det'),
        ('OCR\nMatch', ''),
        ('Node F1', 'Type Acc'),
        ('Edge F1', 'Label Acc')
    ]
    
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02,
                f'{main_metrics[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
        if secondary_metrics[i] is not None:
            ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02,
                    f'{secondary_metrics[i]:.0%}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stage_breakdown.png'), dpi=200,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Charts saved to {output_dir}/")
    print(f"  - pipeline_dashboard.png (overview)")
    print(f"  - detailed_metrics.png (for paper)")
    print(f"  - stage_breakdown.png (pipeline flow)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate pipeline predictions')
    parser.add_argument('--predictions', type=str, default='data/intermediate/arrows',
                       help='Directory with prediction files (*_arrows.json)')
    parser.add_argument('--detection', type=str, default='data/intermediate/detection',
                       help='Directory with detection results')
    parser.add_argument('--ocr', type=str, default='data/intermediate/ocr',
                       help='Directory with OCR results')
    parser.add_argument('--ground-truth', type=str, default='data/input/final_annotations',
                       help='Directory with ground truth annotations')
    parser.add_argument('--output', type=str, default=None,
                       help='Save results to JSON file')
    parser.add_argument('--charts', action='store_true',
                       help='Generate evaluation charts')
    parser.add_argument('--charts-dir', type=str, default='data/output/charts',
                       help='Directory to save charts')
    
    args = parser.parse_args()
    
    det_dir = Path(args.detection)
    ocr_dir = Path(args.ocr)
    arrows_dir = Path(args.predictions)
    gt_dir = Path(args.ground_truth)
    
    # Evaluate each stage
    print("Evaluating Stage 1: Detection...")
    stage1_results = evaluate_stage1(det_dir, gt_dir)
    
    print("Evaluating Stage 2: OCR...")
    stage2_results = evaluate_stage2(ocr_dir, gt_dir)
    
    print("Evaluating Stage 3: Connections...")
    stage3_results = evaluate_stage3(arrows_dir, gt_dir)
    
    # Print results
    print_results(stage1_results, stage2_results, stage3_results)
    
    # Generate charts
    if args.charts:
        generate_charts(stage1_results, stage2_results, stage3_results, args.charts_dir)
    
    # Save results
    if args.output:
        output_data = {
            'stage1': stage1_results,
            'stage2': stage2_results,
            'stage3': stage3_results
        }
        # Convert defaultdicts to regular dicts for JSON serialization
        output_data['stage1']['per_class'] = dict(output_data['stage1']['per_class'])
        output_data['stage3']['per_type'] = dict(output_data['stage3']['per_type'])
        output_data['stage3']['per_label'] = dict(output_data['stage3']['per_label'])
        
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == '__main__':
    main()

