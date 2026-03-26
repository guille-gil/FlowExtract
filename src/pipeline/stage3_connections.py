"""
Stage 3: Connection Derivation

Derives directed graph connections using line tracing and proximity matching.
Outputs nodes (elements) and edges (leads_to relations).
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import os
from pathlib import Path

from ..utils.io_utils import load_json, save_json, ensure_dir



# YOLO class ID to type name mapping (from yolo_data.yaml)
YOLO_CLASS_NAMES = {
    0: 'arrowhead',
    1: 'connector',
    2: 'decision',
    3: 'document',
    4: 'process',
    5: 'terminator'
}

class ArrowDetector:
    """Detector for arrow connections using line tracing and proximity matching."""
    
    def __init__(self, config: Dict):
        """
        Initialize arrow detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        arrow_config = config.get('arrows', {})
        
        # Line detection parameters
        line_config = arrow_config.get('line_detection', {})
        self.hough_threshold = line_config.get('hough_threshold', 50)
        self.min_line_length = line_config.get('min_line_length', 20)
        self.max_line_gap = line_config.get('max_line_gap', 15)
        
        # Connection matching parameters
        self.line_attachment_threshold = arrow_config.get('line_attachment_threshold', 15)
        self.target_touch_threshold = arrow_config.get('target_touch_threshold', 20)
        self.source_trace_threshold = arrow_config.get('source_trace_threshold', 30)
        self.junction_detection_threshold = arrow_config.get('junction_detection_threshold', 15)
        
        # Label assignment parameters
        label_config = arrow_config.get('label_assignment', {})
        self.single_arrow_radius = label_config.get('single_arrow_radius', 100)
        self.branching_radius = label_config.get('branching_radius', 150)
        self.distance_penalty_factor = label_config.get('distance_penalty_factor', 50.0)
        
        # Fallback settings
        self.enable_line_fallback = arrow_config.get('enable_line_fallback', False)
    
    def _identify_arrowhead_ends(
        self,
        arrowhead: Dict,
        image: np.ndarray,
        boxes: List[Dict] = None
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Identify blunt and pointy ends of arrowhead.
        
        The pointy end is the one closest to a box (the target).
        The blunt end is where the line comes from (the source direction).
        
        Args:
            arrowhead: Arrowhead element dict with bbox
            image: Input image
            boxes: List of detected boxes (for orientation detection)
            
        Returns:
            (blunt_end, pointy_end) as (x, y) tuples
        """
        x, y, w, h = arrowhead['bbox']
        center = (x + w // 2, y + h // 2)
        
        # Get all 4 cardinal points around the arrowhead
        candidates = [
            (x, center[1], 'left'),           # Left edge center
            (x + w, center[1], 'right'),      # Right edge center
            (center[0], y, 'top'),            # Top edge center
            (center[0], y + h, 'bottom')      # Bottom edge center
        ]
        
        # If we have boxes, find which end is closest to a box (that's the pointy end)
        if boxes:
            best_pointy = None
            min_dist_to_box = float('inf')
            
            for px, py, direction in candidates:
                for box in boxes:
                    bx, by, bw, bh = box['bbox']
                    # Distance from point to box edge
                    dx = max(bx - px, 0, px - (bx + bw))
                    dy = max(by - py, 0, py - (by + bh))
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist < min_dist_to_box:
                        min_dist_to_box = dist
                        best_pointy = (px, py, direction)
            
            if best_pointy and min_dist_to_box < 50:  # Only if reasonably close
                px, py, pointy_dir = best_pointy
                # Blunt is opposite of pointy
                opposites = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}
                blunt_dir = opposites[pointy_dir]
                blunt = next((c[0], c[1]) for c in candidates if c[2] == blunt_dir)
                return blunt, (px, py)
        
        # Fallback to aspect ratio heuristic
        if w > h:  # Horizontal: pointy on right
            return (x, center[1]), (x + w, center[1])
        else:  # Vertical: pointy on bottom
            return (center[0], y), (center[0], y + h)
    
    def _find_box_touching_point(
        self,
        point: Tuple[int, int],
        boxes: List[Dict],
        threshold: int = 20
    ) -> Optional[int]:
        """
        Find box whose boundary is closest to point.
        
        Args:
            point: (x, y) coordinates
            boxes: List of element boxes
            threshold: Maximum distance in pixels
            
        Returns:
            Element ID or None
        """
        px, py = point
        min_dist = float('inf')
        closest_box = None
        
        for box in boxes:
            x, y, w, h = box['bbox']
            
            # Calculate distance to box boundary
            dx = max(x - px, 0, px - (x + w))
            dy = max(y - py, 0, py - (y + h))
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist and dist <= threshold:
                min_dist = dist
                closest_box = box['id']
        
        return closest_box
    
    def _find_lines_at_point(
        self,
        point: Tuple[int, int],
        line_segments: List[Tuple[int, int, int, int]],
        threshold: int = 15
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find line segments with endpoint near point.
        
        Args:
            point: (x, y) coordinates
            line_segments: List of line segments
            threshold: Distance threshold
            
        Returns:
            List of line segments
        """
        px, py = point
        attached_lines = []
        
        for line in line_segments:
            x1, y1, x2, y2 = line
            dist1 = np.sqrt((x1 - px)**2 + (y1 - py)**2)
            dist2 = np.sqrt((x2 - px)**2 + (y2 - py)**2)
            
            if dist1 <= threshold or dist2 <= threshold:
                attached_lines.append(line)
        
        return attached_lines
    
    def _get_far_endpoint(
        self,
        line: Tuple[int, int, int, int],
        near_point: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Get the endpoint of line that's far from near_point.
        
        Args:
            line: (x1, y1, x2, y2)
            near_point: (x, y)
            
        Returns:
            (x, y) of far endpoint
        """
        x1, y1, x2, y2 = line
        nx, ny = near_point
        
        dist1 = np.sqrt((x1 - nx)**2 + (y1 - ny)**2)
        dist2 = np.sqrt((x2 - nx)**2 + (y2 - ny)**2)
        
        return (x2, y2) if dist1 < dist2 else (x1, y1)
    
    def _trace_line_to_sources(
        self,
        line: Tuple[int, int, int, int],
        start_point: Tuple[int, int],
        line_segments: List[Tuple[int, int, int, int]],
        boxes: List[Dict]
    ) -> List[int]:
        """
        Trace line back to source box(es), handling junctions.
        
        Handles both:
        - Simple case: line directly connects to box
        - Junction case: multiple lines merge before arrowhead
        
        Args:
            line: Starting line segment
            start_point: Point where line attaches to arrowhead
            line_segments: All detected line segments
            boxes: List of element boxes
            
        Returns:
            List of source box IDs
        """
        sources = []
        visited_points = set()
        points_to_explore = [self._get_far_endpoint(line, start_point)]
        explored_lines = {id(line)}  # Track explored lines to avoid loops
        
        while points_to_explore:
            current_point = points_to_explore.pop(0)
            
            # Avoid revisiting points
            point_key = (round(current_point[0]), round(current_point[1]))
            if point_key in visited_points:
                continue
            visited_points.add(point_key)
            
            # Check if this point touches a box
            box_id = self._find_box_touching_point(
                current_point, boxes, threshold=self.source_trace_threshold
            )
            
            if box_id is not None:
                # Found a source box!
                if box_id not in sources:
                    sources.append(box_id)
            else:
                # Not at a box - might be a junction point
                # Find all lines connected to this point
                connected_lines = self._find_lines_at_point(
                    current_point, line_segments, threshold=self.junction_detection_threshold
                )
                
                # Explore each connected line we haven't seen yet
                for connected_line in connected_lines:
                    line_id = id(connected_line)
                    if line_id not in explored_lines:
                        explored_lines.add(line_id)
                        far_end = self._get_far_endpoint(connected_line, current_point)
                        points_to_explore.append(far_end)
        
        return sources
    
    def _mask_elements(self, image: np.ndarray, elements: List[Dict]) -> np.ndarray:
        """Create mask with elements removed."""
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        for element in elements:
            x, y, w, h = element['bbox']
            # Add padding to ensure complete masking
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            mask[y1:y2, x1:x2] = 0
        
        return mask
    
    def _detect_lines(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Detect line segments in masked image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Edge detection
        edges = cv2.Canny(masked, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
        
        # Convert to list of tuples
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_segments.append((int(x1), int(y1), int(x2), int(y2)))
        
        return line_segments
    
    def _get_center(self, bbox: List[int]) -> np.ndarray:
        """Get center point of bounding box."""
        x, y, w, h = bbox
        return np.array([x + w / 2, y + h / 2])
    
    def _calculate_path_midpoint(
        self,
        line_segments: List[Tuple[int, int, int, int]],
        blunt_end: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Calculate midpoint of traced line path.
        
        Args:
            line_segments: List of line segments forming the path
            blunt_end: Starting point of the path
            
        Returns:
            (x, y) midpoint coordinates
        """
        if not line_segments:
            return blunt_end
        
        # For simplicity, use midpoint of first line segment
        # TODO: Could trace full path and find actual midpoint
        line = line_segments[0]
        x1, y1, x2, y2 = line
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _find_labels_near_node(
        self,
        node: Dict,
        labels: List[Dict],
        radius: int = 150
    ) -> List[Dict]:
        """
        Find labels within radius of node center.
        
        Args:
            node: Node dictionary with bbox
            labels: List of label dictionaries
            radius: Search radius in pixels
            
        Returns:
            List of labels within radius
        """
        node_center = self._get_center(node['bbox'])
        nearby_labels = []
        
        for label in labels:
            label_center = np.array(label['center'])
            dist = np.linalg.norm(label_center - node_center)
            
            if dist <= radius:
                nearby_labels.append(label)
        
        return nearby_labels
    
    def _calculate_arrow_direction(
        self,
        source_bbox: List[int],
        target_bbox: List[int]
    ) -> np.ndarray:
        """
        Calculate normalized direction vector from source to target.
        
        Args:
            source_bbox: Source node bounding box
            target_bbox: Target node bounding box
            
        Returns:
            Normalized direction vector
        """
        source_center = self._get_center(source_bbox)
        target_center = self._get_center(target_bbox)
        
        direction = target_center - source_center
        norm = np.linalg.norm(direction)
        
        if norm == 0:
            return np.array([0, 0])
        
        return direction / norm
    
    def _distance_to_line(
        self,
        point: Tuple[int, int],
        line_start: np.ndarray,
        line_end: np.ndarray
    ) -> float:
        """
        Calculate perpendicular distance from point to line segment.
        
        Args:
            point: Point coordinates
            line_start: Line start point
            line_end: Line end point
            
        Returns:
            Distance in pixels
        """
        point = np.array(point)
        
        # Vector from start to end
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return np.linalg.norm(point - line_start)
        
        # Normalized line vector
        line_unit = line_vec / line_len
        
        # Vector from start to point
        point_vec = point - line_start
        
        # Project point onto line
        projection = np.dot(point_vec, line_unit)
        
        # Clamp to line segment
        projection = max(0, min(line_len, projection))
        
        # Closest point on line
        closest = line_start + projection * line_unit
        
        # Distance to closest point
        return np.linalg.norm(point - closest)
    
    def _calculate_alignment_score(
        self,
        arrow: Dict,
        label: Dict,
        source_node: Dict,
        target_node: Dict
    ) -> float:
        """
        Calculate alignment score for (arrow, label) pair.
        
        Args:
            arrow: Arrow dictionary
            label: Label dictionary
            source_node: Source node dictionary
            target_node: Target node dictionary
            
        Returns:
            Alignment score (higher is better)
        """
        # Get arrow direction
        arrow_dir = self._calculate_arrow_direction(
            source_node['bbox'],
            target_node['bbox']
        )
        
        # Get label direction from source
        source_center = self._get_center(source_node['bbox'])
        label_center = np.array(label['center'])
        label_vec = label_center - source_center
        label_norm = np.linalg.norm(label_vec)
        
        if label_norm == 0:
            return -999  # Label at source center, invalid
        
        label_dir = label_vec / label_norm
        
        # Direction alignment (dot product)
        direction_score = np.dot(arrow_dir, label_dir)
        
        # Distance to arrow path
        target_center = self._get_center(target_node['bbox'])
        dist = self._distance_to_line(
            label['center'],
            source_center,
            target_center
        )
        
        distance_penalty = dist / self.distance_penalty_factor
        
        return direction_score - distance_penalty
    

    def _assign_labels_simple(
        self,
        arrows: List[Dict],
        elements: List[Dict],
        labels: List[Dict]
    ) -> Dict[int, str]:
        """
        Assign labels to arrows using direction-based matching.
        
        In troubleshooting diagrams:
        - "nee" arrows typically go RIGHT (to action boxes)
        - "ja" arrows typically go DOWN (to next decision)
        
        We use both distance to midpoint AND direction alignment.
        """
        assignments = {}
        used_labels = set()
        
        for idx, arrow in enumerate(arrows):
            source_node = next((e for e in elements if e['id'] == arrow['source']), None)
            target_node = next((e for e in elements if e['id'] == arrow['target']), None)
            
            if not source_node or not target_node:
                continue
            
            # Calculate arrow direction
            src_center = self._get_center(source_node['bbox'])
            tgt_center = self._get_center(target_node['bbox'])
            arrow_vec = tgt_center - src_center
            arrow_length = np.linalg.norm(arrow_vec)
            
            if arrow_length == 0:
                continue
            
            arrow_dir = arrow_vec / arrow_length
            midpoint = (src_center + tgt_center) / 2
            
            # Score each label
            best_label = None
            best_score = -float('inf')
            best_idx = -1
            
            for i, label in enumerate(labels):
                if i in used_labels:
                    continue
                
                label_center = np.array(label['center'])
                
                # Distance to midpoint (closer is better)
                dist_to_mid = np.linalg.norm(label_center - midpoint)
                
                # Skip if too far from arrow
                if dist_to_mid > 150:
                    continue
                
                # Direction alignment: label should be in direction of arrow
                label_vec = label_center - src_center
                label_dist = np.linalg.norm(label_vec)
                
                if label_dist > 0:
                    label_dir = label_vec / label_dist
                    alignment = np.dot(arrow_dir, label_dir)  # 1.0 = same direction, -1.0 = opposite
                else:
                    alignment = 0
                
                # Score = -distance + alignment_bonus
                # Normalize distance to 0-1 range (assuming max relevant distance is 150px)
                dist_score = 1 - (dist_to_mid / 150)
                score = dist_score * 0.6 + alignment * 0.4  # 60% distance, 40% direction
                
                if score > best_score:
                    best_score = score
                    best_label = label
                    best_idx = i
            
            if best_label and best_score > 0.3:  # Threshold to avoid bad matches
                assignments[idx] = best_label['text']
                used_labels.add(best_idx)
        
        return assignments

    def _assign_labels_to_arrows(
        self,
        arrows: List[Dict],
        elements: List[Dict],
        labels: List[Dict]
    ) -> Dict[int, str]:
        """
        Assign labels to arrows using direction-based matching.
        
        Args:
            arrows: List of arrow dictionaries
            elements: List of element dictionaries
            labels: List of label dictionaries (ja/nee)
            
        Returns:
            Dictionary mapping arrow index to label text
        """
        if not labels:
            return {}
        
        # Group arrows by source
        by_source = {}
        for idx, arrow in enumerate(arrows):
            source_id = arrow['source']
            if source_id not in by_source:
                by_source[source_id] = []
            by_source[source_id].append((idx, arrow))
        
        assignments = {}
        
        for source_id, source_arrows in by_source.items():
            # Get source node
            source_node = next((e for e in elements if e['id'] == source_id), None)
            if not source_node:
                continue
            
            if len(source_arrows) == 1:
                # Simple case: single arrow, find nearest label
                idx, arrow = source_arrows[0]
                target_node = next((e for e in elements if e['id'] == arrow['target']), None)
                
                if target_node:
                    # Find labels near source
                    candidates = self._find_labels_near_node(
                        source_node, labels, radius=self.single_arrow_radius
                    )
                    
                    if candidates:
                        # Get closest to arrow path
                        best_label = None
                        best_score = -999
                        
                        for label in candidates:
                            score = self._calculate_alignment_score(
                                arrow, label, source_node, target_node
                            )
                            if score > best_score:
                                best_score = score
                                best_label = label
                        
                        if best_label:
                            assignments[idx] = best_label['text']
            else:
                # Branching: direction-based assignment
                candidates = self._find_labels_near_node(
                    source_node, labels, radius=self.branching_radius
                )
                
                if not candidates:
                    continue
                
                # Score all (arrow, label) pairs
                scores = []
                for idx, arrow in source_arrows:
                    target_node = next((e for e in elements if e['id'] == arrow['target']), None)
                    if not target_node:
                        continue
                    
                    for label in candidates:
                        score = self._calculate_alignment_score(
                            arrow, label, source_node, target_node
                        )
                        scores.append((score, idx, label))
                
                # Greedy assignment (best score first)
                scores.sort(reverse=True, key=lambda x: x[0])
                used_labels = set()
                
                for score, idx, label in scores:
                    if label['id'] not in used_labels and idx not in assignments:
                        assignments[idx] = label['text']
                        used_labels.add(label['id'])
        
        return assignments
    
    def process_image(
        self,
        image_path: str,
        detection_json_path: str,
        output_dir: str,
        labels: List[Dict] = None,
        save_visualization: bool = True
    ) -> Dict:
        """
        Process single image and detect arrows.
        
        Args:
            image_path: Path to input image
            detection_json_path: Path to detection results JSON
            output_dir: Directory to save outputs
            labels: Optional list of decision labels (ja/nee)
            save_visualization: Whether to save visualization image
            
        Returns:
            Arrow detection results
        """
        # Load detection results
        detection_data = load_json(detection_json_path)
        
        # Detect arrows
        result = self.detect_arrows(image_path, detection_data, labels)
        
        # Save JSON
        ensure_dir(output_dir)
        image_name = Path(image_path).stem
        json_path = os.path.join(output_dir, f"{image_name}_arrows.json")
        save_json(result, json_path)
        
        # Save visualization
        if save_visualization:
            from ..utils.visualization import draw_element_types, draw_arrows
            image = cv2.imread(image_path)
            vis_image = draw_element_types(image, detection_data['elements'])
            vis_image = draw_arrows(vis_image, result['graph']['edges'], detection_data['elements'])
            vis_path = os.path.join(output_dir, f"{image_name}_arrows_vis.png")
            cv2.imwrite(vis_path, vis_image)
        
        return result
    

    def _infer_flow_edges(
        self,
        arrows: List[Dict],
        elements: List[Dict]
    ) -> List[Dict]:
        """
        Infer missing flow edges based on troubleshooting diagram patterns.
        
        Patterns:
        1. Decision boxes have 2 exits: "ja" (down) and "nee" (right)
        2. Process boxes (notes) often connect back to the decision they explain
        
        Returns list of additional arrows to add
        """
        additional_arrows = []
        
        # Sort nodes by Y position (vertical flow)
        sorted_nodes = sorted(elements, key=lambda n: n['bbox'][1])
        
        # Identify decision-like nodes (questions or numbered steps)
        decision_nodes = []
        for node in sorted_nodes:
            text = node.get('text', '').lower()
            if '?' in text or any(f'{i}.' in text for i in range(10, 30)):
                decision_nodes.append(node)
        
        # --- Pattern 1: Decision -> Next Decision (ja path) ---
        for i, decision in enumerate(decision_nodes):
            outgoing = [a for a in arrows if a['source'] == decision['id']]
            
            # If only 1 outgoing edge, try to infer the "ja" path
            if len(outgoing) == 1:
                dy = decision['bbox'][1]
                
                # Look for next decision below
                for nd in decision_nodes[i+1:]:
                    if nd['bbox'][1] > dy + 100:
                        has_edge = any(
                            a['source'] == decision['id'] and a['target'] == nd['id']
                            for a in arrows + additional_arrows
                        )
                        if not has_edge:
                            additional_arrows.append({
                                'source': decision['id'],
                                'target': nd['id'],
                                'inferred': True
                            })
                        break
        
        # --- Pattern 2: Process -> Decision (feedback/info arrows) ---
        # Process boxes (notes like "Xa) ...") are typically LEFT of their decision
        # They have no outgoing arrows but should connect to the decision
        
        for node in sorted_nodes:
            text = node.get('text', '').lower()
            # Process boxes often start with "Na)" patterns
            if ')' in text[:5] and any(c.isdigit() for c in text[:3]):
                # Check if this node has any outgoing arrows
                outgoing = [a for a in arrows + additional_arrows if a['source'] == node['id']]
                
                if not outgoing:
                    # Find the closest decision box at similar Y level or slightly above
                    ny = node['bbox'][1]
                    nx = node['bbox'][0]
                    
                    best_decision = None
                    best_score = -float('inf')
                    
                    for dn in decision_nodes:
                        dy = dn['bbox'][1]
                        dx = dn['bbox'][0]
                        
                        # Decision should be roughly at same Y (within 150px) and to the right
                        if abs(dy - ny) < 150 and dx > nx:
                            # Score: prefer closer in Y, and to the right
                            y_dist = abs(dy - ny)
                            score = -y_dist + (dx - nx) * 0.1
                            
                            if score > best_score:
                                best_score = score
                                best_decision = dn
                    
                    if best_decision:
                        has_edge = any(
                            a['source'] == node['id'] and a['target'] == best_decision['id']
                            for a in arrows + additional_arrows
                        )
                        if not has_edge:
                            additional_arrows.append({
                                'source': node['id'],
                                'target': best_decision['id'],
                                'inferred': True
                            })
        
        return additional_arrows
    
    def _detect_edges_via_lines(
        self,
        boxes: List[Dict],
        line_segments: List[Tuple[int, int, int, int]],
        existing_edges: Set[Tuple[int, int]],
        image: np.ndarray
    ) -> List[Dict]:
        """
        Detect edges by finding lines connecting boxes (fallback for missing arrowheads).
        
        Uses standard flowchart direction conventions (ISO 5807):
        - Vertical flow: top → bottom (source above, target below)
        - Horizontal flow: left → right (source left, target right)
        
        Also handles discontinuous lines by detecting line segments near box boundaries.
        
        Args:
            boxes: List of detected boxes
            line_segments: Detected line segments
            existing_edges: Already detected edges (from arrowheads)
            image: Original image
            
        Returns:
            List of fallback arrows
        """
        fallback_arrows = []
        threshold = self.source_trace_threshold
        
        # For each box, find lines that touch its boundary
        box_lines = {}  # box_id -> list of line endpoints that touch it
        for box in boxes:
            box_id = box['id']
            x, y, w, h = box['bbox']
            box_lines[box_id] = []
            
            # Check each line segment
            for line in line_segments:
                x1, y1, x2, y2 = line
                
                # Check if either endpoint is near box boundary
                for lx, ly in [(x1, y1), (x2, y2)]:
                    # Distance to box boundary
                    dx = max(x - lx, 0, lx - (x + w))
                    dy = max(y - ly, 0, ly - (y + h))
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist <= threshold:
                        # Record the OTHER endpoint (where line goes)
                        if (lx, ly) == (x1, y1):
                            other = (x2, y2)
                        else:
                            other = (x1, y1)
                        box_lines[box_id].append({
                            'touch_point': (lx, ly),
                            'far_point': other,
                            'line': line
                        })
        
        # Find pairs of boxes connected by lines
        for box1 in boxes:
            id1 = box1['id']
            center1 = self._get_center(box1['bbox'])
            
            for box2 in boxes:
                id2 = box2['id']
                if id1 >= id2:  # Avoid duplicates and self-loops
                    continue
                
                # Already detected via arrowhead?
                if (id1, id2) in existing_edges or (id2, id1) in existing_edges:
                    continue
                
                center2 = self._get_center(box2['bbox'])
                
                # === DISTANCE CONSTRAINT ===
                # Skip boxes that are too far apart (no edge expected)
                box_distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                max_edge_distance = 250  # Max distance for connected boxes (domain-agnostic)
                if box_distance > max_edge_distance:
                    continue
                
                # Check if any line from box1 reaches near box2 (or vice versa)
                connected = False
                
                # Direct connection: line endpoint from box1 near box2
                for line_info in box_lines.get(id1, []):
                    far = line_info['far_point']
                    x2, y2, w2, h2 = box2['bbox']
                    dx = max(x2 - far[0], 0, far[0] - (x2 + w2))
                    dy = max(y2 - far[1], 0, far[1] - (y2 + h2))
                    dist_to_box2 = np.sqrt(dx**2 + dy**2)
                    
                    if dist_to_box2 <= threshold * 1.5:  # Conservative for fallback
                        connected = True
                        break
                
                # Also check reverse (line from box2 reaches box1)
                if not connected:
                    for line_info in box_lines.get(id2, []):
                        far = line_info['far_point']
                        x1, y1, w1, h1 = box1['bbox']
                        dx = max(x1 - far[0], 0, far[0] - (x1 + w1))
                        dy = max(y1 - far[1], 0, far[1] - (y1 + h1))
                        dist_to_box1 = np.sqrt(dx**2 + dy**2)
                        
                        if dist_to_box1 <= threshold * 1.5:
                            connected = True
                            break
                
                # Check for discontinuous: line near both boxes but with gap
                if not connected:
                    lines_near_1 = box_lines.get(id1, [])
                    lines_near_2 = box_lines.get(id2, [])
                    
                    if lines_near_1 and lines_near_2:
                        # Check if far endpoints are close (gap bridging)
                        for l1 in lines_near_1:
                            for l2 in lines_near_2:
                                gap_dist = np.sqrt(
                                    (l1['far_point'][0] - l2['far_point'][0])**2 +
                                    (l1['far_point'][1] - l2['far_point'][1])**2
                                )
                                if gap_dist <= threshold * 2:  # Allow small gaps only
                                    connected = True
                                    break
                            if connected:
                                break
                
                if connected:
                    # Determine direction using flowchart heuristics (domain-agnostic)
                    # Standard: top→bottom, left→right
                    dy = center2[1] - center1[1]  # Positive if box2 is below
                    dx = center2[0] - center1[0]  # Positive if box2 is to the right
                    
                    # Primarily vertical layout (typical flowchart)
                    if abs(dy) > abs(dx) * 0.5:  # More vertical than horizontal
                        if dy > 0:  # box2 is below box1
                            source, target = id1, id2
                        else:
                            source, target = id2, id1
                    else:  # Horizontal branch
                        if dx > 0:  # box2 is to the right of box1
                            source, target = id1, id2
                        else:
                            source, target = id2, id1
                    
                    # Avoid duplicates
                    if (source, target) not in existing_edges:
                        fallback_arrows.append({
                            'source': source,
                            'target': target
                        })
                        existing_edges.add((source, target))
        
        return fallback_arrows

    def detect_arrows(
        self,
        image_path: str,
        elements_data: Dict,
        labels: List[Dict] = None
    ) -> Dict:
        """
        Detect arrows using line tracing + proximity matching.
        
        Args:
            image_path: Path to input image
            elements_data: Detection results with elements and arrowheads
            labels: List of decision labels (ja/nee) from OCR
            
        Returns:
            Arrow detection results
        """
        # Load image and separate elements
        image = cv2.imread(image_path)
        elements = elements_data['elements']
        
        arrowheads = [e for e in elements if e.get('class_id') == 0]  # class 0 = arrowhead
        boxes = [e for e in elements if e.get('class_id') != 0]  # All except arrowheads
        
        # Create mask for boxes (exclude arrowheads from mask)
        element_mask = self._mask_elements(image, boxes)
        
        # Detect line segments
        line_segments = self._detect_lines(image, element_mask)
        
        # Process each arrowhead
        arrows = []
        
        for arrowhead in arrowheads:
            # 1. Identify arrowhead ends
            blunt_end, pointy_end = self._identify_arrowhead_ends(arrowhead, image, boxes)
            
            # 2. Find target (box touching pointy end) - PROXIMITY
            target_id = self._find_box_touching_point(
                pointy_end, boxes, threshold=self.target_touch_threshold
            )
            
            if target_id is None:
                continue  # Skip if no target found
            
            # 3. Find source(s) - LINE TRACING with junction detection
            lines_at_blunt = self._find_lines_at_point(
                blunt_end, line_segments, threshold=self.line_attachment_threshold
            )
            
            if not lines_at_blunt:
                # No lines found - might be detection issue
                continue
            
            # Trace each line at blunt end (usually just one, but could be multiple)
            all_sources = []
            for line in lines_at_blunt:
                sources = self._trace_line_to_sources(
                    line, blunt_end, line_segments, boxes
                )
                all_sources.extend(sources)
            
            # Remove duplicates
            all_sources = list(set(all_sources))
            
            # Create arrow for each source → target connection
            for source_id in all_sources:
                # Skip self-loops (node connecting to itself)
                if source_id == target_id:
                    continue
                
                arrows.append({
                    'source': source_id,
                    'target': target_id,
                    'arrowhead_id': arrowhead['id']
                })
        
        # Assign labels to arrows using simple midpoint-based matching
        # Labels are typically ON the arrow line, so we find the closest label to each arrow's midpoint
        label_assignments = {}
        
        # LINE-BASED FALLBACK
        # Find edges without arrowheads (for alternative diagrams and missed arrowheads)
        # Uses standard flowchart direction heuristics (ISO 5807):
        # - Top → bottom for vertical flow
        # - Left → right for horizontal branches
        # NOTE: Disabled by default as arrowhead-only currently achieves better precision.
        # Enable via config: arrows.enable_line_fallback = true
        
        if self.enable_line_fallback:
            arrowhead_edges = set((arrow['source'], arrow['target']) for arrow in arrows)
            fallback_arrows = self._detect_edges_via_lines(
                boxes, line_segments, arrowhead_edges, image
            )
            
            # Combine: arrowhead edges + fallback edges
            for fa in fallback_arrows:
                arrows.append({
                    'source': fa['source'],
                    'target': fa['target'],
                    'method': 'line_fallback'
                })
        
        if labels:
            label_assignments = self._assign_labels_simple(arrows, boxes, labels)
        
        # Prepare output as directed graph
        image_name = Path(image_path).stem
        
        # Create nodes (exclude arrowheads - class 5)
        nodes = []
        for element in boxes:
            nodes.append({
                'id': element['id'],
                'type': YOLO_CLASS_NAMES.get(element.get('class_id'), 'unknown'),
                'text': element.get('text', ''),
                'bbox': element['bbox']
            })
        
        # Create edges with labels
        edges = []
        for idx, arrow in enumerate(arrows):
            edge = {
                'source': arrow['source'],
                'target': arrow['target'],
                'type': 'leads_to'
            }
            # Add label if assigned
            if idx in label_assignments:
                edge['label'] = label_assignments[idx]
            edges.append(edge)
        
        result = {
            'image_path': image_path,
            'image_name': image_name,
            'graph': {
                'nodes': nodes,
                'edges': edges
            },
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }
        
        return result
