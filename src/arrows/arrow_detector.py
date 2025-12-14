"""
Stage 4: Arrow Detection Module

Detects directed connections (arrows) between elements.
Identifies arrow direction and extracts labels (e.g., "ja", "nee").
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

from ..utils.io_utils import load_json, save_json, ensure_dir


class ArrowDetector:
    """Detector for arrows connecting diagram elements."""
    
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
        self.min_line_length = line_config.get('min_line_length', 30)
        self.max_line_gap = line_config.get('max_line_gap', 10)
        
        # Endpoint matching
        endpoint_config = arrow_config.get('endpoint_matching', {})
        self.max_distance = endpoint_config.get('max_distance', 50)
        
        # Arrowhead detection
        arrowhead_config = arrow_config.get('arrowhead', {})
        self.arrowhead_method = arrowhead_config.get('method', 'morphological')
        
        # Label extraction
        label_config = arrow_config.get('label_extraction', {})
        self.search_radius = label_config.get('search_radius', 30)
        self.expected_labels = label_config.get('expected_labels', ['ja', 'nee'])
    
    def _get_element_centroids(self, elements: List[Dict]) -> Dict[int, Tuple[int, int]]:
        """Get centroids of all elements."""
        centroids = {}
        for element in elements:
            x, y, w, h = element['bbox']
            centroid = (x + w // 2, y + h // 2)
            centroids[element['id']] = centroid
        return centroids
    
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
    
    def _detect_lines(self, image: np.ndarray, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
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
    
    def _find_nearest_element(
        self,
        point: Tuple[int, int],
        centroids: Dict[int, Tuple[int, int]]
    ) -> Optional[int]:
        """Find nearest element to a point."""
        min_dist = float('inf')
        nearest_id = None
        
        for element_id, centroid in centroids.items():
            dist = np.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2)
            if dist < min_dist and dist <= self.max_distance:
                min_dist = dist
                nearest_id = element_id
        
        return nearest_id
    
    def _detect_arrowhead_direction(
        self,
        image: np.ndarray,
        line: Tuple[int, int, int, int]
    ) -> str:
        """
        Detect which end of the line has the arrowhead.
        
        Args:
            image: Input image
            line: Line segment (x1, y1, x2, y2)
            
        Returns:
            'start' or 'end' indicating arrowhead position
        """
        x1, y1, x2, y2 = line
        
        if self.arrowhead_method == 'morphological':
            # Simple heuristic: check pixel density near endpoints
            # This is a simplified approach; more sophisticated methods can be added
            
            # For now, assume arrowhead is at the end (x2, y2)
            # This can be improved with actual arrowhead detection
            return 'end'
        
        else:
            # Placeholder for CNN-based arrowhead detection
            return 'end'
    
    def _extract_label_near_point(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        ocr_engine
    ) -> Optional[str]:
        """Extract text label near a point (e.g., arrow midpoint)."""
        x, y = point
        
        # Define search region
        x1 = max(0, x - self.search_radius)
        y1 = max(0, y - self.search_radius)
        x2 = min(image.shape[1], x + self.search_radius)
        y2 = min(image.shape[0], y + self.search_radius)
        
        # Crop region
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        # Extract text
        ocr_result = ocr_engine.extract_text(crop)
        text = ocr_result['text'].strip().lower()
        
        # Check if text matches expected labels
        for label in self.expected_labels:
            if label.lower() in text:
                return label
        
        return None
    
    def detect_arrows(
        self,
        image_path: str,
        ocr_json_path: str,
        output_dir: str,
        ocr_engine=None
    ) -> Dict:
        """
        Detect arrows connecting elements.
        
        Args:
            image_path: Path to original image
            ocr_json_path: Path to OCR results JSON
            output_dir: Directory to save arrow detection results
            ocr_engine: OCR engine instance for label extraction
            
        Returns:
            Arrow detection results dictionary
        """
        # Load image and OCR results
        image = cv2.imread(image_path)
        ocr_results = load_json(ocr_json_path)
        elements = ocr_results['elements']
        
        # Get element centroids
        centroids = self._get_element_centroids(elements)
        
        # Mask out elements
        mask = self._mask_elements(image, elements)
        
        # Detect line segments
        line_segments = self._detect_lines(image, mask)
        
        # Match lines to elements
        arrows = []
        arrow_id = 0
        
        for line in line_segments:
            x1, y1, x2, y2 = line
            
            # Detect arrowhead direction
            direction = self._detect_arrowhead_direction(image, line)
            
            # Determine source and target based on direction
            if direction == 'end':
                source_point = (x1, y1)
                target_point = (x2, y2)
            else:
                source_point = (x2, y2)
                target_point = (x1, y1)
            
            # Find nearest elements
            source_id = self._find_nearest_element(source_point, centroids)
            target_id = self._find_nearest_element(target_point, centroids)
            
            if source_id is not None and target_id is not None:
                # Extract label near midpoint
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                
                label = None
                if ocr_engine is not None:
                    label = self._extract_label_near_point(image, (mid_x, mid_y), ocr_engine)
                
                arrows.append({
                    'id': arrow_id,
                    'source': source_id,
                    'target': target_id,
                    'line': [int(x1), int(y1), int(x2), int(y2)],
                    'label': label
                })
                arrow_id += 1
        
        # Save results
        image_name = Path(image_path).stem
        result = {
            'image_path': image_path,
            'image_name': image_name,
            'elements': elements,
            'arrows': arrows,
            'num_arrows': len(arrows)
        }
        
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, f"{image_name}_arrows.json")
        save_json(result, output_path)
        
        # Save visualization
        from ..utils.visualization import draw_element_types, draw_arrows
        vis_image = draw_element_types(image, elements)
        vis_image = draw_arrows(vis_image, arrows, elements)
        vis_path = os.path.join(output_dir, f"{image_name}_arrows_vis.png")
        cv2.imwrite(vis_path, vis_image)
        
        return result
