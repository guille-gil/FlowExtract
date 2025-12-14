"""
Visualization utilities for the decomposed parsing pipeline.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image
        boxes: List of box dictionaries with 'bbox' key containing [x, y, w, h]
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Image with bounding boxes drawn
    """
    vis_image = image.copy()
    
    for box in boxes:
        bbox = box['bbox']
        x, y, w, h = bbox
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
        
        # Draw element ID if available
        if 'id' in box:
            cv2.putText(
                vis_image,
                str(box['id']),
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
    
    return vis_image


def draw_element_types(
    image: np.ndarray,
    elements: List[Dict]
) -> np.ndarray:
    """
    Draw bounding boxes with different colors for different element types.
    
    Args:
        image: Input image
        elements: List of element dictionaries with 'bbox' and 'type' keys
        
    Returns:
        Image with colored bounding boxes
    """
    vis_image = image.copy()
    
    # Color mapping for element types
    colors = {
        'observation': (0, 255, 0),    # Green
        'decision': (255, 0, 0),        # Blue
        'action': (0, 0, 255)           # Red
    }
    
    for element in elements:
        bbox = element['bbox']
        element_type = element.get('type', 'unknown')
        color = colors.get(element_type, (128, 128, 128))
        
        x, y, w, h = bbox
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # Draw type label
        label = f"{element.get('id', '?')}: {element_type}"
        cv2.putText(
            vis_image,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    return vis_image


def draw_arrows(
    image: np.ndarray,
    arrows: List[Dict],
    elements: List[Dict]
) -> np.ndarray:
    """
    Draw arrows between elements.
    
    Args:
        image: Input image
        arrows: List of arrow dictionaries with 'source' and 'target' keys
        elements: List of element dictionaries with 'id' and 'bbox' keys
        
    Returns:
        Image with arrows drawn
    """
    vis_image = image.copy()
    
    # Create element ID to centroid mapping
    centroids = {}
    for element in elements:
        x, y, w, h = element['bbox']
        centroid = (x + w // 2, y + h // 2)
        centroids[element['id']] = centroid
    
    # Draw arrows
    for arrow in arrows:
        source_id = arrow['source']
        target_id = arrow['target']
        
        if source_id in centroids and target_id in centroids:
            start = centroids[source_id]
            end = centroids[target_id]
            
            # Draw arrow line
            cv2.arrowedLine(
                vis_image,
                start,
                end,
                (255, 0, 255),  # Magenta
                2,
                tipLength=0.3
            )
            
            # Draw label if available
            if 'label' in arrow and arrow['label']:
                mid_x = (start[0] + end[0]) // 2
                mid_y = (start[1] + end[1]) // 2
                cv2.putText(
                    vis_image,
                    arrow['label'],
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 255),
                    1
                )
    
    return vis_image
