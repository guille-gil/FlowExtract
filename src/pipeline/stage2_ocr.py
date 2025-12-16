"""
Stage 2: OCR Module

Extracts text content from detected elements using PaddleOCR or EasyOCR.
Preserves line breaks and bullet point structure.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import os
from pathlib import Path

from ..utils.io_utils import load_json, save_json, ensure_dir


class OCREngine:
    """OCR engine for text extraction from element images."""
    
    def __init__(self, config: Dict):
        """
        Initialize OCR engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        ocr_config = config.get('ocr', {})
        
        self.engine_name = ocr_config.get('engine', 'paddleocr')
        self.language = ocr_config.get('language', 'nl')
        self.use_gpu = ocr_config.get('use_gpu', True)
        self.confidence_threshold = ocr_config.get('confidence_threshold', 0.5)
        self.preserve_line_breaks = ocr_config.get('preserve_line_breaks', True)
        
        self.ocr = self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize OCR engine."""
        if self.engine_name == 'paddleocr':
            from paddleocr import PaddleOCR
            
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.language,
                use_gpu=self.use_gpu,
                show_log=False
            )
            return ocr
        
        elif self.engine_name == 'easyocr':
            import easyocr
            
            ocr = easyocr.Reader(
                [self.language],
                gpu=self.use_gpu
            )
            return ocr
        
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine_name}")
    
    def extract_text(self, image: np.ndarray) -> Dict:
        """
        Extract text from image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if self.engine_name == 'paddleocr':
            return self._extract_paddleocr(image)
        elif self.engine_name == 'easyocr':
            return self._extract_easyocr(image)
    
    def _extract_paddleocr(self, image: np.ndarray) -> Dict:
        """Extract text using PaddleOCR."""
        result = self.ocr.ocr(image, cls=True)
        
        if result is None or len(result) == 0 or result[0] is None:
            return {
                'text': '',
                'lines': [],
                'confidence': 0.0
            }
        
        lines = []
        confidences = []
        
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            
            if confidence >= self.confidence_threshold:
                lines.append(text)
                confidences.append(confidence)
        
        # Combine lines
        if self.preserve_line_breaks:
            full_text = '\n'.join(lines)
        else:
            full_text = ' '.join(lines)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'text': full_text,
            'lines': lines,
            'confidence': float(avg_confidence)
        }
    
    def _extract_easyocr(self, image: np.ndarray) -> Dict:
        """Extract text using EasyOCR."""
        result = self.ocr.readtext(image)
        
        if not result:
            return {
                'text': '',
                'lines': [],
                'confidence': 0.0
            }
        
        lines = []
        confidences = []
        
        for detection in result:
            text = detection[1]
            confidence = detection[2]
            
            if confidence >= self.confidence_threshold:
                lines.append(text)
                confidences.append(confidence)
        
        # Combine lines
        if self.preserve_line_breaks:
            full_text = '\n'.join(lines)
        else:
            full_text = ' '.join(lines)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'text': full_text,
            'lines': lines,
            'confidence': float(avg_confidence)
        }
    
    def process_elements(
        self,
        image_path: str,
        detection_json_path: str,
        output_dir: str
    ) -> Dict:
        """
        Extract text from all detected elements.
        
        Args:
            image_path: Path to original image
            detection_json_path: Path to detection results JSON (YOLO already classified)
            output_dir: Directory to save OCR results
            
        Returns:
            OCR results dictionary
        """
        # Load image and detections
        image = cv2.imread(image_path)
        detections = load_json(detection_json_path)
        
        # Extract text from each element
        elements = detections['elements']
        for element in elements:
            bbox = element['bbox']
            x, y, w, h = bbox
            
            # Crop element
            crop = image[y:y+h, x:x+w]
            
            # Extract text
            ocr_result = self.extract_text(crop)
            element['text'] = ocr_result['text']
            element['text_lines'] = ocr_result['lines']
            element['ocr_confidence'] = ocr_result['confidence']
        
        # Save results
        image_name = Path(image_path).stem
        result = {
            'image_path': image_path,
            'image_name': image_name,
            'elements': elements
        }
        
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, f"{image_name}_ocr.json")
        save_json(result, output_path)
        
        return result
    
    def extract_decision_labels(self, ocr_results: Dict) -> List[Dict]:
        """
        Extract ja/nee decision labels from OCR results.
        
        Args:
            ocr_results: OCR results dictionary with elements
            
        Returns:
            List of label dictionaries with text, bbox, and center
        """
        labels = []
        label_id = 0
        
        for element in ocr_results.get('elements', []):
            text = element.get('text', '').strip().lower()
            
            # Check if text contains ja or nee
            if 'ja' in text or 'nee' in text:
                bbox = element['bbox']
                x, y, w, h = bbox
                
                # Determine which label
                label_text = 'ja' if 'ja' in text else 'nee'
                
                labels.append({
                    'id': label_id,
                    'text': label_text,
                    'bbox': bbox,
                    'center': (x + w // 2, y + h // 2),
                    'element_id': element.get('id')
                })
                label_id += 1
        
        return labels
