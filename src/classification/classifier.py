"""
Stage 2: Element Classification Module

Classifies detected elements into three types:
- Observation (regular rectangles)
- Decision (diamonds)
- Action (rectangles with wavy bottom edge)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from typing import List, Dict, Tuple
import os
from pathlib import Path
from tqdm import tqdm

from ..utils.io_utils import load_json, save_json, ensure_dir


class ElementDataset(Dataset):
    """Dataset for element classification."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of paths to element crop images
            labels: List of integer labels (0: observation, 1: decision, 2: action)
            transform: Optional transforms to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


class ElementClassifier:
    """Element type classifier using CNN."""
    
    def __init__(self, config: Dict):
        """
        Initialize classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        classification_config = config.get('classification', {})
        
        self.model_name = classification_config.get('model_name', 'mobilenet_v2')
        self.num_classes = classification_config.get('num_classes', 3)
        self.input_size = classification_config.get('input_size', 224)
        self.batch_size = classification_config.get('batch_size', 32)
        self.model_path = classification_config.get('model_path')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        
        # Load pretrained weights if available
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class names
        self.class_names = ['observation', 'decision', 'action']
    
    def _build_model(self) -> nn.Module:
        """Build classification model."""
        if self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            # Replace classifier
            model.classifier[1] = nn.Linear(model.last_channel, self.num_classes)
        elif self.model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            # Replace final layer
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return model.to(self.device)
    
    def train(
        self,
        train_data_dir: str,
        val_split: float = 0.2,
        epochs: int = 50,
        learning_rate: float = 0.001
    ):
        """
        Train classifier on labeled data.
        
        Args:
            train_data_dir: Directory with subdirectories for each class
                           (e.g., train_data_dir/observation/, train_data_dir/decision/, etc.)
            val_split: Validation split ratio
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        # Load training data
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(train_data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            for img_file in Path(class_dir).glob('*.png'):
                image_paths.append(str(img_file))
                labels.append(class_idx)
        
        if len(image_paths) == 0:
            raise ValueError(f"No training images found in {train_data_dir}")
        
        print(f"Found {len(image_paths)} training images")
        
        # Split into train/val
        from sklearn.model_selection import train_test_split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=val_split, stratify=labels, random_state=42
        )
        
        # Create datasets
        train_dataset = ElementDataset(train_paths, train_labels, self.transform)
        val_dataset = ElementDataset(val_paths, val_labels, self.transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels_batch.size(0)
                train_correct += predicted.eq(labels_batch).sum().item()
            
            train_acc = 100. * train_correct / train_total
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels_batch in val_loader:
                    images = images.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels_batch)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels_batch.size(0)
                    val_correct += predicted.eq(labels_batch).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, "
                  f"Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(self.model_path)
                print(f"Saved best model with val_acc: {val_acc:.2f}%")
    
    def save_model(self, path: str):
        """Save model weights."""
        ensure_dir(os.path.dirname(path))
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
    
    def classify_element(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Classify single element image.
        
        Args:
            image: Element crop image (BGR format)
            
        Returns:
            Tuple of (class_name, confidence)
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        class_name = self.class_names[predicted.item()]
        confidence_score = confidence.item()
        
        return class_name, confidence_score
    
    def process_detections(
        self,
        image_path: str,
        detection_json_path: str,
        output_dir: str
    ) -> Dict:
        """
        Classify all detected elements in an image.
        
        Args:
            image_path: Path to original image
            detection_json_path: Path to detection results JSON
            output_dir: Directory to save classification results
            
        Returns:
            Classification results dictionary
        """
        # Load image and detections
        image = cv2.imread(image_path)
        detections = load_json(detection_json_path)
        
        # Classify each element
        elements = detections['elements']
        for element in elements:
            bbox = element['bbox']
            x, y, w, h = bbox
            
            # Crop element
            crop = image[y:y+h, x:x+w]
            
            # Classify
            class_name, confidence = self.classify_element(crop)
            element['type'] = class_name
            element['confidence'] = float(confidence)
        
        # Save results
        image_name = Path(image_path).stem
        result = {
            'image_path': image_path,
            'image_name': image_name,
            'elements': elements
        }
        
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, f"{image_name}_classification.json")
        save_json(result, output_path)
        
        # Save visualization
        from ..utils.visualization import draw_element_types
        vis_image = draw_element_types(image, elements)
        vis_path = os.path.join(output_dir, f"{image_name}_classification_vis.png")
        cv2.imwrite(vis_path, vis_image)
        
        return result
