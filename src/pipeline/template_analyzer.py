"""Template image analysis for pose, clothing, and expression extraction"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..models.pose_detector import PoseDetector
from ..models.face_detector import FaceDetector
from ..models.segmenter import Segmenter

logger = get_logger(__name__)


class TemplateAnalyzer:
    """Analyze template image for pose, clothing, expression, and background"""
    
    def __init__(self):
        """Initialize template analyzer"""
        self.pose_detector = PoseDetector()
        self.face_detector = FaceDetector()
        self.segmenter = Segmenter()
    
    def analyze_template(self, template_image: np.ndarray, faces: List[Dict]) -> Dict:
        """
        Analyze template image comprehensively
        
        Args:
            template_image: Template image
            faces: Detected faces in template
        
        Returns:
            Complete template analysis
        """
        result = {
            "pose": None,
            "clothing": {},
            "expression": None,
            "background": None,
            "lighting": None,
            "metadata": {}
        }
        
        # Analyze pose
        pose_data = self.pose_detector.detect_pose(template_image)
        if pose_data:
            result["pose"] = pose_data[0]
        
        # Analyze clothing
        result["clothing"] = self._analyze_clothing(template_image, result["pose"])
        
        # Analyze facial expression
        if faces:
            result["expression"] = self._analyze_expression(template_image, faces[0])
        
        # Analyze background
        result["background"] = self._analyze_background(template_image, result["pose"])
        
        # Analyze lighting
        result["lighting"] = self._analyze_lighting(template_image)
        
        # Store metadata
        result["metadata"] = {
            "image_size": template_image.shape[:2],
            "has_pose": result["pose"] is not None,
            "has_face": len(faces) > 0,
            "num_clothing_items": len(result["clothing"].get("items", []))
        }
        
        return result
    
    def _analyze_clothing(self, image: np.ndarray, pose_data: Optional[Dict]) -> Dict:
        """
        Analyze clothing in template
        
        Args:
            image: Template image
            pose_data: Pose detection result
        
        Returns:
            Clothing analysis result
        """
        clothing = {
            "items": [],
            "masks": {},
            "regions": {}
        }
        
        if pose_data is None:
            logger.warning("No pose data available for clothing analysis")
            return clothing
        
        keypoints = pose_data.get("keypoints", {})
        
        # Segment clothing regions
        body_masks = self.segmenter.segment_body_parts(image, pose_data)
        
        # Identify clothing items based on body parts
        if "torso" in body_masks:
            clothing["items"].append("shirt")
            clothing["masks"]["shirt"] = body_masks["torso"]
            clothing["regions"]["shirt"] = self._get_region_from_mask(body_masks["torso"])
        
        if "left_arm" in body_masks or "right_arm" in body_masks:
            clothing["items"].append("sleeves")
            for side in ["left", "right"]:
                arm_key = f"{side}_arm"
                if arm_key in body_masks:
                    clothing["masks"][arm_key] = body_masks[arm_key]
        
        if "left_leg" in body_masks or "right_leg" in body_masks:
            clothing["items"].append("pants")
            for side in ["left", "right"]:
                leg_key = f"{side}_leg"
                if leg_key in body_masks:
                    clothing["masks"][leg_key] = body_masks[leg_key]
                    if "pants" not in clothing["regions"]:
                        clothing["regions"]["pants"] = []
                    clothing["regions"]["pants"].append(self._get_region_from_mask(body_masks[leg_key]))
        
        # Check for open chest (visible skin)
        if "torso" in body_masks:
            # Analyze torso region for skin vs clothing
            torso_mask = body_masks["torso"]
            torso_region = image[torso_mask > 0]
            if len(torso_region) > 0:
                # Check if it's skin-colored (simple heuristic)
                avg_color = np.mean(torso_region, axis=0)
                if self._is_skin_color(avg_color):
                    clothing["has_open_chest"] = True
                    clothing["visible_body_parts"] = ["chest"]
        
        return clothing
    
    def _analyze_expression(self, image: np.ndarray, face: Dict) -> Dict:
        """
        Analyze facial expression from template
        
        Args:
            image: Template image
            face: Face detection result
        
        Returns:
            Expression analysis result
        """
        landmarks = face.get("landmarks", [])
        
        if len(landmarks) < 5:
            return {"type": "neutral", "landmarks": landmarks}
        
        # Extract key facial features
        expression = {
            "landmarks": landmarks,
            "mouth_open": False,
            "smile": False,
            "eyes_open": True,
            "eyebrow_raised": False
        }
        
        # Analyze mouth
        if len(landmarks) >= 5:
            # Assuming landmarks follow standard format
            # This is simplified - real implementation would use more landmarks
            mouth_points = landmarks[-2:] if len(landmarks) >= 5 else []
            if len(mouth_points) == 2:
                mouth_distance = np.linalg.norm(
                    np.array(mouth_points[0]) - np.array(mouth_points[1])
                )
                expression["mouth_open"] = mouth_distance > 20  # Threshold
        
        # Analyze eyes (simplified)
        if len(landmarks) >= 2:
            eye_points = landmarks[:2] if len(landmarks) >= 2 else []
            # Check if eyes are visible/open
            expression["eyes_open"] = True  # Simplified
        
        # Classify expression type
        if expression["smile"] or (expression["mouth_open"] and not expression["smile"]):
            expression["type"] = "happy" if expression["smile"] else "surprised"
        else:
            expression["type"] = "neutral"
        
        return expression
    
    def _analyze_background(self, image: np.ndarray, pose_data: Optional[Dict]) -> Dict:
        """
        Analyze background
        
        Args:
            image: Template image
            pose_data: Pose detection result
        
        Returns:
            Background analysis result
        """
        # Create foreground mask from pose
        foreground_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if pose_data:
            bbox = pose_data.get("bbox", [0, 0, image.shape[1], image.shape[0]])
            x, y, w, h = bbox
            foreground_mask[y:y+h, x:x+w] = 255
        
        # Get background mask
        background_mask = self.segmenter.segment_background(image, foreground_mask)
        
        # Analyze background
        bg_region = image[background_mask > 0]
        
        background = {
            "mask": background_mask,
            "average_color": np.mean(bg_region, axis=0).tolist() if len(bg_region) > 0 else [0, 0, 0],
            "complexity": "simple"  # Simplified - could analyze texture/complexity
        }
        
        return background
    
    def _analyze_lighting(self, image: np.ndarray) -> Dict:
        """
        Analyze lighting conditions
        
        Args:
            image: Template image
        
        Returns:
            Lighting analysis result
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Calculate statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Determine lighting direction (simplified)
        # Analyze gradient to estimate light direction
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        lighting = {
            "brightness": float(mean_brightness),
            "contrast": float(std_brightness),
            "direction": "frontal",  # Simplified
            "hardness": "medium"  # Simplified
        }
        
        return lighting
    
    def _get_region_from_mask(self, mask: np.ndarray) -> List[int]:
        """Get bounding box region from mask"""
        if mask.sum() == 0:
            return [0, 0, 0, 0]
        
        coords = np.column_stack(np.where(mask > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
    
    def _is_skin_color(self, color: np.ndarray) -> bool:
        """Simple heuristic to check if color is skin-like"""
        # Convert to RGB if needed
        if len(color) == 3:
            r, g, b = color
            # Simple skin color range (can be improved)
            return (r > g > b) and (r > 100) and (g > 50) and (b < 200)
        return False

