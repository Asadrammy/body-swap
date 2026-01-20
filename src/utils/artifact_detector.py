"""Detect and remove artifacts like duplicate faces"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from ..utils.logger import get_logger
from ..models.face_detector import FaceDetector

logger = get_logger(__name__)


class ArtifactDetector:
    """Detect and remove artifacts from generated images"""
    
    def __init__(self):
        """Initialize artifact detector"""
        self.face_detector = FaceDetector()
    
    def detect_duplicate_faces(
        self,
        image: np.ndarray,
        expected_face_regions: List[Tuple[int, int, int, int]]
    ) -> List[Dict]:
        """
        Detect duplicate or misplaced faces in image - ENHANCED to catch all artifacts
        
        Args:
            image: Image to check
            expected_face_regions: List of (x, y, w, h) for expected face locations
        
        Returns:
            List of detected duplicate faces with their locations
        """
        # Detect all faces in image
        faces = self.face_detector.detect_faces(image)
        
        if not faces:
            return []
        
        duplicates = []
        h, w = image.shape[:2]
        
        # Calculate neck line (lowest point of expected faces)
        neck_y = h * 0.3  # Default neck position
        if expected_face_regions:
            neck_y = max([ey + eh for ex, ey, ew, eh in expected_face_regions], default=neck_y) + 30
        
        for face in faces:
            bbox = face.get("bbox", [0, 0, 0, 0])
            fx, fy, fw, fh = bbox
            face_center_y = fy + fh // 2
            
            # Check if this face is in an expected region
            is_expected = False
            for ex, ey, ew, eh in expected_face_regions:
                # Calculate overlap
                overlap_x = max(0, min(fx + fw, ex + ew) - max(fx, ex))
                overlap_y = max(0, min(fy + fh, ey + eh) - max(fy, ey))
                overlap_area = overlap_x * overlap_y
                face_area = fw * fh
                
                # If significant overlap (50% or more), it's expected
                if overlap_area > face_area * 0.5:
                    is_expected = True
                    break
            
            # If not expected, it's a duplicate
            if not is_expected:
                # Check if face is in body region (below neck or on clothing)
                if face_center_y > neck_y:
                    duplicates.append({
                        "bbox": bbox,
                        "type": "duplicate_body_face",
                        "confidence": 1.0,
                        "location": "body_region"
                    })
                    logger.warning(f"Duplicate face detected in body region at ({fx}, {fy}, {fw}, {fh})")
                # Also check if face is unusually small (likely an artifact)
                elif fw * fh < (w * h) * 0.01:  # Less than 1% of image area
                    duplicates.append({
                        "bbox": bbox,
                        "type": "small_artifact_face",
                        "confidence": 0.9,
                        "location": "small_artifact"
                    })
                    logger.warning(f"Small artifact face detected at ({fx}, {fy}, {fw}, {fh})")
        
        return duplicates
    
    def remove_duplicate_faces(
        self,
        image: np.ndarray,
        duplicate_faces: List[Dict],
        expected_face_regions: List[Tuple[int, int, int, int]],
        template_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Remove duplicate faces by inpainting with surrounding texture - ENHANCED
        
        Args:
            image: Image with duplicates
            duplicate_faces: List of duplicate face detections
            expected_face_regions: Expected face locations
            template_image: Original template to restore from
        
        Returns:
            Image with duplicates removed
        """
        if not duplicate_faces:
            return image
        
        result = image.copy()
        
        for dup_face in duplicate_faces:
            bbox = dup_face["bbox"]
            x, y, w, h = bbox
            
            # Create mask for duplicate face - use ellipse for more natural removal
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            center = (x + w // 2, y + h // 2)
            axes = (int(w * 0.6), int(h * 0.6))  # Slightly larger to ensure complete removal
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            
            # Expand mask significantly for better blending
            kernel = np.ones((25, 25), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=3)
            
            # Inpaint using surrounding texture with higher quality
            if template_image is not None and template_image.shape[:2] == image.shape[:2]:
                # Use template image for inpainting (best quality)
                inpainted = cv2.inpaint(result, mask, 5, cv2.INPAINT_TELEA)
                # Blend with original template in that region
                mask_3d = mask[:, :, np.newaxis] / 255.0
                # Use template as base, blend inpainted result
                template_region = template_image * (1 - mask_3d) + inpainted * mask_3d
                result = (result * (1 - mask_3d) + template_region * mask_3d).astype(np.uint8)
            else:
                # Use high-quality inpainting
                result = cv2.inpaint(result, mask, 5, cv2.INPAINT_TELEA)
            
            logger.info(f"Removed duplicate face at ({x}, {y}, {w}, {h})")
        
        return result
    
    def validate_face_placement(
        self,
        image: np.ndarray,
        expected_face_regions: List[Tuple[int, int, int, int]]
    ) -> Tuple[bool, List[Dict]]:
        """
        Validate that faces are only in expected regions
        
        Returns:
            (is_valid, list_of_issues)
        """
        duplicates = self.detect_duplicate_faces(image, expected_face_regions)
        return len(duplicates) == 0, duplicates

