"""Input validation and preprocessing"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from ..utils.image_utils import load_image, resize_image
from ..utils.logger import get_logger
from ..utils.config import get_config
from ..models.face_detector import FaceDetector

logger = get_logger(__name__)


class Preprocessor:
    """Handle input validation and preprocessing"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.face_detector = FaceDetector()
        self.max_image_size = get_config("processing.max_image_size", 1024)
        self.target_size = get_config("processing.image_size", 512)
    
    def validate_image(self, image_path: str) -> Dict:
        """
        Validate image file
        
        Args:
            image_path: Path to image file
        
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": False,
            "error": None,
            "metadata": {}
        }
        
        try:
            image_path = Path(image_path)
            
            # Check if file exists
            if not image_path.exists():
                result["error"] = f"File not found: {image_path}"
                return result
            
            # Check file size
            file_size = image_path.stat().st_size
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                result["error"] = f"File too large: {file_size / (1024*1024):.2f}MB"
                return result
            
            # Try to load image
            image = load_image(image_path)
            
            # Check image dimensions
            h, w = image.shape[:2]
            if h < 64 or w < 64:
                result["error"] = f"Image too small: {w}x{h}"
                return result
            
            # Use config max_image_size, but allow up to 8192 for validation
            # (actual resizing will happen in preprocessing)
            max_dimension = max(self.max_image_size * 4, 8192)  # Allow larger images, will be resized
            if h > max_dimension or w > max_dimension:
                result["error"] = f"Image too large: {w}x{h} (max: {max_dimension})"
                return result
            
            # Extract metadata
            result["metadata"] = {
                "width": w,
                "height": h,
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
                "file_size": file_size,
                "format": image_path.suffix.lower()
            }
            
            result["valid"] = True
            
        except Exception as e:
            result["error"] = f"Validation error: {str(e)}"
            logger.error(f"Image validation failed: {e}")
        
        return result
    
    def preprocess_customer_photos(self, image_paths: List[str]) -> Dict:
        """
        Preprocess customer reference photos
        
        Args:
            image_paths: List of paths to customer photos (1-2 images)
        
        Returns:
            Preprocessing result with images and detected faces
        """
        if len(image_paths) < 1 or len(image_paths) > 2:
            raise ValueError("Must provide 1-2 customer photos")
        
        result = {
            "images": [],
            "faces": [],
            "metadata": [],
            "errors": []
        }
        
        for idx, image_path in enumerate(image_paths):
            # Validate image
            validation = self.validate_image(image_path)
            if not validation["valid"]:
                result["errors"].append(f"Image {idx+1}: {validation['error']}")
                continue
            
            # Load and preprocess image
            try:
                image = load_image(image_path)
                
                # Resize if too large
                h, w = image.shape[:2]
                if max(h, w) > self.max_image_size:
                    scale = self.max_image_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = resize_image(image, (new_w, new_h))
                    logger.info(f"Resized image {idx+1} from {w}x{h} to {new_w}x{new_h}")
                
                # Detect faces
                faces = self.face_detector.detect_faces(image)
                
                if not faces:
                    result["errors"].append(f"Image {idx+1}: No faces detected")
                    continue
                
                # Store results
                result["images"].append(image)
                result["faces"].append(faces)
                result["metadata"].append({
                    "path": str(image_path),
                    "size": image.shape[:2],
                    "num_faces": len(faces)
                })
                
            except Exception as e:
                error_msg = f"Image {idx+1}: Preprocessing error - {str(e)}"
                result["errors"].append(error_msg)
                logger.error(error_msg)
        
        if not result["images"]:
            raise ValueError(f"Failed to process customer photos: {result['errors']}")
        
        return result
    
    def preprocess_template(self, template_path: str) -> Dict:
        """
        Preprocess template image
        
        Args:
            template_path: Path to template image
        
        Returns:
            Preprocessing result with template image
        """
        # Validate template
        validation = self.validate_image(template_path)
        if not validation["valid"]:
            raise ValueError(f"Invalid template image: {validation['error']}")
        
        # Load template
        image = load_image(template_path)
        
        # Resize if needed (keep original aspect ratio)
        h, w = image.shape[:2]
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = resize_image(image, (new_w, new_h))
            logger.info(f"Resized template from {w}x{h} to {new_w}x{new_h}")
        
        # Detect faces in template (for expression matching)
        # Try with more sensitive detection first
        faces = self.face_detector.detect_faces(image, min_size=15)
        
        # If no faces found, try with a larger version (sometimes faces are too small)
        if not faces:
            logger.warning(f"No faces detected in template at original size, trying with upscaled version...")
            h, w = image.shape[:2]
            # Upscale by 1.5x for better face detection
            upscaled = cv2.resize(image, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_LINEAR)
            faces = self.face_detector.detect_faces(upscaled, min_size=15)
            if faces:
                logger.info(f"✅ Found {len(faces)} face(s) in upscaled template")
                # Scale face coordinates back to original size
                for face in faces:
                    if "bbox" in face:
                        x, y, w_face, h_face = face["bbox"]
                        face["bbox"] = [int(x / 1.5), int(y / 1.5), int(w_face / 1.5), int(h_face / 1.5)]
                    if "landmarks" in face:
                        landmarks = face["landmarks"]
                        if isinstance(landmarks, np.ndarray):
                            face["landmarks"] = (landmarks / 1.5).astype(np.int32)
            else:
                logger.warning(f"⚠️  Still no faces detected after upscaling - template may not have visible faces")
                logger.warning(f"   Will proceed with body-only swap (no face swap)")
        
        result = {
            "image": image,
            "faces": faces,
            "metadata": {
                "path": str(template_path),
                "size": image.shape[:2],
                "num_faces": len(faces)
            }
        }
        
        return result
    
    def extract_metadata(self, image: np.ndarray, faces: List[Dict]) -> Dict:
        """
        Extract metadata from image and faces
        
        Args:
            image: Input image
            faces: Detected faces
        
        Returns:
            Metadata dictionary
        """
        h, w = image.shape[:2]
        
        metadata = {
            "image_size": (w, h),
            "num_faces": len(faces),
            "gender_estimate": None,
            "age_estimate": None,
            "body_type_estimate": None
        }
        
        if faces:
            # Aggregate face information
            ages = [f.get("age") for f in faces if f.get("age") is not None]
            genders = [f.get("gender") for f in faces if f.get("gender") is not None]
            
            if ages:
                metadata["age_estimate"] = {
                    "min": min(ages),
                    "max": max(ages),
                    "average": sum(ages) / len(ages)
                }
            
            if genders:
                # 0 = female, 1 = male typically
                male_count = sum(1 for g in genders if g == 1)
                female_count = sum(1 for g in genders if g == 0)
                metadata["gender_estimate"] = {
                    "male": male_count,
                    "female": female_count
                }
        
        return metadata

