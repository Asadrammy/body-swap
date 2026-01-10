"""Detect actual mask boundaries in image and replace them"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.utils.image_utils import load_image, save_image
from src.pipeline.preprocessor import Preprocessor
from src.pipeline.face_processor import FaceProcessor

setup_logger()
logger = get_logger(__name__)


def detect_mask_boundaries(image, face_bboxes):
    """Detect actual mask boundaries using color thresholding"""
    mask_regions = []
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define beige/tan color range (masks are light beige)
    # Beige is typically in the range: low saturation, high value
    lower_beige = np.array([10, 20, 180])  # Light beige
    upper_beige = np.array([30, 80, 255])  # Slightly darker beige
    
    # Create mask for beige regions
    beige_mask = cv2.inRange(hsv, lower_beige, upper_beige)
    
    # Find contours of beige regions
    contours, _ = cv2.findContours(beige_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours near face bboxes - find all masks
    used_contours = set()
    
    for face_bbox in face_bboxes:
        fx, fy, fw, fh = face_bbox
        face_center_x = fx + fw // 2
        face_center_y = fy + fh // 2
        
        best_match = None
        best_distance = float('inf')
        
        # Find closest contour to face center
        for idx, contour in enumerate(contours):
            if idx in used_contours:
                continue
                
            # Get bounding rect of contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate distance from face center to contour center
            contour_center_x = x + w // 2
            contour_center_y = y + h // 2
            distance = np.sqrt((face_center_x - contour_center_x)**2 + (face_center_y - contour_center_y)**2)
            
            # Check if face center is within or near this contour
            if (x <= face_center_x <= x + w and y <= face_center_y <= y + h) or distance < max(w, h):
                if distance < best_distance:
                    best_match = (idx, x, y, w, h)
                    best_distance = distance
        
        if best_match:
            idx, x, y, w, h = best_match
            mask_regions.append({
                'bbox': [x, y, w, h],
                'original_face_bbox': face_bbox
            })
            used_contours.add(idx)
            logger.info(f"Detected mask: ({x}, {y}) size ({w}, {h}) for face at ({fx}, {fy})")
    
    return mask_regions


def main():
    """Detect and replace masks"""
    logger.info("=" * 60)
    logger.info("DETECTING ACTUAL MASK BOUNDARIES AND REPLACING")
    logger.info("=" * 60)
    
    template_path = "IMG20251019131550.jpg"
    customer_path = "1760713603491 (1).jpg"
    output_path = "outputs/mask_boundary_replacement.png"
    
    # Load images
    template_img = load_image(template_path)
    customer_img = load_image(customer_path)
    
    # Preprocess
    preprocessor = Preprocessor()
    template_data = preprocessor.preprocess_template(template_path)
    customer_data = preprocessor.preprocess_customer_photos([customer_path])
    
    template_faces = template_data["faces"]
    customer_faces = customer_data["faces"][0]
    
    if not template_faces or not customer_faces:
        logger.error("Missing faces!")
        return
    
    # Detect actual mask boundaries
    template_face_bboxes = [f.get("bbox", [0, 0, 0, 0]) for f in template_faces]
    mask_regions = detect_mask_boundaries(template_data["image"], template_face_bboxes)
    
    if not mask_regions:
        logger.warning("Could not detect mask boundaries, using expanded face regions")
        # Fallback: expand face regions significantly
        for bbox in template_face_bboxes:
            x, y, w, h = bbox
            expand = 3.0  # Very large expansion
            new_w = int(w * expand)
            new_h = int(h * expand)
            new_x = max(0, x - (new_w - w) // 2)
            new_y = max(0, y - (new_h - h) // 2)
            mask_regions.append({
                'bbox': [new_x, new_y, new_w, new_h],
                'original_face_bbox': bbox
            })
    
    # Extract customer face
    face_processor = FaceProcessor()
    customer_face_identity = face_processor.extract_face_identity(
        customer_data["images"][0], customer_faces[0]
    )
    customer_face_aligned = customer_face_identity["aligned_face"]
    
    # Replace each mask
    result = template_data["image"].copy()
    for i, mask_region in enumerate(mask_regions):
        x, y, w, h = mask_region['bbox']
        logger.info(f"Replacing mask {i+1}: ({x}, {y}) size ({w}, {h})")
        
        # Resize customer face to match mask size
        customer_face_resized = cv2.resize(customer_face_aligned, (w, h))
        
        # Replace entire mask region
        result[y:y+h, x:x+w] = customer_face_resized
    
    # Save
    save_image(result, output_path)
    logger.info(f"Result saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

