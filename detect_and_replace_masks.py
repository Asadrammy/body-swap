"""Detect mask regions and replace them with customer faces"""

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


def detect_mask_regions(image, face_bboxes):
    """Detect solid color mask regions around face bboxes"""
    mask_regions = []
    
    for bbox in face_bboxes:
        x, y, w, h = bbox
        # For masked faces, expand significantly to cover entire mask
        # Masks are typically 2-3x larger than detected face bbox
        expand_factor = 2.5  # Much larger expansion for masks
        expanded_w = int(w * expand_factor)
        expanded_h = int(h * expand_factor)
        expanded_x = max(0, x - (expanded_w - w) // 2)
        expanded_y = max(0, y - (expanded_h - h) // 2)
        expanded_x2 = min(image.shape[1], expanded_x + expanded_w)
        expanded_y2 = min(image.shape[0], expanded_y + expanded_h)
        expanded_w = expanded_x2 - expanded_x
        expanded_h = expanded_y2 - expanded_y
        
        # Always add expanded region (masks need full replacement)
        mask_regions.append({
            'bbox': [expanded_x, expanded_y, expanded_w, expanded_h],
            'original_face_bbox': bbox
        })
        logger.info(f"Expanded mask region: ({expanded_x}, {expanded_y}) size ({expanded_w}, {expanded_h}) from face bbox ({x}, {y}) size ({w}, {h})")
    
    return mask_regions


def replace_mask_with_face(image, mask_region, customer_face_aligned, face_processor, customer_face_data):
    """Replace entire mask region with customer face"""
    x, y, w, h = mask_region['bbox']
    original_face_bbox = mask_region['original_face_bbox']
    
    # Resize customer face to match mask region size
    customer_face_resized = cv2.resize(customer_face_aligned, (w, h))
    
    # Create result
    result = image.copy()
    
    # Replace entire mask region with customer face (100% replacement)
    result[y:y+h, x:x+w] = customer_face_resized
    
    logger.info(f"Replaced mask region at ({x}, {y}) size ({w}, {h}) with customer face")
    
    return result


def main():
    """Main function to replace masks with faces"""
    logger.info("=" * 60)
    logger.info("DETECTING AND REPLACING MASKS WITH FACES")
    logger.info("=" * 60)
    
    # Images
    template_image_path = "IMG20251019131550.jpg"
    customer_image_path = "1760713603491 (1).jpg"
    output_path = "outputs/mask_replacement_result.png"
    
    logger.info(f"Template: {template_image_path}")
    logger.info(f"Customer: {customer_image_path}")
    logger.info(f"Output: {output_path}")
    
    # Load and preprocess
    preprocessor = Preprocessor()
    face_processor = FaceProcessor()
    
    template_data = preprocessor.preprocess_template(template_image_path)
    customer_data = preprocessor.preprocess_customer_photos([customer_image_path])
    
    template_img = template_data["image"]
    template_faces = template_data["faces"]
    
    if not customer_data["faces"] or not customer_data["faces"][0]:
        logger.error("No faces in customer image!")
        return
    
    customer_img = customer_data["images"][0]
    customer_faces = customer_data["faces"][0]
    
    logger.info(f"Template faces: {len(template_faces)}")
    logger.info(f"Customer faces: {len(customer_faces)}")
    
    # Extract customer face identity
    customer_face_identity = face_processor.extract_face_identity(
        customer_img, customer_faces[0]
    )
    customer_face_aligned = customer_face_identity["aligned_face"]
    
    # Detect mask regions
    template_face_bboxes = [f.get("bbox", [0, 0, 0, 0]) for f in template_faces]
    mask_regions = detect_mask_regions(template_img, template_face_bboxes)
    
    if not mask_regions:
        logger.warning("No mask regions detected, using face bboxes")
        mask_regions = [{'bbox': bbox, 'original_face_bbox': bbox} for bbox in template_face_bboxes]
    
    # Replace each mask with customer face
    result = template_img.copy()
    for i, mask_region in enumerate(mask_regions):
        logger.info(f"Processing mask region {i+1}/{len(mask_regions)}...")
        result = replace_mask_with_face(
            result,
            mask_region,
            customer_face_aligned,
            face_processor,
            customer_face_identity
        )
    
    # Save result
    save_image(result, output_path)
    logger.info(f"Result saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

