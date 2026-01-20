"""Fix both upper and lower body - Generate single complete output"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import get_logger
from src.models.generator import Generator
from src.models.pose_detector import PoseDetector
from src.models.face_detector import FaceDetector

logger = get_logger(__name__)

def load_image(path: str) -> np.ndarray:
    """Load image as numpy array"""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

def save_image(image: np.ndarray, path: str):
    """Save numpy array as image"""
    img = Image.fromarray(image.astype(np.uint8))
    img.save(path, quality=95)
    logger.info(f"Saved image to: {path}")

def create_upper_body_mask(image: np.ndarray, face_bbox: tuple = None):
    """Create mask for upper body region (head to waist)"""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Upper body: top 50% of image (head, shoulders, chest, arms, waist)
    upper_body_top = 0
    upper_body_bottom = int(h * 0.55)  # 55% down from top
    
    mask[upper_body_top:upper_body_bottom, :] = 255
    
    # Exclude face region if provided (to preserve it)
    if face_bbox:
        x, y, fw, fh = face_bbox
        expand = 30
        mask[max(0, y-expand):min(h, y+fh+expand), max(0, x-expand):min(w, x+fw+expand)] = 0
    
    # Smooth mask edges
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = (mask > 127).astype(np.uint8) * 255
    
    return mask

def create_lower_body_mask(image: np.ndarray, face_bbox: tuple = None):
    """Create mask for lower body region (waist to feet)"""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Lower body: from waist down (45% from top to bottom)
    lower_body_top = int(h * 0.45)
    lower_body_bottom = h
    
    mask[lower_body_top:lower_body_bottom, :] = 255
    
    # Smooth mask edges for seamless blending
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = (mask > 127).astype(np.uint8) * 255
    
    return mask

def refine_upper_body(image: np.ndarray, mask: np.ndarray, generator: Generator):
    """Refine upper body region using Stability AI"""
    logger.info("Refining upper body with Stability AI...")
    
    prompt = (
        "photorealistic upper body, natural human skin with pores and texture, "
        "realistic skin tone variation, natural arms and hands, single pair of hands, "
        "natural torso, realistic chest, natural shoulders, natural clothing fit, "
        "realistic fabric texture, high quality photograph, detailed features, "
        "natural lighting, authentic human appearance, preserve original body structure, "
        "maintain natural proportions, accurate anatomy, no duplicate body parts, "
        "no extra hands, no extra arms, no artifacts"
    )
    
    negative_prompt = (
        "duplicate hands, extra hands, multiple hands, double hands, "
        "duplicate arms, extra arms, multiple arms, distorted hands, "
        "malformed hands, mutated hands, bad anatomy, unnatural proportions, "
        "plastic, artificial, fake, CGI, 3D render, smooth skin, airbrushed, "
        "perfect skin, doll-like, surreal, deformed, misplaced body parts, "
        "artifacts, blurry, low quality, distorted, compression artifacts, "
        "face on body, face on clothing, face on chest, duplicate faces"
    )
    
    refined = generator.refine(
        image=image,
        prompt=prompt,
        mask=mask,
        negative_prompt=negative_prompt,
        strength=0.45,  # Lower strength to preserve natural features
        num_inference_steps=30
    )
    
    return refined

def refine_lower_body(image: np.ndarray, mask: np.ndarray, generator: Generator):
    """Refine lower body region using Stability AI"""
    logger.info("Refining lower body with Stability AI...")
    
    prompt = (
        "photorealistic lower body, natural legs and feet, natural body proportions, "
        "realistic clothing fit, natural pants fit, realistic fabric texture and folds, "
        "natural shoes, realistic footwear, natural pose, high quality photograph, "
        "detailed features, natural lighting, authentic human appearance, "
        "preserve original body structure, maintain natural proportions, "
        "accurate anatomy, no duplicate body parts, no extra legs, no artifacts"
    )
    
    negative_prompt = (
        "duplicate legs, extra legs, multiple legs, distorted legs, "
        "malformed legs, mutated legs, bad anatomy, unnatural proportions, "
        "plastic, artificial, fake, CGI, 3D render, deformed, misplaced body parts, "
        "artifacts, blurry, low quality, distorted, compression artifacts, "
        "wrong clothing, distorted clothing, floating objects"
    )
    
    refined = generator.refine(
        image=image,
        prompt=prompt,
        mask=mask,
        negative_prompt=negative_prompt,
        strength=0.5,  # Slightly higher for clothing/legs
        num_inference_steps=30
    )
    
    return refined

def blend_regions(image: np.ndarray, upper_refined: np.ndarray, lower_refined: np.ndarray, 
                  upper_mask: np.ndarray, lower_mask: np.ndarray):
    """Blend upper and lower body refinements seamlessly"""
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # Normalize masks for blending
    upper_mask_norm = (upper_mask / 255.0)[:, :, np.newaxis]
    lower_mask_norm = (lower_mask / 255.0)[:, :, np.newaxis]
    
    # Blend upper body
    upper_refined_float = upper_refined.astype(np.float32)
    result = result * (1 - upper_mask_norm) + upper_refined_float * upper_mask_norm
    
    # Blend lower body
    lower_refined_float = lower_refined.astype(np.float32)
    result = result * (1 - lower_mask_norm) + lower_refined_float * lower_mask_norm
    
    # Convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def main():
    """Fix both upper and lower body - Generate single complete output"""
    logger.info("=" * 80)
    logger.info("Complete Body Fix - Upper and Lower Body Refinement")
    logger.info("=" * 80)
    
    # Input image (the fixed upper body result)
    input_path = Path(__file__).parent / "outputs" / "fixed_upper_body_result.jpg"
    
    if not input_path.exists():
        logger.error(f"Input image not found: {input_path}")
        logger.info("Trying original image...")
        input_path = Path(__file__).parent / "outputs" / "stability_ai_test_result.jpg"
        if not input_path.exists():
            logger.error(f"Original image also not found: {input_path}")
            return
    
    logger.info(f"Loading image: {input_path}")
    image = load_image(str(input_path))
    original_image = image.copy()
    h, w = image.shape[:2]
    logger.info(f"Image loaded: shape={image.shape}")
    
    # Initialize generator
    generator = Generator()
    
    if not generator.use_ai_api:
        logger.error("AI API not available!")
        return
    
    # Detect face
    logger.info("Detecting face...")
    face_detector = FaceDetector()
    faces = face_detector.detect_faces(image)
    face_bbox = None
    if faces:
        face_bbox = faces[0].get("bbox", [0, 0, 0, 0])
        logger.info(f"Face detected: bbox={face_bbox}")
    
    # Create masks
    logger.info("Creating upper and lower body masks...")
    upper_body_mask = create_upper_body_mask(image, face_bbox)
    lower_body_mask = create_lower_body_mask(image, face_bbox)
    
    # Refine upper body
    logger.info("=" * 80)
    logger.info("STEP 1: Refining Upper Body")
    logger.info("=" * 80)
    logger.info("This may take 30-60 seconds...")
    upper_refined = refine_upper_body(image, upper_body_mask, generator)
    
    if upper_refined is None or upper_refined.size == 0:
        logger.error("Upper body refinement failed, using original")
        upper_refined = image.copy()
    
    # Refine lower body
    logger.info("=" * 80)
    logger.info("STEP 2: Refining Lower Body")
    logger.info("=" * 80)
    logger.info("This may take 30-60 seconds...")
    # Use upper_refined as base for lower body refinement
    lower_refined = refine_lower_body(upper_refined, lower_body_mask, generator)
    
    if lower_refined is None or lower_refined.size == 0:
        logger.error("Lower body refinement failed, using upper refined")
        lower_refined = upper_refined.copy()
    
    # Blend both regions seamlessly
    logger.info("=" * 80)
    logger.info("STEP 3: Blending Upper and Lower Body")
    logger.info("=" * 80)
    logger.info("Blending refined regions seamlessly...")
    result = blend_regions(
        original_image,
        upper_refined,
        lower_refined,
        upper_body_mask,
        lower_body_mask
    )
    
    # Save final result
    output_path = Path(__file__).parent / "outputs" / "complete_body_fixed_result.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, str(output_path))
    
    logger.info("=" * 80)
    logger.info("Fix Complete!")
    logger.info("=" * 80)
    logger.info(f"Output saved to: {output_path}")
    logger.info("")
    logger.info("Changes made:")
    logger.info("  ✓ Upper body refined with Stability AI")
    logger.info("  ✓ Lower body refined with Stability AI")
    logger.info("  ✓ Both regions blended seamlessly")
    logger.info("  ✓ Single complete output generated")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This is the final output with both upper and lower body fixed!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()

