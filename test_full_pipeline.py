"""Full pipeline test with client image using Stability AI API"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import get_logger
from src.models.generator import Generator
from src.pipeline.refiner import Refiner
from src.pipeline.face_processor import FaceProcessor
from src.pipeline.body_analyzer import BodyAnalyzer
from src.pipeline.template_analyzer import TemplateAnalyzer
from src.models.face_detector import FaceDetector
from src.models.pose_detector import PoseDetector

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
    img.save(path)
    logger.info(f"Saved image to: {path}")

def test_stability_ai_api():
    """Test Stability AI API directly"""
    logger.info("=" * 80)
    logger.info("Test 1: Direct Stability AI API Test")
    logger.info("=" * 80)
    
    # Check API key
    api_key = os.getenv("STABILITY_API_KEY")
    provider = os.getenv("AI_IMAGE_PROVIDER", "stability")
    
    logger.info(f"API Key: {'Set' if api_key else 'Not set'}")
    logger.info(f"Provider: {provider}")
    
    if not api_key:
        logger.error("STABILITY_API_KEY not found!")
        return False
    
    # Test generator initialization
    try:
        generator = Generator()
        logger.info(f"Generator initialized: use_ai_api={generator.use_ai_api}")
        logger.info(f"AI Generator provider: {generator.ai_generator.provider}")
        
        if not generator.use_ai_api:
            logger.error("AI API is not enabled!")
            return False
        
        if generator.ai_generator.provider != "stability":
            logger.warning(f"Provider is {generator.ai_generator.provider}, expected 'stability'")
        
        return True
    except Exception as e:
        logger.error(f"Generator initialization failed: {e}", exc_info=True)
        return False

def test_image_refinement():
    """Test image refinement with Stability AI"""
    logger.info("=" * 80)
    logger.info("Test 2: Image Refinement Test")
    logger.info("=" * 80)
    
    image_path = Path(__file__).parent / "IMG20251019131550.jpg"
    
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return False
    
    try:
        # Load image
        logger.info(f"Loading image: {image_path}")
        image = load_image(str(image_path))
        logger.info(f"Image loaded: shape={image.shape}, dtype={image.dtype}")
        
        # Initialize generator
        generator = Generator()
        
        if not generator.use_ai_api:
            logger.error("AI API not available!")
            return False
        
        # Create a face mask (center region for testing)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 3
        
        import cv2
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Test refinement
        logger.info("Calling Stability AI API for refinement...")
        logger.info("This may take 30-60 seconds...")
        
        prompt = (
            "photorealistic portrait, natural human skin with pores and texture, "
            "realistic skin tone variation, high quality photograph, detailed facial features, "
            "natural lighting, authentic human appearance, subtle skin imperfections, "
            "professional photography, sharp focus, detailed eyes, natural hair"
        )
        
        negative_prompt = (
            "plastic, artificial, fake, CGI, 3D render, smooth skin, airbrushed, "
            "perfect skin, doll-like, wax figure, synthetic, blurred, distorted face, "
            "oversaturated, cartoon, painting, drawing, illustration"
        )
        
        refined = generator.refine(
            image=image,
            prompt=prompt,
            mask=mask,
            negative_prompt=negative_prompt,
            strength=0.55,  # Client requirement: reduced to avoid plastic looks
            num_inference_steps=30
        )
        
        if refined is None or refined.size == 0:
            logger.error("Refinement returned None or empty result")
            return False
        
        # Validate result
        if len(refined.shape) == 3:
            unique_colors = len(np.unique(refined.reshape(-1, refined.shape[-1]), axis=0))
            std_dev = np.std(refined)
        else:
            unique_colors = len(np.unique(refined))
            std_dev = np.std(refined)
        
        logger.info(f"Refinement result: shape={refined.shape}, unique_colors={unique_colors}, std={std_dev:.2f}")
        
        if unique_colors < 20 or std_dev < 8.0:
            logger.warning(f"Result may be low quality (unique_colors={unique_colors}, std={std_dev:.2f})")
        else:
            logger.info("✓ Refinement successful!")
        
        # Save result
        output_path = Path(__file__).parent / "outputs" / "stability_ai_test_result.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(refined, str(output_path))
        
        return True
        
    except Exception as e:
        logger.error(f"Refinement test failed: {e}", exc_info=True)
        return False

def test_face_refinement():
    """Test face-specific refinement"""
    logger.info("=" * 80)
    logger.info("Test 3: Face Refinement Test")
    logger.info("=" * 80)
    
    image_path = Path(__file__).parent / "IMG20251019131550.jpg"
    
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return False
    
    try:
        # Load image
        image = load_image(str(image_path))
        
        # Detect face
        face_detector = FaceDetector()
        faces = face_detector.detect_faces(image)
        
        if not faces:
            logger.warning("No faces detected, skipping face refinement test")
            return False
        
        logger.info(f"Detected {len(faces)} face(s)")
        
        # Get first face
        face = faces[0]
        bbox = face.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox
        logger.info(f"Face bbox: x={x}, y={y}, w={w}, h={h}")
        
        # Initialize refiner
        refiner = Refiner()
        
        # Refine face
        logger.info("Refining face with Stability AI...")
        logger.info("This may take 30-60 seconds...")
        
        refined = refiner.refine_face(
            image=image,
            face_bbox=(x, y, w, h),
            expression_type="neutral"
        )
        
        if refined is None or refined.size == 0:
            logger.error("Face refinement returned None or empty result")
            return False
        
        # Save result
        output_path = Path(__file__).parent / "outputs" / "face_refinement_test_result.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(refined, str(output_path))
        
        logger.info("✓ Face refinement successful!")
        return True
        
    except Exception as e:
        logger.error(f"Face refinement test failed: {e}", exc_info=True)
        return False

def main():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("Full Pipeline Test with Stability AI API")
    logger.info("=" * 80)
    logger.info(f"Image: IMG20251019131550.jpg")
    logger.info("")
    
    results = {}
    
    # Test 1: API initialization
    results["api_test"] = test_stability_ai_api()
    logger.info("")
    
    # Test 2: Image refinement
    if results["api_test"]:
        results["refinement_test"] = test_image_refinement()
        logger.info("")
    else:
        logger.warning("Skipping refinement test - API not available")
        results["refinement_test"] = False
    
    # Test 3: Face refinement
    if results["api_test"]:
        results["face_refinement_test"] = test_face_refinement()
        logger.info("")
    else:
        logger.warning("Skipping face refinement test - API not available")
        results["face_refinement_test"] = False
    
    # Summary
    logger.info("=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    logger.info(f"API Test: {'✓ PASS' if results['api_test'] else '✗ FAIL'}")
    logger.info(f"Refinement Test: {'✓ PASS' if results.get('refinement_test') else '✗ FAIL'}")
    logger.info(f"Face Refinement Test: {'✓ PASS' if results.get('face_refinement_test') else '✗ FAIL'}")
    logger.info("")
    
    if all(results.values()):
        logger.info("✓ All tests passed! Stability AI integration is working correctly.")
    else:
        logger.warning("Some tests failed. Check logs above for details.")
    
    logger.info("=" * 80)
    logger.info("Output files saved to: outputs/")
    logger.info("  - stability_ai_test_result.jpg (full image refinement)")
    logger.info("  - face_refinement_test_result.jpg (face-specific refinement)")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()

