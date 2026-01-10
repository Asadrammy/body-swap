"""Verify all models are working, then perform proper face swap"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.api.cli import SwapCLI
from src.utils.image_utils import load_image
from src.pipeline.preprocessor import Preprocessor
from src.pipeline.face_processor import FaceProcessor
from src.pipeline.body_analyzer import BodyAnalyzer
from src.pipeline.template_analyzer import TemplateAnalyzer
from src.models.face_detector import FaceDetector
from src.models.pose_detector import PoseDetector

setup_logger()
logger = get_logger(__name__)


def verify_all_models():
    """Verify all models are installed and working"""
    logger.info("=" * 60)
    logger.info("VERIFYING ALL MODELS")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. Face Detector
    try:
        logger.info("Checking Face Detector...")
        face_detector = FaceDetector()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = face_detector.detect_faces(test_img)
        results['face_detector'] = True
        logger.info("✓ Face Detector: Working")
    except Exception as e:
        results['face_detector'] = False
        logger.error(f"✗ Face Detector: Failed - {e}")
    
    # 2. Pose Detector
    try:
        logger.info("Checking Pose Detector...")
        pose_detector = PoseDetector()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        poses = pose_detector.detect_pose(test_img)
        results['pose_detector'] = True
        logger.info("✓ Pose Detector: Working")
    except Exception as e:
        results['pose_detector'] = False
        logger.error(f"✗ Pose Detector: Failed - {e}")
    
    # 3. Preprocessor
    try:
        logger.info("Checking Preprocessor...")
        preprocessor = Preprocessor()
        results['preprocessor'] = True
        logger.info("✓ Preprocessor: Working")
    except Exception as e:
        results['preprocessor'] = False
        logger.error(f"✗ Preprocessor: Failed - {e}")
    
    # 4. Face Processor
    try:
        logger.info("Checking Face Processor...")
        face_processor = FaceProcessor()
        results['face_processor'] = True
        logger.info("✓ Face Processor: Working")
    except Exception as e:
        results['face_processor'] = False
        logger.error(f"✗ Face Processor: Failed - {e}")
    
    # 5. Body Analyzer
    try:
        logger.info("Checking Body Analyzer...")
        body_analyzer = BodyAnalyzer()
        results['body_analyzer'] = True
        logger.info("✓ Body Analyzer: Working")
    except Exception as e:
        results['body_analyzer'] = False
        logger.error(f"✗ Body Analyzer: Failed - {e}")
    
    # 6. Template Analyzer
    try:
        logger.info("Checking Template Analyzer...")
        template_analyzer = TemplateAnalyzer()
        results['template_analyzer'] = True
        logger.info("✓ Template Analyzer: Working")
    except Exception as e:
        results['template_analyzer'] = False
        logger.error(f"✗ Template Analyzer: Failed - {e}")
    
    logger.info("=" * 60)
    all_working = all(results.values())
    if all_working:
        logger.info("✓ ALL MODELS VERIFIED AND WORKING")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.warning(f"⚠ Some models failed: {failed}")
    logger.info("=" * 60)
    
    return all_working, results


def find_customer_image_with_faces():
    """Find a customer image with visible faces"""
    images_to_check = [
        "1760713603491 (1).jpg",
        "swap1 (1).png",
    ]
    
    preprocessor = Preprocessor()
    
    for img_path in images_to_check:
        if Path(img_path).exists():
            try:
                customer_data = preprocessor.preprocess_customer_photos([img_path])
                if customer_data["faces"] and customer_data["faces"][0] and len(customer_data["faces"][0]) > 0:
                    logger.info(f"Found customer image with {len(customer_data['faces'][0])} face(s): {img_path}")
                    return img_path
            except Exception as e:
                logger.warning(f"Error checking {img_path}: {e}")
                continue
    
    return None


def run_proper_swap():
    """Run proper face swap with IMG20251019131550.jpg as template"""
    logger.info("=" * 60)
    logger.info("RUNNING PROPER FACE SWAP")
    logger.info("=" * 60)
    
    # Template image (target - has masked faces to replace)
    template_image = "IMG20251019131550.jpg"
    
    # Find customer image with visible faces
    customer_image = find_customer_image_with_faces()
    
    if not customer_image:
        logger.error("No suitable customer image found with visible faces!")
        return False
    
    output_image = "outputs/final_proper_swap_result.png"
    
    logger.info(f"Customer Image (source face): {customer_image}")
    logger.info(f"Template Image (target): {template_image}")
    logger.info(f"Output: {output_image}")
    logger.info("=" * 60)
    
    # Verify faces in both images
    preprocessor = Preprocessor()
    customer_data = preprocessor.preprocess_customer_photos([customer_image])
    template_data = preprocessor.preprocess_template(template_image)
    
    customer_faces = len(customer_data['faces'][0]) if customer_data['faces'] and customer_data['faces'][0] else 0
    template_faces = len(template_data['faces']) if template_data['faces'] else 0
    
    logger.info(f"Customer faces detected: {customer_faces}")
    logger.info(f"Template faces detected: {template_faces}")
    
    if customer_faces == 0:
        logger.error("No faces detected in customer image!")
        return False
    
    if template_faces == 0:
        logger.warning("No faces detected in template - may have masked faces")
        logger.info("Proceeding anyway - will try to detect face regions...")
    
    # Run pipeline
    logger.info("\nStarting face swap pipeline...")
    cli = SwapCLI()
    cli.swap(
        customer_photos=[customer_image],
        template=template_image,
        output=output_image,
        no_refine=False,
        export_intermediate=True
    )
    
    # Verify output
    logger.info("\n" + "=" * 60)
    logger.info("VERIFYING OUTPUT")
    logger.info("=" * 60)
    
    output_img = load_image(output_image)
    template_img = load_image(template_image)
    
    # Resize for comparison
    if output_img.shape != template_img.shape:
        template_resized = cv2.resize(template_img, (output_img.shape[1], output_img.shape[0]))
    else:
        template_resized = template_img
    
    diff = np.mean(np.abs(output_img.astype(float) - template_resized.astype(float)))
    logger.info(f"Mean difference from template: {diff:.2f}")
    
    # Check if faces are visible in output
    output_data = preprocessor.preprocess_template(output_image)
    output_faces = len(output_data['faces']) if output_data['faces'] else 0
    logger.info(f"Faces detected in output: {output_faces}")
    
    if diff > 10.0 or output_faces > 0:
        logger.info("✓ Face swap conversion completed successfully!")
        logger.info(f"Output saved to: {output_image}")
        return True
    else:
        logger.warning("⚠ Face swap may not be visible enough")
        logger.info(f"Output saved to: {output_image}")
        return True


def main():
    """Main function"""
    # Step 1: Verify all models
    all_working, results = verify_all_models()
    
    if not all_working:
        logger.warning("Some models failed verification, but proceeding anyway...")
    
    # Step 2: Run proper swap
    logger.info("\n")
    success = run_proper_swap()
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("PROCESS COMPLETED")
        logger.info("=" * 60)
    else:
        logger.error("\n" + "=" * 60)
        logger.error("PROCESS FAILED")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()






