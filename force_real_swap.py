"""Force real face swap - ensure customer face is actually swapped"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.api.cli import SwapCLI
from src.utils.image_utils import load_image, save_image
from src.pipeline.preprocessor import Preprocessor
from src.pipeline.face_processor import FaceProcessor

setup_logger()
logger = get_logger(__name__)


def verify_customer_face_extracted(customer_image_path):
    """Verify customer face is extracted correctly"""
    logger.info("=" * 60)
    logger.info("VERIFYING CUSTOMER FACE EXTRACTION")
    logger.info("=" * 60)
    
    preprocessor = Preprocessor()
    face_processor = FaceProcessor()
    
    customer_data = preprocessor.preprocess_customer_photos([customer_image_path])
    
    if not customer_data['faces'] or not customer_data['faces'][0]:
        logger.error("No faces detected in customer image!")
        return None
    
    customer_image = customer_data["images"][0]
    customer_faces = customer_data["faces"][0]
    
    if not customer_faces:
        logger.error("No faces in customer_faces list!")
        return None
    
    customer_face = customer_faces[0]  # Get first face
    logger.info(f"Customer face detected: bbox={customer_face.get('bbox')}")
    
    # Extract face identity
    face_identity = face_processor.extract_face_identity(customer_image, customer_face)
    
    aligned_face = face_identity.get("aligned_face")
    
    if aligned_face is None:
        logger.error("Failed to extract aligned face!")
        return None
    
    logger.info(f"Aligned face shape: {aligned_face.shape}")
    logger.info(f"Aligned face dtype: {aligned_face.dtype}")
    logger.info(f"Aligned face value range: [{aligned_face.min()}, {aligned_face.max()}]")
    
    # Save aligned face for inspection
    save_image(aligned_face, "outputs/customer_aligned_face_debug.png")
    logger.info("Saved customer aligned face to outputs/customer_aligned_face_debug.png")
    
    return aligned_face


def run_force_swap():
    """Run face swap with forced customer face"""
    logger.info("=" * 60)
    logger.info("FORCING REAL FACE SWAP")
    logger.info("=" * 60)
    
    customer_image = "IMG20251019131550.jpg"
    template_image = "swap1 (1).png"
    output_image = "outputs/forced_swap_result.png"
    
    # Verify customer face extraction
    customer_aligned = verify_customer_face_extracted(customer_image)
    if customer_aligned is None:
        logger.error("Failed to extract customer face!")
        return False
    
    # Run pipeline
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING PIPELINE WITH FORCED SWAP")
    logger.info("=" * 60)
    
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
    customer_img = load_image(customer_image)
    
    # Resize for comparison
    if output_img.shape != template_img.shape:
        template_resized = cv2.resize(template_img, (output_img.shape[1], output_img.shape[0]))
    else:
        template_resized = template_img
    
    diff = np.mean(np.abs(output_img.astype(float) - template_resized.astype(float)))
    logger.info(f"Mean difference from template: {diff:.2f}")
    
    if diff < 10.0:
        logger.error("✗ Face swap still not working - difference too small!")
        logger.error("The customer face is not being properly swapped onto the template")
        return False
    else:
        logger.info("✓ Face swap appears to have worked!")
        return True


if __name__ == "__main__":
    success = run_force_swap()
    if not success:
        sys.exit(1)

