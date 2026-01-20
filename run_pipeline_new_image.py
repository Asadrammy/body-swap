"""Run pipeline on new image using sample output as template reference"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.api.cli import SwapCLI
from src.utils.image_utils import load_image, save_image

setup_logger()
logger = get_logger(__name__)


def verify_output_different(output_path, customer_path, template_path):
    """Verify that output is different from both inputs"""
    try:
        output_img = load_image(output_path)
        customer_img = load_image(customer_path)
        template_img = load_image(template_path)
        
        # Resize for comparison
        if output_img.shape != template_img.shape:
            output_resized = cv2.resize(output_img, (template_img.shape[1], template_img.shape[0]))
        else:
            output_resized = output_img
        
        if output_img.shape != customer_img.shape:
            customer_resized = cv2.resize(customer_img, (output_img.shape[1], output_img.shape[0]))
        else:
            customer_resized = customer_img
        
        # Check if identical to template
        if np.array_equal(output_resized, template_img):
            logger.error("✗ Output is IDENTICAL to template - face swap did NOT happen!")
            return False
        
        # Check if identical to customer
        if np.array_equal(output_img, customer_resized):
            logger.error("✗ Output is IDENTICAL to customer image - pipeline just copied input!")
            return False
        
        # Calculate differences
        diff_template = np.mean(np.abs(output_resized.astype(float) - template_img.astype(float)))
        diff_customer = np.mean(np.abs(output_img.astype(float) - customer_resized.astype(float)))
        
        logger.info(f"Mean difference from template: {diff_template:.2f}")
        logger.info(f"Mean difference from customer: {diff_customer:.2f}")
        
        if diff_template < 5.0:
            logger.warning("⚠ Warning: Output is very similar to template - face swap may not have worked")
            return False
        
        logger.info("✓ Output is different from both inputs - transformation occurred")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    """Run the pipeline on new image"""
    logger.info("=" * 60)
    logger.info("RUNNING FACE-BODY SWAP ON NEW IMAGE")
    logger.info("=" * 60)
    
    # File paths
    customer_image = "IMG20251019131550.jpg"
    template_image = "swap1 (1).png"  # Sample output as template
    output_image = "outputs/new_image_conversion_result.png"
    
    # Verify files exist
    if not Path(customer_image).exists():
        logger.error(f"Customer image not found: {customer_image}")
        return
    
    if not Path(template_image).exists():
        logger.error(f"Template image not found: {template_image}")
        return
    
    # Ensure output directory exists
    Path(output_image).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Customer Image: {customer_image}")
    logger.info(f"Template Image: {template_image}")
    logger.info(f"Output Image: {output_image}")
    logger.info("=" * 60)
    
    # Run the pipeline
    logger.info("\nStarting pipeline...")
    cli = SwapCLI()
    
    try:
        cli.swap(
            customer_photos=[customer_image],
            template=template_image,
            output=output_image,
            no_refine=False,
            export_intermediate=True
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE EXECUTION COMPLETED")
        logger.info("=" * 60)
        
        # Verify output
        logger.info("\nVerifying output...")
        if verify_output_different(output_image, customer_image, template_image):
            logger.info("\n✓ SUCCESS: Face swap verification passed!")
            logger.info(f"Output saved to: {output_image}")
            logger.info("\nPlease check the output image to confirm the conversion is correct.")
        else:
            logger.error("\n✗ FAILED: Face swap verification failed!")
            logger.error("The output may be identical to the input. Please check the pipeline.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()













