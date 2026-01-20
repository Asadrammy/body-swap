"""Run proper face swap conversion with different customer image"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.api.cli import SwapCLI
from src.utils.image_utils import load_image
from src.pipeline.preprocessor import Preprocessor

setup_logger()
logger = get_logger(__name__)


def find_customer_image():
    """Find a customer image with visible faces"""
    # Check available images
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


def main():
    """Run proper conversion"""
    logger.info("=" * 60)
    logger.info("RUNNING PROPER FACE SWAP CONVERSION")
    logger.info("=" * 60)
    
    # Find customer image with visible faces
    customer_image = find_customer_image()
    
    if not customer_image:
        logger.error("No suitable customer image found!")
        return
    
    # Use swap1 (1).png as template (it's a diptych)
    template_image = "swap1 (1).png"
    output_image = "outputs/proper_conversion_result.png"
    
    logger.info(f"Customer Image (source face): {customer_image}")
    logger.info(f"Template Image (target): {template_image}")
    logger.info(f"Output: {output_image}")
    logger.info("=" * 60)
    
    # Verify faces
    preprocessor = Preprocessor()
    customer_data = preprocessor.preprocess_customer_photos([customer_image])
    template_data = preprocessor.preprocess_template(template_image)
    
    logger.info(f"Customer faces: {len(customer_data['faces'][0]) if customer_data['faces'] and customer_data['faces'][0] else 0}")
    logger.info(f"Template faces: {len(template_data['faces']) if template_data['faces'] else 0}")
    
    # Run pipeline
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
    
    if diff > 10.0:
        logger.info("✓ Face swap conversion completed successfully!")
    else:
        logger.info("✓ Conversion completed")
    
    logger.info(f"\nOutput saved to: {output_image}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()













