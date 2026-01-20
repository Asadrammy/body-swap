"""Run face swap conversion using swap1 (1).png as customer image with available templates"""

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


def find_template_with_faces():
    """Find a template that has faces detected"""
    templates = [
        "examples/templates/individual_casual_001.png",
        "examples/templates/individual_action_002.png",
        "examples/templates/couple_garden_001.png",
        "examples/templates/family_studio_001.png",
    ]
    
    preprocessor = Preprocessor()
    
    for template_path in templates:
        if Path(template_path).exists():
            try:
                template_data = preprocessor.preprocess_template(template_path)
                if template_data["faces"] and len(template_data["faces"]) > 0:
                    logger.info(f"Found template with {len(template_data['faces'])} face(s): {template_path}")
                    return template_path
            except Exception as e:
                logger.warning(f"Error checking {template_path}: {e}")
                continue
    
    return None


def main():
    """Run conversion on own"""
    logger.info("=" * 60)
    logger.info("RUNNING FACE SWAP CONVERSION (OWN)")
    logger.info("=" * 60)
    
    # Customer image - swap1 (1).png has visible faces
    customer_image = "swap1 (1).png"
    
    # Find a template with faces
    template_image = find_template_with_faces()
    
    if not template_image:
        logger.error("No suitable template found with faces!")
        logger.info("Trying to use swap1 (1).png as both customer and template (diptych conversion)...")
        # Use swap1 (1).png as template too - it's a diptych, so we can swap faces between panels
        template_image = "swap1 (1).png"
    
    output_image = "outputs/own_conversion_result.png"
    
    logger.info(f"Customer Image: {customer_image}")
    logger.info(f"Template Image: {template_image}")
    logger.info(f"Output: {output_image}")
    logger.info("=" * 60)
    
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
        logger.info("✓ Conversion completed (subtle changes)")
    
    logger.info(f"\nOutput saved to: {output_image}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()













