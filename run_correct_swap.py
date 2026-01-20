"""Run face swap with correct image roles"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.api.cli import SwapCLI
from src.utils.image_utils import load_image
import numpy as np
import cv2

setup_logger()
logger = get_logger(__name__)


def main():
    """Run face swap with correct roles"""
    logger.info("=" * 60)
    logger.info("RUNNING FACE SWAP WITH CORRECT IMAGE ROLES")
    logger.info("=" * 60)
    
    # CORRECT ROLES:
    # Customer image: swap1 (1).png - has visible faces to extract
    # Template image: IMG20251019131550.jpg - has masked faces to replace
    customer_image = "swap1 (1).png"  # Source of face (has visible faces)
    template_image = "IMG20251019131550.jpg"  # Target (has masked faces)
    output_image = "outputs/correct_swap_result.png"
    
    logger.info(f"Customer Image (source face): {customer_image}")
    logger.info(f"Template Image (target): {template_image}")
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
    customer_img = load_image(customer_image)
    
    # Resize for comparison
    if output_img.shape != template_img.shape:
        template_resized = cv2.resize(template_img, (output_img.shape[1], output_img.shape[0]))
    else:
        template_resized = template_img
    
    diff = np.mean(np.abs(output_img.astype(float) - template_resized.astype(float)))
    logger.info(f"Mean difference from template: {diff:.2f}")
    
    if diff > 10.0:
        logger.info("✓ Face swap appears to have worked!")
    else:
        logger.warning("⚠ Face swap may not be visible enough")
    
    logger.info(f"\nOutput saved to: {output_image}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()













