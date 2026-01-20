"""
Test script for client image according to CLIENT_REQUIREMENTS_IMPLEMENTATION.md

This script tests the face-body swap pipeline with the provided client image,
ensuring all client requirements are met:
1. Body conditioning for open chest shirts
2. No plastic-looking faces
3. Action photos support
4. Manual touch-ups capability
5. Multiple subjects support
6. Quality assurance
"""

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


def verify_image_exists(image_path: str) -> bool:
    """Verify that the image file exists"""
    path = Path(image_path)
    if not path.exists():
        logger.error(f"Image not found: {image_path}")
        return False
    logger.info(f"✓ Found image: {image_path}")
    return True


def select_template_for_testing():
    """
    Select an appropriate template for testing client requirements.
    Prefer templates with open chest to test body conditioning.
    """
    templates = [
        {
            "path": "examples/templates/individual_action_002.png",
            "name": "Action Shot (Open Chest)",
            "features": ["action", "open_chest", "torso_visible"]
        },
        {
            "path": "examples/templates/individual_casual_001.png",
            "name": "Casual Portrait",
            "features": ["casual", "full_body"]
        },
        {
            "path": "examples/templates/couple_action_002.png",
            "name": "Couple Action (Open Chest)",
            "features": ["couple", "action", "open_chest"]
        }
    ]
    
    # Try to find a template that exists
    for template in templates:
        if Path(template["path"]).exists():
            logger.info(f"Selected template: {template['name']} ({template['path']})")
            logger.info(f"  Features: {', '.join(template['features'])}")
            return template["path"]
    
    # If no template found, check if IMG20251019131550.jpg exists (might be a template)
    fallback = "IMG20251019131550.jpg"
    if Path(fallback).exists():
        logger.info(f"Using fallback template: {fallback}")
        return fallback
    
    logger.error("No suitable template found!")
    return None


def main():
    """Run the test with client image according to requirements"""
    logger.info("=" * 80)
    logger.info("CLIENT REQUIREMENTS TEST - FACE-BODY SWAP PIPELINE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Testing according to CLIENT_REQUIREMENTS_IMPLEMENTATION.md:")
    logger.info("  1. Body conditioning for open chest shirts")
    logger.info("  2. No plastic-looking faces")
    logger.info("  3. Action photos support")
    logger.info("  4. Manual touch-ups capability")
    logger.info("  5. Multiple subjects support")
    logger.info("  6. Quality assurance")
    logger.info("")
    logger.info("=" * 80)
    
    # Customer image provided by client
    customer_image = "1760713603491 (1).jpg"
    
    # Verify customer image exists
    if not verify_image_exists(customer_image):
        logger.error("Customer image not found! Please check the path.")
        return False
    
    # Select template
    template_image = select_template_for_testing()
    if not template_image:
        logger.error("No template found! Cannot proceed.")
        return False
    
    if not verify_image_exists(template_image):
        logger.error("Template image not found!")
        return False
    
    # Output path
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image = str(output_dir / "client_test_result.png")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Customer Image: {customer_image}")
    logger.info(f"Template Image: {template_image}")
    logger.info(f"Output Image: {output_image}")
    logger.info("")
    logger.info("Configuration:")
    logger.info("  - Export intermediate results: YES (for manual touch-ups)")
    logger.info("  - Refinement: ENABLED (with AI image generation)")
    logger.info("  - Quality assessment: ENABLED (with Google AI)")
    logger.info("  - Body conditioning: AUTO (if open chest detected)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    
    # Run pipeline
    logger.info("Starting face-body swap pipeline...")
    logger.info("")
    
    try:
        cli = SwapCLI()
        cli.swap(
            customer_photos=[customer_image],
            template=template_image,
            output=output_image,
            no_refine=False,  # Enable AI refinement
            export_intermediate=True  # Export for manual touch-ups
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        # Verify output
        if Path(output_image).exists():
            logger.info(f"✓ Output saved: {output_image}")
            
            # Load and check output
            output_img = load_image(output_image)
            if output_img is not None:
                logger.info(f"✓ Output image valid: shape={output_img.shape}, dtype={output_img.dtype}")
                
                # Check for intermediate results
                intermediate_dir = output_dir / "client_test_result_intermediate"
                if intermediate_dir.exists():
                    logger.info(f"✓ Intermediate results exported: {intermediate_dir}")
                    logger.info("  These can be used for manual touch-ups if needed")
                
                logger.info("")
                logger.info("=" * 80)
                logger.info("QUALITY CHECK")
                logger.info("=" * 80)
                logger.info("Please review the output image to verify:")
                logger.info("  1. Face looks natural (not plastic)")
                logger.info("  2. Body proportions match customer")
                logger.info("  3. Clothing fits properly")
                logger.info("  4. Skin tone matches (if open chest)")
                logger.info("  5. Overall quality meets standards")
                logger.info("")
                logger.info(f"Output: {output_image}")
                logger.info("=" * 80)
                
                return True
            else:
                logger.error("Output image is invalid!")
                return False
        else:
            logger.error("Output file was not created!")
            return False
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error("  1. Check that all models are downloaded")
        logger.error("  2. Verify GPU/CUDA is available (or set DEVICE=cpu in .env)")
        logger.error("  3. Check that Google AI API key is set in .env")
        logger.error("  4. Review logs for specific errors")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

