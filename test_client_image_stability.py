"""Test the face-body swap pipeline with client image using Stability AI API"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api.cli import SwapCLI
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Test pipeline with client image"""
    
    # Client image path - Use relative path (portable across platforms)
    project_root = Path(__file__).parent
    customer_image = project_root / "IMG20251019131550.jpg"
    customer_image = os.getenv("TEST_CUSTOMER_IMAGE", str(customer_image))
    
    # Check if image exists
    if not os.path.exists(customer_image):
        logger.error(f"Customer image not found: {customer_image}")
        return
    
    logger.info("=" * 80)
    logger.info("Testing Face-Body Swap Pipeline with Stability AI API")
    logger.info("=" * 80)
    logger.info(f"Customer Image: {customer_image}")
    
    # Check API key
    stability_key = os.getenv("STABILITY_API_KEY")
    if not stability_key:
        logger.error("STABILITY_API_KEY not found in environment")
        logger.info("Please set STABILITY_API_KEY in .env file")
        return
    
    logger.info(f"✓ Stability AI API Key: {'Set' if stability_key else 'Not set'}")
    logger.info(f"✓ AI Provider: {os.getenv('AI_IMAGE_PROVIDER', 'stability')}")
    logger.info(f"✓ Use AI API: {os.getenv('USE_AI_API', 'true')}")
    
    # Initialize CLI
    cli = SwapCLI()
    
    # For testing, we need a template
    # Let's check if there are any templates available
    template_dir = Path(__file__).parent / "examples" / "templates"
    templates = list(template_dir.glob("*.jpg")) + list(template_dir.glob("*.png"))
    
    if not templates:
        logger.warning("No templates found. Creating a simple test...")
        logger.info("You can test the pipeline by providing a template image")
        logger.info("Usage: python test_client_image_stability.py <template_image>")
        
        # Try to use the customer image as both customer and template for testing
        logger.info("Testing with customer image as template (for API verification)...")
        output_path = Path(__file__).parent / "outputs" / "test_stability_result.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Test API connection by trying to refine the image
            from src.models.generator import Generator
            import numpy as np
            from PIL import Image
            
            logger.info("Testing Stability AI API connection...")
            generator = Generator()
            
            if not generator.use_ai_api:
                logger.error("AI API is not enabled or not available")
                return
            
            # Load and test refine
            img = Image.open(customer_image)
            img_array = np.array(img)
            
            logger.info("Testing image refinement with Stability AI...")
            logger.info("This will verify the API key works correctly")
            
            # Create a simple mask (center region)
            h, w = img_array.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center_x, center_y = w // 2, h // 2
            radius = min(w, h) // 4
            import cv2
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            # Test refine
            prompt = "photorealistic portrait, natural human skin with pores and texture, realistic skin tone variation, high quality photograph, detailed facial features, natural lighting, authentic human appearance"
            negative_prompt = "plastic, artificial, fake, CGI, 3D render, smooth skin, airbrushed, perfect skin, doll-like"
            
            logger.info("Calling Stability AI API for refinement...")
            refined = generator.refine(
                image=img_array,
                prompt=prompt,
                mask=mask,
                negative_prompt=negative_prompt,
                strength=0.55,
                num_inference_steps=30
            )
            
            if refined is not None and refined.size > 0:
                result_img = Image.fromarray(refined)
                result_img.save(output_path)
                logger.info(f"✓ Success! Result saved to: {output_path}")
                logger.info("Stability AI API is working correctly!")
            else:
                logger.error("Refinement returned invalid result")
                
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            return
    else:
        # Use first template found
        template = str(templates[0])
        logger.info(f"Using template: {template}")
        
        output_path = Path(__file__).parent / "outputs" / "test_client_result.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Running face-body swap pipeline...")
            cli.swap(
                customer_photos=[customer_image],
                template=template,
                output=str(output_path),
                export_intermediate=True
            )
            logger.info(f"✓ Success! Result saved to: {output_path}")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return
    
    logger.info("=" * 80)
    logger.info("Test Complete")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()

