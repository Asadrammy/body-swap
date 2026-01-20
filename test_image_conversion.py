#!/usr/bin/env python3
"""
Test script to load all models and process the user's image
"""
import sys
import os
from pathlib import Path
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.api.cli import SwapCLI

# Setup logger
setup_logger()
logger = get_logger(__name__)

def main():
    """Test image conversion with model loading"""
    logger.info("=" * 80)
    logger.info("🚀 STARTING IMAGE CONVERSION TEST")
    logger.info("=" * 80)
    
    # Paths - Use relative paths (portable across platforms)
    project_root = Path(__file__).parent
    customer_photo = project_root / "1760713603491 (1).jpg"
    template = project_root / "examples" / "templates" / "individual_casual_001.png"
    output = project_root / "outputs" / "test_result.png"
    
    # Allow override via environment variables
    customer_photo = os.getenv("TEST_CUSTOMER_IMAGE", str(customer_photo))
    template = os.getenv("TEST_TEMPLATE_IMAGE", str(template))
    output = os.getenv("TEST_OUTPUT_IMAGE", str(output))
    
    # Ensure output directory exists
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Customer photo: {customer_photo}")
    logger.info(f"Template: {template}")
    logger.info(f"Output: {output}")
    
    # Verify files exist
    if not Path(customer_photo).exists():
        logger.error(f"❌ Customer photo not found: {customer_photo}")
        return False
    
    if not Path(template).exists():
        logger.error(f"❌ Template not found: {template}")
        return False
    
    logger.info("✅ Input files verified")
    
    try:
        # Initialize CLI (this will load models)
        logger.info("\n" + "=" * 80)
        logger.info("📦 INITIALIZING PIPELINE (LOADING MODELS)")
        logger.info("=" * 80)
        
        cli = SwapCLI()
        
        logger.info("\n" + "=" * 80)
        logger.info("🔄 STARTING IMAGE CONVERSION")
        logger.info("=" * 80)
        
        # Run swap without refinement to see base composition
        cli.swap(
            customer_photos=[customer_photo],
            template=template,
            output=output,
            export_intermediate=False,  # Disable to avoid export bug
            no_refine=True  # Skip refinement to see base composition
        )
        
        # Verify output
        if Path(output).exists():
            logger.info("\n" + "=" * 80)
            logger.info("✅ CONVERSION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Output saved to: {output}")
            
            # Check file size
            file_size = Path(output).stat().st_size
            logger.info(f"Output file size: {file_size / 1024:.2f} KB")
            
            if file_size > 0:
                logger.info("✅ Output file is valid (non-empty)")
                return True
            else:
                logger.error("❌ Output file is empty!")
                return False
        else:
            logger.error("❌ Output file was not created!")
            return False
            
    except Exception as e:
        logger.error(f"\n❌ CONVERSION FAILED: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

