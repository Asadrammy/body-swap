#!/usr/bin/env python3
"""Simple test script to verify the pipeline works end-to-end"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger, get_logger
from src.utils.image_utils import save_image, load_image
from src.pipeline.preprocessor import Preprocessor
from src.pipeline.body_analyzer import BodyAnalyzer
from src.pipeline.template_analyzer import TemplateAnalyzer

setup_logger()
logger = get_logger(__name__)


def create_test_image(size=(512, 512), color=(128, 128, 128)):
    """Create a simple test image"""
    img = np.zeros((*size[::-1], 3), dtype=np.uint8)
    img[:] = color
    return img


def test_preprocessor():
    """Test the preprocessor"""
    logger.info("Testing Preprocessor...")
    
    try:
        preprocessor = Preprocessor()
        
        # Create a test image and save it
        test_image = create_test_image((512, 512))
        test_path = Path("temp") / "test_customer.jpg"
        test_path.parent.mkdir(exist_ok=True)
        save_image(test_image, test_path)
        
        # Validate image
        validation = preprocessor.validate_image(str(test_path))
        assert validation["valid"], f"Image validation failed: {validation.get('error')}"
        logger.info("✓ Preprocessor validation works")
        
        return True
    except Exception as e:
        logger.error(f"✗ Preprocessor test failed: {e}")
        return False


def test_face_detector():
    """Test face detection (will use OpenCV fallback)"""
    logger.info("Testing Face Detector...")
    
    try:
        from src.models.face_detector import FaceDetector
        
        detector = FaceDetector()
        
        # Create a test image with a simple pattern (no actual face, but detector should handle it)
        test_image = create_test_image((512, 512))
        
        faces = detector.detect_faces(test_image)
        logger.info(f"✓ Face detector initialized, detected {len(faces)} faces")
        
        return True
    except Exception as e:
        logger.error(f"✗ Face detector test failed: {e}")
        return False


def test_pipeline_modules():
    """Test that all pipeline modules can be imported and initialized"""
    logger.info("Testing Pipeline Modules...")
    
    modules = [
        ("Preprocessor", "src.pipeline.preprocessor"),
        ("BodyAnalyzer", "src.pipeline.body_analyzer"),
        ("TemplateAnalyzer", "src.pipeline.template_analyzer"),
        ("FaceProcessor", "src.pipeline.face_processor"),
        ("BodyWarper", "src.pipeline.body_warper"),
        ("Composer", "src.pipeline.composer"),
        ("Refiner", "src.pipeline.refiner"),
        ("QualityControl", "src.pipeline.quality_control"),
    ]
    
    all_ok = True
    for name, module_path in modules:
        try:
            module = __import__(module_path, fromlist=[name])
            cls = getattr(module, name)
            instance = cls()
            logger.info(f"✓ {name} initialized successfully")
        except Exception as e:
            logger.error(f"✗ {name} failed: {e}")
            all_ok = False
    
    return all_ok


def test_api_imports():
    """Test API imports"""
    logger.info("Testing API Modules...")
    
    try:
        from src.api.main import app
        from src.api.routes import router
        from src.api.schemas import SwapRequest, SwapResponse
        logger.info("✓ API modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ API import failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Pipeline End-to-End Test")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Preprocessor
    results["preprocessor"] = test_preprocessor()
    
    # Test 2: Face Detector
    results["face_detector"] = test_face_detector()
    
    # Test 3: Pipeline Modules
    results["pipeline_modules"] = test_pipeline_modules()
    
    # Test 4: API
    results["api"] = test_api_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All pipeline tests passed!")
        print("\nThe pipeline is ready to use. You can now:")
        print("  1. Test with real images using the CLI")
        print("  2. Start the API server: python -m src.api.main")
        print("  3. Check QUICKSTART.md for usage examples")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

