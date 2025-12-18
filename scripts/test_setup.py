#!/usr/bin/env python3
"""Test script to verify installation and setup"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    errors = []
    
    # Core Python packages
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        errors.append(f"✗ numpy: {e}")
        print(f"✗ numpy: {e}")
    
    try:
        import cv2
        print("✓ opencv-python")
    except ImportError as e:
        errors.append(f"✗ opencv-python: {e}")
        print(f"✗ opencv-python: {e}")
    
    try:
        from PIL import Image
        print("✓ Pillow")
    except ImportError as e:
        errors.append(f"✗ Pillow: {e}")
        print(f"✗ Pillow: {e}")
    
    # Deep learning
    try:
        import torch
        print(f"✓ torch (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available (device: {torch.cuda.get_device_name(0)})")
        else:
            print("  ⚠ CUDA not available (CPU mode)")
    except ImportError as e:
        errors.append(f"✗ torch: {e}")
        print(f"✗ torch: {e}")
    
    try:
        import diffusers
        print("✓ diffusers")
    except ImportError as e:
        errors.append(f"✗ diffusers: {e}")
        print(f"✗ diffusers: {e}")
    
    # Face detection
    try:
        import insightface
        print("✓ insightface")
    except ImportError as e:
        print(f"⚠ insightface: {e} (optional, will use fallback)")
    
    # Pose detection
    try:
        import mediapipe
        print("✓ mediapipe")
    except ImportError as e:
        errors.append(f"✗ mediapipe: {e}")
        print(f"✗ mediapipe: {e}")
    
    # API
    try:
        import fastapi
        print("✓ fastapi")
    except ImportError as e:
        errors.append(f"✗ fastapi: {e}")
        print(f"✗ fastapi: {e}")
    
    # Project modules
    try:
        from src.utils import load_image, get_config, setup_logger
        print("✓ src.utils")
    except ImportError as e:
        errors.append(f"✗ src.utils: {e}")
        print(f"✗ src.utils: {e}")
    
    try:
        from src.models import FaceDetector, PoseDetector
        print("✓ src.models")
    except ImportError as e:
        errors.append(f"✗ src.models: {e}")
        print(f"✗ src.models: {e}")
    
    try:
        from src.pipeline import Preprocessor
        print("✓ src.pipeline")
    except ImportError as e:
        errors.append(f"✗ src.pipeline: {e}")
        print(f"✗ src.pipeline: {e}")
    
    try:
        from src.api import app
        print("✓ src.api")
    except ImportError as e:
        errors.append(f"✗ src.api: {e}")
        print(f"✗ src.api: {e}")
    
    return len(errors) == 0, errors


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config import get_config, load_config
        config = load_config()
        print("✓ Configuration loaded successfully")
        print(f"  Image size: {config.get('processing', {}).get('image_size', 'N/A')}")
        print(f"  Device: {config.get('models', {}).get('generator', {}).get('device', 'N/A')}")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_directory_structure():
    """Test directory structure"""
    print("\nTesting directory structure...")
    
    base_path = Path(__file__).parent.parent
    required_dirs = [
        "src",
        "src/pipeline",
        "src/models",
        "src/utils",
        "src/api",
        "configs",
        "examples",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests"""
    print("=" * 60)
    print("Face and Body Swap Pipeline - Setup Verification")
    print("=" * 60)
    
    # Test directory structure
    dirs_ok = test_directory_structure()
    
    # Test imports
    imports_ok, import_errors = test_imports()
    
    # Test configuration
    config_ok = test_config()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Directory structure: {'✓ PASS' if dirs_ok else '✗ FAIL'}")
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Configuration: {'✓ PASS' if config_ok else '✗ FAIL'}")
    
    if imports_ok and config_ok and dirs_ok:
        print("\n✓ All tests passed! Setup is complete.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        if import_errors:
            print("\nMissing dependencies. Install with:")
            print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

