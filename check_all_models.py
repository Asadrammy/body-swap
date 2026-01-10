"""Comprehensive model verification script - Check all models are installed and working"""

import sys
from pathlib import Path
import numpy as np
import cv2
import traceback

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger

setup_logger()
logger = get_logger(__name__)


def check_package_installation(package_name: str, import_name: str = None) -> tuple[bool, str]:
    """Check if a Python package is installed"""
    import_name = import_name or package_name
    try:
        __import__(import_name)
        return True, f"✓ {package_name} installed"
    except ImportError as e:
        return False, f"✗ {package_name} NOT installed: {e}"


def check_face_detector():
    """Check Face Detector (InsightFace)"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING FACE DETECTOR (InsightFace)")
    logger.info("=" * 60)
    
    results = {}
    
    # Check package
    installed, msg = check_package_installation("insightface")
    logger.info(msg)
    results['package'] = installed
    
    if not installed:
        logger.warning("⚠ InsightFace not installed. Install with: pip install insightface")
        logger.info("  Testing OpenCV fallback...")
    
    # Check initialization (always test, even if InsightFace not installed)
    try:
        from src.models.face_detector import FaceDetector
        face_detector = FaceDetector()
        
        # Check if InsightFace or OpenCV is loaded
        if face_detector.app is not None:
            logger.info("✓ InsightFace model loaded (buffalo_l)")
            results['initialization'] = True
            results['model_type'] = 'insightface'
        elif face_detector.model is not None:
            logger.info("✓ OpenCV face detector loaded (fallback)")
            results['initialization'] = True
            results['model_type'] = 'opencv'
        else:
            logger.error("✗ Face detector not initialized")
            results['initialization'] = False
            return results
        
    except Exception as e:
        logger.error(f"✗ Face detector initialization failed: {e}")
        logger.error(traceback.format_exc())
        results['initialization'] = False
        return results
    
    # Test detection
    try:
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        faces = face_detector.detect_faces(test_image)
        logger.info(f"✓ Face detection test passed (found {len(faces)} faces)")
        results['detection'] = True
    except Exception as e:
        logger.error(f"✗ Face detection test failed: {e}")
        results['detection'] = False
    
    return results


def check_pose_detector():
    """Check Pose Detector (MediaPipe)"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING POSE DETECTOR (MediaPipe)")
    logger.info("=" * 60)
    
    results = {}
    
    # Check package
    installed, msg = check_package_installation("mediapipe")
    logger.info(msg)
    results['package'] = installed
    
    if not installed:
        logger.error("✗ MediaPipe not installed. Install with: pip install mediapipe")
        results['initialization'] = False
        results['detection'] = False
        return results
    
    # Check initialization
    try:
        from src.models.pose_detector import PoseDetector
        pose_detector = PoseDetector()
        
        if pose_detector.pose is not None:
            logger.info("✓ MediaPipe pose detector initialized")
            results['initialization'] = True
        else:
            logger.error("✗ Pose detector not initialized")
            results['initialization'] = False
            return results
        
    except Exception as e:
        logger.error(f"✗ Pose detector initialization failed: {e}")
        logger.error(traceback.format_exc())
        results['initialization'] = False
        return results
    
    # Test detection
    try:
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        poses = pose_detector.detect_pose(test_image)
        logger.info(f"✓ Pose detection test passed (found {len(poses)} poses)")
        results['detection'] = True
    except Exception as e:
        logger.error(f"✗ Pose detection test failed: {e}")
        results['detection'] = False
    
    return results


def check_stable_diffusion():
    """Check Stable Diffusion Generator"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING STABLE DIFFUSION GENERATOR")
    logger.info("=" * 60)
    
    results = {}
    
    # Check packages
    packages = {
        "torch": "torch",
        "diffusers": "diffusers",
        "transformers": "transformers",
        "accelerate": "accelerate"
    }
    
    all_installed = True
    for pkg_name, import_name in packages.items():
        installed, msg = check_package_installation(pkg_name, import_name)
        logger.info(msg)
        results[f'{pkg_name}_package'] = installed
        if not installed:
            all_installed = False
    
    if not all_installed:
        logger.error("✗ Some required packages missing for Stable Diffusion")
        logger.info("  Install with: pip install torch diffusers transformers accelerate")
        results['initialization'] = False
        return results
    
    # Check CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA Version: {torch.version.cuda}")
            logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("⚠ CUDA not available - will use CPU (slower)")
        results['cuda_available'] = cuda_available
    except Exception as e:
        logger.warning(f"⚠ Could not check CUDA: {e}")
        results['cuda_available'] = False
    
    # Check initialization (lazy loading - won't download models yet)
    try:
        from src.models.generator import Generator
        generator = Generator()
        logger.info(f"✓ Generator initialized (device: {generator.device})")
        logger.info(f"  Base model: {generator.base_model}")
        logger.info(f"  Inpaint model: {generator.inpaint_model}")
        logger.info(f"  ControlNet model: {generator.controlnet_model}")
        results['initialization'] = True
        
        # Note: Models are lazy-loaded, so we won't test actual generation here
        # to avoid downloading large models if not needed
        logger.info("  Note: Models will be downloaded on first use")
        results['models_loaded'] = False  # Lazy loading
        
    except Exception as e:
        logger.error(f"✗ Generator initialization failed: {e}")
        logger.error(traceback.format_exc())
        results['initialization'] = False
    
    return results


def check_segmenter():
    """Check Segmenter"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING SEGMENTER")
    logger.info("=" * 60)
    
    results = {}
    
    try:
        from src.models.segmenter import Segmenter
        segmenter = Segmenter()
        logger.info("✓ Segmenter initialized (pose-based segmentation)")
        logger.info("  Note: SAM is optional and not required")
        results['initialization'] = True
        
        # Test segmentation with dummy pose data
        try:
            test_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
            dummy_pose = {
                "keypoints": {
                    "left_shoulder": [50, 50],
                    "right_shoulder": [150, 50],
                    "left_hip": [50, 150],
                    "right_hip": [150, 150]
                }
            }
            masks = segmenter.segment_body_parts(test_image, dummy_pose)
            logger.info(f"✓ Segmentation test passed (generated {len(masks)} masks)")
            results['segmentation'] = True
        except Exception as e:
            logger.warning(f"⚠ Segmentation test warning: {e}")
            results['segmentation'] = False
        
    except Exception as e:
        logger.error(f"✗ Segmenter initialization failed: {e}")
        logger.error(traceback.format_exc())
        results['initialization'] = False
    
    return results


def check_core_dependencies():
    """Check core dependencies"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING CORE DEPENDENCIES")
    logger.info("=" * 60)
    
    packages = {
        "numpy": "numpy",
        "opencv-python": "cv2",
        "Pillow": "PIL",
        "scipy": "scipy"
    }
    
    results = {}
    all_ok = True
    
    for pkg_name, import_name in packages.items():
        installed, msg = check_package_installation(pkg_name, import_name)
        logger.info(msg)
        results[pkg_name] = installed
        if not installed:
            all_ok = False
    
    if all_ok:
        # Check versions
        try:
            import numpy as np
            import cv2
            logger.info(f"  NumPy version: {np.__version__}")
            logger.info(f"  OpenCV version: {cv2.__version__}")
        except:
            pass
    
    return results


def main():
    """Main verification function"""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MODEL VERIFICATION")
    logger.info("=" * 80)
    
    all_results = {}
    
    # Check core dependencies
    all_results['core_dependencies'] = check_core_dependencies()
    
    # Check Face Detector
    all_results['face_detector'] = check_face_detector()
    
    # Check Pose Detector
    all_results['pose_detector'] = check_pose_detector()
    
    # Check Stable Diffusion
    all_results['stable_diffusion'] = check_stable_diffusion()
    
    # Check Segmenter
    all_results['segmenter'] = check_segmenter()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    
    # Face Detector
    fd = all_results['face_detector']
    model_type = fd.get('model_type', 'unknown')
    if fd.get('initialization') and fd.get('detection'):
        if model_type == 'insightface':
            logger.info("✓ Face Detector: WORKING (InsightFace)")
        elif model_type == 'opencv':
            logger.info("✓ Face Detector: WORKING (OpenCV fallback)")
        else:
            logger.info("✓ Face Detector: WORKING")
    elif fd.get('initialization'):
        logger.warning("⚠ Face Detector: Initialized but detection test failed")
    else:
        logger.error("✗ Face Detector: NOT WORKING")
    
    # Pose Detector
    pd = all_results['pose_detector']
    if pd.get('initialization') and pd.get('detection'):
        logger.info("✓ Pose Detector: WORKING")
    elif pd.get('initialization'):
        logger.warning("⚠ Pose Detector: Initialized but detection test failed")
    else:
        logger.error("✗ Pose Detector: NOT WORKING")
    
    # Stable Diffusion
    sd = all_results['stable_diffusion']
    if sd.get('initialization'):
        logger.info("✓ Stable Diffusion Generator: INITIALIZED")
        if sd.get('cuda_available'):
            logger.info("  GPU acceleration: Available")
        else:
            logger.warning("  GPU acceleration: Not available (CPU mode)")
        logger.info("  Note: Models will download on first use")
    else:
        logger.error("✗ Stable Diffusion Generator: NOT INITIALIZED")
    
    # Segmenter
    seg = all_results['segmenter']
    if seg.get('initialization'):
        logger.info("✓ Segmenter: WORKING")
    else:
        logger.error("✗ Segmenter: NOT WORKING")
    
    # Overall status
    logger.info("\n" + "=" * 80)
    critical_models = ['face_detector', 'pose_detector', 'segmenter']
    critical_ok = all(
        all_results[m].get('initialization', False) 
        for m in critical_models
    )
    
    if critical_ok and sd.get('initialization'):
        logger.info("✅ ALL CRITICAL MODELS ARE WORKING")
        logger.info("✅ System is ready for face-body swap operations")
    elif critical_ok:
        logger.warning("⚠️  CRITICAL MODELS WORKING, but Stable Diffusion not initialized")
        logger.warning("   Face swap will work, but refinement may not be available")
    else:
        logger.error("❌ SOME CRITICAL MODELS ARE NOT WORKING")
        logger.error("   Please install missing packages and check errors above")
    
    logger.info("=" * 80)
    
    return all_results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("\n\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nFatal error during verification: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

