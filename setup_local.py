#!/usr/bin/env python3
"""
Local Setup Script for Face-Body Swap
Creates directories, verifies dependencies, and validates configuration
"""

import sys
import os
from pathlib import Path
import shutil

# Colors for terminal output (Windows compatible)
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg):
    print(f"{Colors.GREEN}[OK]{Colors.END} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARN]{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.END} {msg}")

def print_info(msg):
    print(f"{Colors.BLUE}[INFO]{Colors.END} {msg}")

def print_header(msg):
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}\n")


def check_python_version():
    """Check Python version >= 3.8"""
    print_header("Checking Python Version")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} (requires >= 3.8)")
        print_info("Please upgrade Python to version 3.8 or higher")
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    print_header("Checking Dependencies")
    
    required_packages = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "cv2": "OpenCV",
        "PIL": "Pillow",
        "diffusers": "Diffusers",
        "transformers": "Transformers",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "mediapipe": "MediaPipe",
        "yaml": "PyYAML",
        "loguru": "Loguru",
    }
    
    optional_packages = {
        "insightface": "InsightFace (optional, has fallback)",
        "xformers": "XFormers (optional, for memory efficiency)",
    }
    
    all_ok = True
    missing = []
    
    for module, name in required_packages.items():
        try:
            __import__(module)
            print_success(f"{name}")
        except ImportError:
            print_error(f"{name} - not installed")
            missing.append(name)
            all_ok = False
    
    for module, name in optional_packages.items():
        try:
            __import__(module)
            print_success(f"{name} (optional)")
        except ImportError:
            print_warning(f"{name} - not installed (optional)")
    
    if missing:
        print_info("\nInstall missing dependencies with:")
        print("  pip install -r requirements.txt")
    
    return all_ok, missing


def create_directories():
    """Create required directories"""
    print_header("Creating Required Directories")
    
    project_root = Path(__file__).parent
    required_dirs = [
        "models",
        "outputs",
        "temp",
        "logs",
    ]
    
    all_created = True
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print_success(f"Created/verified: {dir_name}/")
        except Exception as e:
            print_error(f"Failed to create {dir_name}/: {e}")
            all_created = False
    
    return all_created


def check_directory_structure():
    """Check if project structure is correct"""
    print_header("Checking Project Structure")
    
    project_root = Path(__file__).parent
    required_paths = [
        "src",
        "src/pipeline",
        "src/models",
        "src/utils",
        "src/api",
        "configs",
        "requirements.txt",
    ]
    
    all_exist = True
    for path_str in required_paths:
        path = project_root / path_str
        if path.exists():
            print_success(f"{path_str}/")
        else:
            print_error(f"{path_str}/ - missing")
            all_exist = False
    
    return all_exist


def test_imports():
    """Test if project modules can be imported"""
    print_header("Testing Project Imports")
    
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    test_modules = [
        ("src.utils.config", "get_config"),
        ("src.utils.logger", "setup_logger"),
        ("src.models.generator", "Generator"),
        ("src.pipeline.preprocessor", "Preprocessor"),
    ]
    
    all_ok = True
    for module_name, attr_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            getattr(module, attr_name)
            print_success(f"{module_name}.{attr_name}")
        except Exception as e:
            print_error(f"{module_name}.{attr_name} - {e}")
            all_ok = False
    
    return all_ok


def check_gpu_detection():
    """Check GPU/CPU detection"""
    print_header("Checking GPU/CPU Detection")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print_success(f"CUDA available - {device_count} device(s)")
            print_info(f"  Device 0: {device_name}")
            print_info(f"  Recommended: device=cuda")
        else:
            print_warning("CUDA not available - will use CPU")
            print_info("  Recommended: device=cpu")
        
        return True
    except ImportError:
        print_warning("PyTorch not installed - cannot detect GPU")
        print_info("  Install PyTorch: pip install torch")
        return False
    except Exception as e:
        print_error(f"Error detecting GPU: {e}")
        return False


def test_configuration():
    """Test configuration loading"""
    print_header("Testing Configuration")
    
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        from src.utils.config import get_config, load_config
        
        config = load_config()
        device = config.get("models", {}).get("generator", {}).get("device", "unknown")
        image_size = config.get("processing", {}).get("image_size", "unknown")
        
        print_success("Configuration loaded successfully")
        print_info(f"  Device: {device}")
        print_info(f"  Image size: {image_size}")
        
        return True
    except Exception as e:
        print_error(f"Configuration error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_env_file():
    """Create .env file from env.example if it doesn't exist"""
    print_header("Checking Environment Configuration")
    
    project_root = Path(__file__).parent
    env_example = project_root / "env.example"
    env_file = project_root / ".env"
    
    if env_file.exists():
        print_success(".env file already exists")
        return True
    
    if not env_example.exists():
        print_warning("env.example not found - creating basic .env")
        try:
            # Create basic .env
            env_content = """# Face and Body Swap Pipeline Configuration

# Device Configuration (auto-detected, can override)
# DEVICE=cuda  # or cpu

# Processing Configuration
IMAGE_SIZE=512
MAX_IMAGE_SIZE=1024

# Paths
MODELS_DIR=models
OUTPUTS_DIR=outputs
TEMP_DIR=temp

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
"""
            env_file.write_text(env_content)
            print_success("Created .env file")
            return True
        except Exception as e:
            print_error(f"Failed to create .env: {e}")
            return False
    
    try:
        # Detect device
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except:
            pass
        
        # Read example and update device
        content = env_example.read_text()
        # Update device if not set
        if "DEVICE=" not in content:
            content = content.replace("# Device Configuration", f"# Device Configuration\nDEVICE={device}")
        else:
            # Replace existing DEVICE line
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("DEVICE=") or line.startswith("# DEVICE="):
                    lines[i] = f"DEVICE={device}"
                    break
            content = '\n'.join(lines)
        
        env_file.write_text(content)
        print_success(f"Created .env file from env.example (device={device})")
        return True
    except Exception as e:
        print_error(f"Failed to create .env: {e}")
        return False


def main():
    """Run all setup checks"""
    print_header("Face-Body Swap - Local Setup Verification")
    
    results = {
        "python_version": False,
        "dependencies": False,
        "directories": False,
        "structure": False,
        "imports": False,
        "gpu_detection": False,
        "configuration": False,
        "env_file": False,
    }
    
    # Run checks
    results["python_version"] = check_python_version()
    if not results["python_version"]:
        print_error("\nPython version check failed. Please upgrade Python.")
        return 1
    
    deps_ok, missing = check_dependencies()
    results["dependencies"] = deps_ok
    
    results["directories"] = create_directories()
    results["structure"] = check_directory_structure()
    
    if deps_ok:
        results["imports"] = test_imports()
        results["gpu_detection"] = check_gpu_detection()
        results["configuration"] = test_configuration()
    
    results["env_file"] = create_env_file()
    
    # Summary
    print_header("Setup Summary")
    
    all_passed = True
    for check, passed in results.items():
        if passed:
            print_success(f"{check.replace('_', ' ').title()}")
        else:
            print_error(f"{check.replace('_', ' ').title()}")
            all_passed = False
    
    print()
    if all_passed:
        print_success("All checks passed! Setup is complete.")
        print_info("\nNext steps:")
        print_info("  1. Review .env file and adjust if needed")
        print_info("  2. Run: python app.py")
        print_info("  3. Or use: start_local.bat")
        return 0
    else:
        print_error("Some checks failed. Please fix the issues above.")
        if missing:
            print_info("\nTo install missing dependencies:")
            print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

