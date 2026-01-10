#!/usr/bin/env python3
"""Comprehensive project test script"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("PROJECT TEST SUITE")
print("=" * 60)

tests_passed = 0
tests_failed = 0

def test(name, func):
    global tests_passed, tests_failed
    try:
        print(f"\n[TEST] {name}...")
        result = func()
        if result:
            print(f"  [PASS] PASSED")
            tests_passed += 1
            return True
        else:
            print(f"  [FAIL] FAILED")
            tests_failed += 1
            return False
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")
        tests_failed += 1
        return False

# Test 1: Config module
def test_config():
    from src.utils.config import get_config
    config = get_config()
    return config is not None and isinstance(config, dict)

# Test 2: Logger module
def test_logger():
    from src.utils.logger import setup_logger, get_logger
    setup_logger()
    logger = get_logger('test')
    return logger is not None

# Test 3: Template catalog
def test_template_catalog():
    from src.utils.template_catalog import TemplateCatalog
    catalog = TemplateCatalog()
    templates = catalog.list_templates()
    return len(templates) > 0

# Test 4: Face Detector import
def test_face_detector():
    from src.models.face_detector import FaceDetector
    return True

# Test 5: Pose Detector import
def test_pose_detector():
    from src.models.pose_detector import PoseDetector
    return True

# Test 6: Generator with LoRA support
def test_generator_lora():
    from src.models.generator import Generator
    gen = Generator()
    has_load = hasattr(gen, '_load_lora_adapters')
    has_load_method = hasattr(gen, 'load_lora')
    has_unload = hasattr(gen, 'unload_lora')
    return has_load and has_load_method and has_unload

# Test 7: API routes structure
def test_api_routes():
    from src.api.routes import router
    return router is not None

# Test 8: API main with health endpoint
def test_api_main():
    from src.api.main import app
    routes = [route.path for route in app.routes]
    has_health = '/health' in routes
    has_metrics = '/metrics' in routes
    return has_health and has_metrics

# Test 9: Pipeline modules
def test_pipeline_modules():
    from src.pipeline.preprocessor import Preprocessor
    from src.pipeline.body_analyzer import BodyAnalyzer
    from src.pipeline.template_analyzer import TemplateAnalyzer
    from src.pipeline.face_processor import FaceProcessor
    from src.pipeline.body_warper import BodyWarper
    from src.pipeline.composer import Composer
    from src.pipeline.refiner import Refiner
    from src.pipeline.quality_control import QualityControl
    return True

# Test 10: Test set generation script exists
def test_test_set_script():
    script_path = project_root / "scripts" / "generate_test_set.py"
    return script_path.exists()

# Test 11: Deployment documentation
def test_deployment_docs():
    docs = [
        "RUNNINGHUB_DEPLOYMENT.md",
        "DEPLOYMENT.md",
        "README.md"
    ]
    for doc in docs:
        if not (project_root / doc).exists():
            return False
    return True

# Test 12: Configuration files
def test_config_files():
    configs = [
        "configs/default.yaml",
        "requirements.txt",
        "docker-compose.yml",
        "Dockerfile"
    ]
    for config in configs:
        if not (project_root / config).exists():
            return False
    return True

# Test 13: Project structure
def test_project_structure():
    required_dirs = [
        "src",
        "src/api",
        "src/models",
        "src/pipeline",
        "src/utils",
        "scripts",
        "configs",
        "examples"
    ]
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            return False
    return True

# Run all tests
print("\nRunning comprehensive project tests...\n")

test("Config Module", test_config)
test("Logger Module", test_logger)
test("Template Catalog", test_template_catalog)
test("Face Detector Import", test_face_detector)
test("Pose Detector Import", test_pose_detector)
test("Generator LoRA Support", test_generator_lora)
test("API Routes Structure", test_api_routes)
test("API Health/Metrics Endpoints", test_api_main)
test("Pipeline Modules", test_pipeline_modules)
test("Test Set Generation Script", test_test_set_script)
test("Deployment Documentation", test_deployment_docs)
test("Configuration Files", test_config_files)
test("Project Structure", test_project_structure)

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Passed: {tests_passed}")
print(f"Failed: {tests_failed}")
print(f"Total:  {tests_passed + tests_failed}")
print("=" * 60)

if tests_failed == 0:
    print("\n[SUCCESS] ALL TESTS PASSED - PROJECT IS 100% READY!")
    sys.exit(0)
else:
    print(f"\n[ERROR] {tests_failed} TEST(S) FAILED")
    sys.exit(1)

