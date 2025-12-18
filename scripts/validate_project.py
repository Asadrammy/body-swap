#!/usr/bin/env python3
"""Validate project structure and code"""

import sys
from pathlib import Path
import ast

def check_file_structure():
    """Check if all expected files exist"""
    base = Path(__file__).parent.parent
    
    required_files = [
        "src/__init__.py",
        "src/pipeline/__init__.py",
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "src/api/__init__.py",
        "src/pipeline/preprocessor.py",
        "src/pipeline/body_analyzer.py",
        "src/pipeline/template_analyzer.py",
        "src/pipeline/face_processor.py",
        "src/pipeline/body_warper.py",
        "src/pipeline/composer.py",
        "src/pipeline/refiner.py",
        "src/pipeline/quality_control.py",
        "src/models/face_detector.py",
        "src/models/pose_detector.py",
        "src/models/segmenter.py",
        "src/models/generator.py",
        "src/utils/config.py",
        "src/utils/logger.py",
        "src/utils/image_utils.py",
        "src/utils/warp_utils.py",
        "src/api/main.py",
        "src/api/routes.py",
        "src/api/schemas.py",
        "src/api/cli.py",
        "requirements.txt",
        "README.md",
        "Dockerfile",
        "docker-compose.yml",
    ]
    
    missing = []
    for file_path in required_files:
        full_path = base / file_path
        if not full_path.exists():
            missing.append(file_path)
    
    return missing


def check_python_syntax():
    """Check Python syntax of all .py files"""
    base = Path(__file__).parent.parent
    errors = []
    
    for py_file in base.rglob("*.py"):
        # Skip __pycache__ and venv
        if "__pycache__" in str(py_file) or "venv" in str(py_file):
            continue
        
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                ast.parse(f.read(), filename=str(py_file))
        except SyntaxError as e:
            errors.append((py_file, str(e)))
    
    return errors


def main():
    """Run validation"""
    print("Validating project structure...")
    
    # Check file structure
    missing = check_file_structure()
    if missing:
        print(f"✗ Missing files: {missing}")
        return 1
    else:
        print("✓ All required files present")
    
    # Check syntax
    print("\nValidating Python syntax...")
    syntax_errors = check_python_syntax()
    if syntax_errors:
        print(f"✗ Syntax errors found:")
        for file_path, error in syntax_errors:
            print(f"  {file_path}: {error}")
        return 1
    else:
        print("✓ No syntax errors")
    
    print("\n✓ Project validation passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

