"""Setup script for face and body swap pipeline"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="face-body-swap",
    version="1.0.0",
    description="Automated face and body swap pipeline using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/face-body-swap",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scipy>=1.10.0",
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "insightface>=0.7.3",
        "onnxruntime>=1.15.0",
        "mediapipe>=0.10.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "onnxruntime-gpu>=1.15.0",
            "xformers>=0.0.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "face-body-swap=src.api.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

