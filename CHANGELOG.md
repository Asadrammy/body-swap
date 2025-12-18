# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-01-XX

### Added
- Initial release of face and body swap pipeline
- Core pipeline stages: preprocessing, body analysis, template analysis, face processing, body warping, composition, refinement, and quality control
- Face detection using InsightFace, dlib, and OpenCV fallback
- Pose detection using MediaPipe and OpenPose support
- Body shape analysis and classification
- Clothing adaptation to different body types
- Expression matching and transfer
- Stable Diffusion + ControlNet integration for refinement
- REST API with FastAPI
- CLI interface for batch processing
- Docker support
- Configuration management (YAML and environment variables)
- Quality control and manual refinement mask generation
- Support for single and multiple subjects (couples, families)

### Technical Stack
- PyTorch for deep learning
- Diffusers for Stable Diffusion
- InsightFace for face recognition
- MediaPipe for pose detection
- FastAPI for REST API
- Docker for containerization

