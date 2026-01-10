# Model Wrappers Implementation

This document describes the implementation of the model wrapper classes that were missing from the project.

## Overview

Four model wrapper classes have been implemented to integrate with the face-body-swap pipeline:

1. **FaceDetector** - Face detection and recognition
2. **PoseDetector** - Human pose detection
3. **Segmenter** - Body part and background segmentation
4. **Generator** - Stable Diffusion image refinement

## Implementation Details

### 1. FaceDetector (`src/models/face_detector.py`)

**Purpose**: Detect faces, extract embeddings, and align faces for processing.

**Features**:
- Primary backend: **InsightFace** (buffalo_l model)
- Fallback: **OpenCV Haar Cascade**
- Optional: dlib support (placeholder)

**Key Methods**:
- `detect_faces(image)` - Returns list of face detections with:
  - Bounding boxes
  - Landmarks (5 points: eyes, nose, mouth)
  - Face embeddings (512-dim vectors)
  - Age and gender estimates (when available)
- `align_face(image, landmarks, size)` - Aligns face to standard 112x112 format
- `extract_face_embedding(image, bbox)` - Extracts face embedding vector

**Configuration**:
- Set `models.face_detector` in config (options: "insightface", "dlib", "opencv")

### 2. PoseDetector (`src/models/pose_detector.py`)

**Purpose**: Detect human body pose and extract keypoints for body warping.

**Features**:
- Backend: **MediaPipe Pose** (model_complexity=2 for best accuracy)
- Extracts 13+ keypoints including:
  - Head: nose, neck
  - Upper body: shoulders, elbows, wrists
  - Lower body: hips, knees, ankles

**Key Methods**:
- `detect_pose(image)` - Returns list of pose detections with:
  - Keypoints dictionary (name -> [x, y] coordinates)
  - Bounding box
  - Confidence score

**Keypoints Extracted**:
- nose, neck, left_shoulder, right_shoulder
- left_elbow, right_elbow, left_wrist, right_wrist
- left_hip, right_hip, left_knee, right_knee
- left_ankle, right_ankle
- mid_hip (calculated from hip positions)

**Configuration**:
- Set `models.pose_detector` in config (currently only "mediapipe" supported)

### 3. Segmenter (`src/models/segmenter.py`)

**Purpose**: Segment body parts and background for targeted processing.

**Features**:
- Pose-based body part segmentation
- Optional SAM (Segment Anything Model) support (placeholder)
- Heuristic-based segmentation using pose keypoints

**Key Methods**:
- `segment_body_parts(image, pose_data)` - Returns dictionary of masks:
  - `torso` - Torso/chest region
  - `left_arm`, `right_arm` - Arm regions
  - `left_leg`, `right_leg` - Leg regions
- `segment_background(image, foreground_mask)` - Returns background mask

**Segmentation Approach**:
- Uses pose keypoints to create polygon regions
- Applies Gaussian blur for smooth masks
- Estimates body part widths based on pose geometry

**Configuration**:
- Set `models.segmenter` in config (currently uses pose-based method)

### 4. Generator (`src/models/generator.py`)

**Purpose**: Refine images using Stable Diffusion for photorealistic results.

**Features**:
- **Stable Diffusion Inpainting** pipeline
- Automatic device selection (CUDA/CPU)
- Memory-efficient attention (xformers when available)
- Configurable refinement strength

**Key Methods**:
- `refine(image, prompt, mask, negative_prompt, strength, num_inference_steps)`:
  - Refines masked regions of image
  - Uses text prompts to guide generation
  - Returns refined image
- `guided_face_refine(face_patch, control_map, prompt, negative_prompt)`:
  - Specialized face refinement with control map guidance
  - Uses landmark-based control maps

**Models Used**:
- Base: `runwayml/stable-diffusion-v1-5`
- Inpainting: `runwayml/stable-diffusion-inpainting`
- ControlNet: `lllyasviel/sd-controlnet-openpose` (referenced, not yet integrated)

**Configuration**:
- Set `models.generator.device` in config ("cuda" or "cpu")
- Set `models.generator.base_model` for base SD model
- Set `models.generator.inpaint_model` for inpainting model

## Error Handling

All model classes implement robust error handling:
- Graceful fallbacks when primary backends fail
- Logging of errors and warnings
- Returns empty results or original images on failure (never crashes)

## Dependencies

### Required:
- `torch` - PyTorch for Stable Diffusion
- `diffusers` - HuggingFace Diffusers library
- `transformers` - For model loading
- `mediapipe` - For pose detection
- `opencv-python` - For image processing
- `numpy` - For array operations
- `Pillow` - For image I/O

### Optional:
- `insightface` - For advanced face detection (recommended)
- `onnxruntime` or `onnxruntime-gpu` - For InsightFace
- `xformers` - For memory-efficient attention (recommended for GPU)

## Installation Notes

1. **InsightFace**: Automatically downloads models on first use
2. **MediaPipe**: Installs via pip, no additional setup needed
3. **Stable Diffusion**: Downloads models from HuggingFace on first use (requires internet)
4. **CUDA**: For GPU support, ensure PyTorch with CUDA is installed

## Usage Example

```python
from src.models import FaceDetector, PoseDetector, Segmenter, Generator

# Initialize models
face_detector = FaceDetector()
pose_detector = PoseDetector()
segmenter = Segmenter()
generator = Generator()

# Detect faces
faces = face_detector.detect_faces(image)

# Detect pose
poses = pose_detector.detect_pose(image)

# Segment body parts
masks = segmenter.segment_body_parts(image, poses[0])

# Refine image
refined = generator.refine(
    image=image,
    prompt="photorealistic, high quality",
    mask=refinement_mask,
    strength=0.8
)
```

## Performance Considerations

1. **Face Detection**: InsightFace is fast on GPU, slower on CPU
2. **Pose Detection**: MediaPipe is optimized and runs well on CPU
3. **Segmentation**: Pose-based method is fast, no GPU needed
4. **Generation**: Stable Diffusion requires GPU for reasonable speed (30-120s per image)

## Future Enhancements

- [ ] Full SAM integration for advanced segmentation
- [ ] ControlNet integration for pose-guided generation
- [ ] LoRA support for custom model fine-tuning
- [ ] Batch processing optimizations
- [ ] Model quantization for faster CPU inference

## Notes

- All models are initialized lazily (only when needed)
- Models are cached in memory after first use
- GPU memory is managed automatically
- Models download automatically from HuggingFace/InsightFace on first use

