# Project Overview - Face and Body Swap Pipeline

## Project Structure

```
face-body-swap/
├── src/
│   ├── api/              # FastAPI REST API and CLI
│   │   ├── main.py       # FastAPI app initialization
│   │   ├── routes.py     # API endpoints and main pipeline (CRITICAL)
│   │   ├── schemas.py    # Pydantic models
│   │   └── cli.py        # Command-line interface
│   ├── models/           # ML model wrappers
│   │   ├── generator.py  # Stable Diffusion wrapper (CRITICAL - has solid color issue)
│   │   ├── face_detector.py
│   │   ├── pose_detector.py
│   │   └── segmenter.py
│   ├── pipeline/         # Processing pipeline stages
│   │   ├── preprocessor.py
│   │   ├── body_analyzer.py
│   │   ├── template_analyzer.py
│   │   ├── face_processor.py
│   │   ├── body_warper.py
│   │   ├── composer.py
│   │   ├── refiner.py    # Image refinement orchestrator (CRITICAL)
│   │   └── quality_control.py
│   └── utils/            # Utilities
│       ├── config.py      # Configuration loader
│       ├── logger.py      # Logging setup
│       └── image_utils.py # Image I/O helpers
├── configs/
│   └── default.yaml      # Main configuration file
├── app.py                # Application entry point
├── requirements.txt      # Python dependencies
└── server_logs.txt       # Recent error logs
```

## Key Components

### 1. Generator (`src/models/generator.py`)
**Purpose**: Wraps Stable Diffusion inpainting model for image refinement

**Key Methods:**
- `refine()`: Main refinement method (lines 271-429) - **HAS SOLID COLOR ISSUE**
- `guided_face_refine()`: Face-specific refinement
- `_init_inpaint_pipeline()`: Model loading

**Current Issues:**
- Produces solid color images on CPU
- Validation thresholds may be too lenient
- Model loading has safetensors warnings

### 2. Refiner (`src/pipeline/refiner.py`)
**Purpose**: Orchestrates refinement passes for different image regions

**Key Methods:**
- `refine_composition()`: Main composition refinement (lines 32-182)
- `refine_face()`: Face-specific refinement (lines 231-347)
- `_is_valid_image()`: Image validation helper

**Current Issues:**
- CPU detection may not be working correctly
- Face refinement produces solid colors
- Validation catches errors but doesn't prevent them

### 3. Routes (`src/api/routes.py`)
**Purpose**: Main pipeline orchestrator and API endpoints

**Key Class:**
- `SwapPipeline.process()`: Main processing pipeline (lines 68-401)

**Pipeline Flow:**
1. Preprocess inputs
2. Analyze body shapes
3. Analyze template
4. Process faces
5. Warp body
6. Compose image
7. **Refine (HAS ISSUES HERE)**
8. Quality control
9. Save result

## Configuration

**File**: `configs/default.yaml`

**Key Settings:**
```yaml
models:
  generator:
    device: cuda  # Falls back to CPU if no GPU
    inpaint_model: runwayml/stable-diffusion-inpainting

processing:
  num_inference_steps: 40
  guidance_scale: 9.0
  refinement_strength: 0.8
  region_refine_strengths:
    face: 0.65
    body: 0.55
    edges: 0.45
    problems: 0.7
```

## Dependencies

**Key Libraries:**
- `diffusers`: Stable Diffusion models
- `torch`: PyTorch (CPU version)
- `transformers`: HuggingFace transformers
- `opencv-python`: Image processing
- `mediapipe`: Pose detection
- `fastapi`: Web API framework

## Error Flow

1. **User uploads images** → API receives request
2. **Pipeline processes** → Face swap, body warp, composition
3. **Refinement called** → `refiner.refine_composition()`
4. **Generator called** → `generator.refine()`
5. **Stable Diffusion runs** → **PRODUCES SOLID COLOR**
6. **Validation catches** → Falls back to original
7. **Result saved** → May still be problematic

## Current Error Location

**Primary Issue**: `src/models/generator.py:refine()` method
- Line 357: Stable Diffusion inference call
- Line 380-424: Validation logic (may be too lenient)
- **Problem**: Inference produces solid colors

**Secondary Issue**: `src/pipeline/refiner.py:refine_composition()`
- Line 79-82: CPU detection (may not be working)
- Line 97-123: Global refinement (produces near-solid colors)
- Line 144-173: Region refinement (face produces 1 unique color)

## Validation Logic

**Generator (`generator.py`):**
- Unique colors: >= 15 (but logs show 1)
- Std dev: >= 6.0 (but logs show 35.52 for face)
- Per-channel std: >= 3.0 (but logs show 0.0 for face)

**Refiner (`refiner.py`):**
- Unique colors: >= 20
- Std dev: >= 8.0
- Per-channel std: >= 5.0

**Mismatch**: Generator allows lower thresholds than refiner expects

## Log Analysis

**Key Log Messages:**
```
[WARNING] Generator returned near-solid color (channel_stds=[1.82, 1.48, 1.52])
[WARNING] Refined region 'face' is solid color (unique_colors=1, std=35.52, channel_stds=[0.0, 0.0, 0.0])
[WARNING] Safetensors loading failed, trying without safetensors
[WARNING] Running on CPU - this will be slower
```

## Files to Focus On

1. **`src/models/generator.py`** - Fix solid color generation
2. **`src/pipeline/refiner.py`** - Fix CPU detection and validation
3. **`configs/default.yaml`** - Adjust parameters if needed

## Testing

**Test Case:**
- Input: Customer photo + template
- Expected: Realistic face/body swapped image
- Actual: Solid color image (blue/pink/red)

**Validation:**
- System detects solid colors
- Falls back to original/composed image
- But root cause (why Stable Diffusion produces solid colors) not fixed

---

This overview provides context for the error report. The main issue is in the Stable Diffusion inference producing solid colors instead of realistic images.


















