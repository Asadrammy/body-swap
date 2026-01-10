# Face and Body Swap Pipeline

A fully automated face-and-body swap pipeline that transplants customers into template images while preserving clothing style, pose, background, and facial expressions. The system supports individuals, couples, families, and various body types.

## Features

- **Automated Processing**: Fully automated pipeline from input to output
- **Multiple Subjects**: Support for individuals, couples, families, and groups
- **Body Type Adaptation**: Automatically adapts clothing to different body sizes (average, plus-size, athletic, etc.)
- **Body Conditioning**: Enhanced support for open chest shirts with realistic skin synthesis (male, female, children)
- **Expression Matching**: Preserves and matches facial expressions from templates
- **Action Photo Support**: Automatic detection and handling of action poses with dynamic expressions
- **Pose Alignment**: Accurate body pose matching and warping
- **Natural Face Refinement**: Enhanced face processing to avoid plastic-looking results
- **Quality Control**: Built-in quality assessment and refinement capabilities
- **Manual Touch-ups**: Enhanced mask generation for precise selective refinement
- **Full Workflow Control**: Comprehensive manual intervention points at every stage
- **REST API**: Full REST API for integration with web applications
- **CLI Interface**: Command-line interface for batch processing

## Architecture

The pipeline consists of several stages:

1. **Input Validation & Preprocessing**: Validates and preprocesses customer photos and templates
2. **Body Shape Analysis**: Extracts body proportions and classifies body type
3. **Template Analysis**: Analyzes pose, clothing, expression, and background
4. **Face Processing**: Detects, aligns, and matches faces and expressions
5. **Body Warping**: Warps customer body to match template pose
6. **Clothing Adaptation**: Adapts clothing to customer's body proportions
7. **Composition**: Blends warped body into template background
8. **Generative Refinement**: Uses Stable Diffusion + ControlNet for photorealistic results
9. **Quality Control**: Assesses quality and generates refinement masks

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)
- 20GB+ free disk space for models

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd face-and-body-swap
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support, ensure you have the appropriate PyTorch CUDA version installed.

### Step 4: Download Models

Models will be downloaded automatically on first use, or you can download them manually:

- **InsightFace**: Automatically downloaded
- **MediaPipe**: Automatically included
- **Stable Diffusion**: Automatically downloaded from HuggingFace
- **SAM** (optional): Download from [Segment Anything Model](https://github.com/facebookresearch/segment-anything)

### Step 5: Configure

Copy `.env.example` to `.env` and adjust settings:

```bash
cp .env.example .env
```

Edit `.env` to configure paths, device settings, etc.

## Usage

### Web Interface (Frontend)

The easiest way to use the service is through the web interface:

1. **Start the API server**:
   ```bash
   python -m src.api.main
   ```

2. **Open your browser** and visit:
   ```
   http://localhost:8000
   ```

3. **Use the interface**:
   - Upload your photos (1-2 images)
   - Select a template
   - Watch the processing progress
   - Download your result

The frontend is a modern, responsive web interface with drag-and-drop upload, real-time progress tracking, and quality metrics display.

### CLI Usage

Basic usage with single customer photo:

```bash
python -m src.api.cli swap \
    --customer-photos path/to/customer.jpg \
    --template path/to/template.jpg \
    --output path/to/result.jpg
```

Multiple customer photos (for couples/families):

```bash
python -m src.api.cli swap \
    --customer-photos photo1.jpg photo2.jpg \
    --template template.jpg \
    --output result.jpg
```

With refinement mask:

```bash
python -m src.api.cli swap \
    --customer-photos customer.jpg \
    --template template.jpg \
    --output result.jpg \
    --refine-mask mask.png
```

Export intermediate results:

```bash
python -m src.api.cli swap \
    --customer-photos customer.jpg \
    --template template.jpg \
    --output result.jpg \
    --export-intermediate
```

Skip refinement (faster, lower quality):

```bash
python -m src.api.cli swap \
    --customer-photos customer.jpg \
    --template template.jpg \
    --output result.jpg \
    --no-refine
```

### REST API Usage

Start the API server:

```bash
python -m src.api.main
```

Or with uvicorn directly:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API documentation will be available at `http://localhost:8000/docs`

#### Example API Requests

**Create Swap Job:**

```bash
curl -X POST "http://localhost:8000/api/v1/swap" \
  -F "customer_photos=@customer1.jpg" \
  -F "customer_photos=@customer2.jpg" \
  -F "template=@template.jpg"
```

**Check Job Status:**

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}"
```

**Download Result:**

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}/result" --output result.png
```

### Docker Usage

Build the Docker image:

```bash
docker build -t face-body-swap .
```

Run with docker-compose:

```bash
docker-compose up -d
```

Or run directly:

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  face-body-swap
```

## Configuration

Configuration can be managed through:

1. **YAML files** in `configs/` directory
2. **Environment variables** in `.env` file
3. **Command-line arguments** (for CLI)

Key configuration options:

- `IMAGE_SIZE`: Target processing size (default: 512)
- `MAX_IMAGE_SIZE`: Maximum input size (default: 1024)
- `DEVICE`: Processing device (cuda/cpu)
- `REFINEMENT_STRENGTH`: Refinement strength (0-1)

## Project Structure

```
face-and-body-swap/
├── src/
│   ├── pipeline/          # Core pipeline stages
│   ├── models/            # Model wrappers
│   ├── utils/             # Utility functions
│   └── api/               # API and CLI
├── configs/               # Configuration files
├── tests/                 # Test suite
├── examples/              # Example inputs/outputs
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose config
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Documentation

- **[WORKFLOW_DOCUMENTATION.md](WORKFLOW_DOCUMENTATION.md)**: Complete workflow logic explanation with all stages, control points, and manual intervention options
- **[TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)**: Comprehensive troubleshooting guide for common issues
- **[MODELS_IMPLEMENTATION.md](MODELS_IMPLEMENTATION.md)**: Details about model wrappers and their implementation

## Troubleshooting

For detailed troubleshooting, see **[TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)**.

### Quick Fixes

**Plastic-Looking Face**:
- Reduce face refinement strength to 0.5-0.6 in config
- Check negative prompts include "plastic, artificial, CGI"
- Verify post-processing is preserving original texture

**Open Chest Shirt Issues**:
- Ensure customer photos show visible skin
- Check skin profile extraction is working
- Use chest_skin mask for targeted refinement

**Body Size Mismatch**:
- Verify pose detection is working
- Check body measurements are accurate
- Adjust scale factors manually if needed

**Blending Seams**:
- Use edge refinement mask
- Increase edge refinement strength
- Check background segmentation

**Action Pose Issues**:
- Verify action pose detection is working
- Check expression matching for dynamic poses
- Adjust warp parameters for dynamic poses

### Common Issues

### GPU Not Detected

Ensure CUDA is properly installed and PyTorch can detect it:

```python
import torch
print(torch.cuda.is_available())
```

### Out of Memory Errors

- Reduce `IMAGE_SIZE` in configuration
- Process images in smaller batches
- Use CPU mode (slower but less memory)

### Model Download Issues

- Check internet connection
- Set `HF_TOKEN` in `.env` for private models
- Manually download models to `models/` directory

### Face Detection Fails

- Ensure faces are clearly visible in input images
- Try different face detector (`face_detector: dlib` in config)
- Check image quality and resolution

### Poor Quality Results

- Increase `REFINEMENT_STRENGTH`
- Use higher `IMAGE_SIZE`
- Ensure good quality input images
- Check template image quality
- Review quality assessment and use recommended refinements

## Performance

- **Processing Time**: 30-120 seconds per image (depends on GPU)
- **Memory Usage**: 4-8GB GPU memory
- **Supported Formats**: JPEG, PNG
- **Output Resolution**: Up to 2048x2048 (configurable)

## Limitations

- Requires clear face visibility in input images
- Best results with front-facing or slightly angled poses
- Template pose must be detectable (not heavily occluded)
- Processing time increases with image size
- GPU recommended for reasonable processing times

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Your License Here]

## Support

For issues, questions, or support:
- Open an issue on GitHub
- Check the documentation
- Review troubleshooting section

## Acknowledgments

This project uses:
- Stable Diffusion (Stability AI)
- ControlNet (lllyasviel)
- InsightFace (deepinsight)
- MediaPipe (Google)
- Segment Anything (Meta)

## Roadmap

- [ ] Video support
- [ ] Real-time processing
- [ ] Web UI interface
- [ ] Batch processing improvements
- [ ] Additional body type classifications
- [ ] Advanced expression transfer
- [ ] Style transfer options

