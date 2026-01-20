# Google AI Studio Image Generation Setup

## ✅ Google AI Studio Support Added!

The system now supports **Google AI Studio (Imagen)** for image generation using your existing API key!

## Your API Key

You already have a working Google AI Studio API key:
```
AIzaSyCioMnUARXoWlLmQZDKS-wrUZhoulS6hPU
```

## Quick Setup

### 1. Configure `.env` File

Add these lines to your `.env` file:

```bash
# Enable AI API usage
USE_AI_API=true

# Use Google AI Studio (your existing key)
GOOGLE_AI_API_KEY=AIzaSyCioMnUARXoWlLmQZDKS-wrUZhoulS6hPU

# Set provider to Google
AI_IMAGE_PROVIDER=google
```

### 2. Test It

Run the test script:

```bash
python test_ai_generation.py --customer-image "D:\projects\image\face-body-swap\1760713603491 (1).jpg"
```

## How It Works

1. **Automatic Detection**: The system automatically detects your Google AI API key
2. **Imagen Models**: Uses Google's Imagen 4.0 models for image generation:
   - `imagen-4.0-fast-generate-001` (fast, recommended)
   - `imagen-4.0-generate-001` (standard quality)
   - `imagen-4.0-ultra-generate-001` (highest quality)
3. **Inpainting Support**: Supports inpainting with masks for face/body refinement
4. **Fallback**: Falls back to other providers if Google AI is unavailable

## Features

✅ **Text-to-Image**: Generate images from prompts  
✅ **Image Inpainting**: Refine specific regions using masks  
✅ **High Quality**: Uses Google's latest Imagen 4.0 models  
✅ **Fast Processing**: Fast generation model available  
✅ **No Local Models**: No need to download large model files  

## API Usage

The system uses Google's Imagen REST API:
- **Endpoint**: `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateImages`
- **Method**: POST
- **Authentication**: API key in query parameter

## Cost

- **Free Tier**: Available for testing
- **Pay-per-use**: Typically $0.01-0.03 per image
- **Check Usage**: Monitor at https://ai.google.dev/aistudio/

## Comparison with Other Providers

| Feature | Google AI | Stability AI | OpenAI |
|---------|-----------|--------------|--------|
| **Image Generation** | ✅ Excellent | ✅ Good | ✅ Good |
| **Inpainting** | ✅ Supported | ✅ Excellent | ⚠️ Limited |
| **Speed** | ✅ Fast | ✅ Fast | ⚠️ Slower |
| **Cost** | ✅ Low | ✅ Low | ⚠️ Higher |
| **Already Configured** | ✅ Yes | ❌ No | ❌ No |

## Troubleshooting

### API Key Not Working
- Verify key is correct: `AIzaSyCioMnUARXoWlLmQZDKS-wrUZhoulS6hPU`
- Check key is set in `.env` file
- Ensure `USE_AI_API=true` is set

### Generation Fails
- Check internet connection
- Verify API quota/credits
- Check logs for specific error messages
- System will fallback to other providers automatically

### Slow Generation
- Try `imagen-4.0-fast-generate-001` model (faster)
- Reduce image size if possible
- Check API status at https://status.cloud.google.com/

## Example Usage

```python
from src.models.ai_image_generator import AIImageGenerator

# Initialize (automatically uses Google AI if key is set)
generator = AIImageGenerator()

# Refine image
refined = generator.refine(
    image=your_image,
    prompt="photorealistic portrait, natural skin texture",
    mask=face_mask,  # Optional
    strength=0.8
)
```

## Next Steps

1. ✅ **Already Done**: API key is configured
2. ✅ **Already Done**: Code is updated
3. **Test**: Run test script with your image
4. **Monitor**: Check API usage and costs
5. **Optimize**: Adjust prompts for best results

---

**Note**: The system automatically prioritizes Google AI Studio when the API key is available, making it the default provider for image generation!








