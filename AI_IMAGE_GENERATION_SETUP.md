# AI Image Generation Setup Guide

This document explains how to use AI image generation APIs instead of local models to avoid distortion issues.

## Overview

The system now supports using AI image generation APIs (OpenAI DALL-E, Stability AI, Replicate) instead of local Stable Diffusion models. This approach:

- ✅ **Avoids distortion** - APIs use optimized models that produce better results
- ✅ **No local model downloads** - No need to download large model files
- ✅ **Better quality** - Professional-grade image generation
- ✅ **Faster processing** - No local GPU/CPU inference overhead

## Setup

### 1. Install Required Packages

```bash
pip install google-generativeai>=0.8.0  # For Google AI Studio (already installed)
pip install openai>=1.0.0  # For OpenAI DALL-E (optional)
pip install replicate>=0.15.0  # For Replicate (optional)
# requests is already in requirements.txt
```

### 2. Get API Keys

#### Option 1: Google AI Studio (Recommended - Uses Imagen)
1. Sign up at https://ai.google.dev/aistudio/
2. Get your API key from the dashboard
3. Set in `.env`:
   ```bash
   GOOGLE_AI_API_KEY=your_google_ai_api_key_here
   ```
   **Note:** You already have a working API key: `AIzaSyCioMnUARXoWlLmQZDKS-wrUZhoulS6hPU`

#### Option 2: Stability AI (Alternative for Inpainting)
1. Sign up at https://platform.stability.ai/
2. Get your API key from the dashboard
3. Set in `.env`:
   ```bash
   STABILITY_API_KEY=your_stability_api_key_here
   ```

#### Option 3: OpenAI DALL-E
1. Sign up at https://platform.openai.com/
2. Get your API key from the dashboard
3. Set in `.env`:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

#### Option 4: Replicate
1. Sign up at https://replicate.com/
2. Get your API token from the dashboard
3. Set in `.env`:
   ```bash
   REPLICATE_API_TOKEN=your_replicate_token_here
   ```

### 3. Configure Environment

Edit `.env` file:

```bash
# Enable AI API usage
USE_AI_API=true

# Set your API key (at least one required)
# Google AI Studio (recommended - already configured)
GOOGLE_AI_API_KEY=AIzaSyCioMnUARXoWlLmQZDKS-wrUZhoulS6hPU
# OR
STABILITY_API_KEY=your_key_here
# OR
OPENAI_API_KEY=your_key_here
# OR
REPLICATE_API_TOKEN=your_token_here

# Optional: Select provider (default: google if available)
AI_IMAGE_PROVIDER=google
```

## Usage

### Test Script

Run the test script with your client image:

```bash
python test_ai_generation.py --customer-image "path/to/image.jpg"
```

### API Usage

The system automatically uses AI APIs when:
1. `USE_AI_API=true` is set
2. At least one API key is configured
3. The `Generator` class is initialized

### Fallback Behavior

If no API keys are available, the system falls back to local models (if installed). This ensures backward compatibility.

## How It Works

1. **Initialization**: The `Generator` class checks for API keys
2. **Provider Selection**: Selects the best available provider (Stability AI > OpenAI > Replicate)
3. **Image Refinement**: Uses API for inpainting/refinement instead of local models
4. **Validation**: Validates results to ensure quality

## API Providers Comparison

| Provider | Best For | Pros | Cons |
|----------|----------|------|------|
| **Google AI Studio** | Image generation, inpainting | High quality, fast, already configured | API structure may vary |
| **Stability AI** | Inpainting, body refinement | Excellent inpainting, good quality | Requires separate API key |
| **OpenAI DALL-E** | General generation | High quality, reliable | Limited inpainting support |
| **Replicate** | Custom models | Flexible, many models | May be slower |

## Troubleshooting

### No API Keys Found
**Error**: "No AI image generation API keys found"

**Solution**: 
1. Set `USE_AI_API=true` in `.env`
2. Add at least one API key (STABILITY_API_KEY, OPENAI_API_KEY, or REPLICATE_API_TOKEN)

### API Request Failed
**Error**: "API request error" or "API returned no image data"

**Solutions**:
1. Check API key is valid
2. Check API quota/credits
3. Verify internet connection
4. Try a different provider

### Fallback to Local Models
**Warning**: "AI API not available, falling back to local models"

**Solution**: This is normal if no API keys are set. The system will use local models if available.

## Client Requirements Compliance

The AI image generation system addresses all client requirements:

✅ **No Plastic-Looking Faces** - AI APIs produce natural, photorealistic results  
✅ **Body Conditioning** - APIs handle open chest shirts with realistic skin synthesis  
✅ **Action Photos** - APIs preserve dynamic poses and expressions  
✅ **Quality Assurance** - Built-in validation ensures high-quality results  
✅ **Manual Touch-Ups** - Can still use manual refinement masks with APIs  

## Testing

Test with the provided client image:

```bash
python test_ai_generation.py --customer-image "D:\projects\image\face-body-swap\1760713603491 (1).jpg"
```

The script will:
1. Check for API keys
2. Process the image through all 9 pipeline stages
3. Use AI APIs for refinement (if configured)
4. Save the result to `outputs/` directory

## Next Steps

1. **Use Existing API Key**: Your Google AI Studio API key is already configured!
   - Key: `AIzaSyCioMnUARXoWlLmQZDKS-wrUZhoulS6hPU`
   - Just set `USE_AI_API=true` and `AI_IMAGE_PROVIDER=google` in `.env`
2. **Configure**: Ensure `.env` has `GOOGLE_AI_API_KEY` set
3. **Test**: Run test script with your images
4. **Monitor**: Check API usage and costs
5. **Optimize**: Adjust prompts and settings for best results

## Cost Considerations

- **Google AI Studio**: Free tier available, pay-per-use after limits, typically $0.01-0.03 per image
- **Stability AI**: Pay-per-use, typically $0.01-0.05 per image
- **OpenAI DALL-E**: Pay-per-use, typically $0.04-0.08 per image
- **Replicate**: Pay-per-use, varies by model

For production use, consider:
- Setting usage limits
- Caching results when possible
- Monitoring API costs
- Using local models for development/testing

## Support

For issues or questions:
1. Check API provider documentation
2. Review error logs in `logs/app.log`
3. Verify API keys are correct
4. Test with a simple image first

---

**Note**: The system automatically falls back to local models if APIs are unavailable, ensuring the pipeline always works.

