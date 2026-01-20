"""Check .env configuration and test image generation"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"[OK] Loaded .env file from: {env_path}")
else:
    print(f"[WARN] .env file not found at: {env_path}")
    print("   Loading from environment variables...")

print("\n" + "=" * 80)
print("Environment Configuration Check")
print("=" * 80)

# Check required variables
required_vars = {
    "USE_AI_API": os.getenv("USE_AI_API", "Not set"),
    "GOOGLE_AI_API_KEY": os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY") or "Not set",
    "AI_IMAGE_PROVIDER": os.getenv("AI_IMAGE_PROVIDER", "Not set"),
}

print("\nRequired Configuration:")
for var_name, var_value in required_vars.items():
    if var_value == "Not set":
        status = "[FAIL] NOT SET"
    elif var_name == "GOOGLE_AI_API_KEY" and var_value != "Not set":
        # Mask the API key
        masked = var_value[:20] + "..." + var_value[-10:] if len(var_value) > 30 else "***"
        status = f"[OK] Set ({masked})"
    else:
        status = f"[OK] Set ({var_value})"
    print(f"  {var_name}: {status}")

# Check optional variables
optional_vars = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "STABILITY_API_KEY": os.getenv("STABILITY_API_KEY") or os.getenv("STABILITY_AI_API_KEY"),
    "REPLICATE_API_TOKEN": os.getenv("REPLICATE_API_TOKEN"),
}

print("\nOptional Configuration (Other Providers):")
for var_name, var_value in optional_vars.items():
    if var_value:
        masked = var_value[:10] + "..." + var_value[-5:] if len(var_value) > 15 else "***"
        print(f"  {var_name}: [OK] Set ({masked})")
    else:
        print(f"  {var_name}: [INFO] Not set")

# Validate configuration
print("\n" + "=" * 80)
print("Configuration Validation")
print("=" * 80)

use_ai_api = os.getenv("USE_AI_API", "false").lower() == "true"
google_key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not use_ai_api:
    print("[FAIL] USE_AI_API is not set to 'true'")
    print("   Recommendation: Set USE_AI_API=true in .env")
    valid = False
else:
    print("[OK] USE_AI_API is enabled")

if not google_key:
    print("[FAIL] GOOGLE_AI_API_KEY is not set")
    print("   Recommendation: Set GOOGLE_AI_API_KEY=your_key in .env")
    valid = False
else:
    print(f"[OK] GOOGLE_AI_API_KEY is set (length: {len(google_key)})")
    # Check if it's the known key
    if "AIzaSyCioMnUARXoWlLmQZDKS-wrUZhoulS6hPU" in google_key:
        print("  [OK] Using known Google AI Studio API key")

provider = os.getenv("AI_IMAGE_PROVIDER", "google").lower()
if provider not in ["google", "openai", "stability", "replicate"]:
    print(f"[WARN] AI_IMAGE_PROVIDER is set to '{provider}' (should be: google, openai, stability, or replicate)")
else:
    print(f"[OK] AI_IMAGE_PROVIDER is set to '{provider}'")

# Test AI Image Generator
print("\n" + "=" * 80)
print("Testing AI Image Generator")
print("=" * 80)

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from src.models.ai_image_generator import AIImageGenerator
    
    print("Initializing AI Image Generator...")
    generator = AIImageGenerator()
    
    if generator._has_api_key():
        print("[OK] AI Image Generator initialized successfully")
        available = [p for p in ['google', 'openai', 'stability', 'replicate'] if getattr(generator, f'{p}_api_key', None)]
        print(f"  Available providers: {', '.join(available)}")
        print(f"  Selected provider: {generator.provider}")
        
        # Test with a simple image
        print("\nTesting image generation...")
        try:
            import numpy as np
            from PIL import Image
            
            # Create a simple test image
            test_image = np.ones((512, 512, 3), dtype=np.uint8) * 128
            test_pil = Image.fromarray(test_image)
            
            print("  Creating test image (512x512 gray)...")
            print("  Attempting refinement with Google AI...")
            
            # Try a simple refinement
            result = generator.refine(
                image=test_image,
                prompt="photorealistic portrait, natural skin texture, high quality",
                mask=None,
                negative_prompt="blurry, low quality",
                strength=0.7,
                num_inference_steps=30
            )
            
            if result is not None and not np.array_equal(result, test_image):
                print("  [OK] Image generation test successful!")
                print(f"  Result shape: {result.shape}")
                print(f"  Result dtype: {result.dtype}")
                
                # Check if result is valid
                if len(result.shape) == 3 and result.shape[2] == 3:
                    unique_colors = len(np.unique(result.reshape(-1, 3), axis=0))
                    std_dev = np.std(result)
                    print(f"  Unique colors: {unique_colors}")
                    print(f"  Standard deviation: {std_dev:.2f}")
                    
                    if unique_colors > 20 and std_dev > 8.0:
                        print("  [OK] Generated image is valid (not solid color)")
                    else:
                        print("  [WARN] Generated image may be invalid (low color variety)")
                else:
                    print("  [WARN] Generated image has unexpected shape")
            else:
                print("  [WARN] Image generation returned original image (may indicate API issue)")
                print("  This could mean:")
                print("    - API key is invalid")
                print("    - API quota exceeded")
                print("    - Network issue")
                print("    - API endpoint changed")
                
        except Exception as e:
            print(f"  [FAIL] Image generation test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[FAIL] No API keys found in AI Image Generator")
        print("   Please set at least one API key in .env")
        
except Exception as e:
    print(f"[FAIL] Failed to initialize AI Image Generator: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

if use_ai_api and google_key:
    print("[OK] Configuration looks good!")
    print("  - USE_AI_API is enabled")
    print("  - Google AI API key is set")
    print("\nYou can now test with:")
    print('  python test_ai_generation.py --customer-image "D:\\projects\\image\\face-body-swap\\1760713603491 (1).jpg"')
else:
    print("[WARN] Configuration needs attention:")
    if not use_ai_api:
        print("  - Set USE_AI_API=true in .env")
    if not google_key:
        print("  - Set GOOGLE_AI_API_KEY=your_key in .env")

print("\n" + "=" * 80)

