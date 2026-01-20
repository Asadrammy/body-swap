"""List all available Google AI models and their capabilities"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

api_key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: API key not found")
    sys.exit(1)

print("=" * 80)
print("Listing Available Google AI Models")
print("=" * 80)

try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    print("\nFetching models...")
    models = list(genai.list_models())
    
    print(f"\nTotal models found: {len(models)}")
    print("\n" + "=" * 80)
    print("Models with Image Generation Capabilities")
    print("=" * 80)
    
    image_models = []
    for model in models:
        methods = model.supported_generation_methods
        if 'generateContent' in methods or 'generateImages' in methods or 'imagen' in model.name.lower():
            image_models.append(model)
            print(f"\nModel: {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Description: {model.description[:100] if model.description else 'N/A'}")
            print(f"  Supported Methods: {methods}")
            print(f"  Input Token Limit: {getattr(model, 'input_token_limit', 'N/A')}")
            print(f"  Output Token Limit: {getattr(model, 'output_token_limit', 'N/A')}")
    
    if not image_models:
        print("\n[WARNING] No image generation models found!")
        print("\nAll available models:")
        for model in models[:20]:  # Show first 20
            print(f"  - {model.name} ({', '.join(model.supported_generation_methods)})")
    
    print("\n" + "=" * 80)
    print("Testing Image Generation with Available Models")
    print("=" * 80)
    
    # Try to generate image with any model that supports generateContent
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            print(f"\nTrying model: {model.name}")
            try:
                gen_model = genai.GenerativeModel(model.name)
                
                # Try with image generation prompt
                prompt = "Generate a photorealistic portrait image of a person"
                response = gen_model.generate_content(prompt)
                
                print(f"  Response type: {type(response)}")
                if hasattr(response, 'candidates'):
                    print(f"  Candidates: {len(response.candidates)}")
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content'):
                            parts = candidate.content.parts
                            print(f"  Content parts: {len(parts)}")
                            for part in parts:
                                print(f"    Part type: {type(part)}")
                                if hasattr(part, 'inline_data'):
                                    print("    [SUCCESS] Found image data!")
                                    import base64
                                    from PIL import Image
                                    import io
                                    img_data = part.inline_data.data
                                    img_bytes = base64.b64decode(img_data)
                                    img = Image.open(io.BytesIO(img_bytes))
                                    print(f"    Image size: {img.size}")
                                    img.save(f"generated_{model.name.replace('/', '_')}.png")
                                    print(f"    Saved to: generated_{model.name.replace('/', '_')}.png")
                                    break
                
            except Exception as e:
                print(f"  Error: {e}")
                if "not supported" not in str(e).lower():
                    import traceback
                    traceback.print_exc()
            break  # Only test first model for now
    
except ImportError:
    print("google-generativeai not installed")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("\nBased on the results:")
print("1. If no image generation models are listed, Imagen may require special access")
print("2. You may need to use a different API (Vertex AI) for Imagen")
print("3. Consider using alternative providers (Stability AI, OpenAI) for image generation")
print("=" * 80)








