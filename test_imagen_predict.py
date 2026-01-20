"""Test Imagen models using predict method"""

import os
import sys
import requests
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io

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
print("Testing Imagen Models with PREDICT Method")
print("=" * 80)

# Imagen models that use 'predict' method
imagen_models = [
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-4.0-generate-preview-06-06",
]

# Also test Gemini image models
gemini_image_models = [
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview",
    "gemini-2.0-flash-exp-image-generation",
]

print("\n[Test 1] Testing Imagen models with predict endpoint")
print("-" * 80)

for model_name in imagen_models:
    print(f"\nTesting: {model_name}")
    
    # Try Vertex AI style predict endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:predict"
    
    # Payload for predict method
    payload = {
        "instances": [{
            "prompt": "A photorealistic portrait of a person with natural skin texture, high quality, professional photography"
        }],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "1:1"
        }
    }
    
    try:
        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}
        
        response = requests.post(url, headers=headers, json=payload, params=params, timeout=60)
        
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"  [SUCCESS] Response received!")
            print(f"  Response keys: {list(result.keys())}")
            
            # Try to extract image
            if "predictions" in result:
                for pred in result["predictions"]:
                    if "bytesBase64Encoded" in pred:
                        img_bytes = base64.b64decode(pred["bytesBase64Encoded"])
                        img = Image.open(io.BytesIO(img_bytes))
                        print(f"  [SUCCESS] Generated image: {img.size}")
                        img.save(f"imagen_{model_name.replace('/', '_')}.png")
                        print(f"  Saved to: imagen_{model_name.replace('/', '_')}.png")
                        break
            else:
                print(f"  Response: {json.dumps(result, indent=2)[:500]}")
        else:
            print(f"  Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Message: {error_data.get('error', {}).get('message', 'Unknown')}")
            except:
                print(f"  Response: {response.text[:300]}")
                
    except Exception as e:
        print(f"  Exception: {e}")

print("\n" + "=" * 80)
print("[Test 2] Testing Gemini Image Generation Models")
print("=" * 80)

try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    for model_name in gemini_image_models:
        print(f"\nTesting: {model_name}")
        try:
            model = genai.GenerativeModel(model_name)
            
            # Try image generation prompt
            prompt = "Generate a photorealistic portrait image of a person with natural skin texture"
            response = model.generate_content(prompt)
            
            print(f"  Response received")
            if hasattr(response, 'candidates'):
                for candidate in response.candidates:
                    if hasattr(candidate, 'content'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data'):
                                print("  [SUCCESS] Found image data!")
                                try:
                                    # Try different data formats
                                    img_data = part.inline_data.data
                                    if isinstance(img_data, str):
                                        img_bytes = base64.b64decode(img_data)
                                    else:
                                        img_bytes = img_data
                                    
                                    img = Image.open(io.BytesIO(img_bytes))
                                    print(f"  Image size: {img.size}, mode: {img.mode}")
                                    output_name = f"gemini_{model_name.replace('/', '_')}.png"
                                    img.save(output_name)
                                    print(f"  Saved to: {output_name}")
                                except Exception as e:
                                    print(f"  Error reading image: {e}")
                                    print(f"  Data type: {type(img_data)}, length: {len(img_data) if hasattr(img_data, '__len__') else 'N/A'}")
                                    
        except Exception as e:
            print(f"  Error: {e}")
            if "not found" not in str(e).lower() and "not supported" not in str(e).lower():
                import traceback
                traceback.print_exc()
                
except ImportError:
    print("google-generativeai not installed")
except Exception as e:
    print(f"SDK test failed: {e}")

print("\n" + "=" * 80)
print("Test Complete")
print("=" * 80)








