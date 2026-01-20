"""Test Google Imagen API directly to find correct endpoint"""

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

# Get API key
api_key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: API key not found in .env")
    sys.exit(1)

print(f"API Key: {api_key[:20]}...{api_key[-10:]}")
print("\n" + "=" * 80)
print("Testing Google Imagen API Endpoints")
print("=" * 80)

# Test different endpoint formats based on documentation
test_configs = [
    {
        "name": "Imagen via Gemini API (generateContent)",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-fast-generate-001:generateContent",
        "payload": {
            "contents": [{
                "parts": [{
                    "text": "A photorealistic portrait of a person with natural skin texture, high quality"
                }]
            }]
        }
    },
    {
        "name": "Imagen direct generateImages",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-fast-generate-001:generateImages",
        "payload": {
            "prompt": "A photorealistic portrait of a person with natural skin texture, high quality",
            "numberOfImages": 1
        }
    },
    {
        "name": "Imagen via ai.google.dev",
        "url": "https://ai.google.dev/api/rest/v1beta/models/imagen-4.0-fast-generate-001:generateImages",
        "payload": {
            "prompt": "A photorealistic portrait of a person with natural skin texture, high quality",
            "numberOfImages": 1
        }
    }
]

for i, config in enumerate(test_configs, 1):
    print(f"\n[Test {i}] {config['name']}")
    print(f"URL: {config['url']}")
    print("-" * 80)
    
    try:
        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}
        
        response = requests.post(
            config['url'],
            headers=headers,
            json=config['payload'],
            params=params,
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("[SUCCESS] Request successful!")
            result = response.json()
            print(f"Response keys: {list(result.keys())}")
            
            # Try different response structures
            image_found = False
            
            # Check for generatedImages
            if "generatedImages" in result:
                images = result["generatedImages"]
                print(f"Found {len(images)} generated images")
                if len(images) > 0:
                    img_data = images[0]
                    if "bytesBase64Encoded" in img_data:
                        img_bytes = base64.b64decode(img_data["bytesBase64Encoded"])
                        img = Image.open(io.BytesIO(img_bytes))
                        print(f"[SUCCESS] Generated image: {img.size}, mode: {img.mode}")
                        output_path = f"test_imagen_{i}.png"
                        img.save(output_path)
                        print(f"Saved to: {output_path}")
                        image_found = True
            
            # Check for candidates (Gemini style)
            if not image_found and "candidates" in result:
                print("Found candidates in response")
                for candidate in result["candidates"]:
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            if "inlineData" in part:
                                img_data = part["inlineData"]["data"]
                                img_bytes = base64.b64decode(img_data)
                                img = Image.open(io.BytesIO(img_bytes))
                                print(f"[SUCCESS] Generated image: {img.size}")
                                output_path = f"test_imagen_{i}.png"
                                img.save(output_path)
                                print(f"Saved to: {output_path}")
                                image_found = True
                                break
            
            if not image_found:
                print("Response structure:")
                print(json.dumps(result, indent=2)[:1000])
                
        elif response.status_code == 404:
            print("[404] Endpoint not found - this endpoint may not exist")
        elif response.status_code == 401:
            print("[401] Unauthorized - check API key")
        elif response.status_code == 403:
            print("[403] Forbidden - may need special access or quota")
            try:
                error_data = response.json()
                print(f"Error message: {error_data.get('error', {}).get('message', 'Unknown')}")
            except:
                print(f"Response: {response.text[:500]}")
        else:
            print(f"[ERROR] Status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)[:500]}")
            except:
                print(f"Response: {response.text[:500]}")
                
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        import traceback
        traceback.print_exc()

# Test with SDK
print("\n" + "=" * 80)
print("Testing with google-generativeai SDK")
print("=" * 80)

try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    # Try to use Imagen model directly
    model_name = "imagen-4.0-fast-generate-001"
    print(f"\nTrying to use model: {model_name}")
    
    try:
        model = genai.GenerativeModel(model_name)
        print("Model object created")
        
        # Try generate_content
        response = model.generate_content(
            "Generate a photorealistic portrait of a person with natural skin texture",
            generation_config={
                "temperature": 0.7,
            }
        )
        
        print(f"Response received: {type(response)}")
        
        # Check response structure
        if hasattr(response, 'candidates'):
            print(f"Candidates: {len(response.candidates)}")
            for candidate in response.candidates:
                if hasattr(candidate, 'content'):
                    parts = candidate.content.parts
                    print(f"Content parts: {len(parts)}")
                    for part in parts:
                        print(f"  Part type: {type(part)}")
                        if hasattr(part, 'inline_data'):
                            print("  Found inline_data!")
                            img_data = part.inline_data.data
                            img_bytes = base64.b64decode(img_data)
                            img = Image.open(io.BytesIO(img_bytes))
                            print(f"  [SUCCESS] Generated image: {img.size}")
                            img.save("test_imagen_sdk.png")
                            print("  Saved to: test_imagen_sdk.png")
        
    except Exception as e:
        print(f"Error using model: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError:
    print("google-generativeai not installed")
except Exception as e:
    print(f"SDK test failed: {e}")

print("\n" + "=" * 80)
print("Test Complete")
print("=" * 80)








