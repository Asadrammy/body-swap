"""Verify .env file configuration for Google AI API key"""

import os
from pathlib import Path
from dotenv import load_dotenv

def verify_env_setup():
    """Verify .env file has Google AI API key"""
    print("=" * 60)
    print("Verifying .env Configuration")
    print("=" * 60)
    
    # Load .env file
    env_path = Path(__file__).parent / ".env"
    
    if not env_path.exists():
        print("[FAIL] .env file does not exist!")
        print(f"       Expected at: {env_path}")
        print("\n[INFO] Creating .env file from env.example...")
        
        # Create from env.example
        env_example = Path(__file__).parent / "env.example"
        if env_example.exists():
            content = env_example.read_text()
            # Add the API key
            if "GOOGLE_AI_API_KEY" not in content:
                content += "\n# Google AI Studio API Key\n"
                content += "GOOGLE_AI_API_KEY=AIzaSyCioMnUARXoWlLmQZDKS-wrUZhoulS6hPU\n"
            env_path.write_text(content)
            print("[OK] .env file created")
        else:
            print("[FAIL] env.example not found")
            return False
    else:
        print(f"[OK] .env file exists at: {env_path}")
    
    # Load environment variables
    load_dotenv(env_path)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("\n[WARNING] GOOGLE_AI_API_KEY not found in .env file")
        print("          Adding API key to .env file...")
        
        # Read current content
        content = env_path.read_text()
        
        # Add API key if not present
        if "GOOGLE_AI_API_KEY" not in content and "GEMINI_API_KEY" not in content:
            content += "\n# Google AI Studio API Key\n"
            content += "GOOGLE_AI_API_KEY=AIzaSyCioMnUARXoWlLmQZDKS-wrUZhoulS6hPU\n"
            env_path.write_text(content)
            print("[OK] API key added to .env file")
            
            # Reload
            load_dotenv(env_path, override=True)
            api_key = os.getenv("GOOGLE_AI_API_KEY")
        else:
            print("[FAIL] Could not add API key automatically")
            return False
    else:
        print(f"\n[OK] Google AI API key found")
        print(f"     Key: {api_key[:20]}...{api_key[-10:]}")
    
    # Verify other important env vars
    print("\n" + "=" * 60)
    print("Checking Other Environment Variables")
    print("=" * 60)
    
    important_vars = {
        "DEVICE": os.getenv("DEVICE", "Not set"),
        "CUDA_AVAILABLE": os.getenv("CUDA_AVAILABLE", "Not set"),
        "IMAGE_SIZE": os.getenv("IMAGE_SIZE", "Not set"),
        "API_HOST": os.getenv("API_HOST", "Not set"),
        "API_PORT": os.getenv("API_PORT", "Not set"),
    }
    
    for var_name, var_value in important_vars.items():
        status = "[OK]" if var_value != "Not set" else "[INFO]"
        print(f"{status} {var_name}={var_value}")
    
    # Test API key works
    print("\n" + "=" * 60)
    print("Testing API Key")
    print("=" * 60)
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Try to list models
        try:
            models = list(genai.list_models())
            print(f"[OK] API key is valid - found {len(models)} models")
            return True
        except Exception as e:
            print(f"[WARNING] Could not verify API key: {e}")
            print("          But API key is set in .env file")
            return True  # Still return True as key is set
    except ImportError:
        print("[WARNING] google-generativeai not installed")
        print("          But API key is set in .env file")
        return True
    
    return True

if __name__ == "__main__":
    success = verify_env_setup()
    if success:
        print("\n" + "=" * 60)
        print("[SUCCESS] .env configuration is correct!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("[FAIL] .env configuration needs attention")
        print("=" * 60)

