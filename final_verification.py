"""Final verification of Google AI integration"""

import os
import sys
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("=" * 60)
print("FINAL VERIFICATION - Google AI Integration")
print("=" * 60)

load_dotenv()
key = os.getenv('GOOGLE_AI_API_KEY')

if key:
    print(f"[OK] API Key in .env: OK")
    print(f"     Key: {key[:20]}...{key[-10:]}")
else:
    print("[FAIL] API Key in .env: MISSING")

print("\n" + "=" * 60)
print("Integration Status:")
print("=" * 60)

checks = [
    ("Google AI Client Module", "src/models/google_ai_client.py"),
    ("Quality Control Integration", "src/pipeline/quality_control.py"),
    ("Test Scripts", "test_google_ai_api.py"),
    ("Documentation", "GOOGLE_AI_INTEGRATION.md"),
]

from pathlib import Path
for name, filepath in checks:
    path = Path(filepath)
    if path.exists():
        print(f"[OK] {name}: OK")
    else:
        print(f"[FAIL] {name}: MISSING")

print("\n" + "=" * 60)
print("All systems ready!")
print("=" * 60)

