"""Update .env file with Stability AI key"""

import os
from pathlib import Path

env_path = Path(__file__).parent / ".env"

# Read existing .env
lines = []
if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

# Remove existing STABILITY_API_KEY and AI_IMAGE_PROVIDER lines
filtered_lines = []
for line in lines:
    if not line.strip().startswith('STABILITY_API_KEY') and not line.strip().startswith('AI_IMAGE_PROVIDER'):
        filtered_lines.append(line)

# Add new lines
filtered_lines.append('\n# Stability AI API Key (for image generation)\n')
filtered_lines.append('STABILITY_API_KEY=sk-tqwRFUrF9pd2T7FR8CpApfZgC5MD12wwkSyTcgF8QtOGamDO\n')
filtered_lines.append('AI_IMAGE_PROVIDER=stability\n')

# Write back
with open(env_path, 'w', encoding='utf-8') as f:
    f.writelines(filtered_lines)

print(f"Updated .env file: {env_path}")
print("Added:")
print("  STABILITY_API_KEY=sk-tqwRFUrF9pd2T7FR8CpApfZgC5MD12wwkSyTcgF8QtOGamDO")
print("  AI_IMAGE_PROVIDER=stability")








