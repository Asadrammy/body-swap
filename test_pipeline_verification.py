#!/usr/bin/env python3
"""
Quick test script to verify pipeline produces real images
Runs swap and validates output quality
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Pipeline Verification Test")
print("=" * 60)

# Step 1: Check dependencies
print("\n[1/5] Checking dependencies...")
try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    print(f"  ✓ Device will be: {'cuda' if torch.cuda.is_available() else 'cpu'}")
except ImportError as e:
    print(f"  ✗ torch not installed: {e}")
    print("  Run: pip install torch")
    sys.exit(1)

try:
    import diffusers
    print(f"  ✓ diffusers {diffusers.__version__}")
except ImportError as e:
    print(f"  ✗ diffusers not installed: {e}")
    print("  Run: pip install diffusers")
    sys.exit(1)

try:
    import transformers
    print(f"  ✓ transformers {transformers.__version__}")
except ImportError as e:
    print(f"  ✗ transformers not installed: {e}")
    sys.exit(1)

try:
    import accelerate
    print(f"  ✓ accelerate {accelerate.__version__}")
except ImportError as e:
    print(f"  ✗ accelerate not installed: {e}")
    sys.exit(1)

# Step 2: Check input files
print("\n[2/5] Checking input files...")
customer_image = project_root / "1760713603491 (1).jpg"
template_image = project_root / "examples" / "templates" / "individual_action_002.png"
output_path = project_root / "outputs" / "verification_test_result.png"

if not customer_image.exists():
    print(f"  ✗ Customer image not found: {customer_image}")
    sys.exit(1)
print(f"  ✓ Customer image: {customer_image}")

if not template_image.exists():
    print(f"  ✗ Template image not found: {template_image}")
    sys.exit(1)
print(f"  ✓ Template image: {template_image}")

# Ensure output directory exists
output_path.parent.mkdir(parents=True, exist_ok=True)
print(f"  ✓ Output path: {output_path}")

# Step 3: Import pipeline components
print("\n[3/5] Loading pipeline components...")
try:
    from src.api.cli import SwapCLI
    print("  ✓ SwapCLI imported")
except Exception as e:
    print(f"  ✗ Failed to import SwapCLI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Run swap
print("\n[4/5] Running swap pipeline...")
print("  This may take several minutes (models download on first use)...")
print("  Customer image:", str(customer_image))
print("  Template:", str(template_image))
print("  Output:", str(output_path))

try:
    cli = SwapCLI()
    cli.swap(
        customer_photos=[str(customer_image)],
        template=str(template_image),
        output=str(output_path),
        export_intermediate=True
    )
    print("  ✓ Swap completed!")
except Exception as e:
    print(f"  ✗ Swap failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Validate output
print("\n[5/5] Validating output image...")
import numpy as np
from PIL import Image

if not output_path.exists():
    print(f"  ✗ Output file not created: {output_path}")
    sys.exit(1)

try:
    img = Image.open(output_path)
    img_array = np.array(img)
    
    print(f"  ✓ Image loaded: {img_array.shape}")
    
    # Check for solid color
    if len(img_array.shape) == 3:
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
        std_dev = np.std(img_array)
        mean_rgb = np.mean(img_array, axis=(0, 1))
    else:
        unique_colors = len(np.unique(img_array))
        std_dev = np.std(img_array)
        mean_rgb = np.mean(img_array)
    
    print(f"  ✓ Unique colors: {unique_colors}")
    print(f"  ✓ Std deviation: {std_dev:.2f}")
    print(f"  ✓ Mean RGB: {mean_rgb}")
    
    # Validation thresholds
    if unique_colors < 50:
        print(f"  ⚠ WARNING: Low color variety (unique_colors={unique_colors} < 50)")
        print("     This may indicate a solid color issue!")
    else:
        print(f"  ✓ Color variety is good (>= 50)")
    
    if std_dev < 10.0:
        print(f"  ⚠ WARNING: Low variance (std={std_dev:.2f} < 10.0)")
        print("     This may indicate a solid color issue!")
    else:
        print(f"  ✓ Variance is good (>= 10.0)")
    
    if unique_colors >= 50 and std_dev >= 10.0:
        print("\n" + "=" * 60)
        print("✓ VERIFICATION PASSED: Image appears to be real (not solid color)")
        print("=" * 60)
        print(f"\nOutput saved to: {output_path}")
        print("\nNext steps:")
        print("  1. Visually inspect the image for photorealistic quality")
        print("  2. Check if face looks natural (not plastic)")
        print("  3. Verify skin tone matching")
        print("  4. Ensure overall image quality meets client requirements")
    else:
        print("\n" + "=" * 60)
        print("⚠ VERIFICATION FAILED: Image may be solid color")
        print("=" * 60)
        print("\nIssues detected:")
        if unique_colors < 50:
            print(f"  - Low unique colors: {unique_colors} (need >= 50)")
        if std_dev < 10.0:
            print(f"  - Low std deviation: {std_dev:.2f} (need >= 10.0)")
        print("\nCheck generator.py settings:")
        print("  - guidance_scale (currently 7.0)")
        print("  - num_inference_steps (currently 25-40)")
        print("  - strength (currently 0.65-0.95)")
        print("  - negative prompts (should exclude solid colors)")
        
except Exception as e:
    print(f"  ✗ Validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTest completed!")








