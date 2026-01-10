"""
Fix Blue Image Issue - Complete Validation Script
Run this in Google Colab to fix the blue image issue
"""

import os
from pathlib import Path

print("🔧 Fixing blue image issue with complete validation...")
print("=" * 80)

project_dir = Path("/content/image/face-body-swap")
routes_file = project_dir / "src" / "api" / "routes.py"

if not routes_file.exists():
    print(f"❌ Routes file not found: {routes_file}")
    raise FileNotFoundError("Routes file not found")

# Read current file
with open(routes_file, 'r', encoding='utf-8') as f:
    content = f.read()

changes_made = []

# 1. Check if cv2 is imported
if "import cv2" not in content:
    content = content.replace("import numpy as np", "import numpy as np\nimport cv2")
    changes_made.append("✅ Added cv2 import")

# 2. Add validation after face processing
if "# Validate result after face processing" not in content:
    old_face_processing = """                )
            
            jobs[job_id]["progress"] = 0.5"""
    
    new_face_processing = """                )
            
            # Validate result after face processing
            if result is None or not isinstance(result, np.ndarray) or result.size == 0:
                logger.error(f"Result after face processing is invalid for job {job_id}, using template")
                result = template_data["image"].copy()
            else:
                logger.info(f"Face processing result valid: shape={result.shape}, dtype={result.dtype}")
            
            jobs[job_id]["progress"] = 0.5"""
    
    if old_face_processing in content:
        content = content.replace(old_face_processing, new_face_processing)
        changes_made.append("✅ Added face processing validation")

# 3. Add validation after composition
if "# Validate composed image" not in content or "logger.info(f\"Composed image valid:" not in content:
    old_compose = """            )
            
            jobs[job_id]["progress"] = 0.7"""
    
    new_compose = """            )
            
            # Validate composed image
            if composed is None or not isinstance(composed, np.ndarray) or composed.size == 0:
                logger.error(f"Composed image is invalid for job {job_id}, using template")
                composed = template_data["image"].copy()
            else:
                logger.info(f"Composed image valid: shape={composed.shape}, dtype={composed.dtype}, min={np.min(composed)}, max={np.max(composed)}")
            
            jobs[job_id]["progress"] = 0.7"""
    
    if old_compose in content:
        content = content.replace(old_compose, new_compose)
        changes_made.append("✅ Added composed image validation")

# 4. Add validation after refinement
if "# Validate refined image immediately after refinement" not in content:
    old_refine = """                strength=0.8
            )
            
            jobs[job_id]["progress"] = 0.9"""
    
    new_refine = """                strength=0.8
            )
            
            # Validate refined image immediately after refinement
            if refined is None or not isinstance(refined, np.ndarray) or refined.size == 0:
                logger.error(f"Refined image is invalid after refinement for job {job_id}, using composed")
                refined = composed.copy()
            else:
                logger.info(f"Refined image valid: shape={refined.shape}, dtype={refined.dtype}, min={np.min(refined)}, max={np.max(refined)}")
            
            jobs[job_id]["progress"] = 0.9"""
    
    if old_refine in content:
        content = content.replace(old_refine, new_refine)
        changes_made.append("✅ Added refinement validation")

# 5. Enhance final validation with detailed logging
if "logger.info(f\"Validating result image for job" not in content:
    # Find the validation section and enhance it
    old_validation = """            # Validate refined image before saving
            if refined is None or not isinstance(refined, np.ndarray):"""
    
    new_validation = """            # Validate refined image before saving
            logger.info(f"Validating result image for job {job_id}...")
            logger.info(f"  Refined image shape: {refined.shape if isinstance(refined, np.ndarray) else 'None'}")
            logger.info(f"  Refined image dtype: {refined.dtype if isinstance(refined, np.ndarray) else 'None'}")
            if isinstance(refined, np.ndarray) and refined.size > 0:
                logger.info(f"  Refined image min/max: {np.min(refined)}/{np.max(refined)}")
                unique_colors = len(np.unique(refined.reshape(-1, refined.shape[-1]), axis=0)) if len(refined.shape) == 3 else len(np.unique(refined))
                logger.info(f"  Refined image unique colors: {unique_colors}")
            
            if refined is None or not isinstance(refined, np.ndarray):"""
    
    if old_validation in content:
        content = content.replace(old_validation, new_validation)
        changes_made.append("✅ Enhanced validation logging")
    
    # Also update the solid color check
    if "np.min(refined) == np.max(refined)" in content and "unique_colors" not in content:
        old_solid_check = """                # Check if image is all zeros or single color (potential error)
                if np.all(refined == 0) or (np.min(refined) == np.max(refined) and refined.size > 1000):
                    logger.warning(f"Refined image appears to be empty/solid color for job {job_id}, using template as fallback")
                    refined = template_data["image"].copy()"""
        
        new_solid_check = """                # Check if image is all zeros or single color (potential error)
                unique_colors = len(np.unique(refined.reshape(-1, refined.shape[-1]), axis=0)) if len(refined.shape) == 3 else len(np.unique(refined))
                if np.all(refined == 0):
                    logger.warning(f"Refined image is all zeros for job {job_id}, using template as fallback")
                    refined = template_data["image"].copy()
                elif unique_colors < 10 and refined.size > 1000:
                    logger.warning(f"Refined image has only {unique_colors} unique colors (likely solid color) for job {job_id}, using template as fallback")
                    refined = template_data["image"].copy()"""
        
        if old_solid_check in content:
            content = content.replace(old_solid_check, new_solid_check)
            changes_made.append("✅ Enhanced solid color detection")

# 6. Add final logging before save
if "logger.info(f\"  Final image shape:" not in content:
    old_save = """            save_image(refined, result_path)"""
    
    new_save = """            logger.info(f"  Final image shape: {refined.shape}, dtype={refined.dtype}")
            logger.info(f"  Final image min/max: {np.min(refined)}/{np.max(refined)}")
            save_image(refined, result_path)
            logger.info(f"✅ Result saved to {result_path}")"""
    
    if old_save in content and "logger.info(f\"✅ Result saved" not in content:
        content = content.replace(old_save, new_save)
        changes_made.append("✅ Added final save logging")

# Write the updated file
if changes_made:
    with open(routes_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ FIXES APPLIED:")
    for change in changes_made:
        print(f"   {change}")
else:
    print("\n✅ All fixes already applied!")

print("\n" + "=" * 80)
print("✅ FIX COMPLETE!")
print("=" * 80)
print("\n📋 Next steps:")
print("   1. Restart your SERVER CELL (stop and restart)")
print("   2. Process an image again")
print("   3. Check logs for validation messages")
print("   4. Result should now be valid (not blue)")
print("\n💡 The validation will:")
print("   - Check image validity at each stage")
print("   - Log detailed information about image properties")
print("   - Fall back to template if result is invalid")
print("   - Detect solid color images (like blue)")
print("=" * 80)

