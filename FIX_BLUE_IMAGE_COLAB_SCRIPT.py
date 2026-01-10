"""
🚀 COMPLETE FIX SCRIPT FOR BLUE IMAGE ISSUE
===========================================
Paste this entire script into a Colab cell and run it.
It will apply all validation fixes to prevent blue images.
"""

import os
from pathlib import Path

print("🔧 APPLYING ALL FIXES FOR BLUE IMAGE ISSUE...")
print("=" * 80)

project_dir = Path("/content/image/face-body-swap")
if not project_dir.exists():
    print(f"❌ Project directory not found: {project_dir}")
    print("   Make sure your project is extracted to /content/image/face-body-swap")
    raise FileNotFoundError("Project directory not found")

changes_made = []

# ============================================================================
# FIX 1: Fix generator.py - Add solid color detection
# ============================================================================
generator_file = project_dir / "src" / "models" / "generator.py"
if generator_file.exists():
    print("\n📝 Fixing generator.py...")
    with open(generator_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Find and replace the return statement in refine method
    old_return = """            # Convert back to numpy array
            result_array = np.array(result).astype(np.uint8)
            
            return result_array
            
        except Exception as e:
            logger.error(f"Image refinement error: {e}")
            return image"""
    
    new_return = """            # Convert back to numpy array
            result_array = np.array(result).astype(np.uint8)
            
            # Validate result - check if it's a solid color (indicates failure)
            if result_array.size > 0:
                unique_colors = len(np.unique(result_array.reshape(-1, result_array.shape[-1]), axis=0)) if len(result_array.shape) == 3 else len(np.unique(result_array))
                std_dev = np.std(result_array)
                # If result is solid color (very few unique colors and low std dev), return original
                if unique_colors < 10 and std_dev < 5.0:
                    logger.warning(f"Generator returned solid color image (unique_colors={unique_colors}, std={std_dev:.2f}), using original")
                    return image
            
            return result_array
            
        except Exception as e:
            logger.error(f"Image refinement error: {e}")
            return image"""
    
    if old_return in content:
        content = content.replace(old_return, new_return)
        changes_made.append("✅ Fixed generator.py - Added solid color detection")
    elif "# Validate result - check if it's a solid color" in content:
        print("   ℹ️  Generator fix already applied")
    else:
        print("   ⚠️  Could not find exact pattern in generator.py")
    
    if content != original_content:
        with open(generator_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("   ✅ generator.py updated")
else:
    print(f"   ⚠️  generator.py not found: {generator_file}")

# ============================================================================
# FIX 2: Fix refiner.py - Add validation for global refinement
# ============================================================================
refiner_file = project_dir / "src" / "pipeline" / "refiner.py"
if refiner_file.exists():
    print("\n📝 Fixing refiner.py - Global refinement validation...")
    with open(refiner_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix global refinement validation
    old_global = """                working = self.generator.refine(
                    image=working,
                    prompt=prompt,
                    mask=refinement_mask,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    num_inference_steps=inference_steps
                )
                logger.info("Global refinement pass completed")"""
    
    new_global = """                refined_global = self.generator.refine(
                    image=working,
                    prompt=prompt,
                    mask=refinement_mask,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    num_inference_steps=inference_steps
                )
                # Validate refined result - if it's invalid or solid color, keep original
                if refined_global is not None and isinstance(refined_global, np.ndarray) and refined_global.size > 0:
                    unique_colors = len(np.unique(refined_global.reshape(-1, refined_global.shape[-1]), axis=0)) if len(refined_global.shape) == 3 else len(np.unique(refined_global))
                    std_dev = np.std(refined_global)
                    if unique_colors >= 10 or std_dev >= 5.0:
                        working = refined_global
                        logger.info("Global refinement pass completed")
                    else:
                        logger.warning(f"Global refinement returned solid color (unique_colors={unique_colors}, std={std_dev:.2f}), keeping original")
                else:
                    logger.warning("Global refinement returned invalid result, keeping original")"""
    
    if old_global in content:
        content = content.replace(old_global, new_global)
        changes_made.append("✅ Fixed refiner.py - Added global refinement validation")
    elif "# Validate refined result - if it's invalid or solid color" in content and "refined_global" in content:
        print("   ℹ️  Global refinement fix already applied")
    else:
        print("   ⚠️  Could not find exact pattern for global refinement")
    
    # Fix region refinement validation
    old_region = """                    working = self.generator.refine(
                        image=working,
                        prompt=region_prompt,
                        mask=mask,
                        negative_prompt=negative_prompt,
                        strength=region_strength,
                        num_inference_steps=inference_steps
                    )
                    logger.info(f"Refined region '{region_name}' with strength {region_strength}")"""
    
    new_region = """                    refined_region = self.generator.refine(
                        image=working,
                        prompt=region_prompt,
                        mask=mask,
                        negative_prompt=negative_prompt,
                        strength=region_strength,
                        num_inference_steps=inference_steps
                    )
                    # Validate refined result - if it's invalid or solid color, keep original
                    if refined_region is not None and isinstance(refined_region, np.ndarray) and refined_region.size > 0:
                        unique_colors = len(np.unique(refined_region.reshape(-1, refined_region.shape[-1]), axis=0)) if len(refined_region.shape) == 3 else len(np.unique(refined_region))
                        std_dev = np.std(refined_region)
                        if unique_colors >= 10 or std_dev >= 5.0:
                            working = refined_region
                            logger.info(f"Refined region '{region_name}' with strength {region_strength}")
                        else:
                            logger.warning(f"Refined region '{region_name}' is solid color (unique_colors={unique_colors}, std={std_dev:.2f}), keeping original")
                    else:
                        logger.warning(f"Refined region '{region_name}' is invalid, keeping original")"""
    
    if old_region in content:
        content = content.replace(old_region, new_region)
        changes_made.append("✅ Fixed refiner.py - Added region refinement validation")
    elif "# Validate refined result - if it's invalid or solid color" in content and "refined_region" in content:
        print("   ℹ️  Region refinement fix already applied")
    else:
        print("   ⚠️  Could not find exact pattern for region refinement")
    
    # Fix face refinement validation
    old_face = """            refined = self.generator.refine(
                image=image,
                prompt=prompt,
                mask=mask,
                strength=0.55,  # Reduced from 0.7 to preserve natural texture
                num_inference_steps=30  # More steps for better quality
            )
            
            # Post-process to enhance natural appearance
            refined = self._post_process_face(refined, mask, image)
            
            return refined"""
    
    new_face = """            refined = self.generator.refine(
                image=image,
                prompt=prompt,
                mask=mask,
                strength=0.55,  # Reduced from 0.7 to preserve natural texture
                num_inference_steps=30  # More steps for better quality
            )
            
            # Validate refined result - if it's invalid or solid color, return original
            if refined is not None and isinstance(refined, np.ndarray) and refined.size > 0:
                unique_colors = len(np.unique(refined.reshape(-1, refined.shape[-1]), axis=0)) if len(refined.shape) == 3 else len(np.unique(refined))
                std_dev = np.std(refined)
                if unique_colors >= 10 and std_dev >= 5.0:
                    # Post-process to enhance natural appearance
                    refined = self._post_process_face(refined, mask, image)
                    return refined
                else:
                    logger.warning(f"Face refinement returned solid color (unique_colors={unique_colors}, std={std_dev:.2f}), using original")
                    return image
            else:
                logger.warning("Face refinement returned invalid result, using original")
                return image"""
    
    if old_face in content:
        content = content.replace(old_face, new_face)
        changes_made.append("✅ Fixed refiner.py - Added face refinement validation")
    elif "# Validate refined result - if it's invalid or solid color" in content and "Face refinement returned solid color" in content:
        print("   ℹ️  Face refinement fix already applied")
    else:
        print("   ⚠️  Could not find exact pattern for face refinement")
    
    if content != original_content:
        with open(refiner_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("   ✅ refiner.py updated")
else:
    print(f"   ⚠️  refiner.py not found: {refiner_file}")

# ============================================================================
# FIX 3: Fix routes.py - Enhanced solid color detection
# ============================================================================
routes_file = project_dir / "src" / "api" / "routes.py"
if routes_file.exists():
    print("\n📝 Fixing routes.py - Enhanced solid color detection...")
    with open(routes_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Enhanced solid color detection
    old_solid = """                # Check if image is all zeros or single color (potential error)
                unique_colors = len(np.unique(refined.reshape(-1, refined.shape[-1]), axis=0)) if len(refined.shape) == 3 else len(np.unique(refined))
                if np.all(refined == 0):
                    logger.warning(f"Refined image is all zeros for job {job_id}, using template as fallback")
                    refined = template_data["image"].copy()
                elif unique_colors < 10 and refined.size > 1000:
                    logger.warning(f"Refined image has only {unique_colors} unique colors (likely solid color) for job {job_id}, using template as fallback")
                    refined = template_data["image"].copy()"""
    
    new_solid = """                # Check if image is all zeros or single color (potential error)
                unique_colors = len(np.unique(refined.reshape(-1, refined.shape[-1]), axis=0)) if len(refined.shape) == 3 else len(np.unique(refined))
                std_dev = np.std(refined)
                mean_rgb = np.mean(refined, axis=(0, 1)) if len(refined.shape) == 3 else np.mean(refined)
                
                if np.all(refined == 0):
                    logger.warning(f"Refined image is all zeros for job {job_id}, using template as fallback")
                    refined = template_data["image"].copy()
                elif unique_colors < 10 and refined.size > 1000:
                    logger.warning(f"Refined image has only {unique_colors} unique colors (likely solid color) for job {job_id}, using template as fallback")
                    logger.warning(f"  Color stats - Mean RGB: {mean_rgb}, Std RGB: {std_dev:.2f}")
                    refined = template_data["image"].copy()
                elif std_dev < 5.0 and refined.size > 1000:
                    # Very low standard deviation indicates solid/near-solid color
                    logger.warning(f"Refined image appears to be solid/near-solid color for job {job_id}, using template as fallback")
                    logger.warning(f"  Color stats - Mean RGB: {mean_rgb}, Std RGB: {std_dev:.2f}")
                    refined = template_data["image"].copy()"""
    
    if old_solid in content:
        content = content.replace(old_solid, new_solid)
        changes_made.append("✅ Fixed routes.py - Enhanced solid color detection with std dev check")
    elif "std_dev < 5.0" in content and "Color stats - Mean RGB" in content:
        print("   ℹ️  Enhanced solid color detection already applied")
    else:
        print("   ⚠️  Could not find exact pattern for solid color detection")
    
    if content != original_content:
        with open(routes_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("   ✅ routes.py updated")
else:
    print(f"   ⚠️  routes.py not found: {routes_file}")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("🔍 VERIFICATION")
print("=" * 80)

verification_passed = True

# Verify generator.py
if generator_file.exists():
    with open(generator_file, 'r', encoding='utf-8') as f:
        gen_content = f.read()
    if "# Validate result - check if it's a solid color" in gen_content:
        print("✅ generator.py - Solid color validation: FOUND")
    else:
        print("❌ generator.py - Solid color validation: NOT FOUND")
        verification_passed = False

# Verify refiner.py
if refiner_file.exists():
    with open(refiner_file, 'r', encoding='utf-8') as f:
        ref_content = f.read()
    checks = [
        ("refined_global", "Global refinement validation"),
        ("refined_region", "Region refinement validation"),
        ("Face refinement returned solid color", "Face refinement validation")
    ]
    for check, name in checks:
        if check in ref_content:
            print(f"✅ refiner.py - {name}: FOUND")
        else:
            print(f"❌ refiner.py - {name}: NOT FOUND")
            verification_passed = False

# Verify routes.py
if routes_file.exists():
    with open(routes_file, 'r', encoding='utf-8') as f:
        routes_content = f.read()
    if "std_dev < 5.0" in routes_content and "Color stats - Mean RGB" in routes_content:
        print("✅ routes.py - Enhanced solid color detection: FOUND")
    else:
        print("❌ routes.py - Enhanced solid color detection: NOT FOUND")
        verification_passed = False

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
if changes_made:
    print("✅ FIXES APPLIED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📋 Changes made:")
    for i, change in enumerate(changes_made, 1):
        print(f"   {i}. {change}")
else:
    print("✅ ALL FIXES ALREADY APPLIED!")
    print("=" * 80)

if verification_passed:
    print("\n✅ VERIFICATION: ALL CHECKS PASSED")
else:
    print("\n⚠️  VERIFICATION: SOME CHECKS FAILED - Please review manually")

print("\n" + "=" * 80)
print("📋 NEXT STEPS:")
print("=" * 80)
print("   1. ⏹️  STOP your current server cell (if running)")
print("   2. 🔄 RESTART your server cell")
print("   3. 🖼️  Process an image again")
print("   4. 📊 Check logs for:")
print("      - 'Generator returned solid color image' (if generator fails)")
print("      - 'Global refinement returned solid color' (if refinement fails)")
print("      - 'Refined image appears to be solid/near-solid color' (final check)")
print("      - 'Color stats - Mean RGB: [...], Std RGB: X.XX' (color analysis)")
print("   5. ✅ Result should now be valid (not blue) - falls back to template if invalid")
print("\n💡 The fixes will:")
print("   ✅ Detect solid color images at generator level")
print("   ✅ Validate refinement results before using them")
print("   ✅ Enhanced detection using standard deviation")
print("   ✅ Fall back to template image when invalid")
print("=" * 80)
print("\n🎉 Ready to test! Restart your server and process an image.")

