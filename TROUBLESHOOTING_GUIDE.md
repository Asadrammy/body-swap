# Troubleshooting Guide

This guide helps you troubleshoot issues in the face-body swap workflow. Each issue includes symptoms, causes, and step-by-step solutions.

## Quick Reference

| Issue | Quick Fix | Full Solution |
|-------|-----------|---------------|
| Plastic-looking face | Reduce face refinement strength to 0.5 | See "Plastic-Looking Face" |
| Wrong skin tone (open chest) | Check skin profile extraction | See "Open Chest Skin Issues" |
| Body size mismatch | Verify pose detection | See "Body Size Issues" |
| Blending seams | Use edge refinement mask | See "Blending Issues" |
| Action pose fails | Check action pose detection | See "Action Pose Issues" |
| Poor quality | Increase refinement strength | See "Quality Issues" |

---

## Common Issues and Solutions

### 1. Plastic-Looking Face

**Symptoms**:
- Face appears smooth, artificial, or doll-like
- Lacks natural skin texture
- Looks like CGI or 3D render

**Causes**:
- Refinement strength too high
- Over-processing removes natural texture
- Negative prompts not working correctly

**Solutions**:

**Step 1: Check Current Settings**
```python
# In configs/default.yaml
processing:
  region_refine_strengths:
    face: 0.55  # Should be 0.5-0.6, not higher
```

**Step 2: Verify Negative Prompts**
- Ensure negative prompts include: "plastic, artificial, CGI, 3D render, smooth skin, airbrushed"
- Check that generator is using correct prompts

**Step 3: Reduce Refinement Strength**
```python
# Manually adjust face refinement
refiner.refine_face(
    image=image,
    face_bbox=bbox,
    expression_type="neutral"
)
# Internally uses strength=0.55
```

**Step 4: Enable Post-Processing**
- Verify `_post_process_face()` is running
- This blends 15% original texture back to preserve natural details

**Step 5: Check Input Quality**
- Ensure customer photo has good skin detail
- Avoid over-processed or heavily filtered input images

**Prevention**:
- Always use face refinement strength ≤ 0.6
- Keep original texture preservation enabled
- Use high-quality input images

---

### 2. Open Chest Shirt - Wrong Skin Tone

**Symptoms**:
- Chest/arms show incorrect skin color
- Skin looks flat or painted on
- Doesn't match customer's actual skin tone

**Causes**:
- Skin profile not extracted correctly
- Visible skin regions not detected
- Face reference not available for texture

**Solutions**:

**Step 1: Verify Skin Profile Extraction**
```python
# Check body analyzer output
body_shape = body_analyzer.analyze_body_shape(image, faces)
skin_profile = body_shape.get("skin_profile", {})

# Verify skin tone is extracted
tone = skin_profile.get("tone")
if tone is None:
    # Problem: Skin tone not extracted
    # Solution: Ensure face is visible and body has visible skin
```

**Step 2: Check Visible Skin Detection**
```python
# Check if visible regions are detected
visible_regions = skin_profile.get("visible_body_regions", {})
if "chest" not in visible_regions:
    # Problem: Chest region not detected
    # Solution: Ensure customer photo shows chest/arms
```

**Step 3: Verify Face Reference**
```python
# Face reference is used for texture
face_reference = skin_profile.get("face_reference")
if face_reference is None:
    # Problem: No face reference
    # Solution: Ensure face detection is working
```

**Step 4: Manual Skin Tone Adjustment**
```python
# If automatic extraction fails, manually set skin tone
skin_profile = {
    "tone": [180, 150, 120],  # BGR format, adjust to match customer
    "face_reference": face_patch
}
```

**Step 5: Use Chest Skin Mask for Refinement**
```python
# Generate refinement mask for chest
masks = quality_control.generate_refinement_masks(
    result_image,
    quality_assessment,
    face_regions,
    body_mask,
    template_analysis
)

# Refine chest region specifically
if "chest_skin" in masks:
    refined = refiner.refine_composition(
        image,
        template_analysis,
        customer_body_shape,
        refinement_mask=masks["chest_skin"],
        strength=0.5
    )
```

**Prevention**:
- Ensure customer photos show visible skin if template has open chest
- Use high-quality photos with good lighting
- Verify face detection is working (needed for skin tone reference)

---

### 3. Body Size Mismatch

**Symptoms**:
- Customer body doesn't match template size
- Clothing looks too big or too small
- Body proportions are wrong

**Causes**:
- Pose detection inaccurate
- Measurements incorrect
- Scale factors not calculated properly

**Solutions**:

**Step 1: Verify Pose Detection**
```python
# Check if pose is detected
pose_data = pose_detector.detect_pose(image)
if not pose_data:
    # Problem: No pose detected
    # Solution: Ensure full body is visible in photo
```

**Step 2: Check Measurements**
```python
# Verify body measurements
measurements = body_shape.get("measurements", {})
required = ["shoulder_width", "hip_width", "waist_width", "torso_height"]

for key in required:
    if key not in measurements:
        # Problem: Missing measurement
        # Solution: Check pose keypoints are detected
```

**Step 3: Verify Scale Factors**
```python
# Check scale map calculation
scale_map = body_warper._derive_scale_map(
    customer_measurements,
    template_measurements
)

# Scale factors should be reasonable (0.7 - 1.5)
for key, value in scale_map.items():
    if value < 0.5 or value > 2.0:
        # Problem: Unrealistic scale
        # Solution: Check measurements are correct
```

**Step 4: Manual Scale Adjustment**
```python
# If automatic scaling fails, manually adjust
blueprint = body_warper.build_warp_blueprint(
    customer_body_shape,
    template_pose
)

# Manually adjust scale factor
blueprint["size_scale_factor"] = 1.2  # Adjust as needed
```

**Prevention**:
- Ensure full body is visible in customer photos
- Use photos with clear pose (not heavily occluded)
- Verify pose detection confidence > 0.7

---

### 4. Blending Seams Visible

**Symptoms**:
- Visible edges between customer and background
- Halo effects around subject
- Abrupt color transitions

**Causes**:
- Edge refinement not applied
- Mask boundaries too sharp
- Background segmentation incorrect

**Solutions**:

**Step 1: Use Edge Refinement Mask**
```python
# Generate edge mask
masks = quality_control.generate_refinement_masks(...)
edge_mask = masks.get("edges")

# Refine edges
refined = refiner.refine_composition(
    image,
    template_analysis,
    customer_body_shape,
    refinement_mask=edge_mask,
    strength=0.45
)
```

**Step 2: Check Mask Feathering**
```python
# Verify masks have feathered edges
# In composer.py, masks should be blurred
mask = cv2.GaussianBlur(mask, (11, 11), 0)
```

**Step 3: Verify Background Segmentation**
```python
# Check background mask is correct
background_mask = segmenter.segment_background(image, foreground_mask)

# Background should not include subject
# If incorrect, manually adjust
```

**Step 4: Increase Edge Refinement Strength**
```python
# In configs/default.yaml
processing:
  region_refine_strengths:
    edges: 0.5  # Increase from 0.45 if needed
```

**Prevention**:
- Always use edge refinement
- Ensure masks are properly feathered
- Verify background segmentation is accurate

---

### 5. Action Pose Not Handled Correctly

**Symptoms**:
- Dynamic poses look static
- Expression doesn't match action
- Body style in action not preserved

**Causes**:
- Action pose not detected
- Expression matching fails for action
- Warping doesn't account for dynamics

**Solutions**:

**Step 1: Verify Action Pose Detection**
```python
# Check if action pose is detected
template_analysis = template_analyzer.analyze_template(template, faces)
is_action = template_analysis.get("pose", {}).get("is_action_pose", False)

if not is_action:
    # Problem: Action pose not detected
    # Check: Are arms raised? Legs spread? Body leaning?
```

**Step 2: Check Expression Matching**
```python
# For action poses, expression should match action
expression = template_analysis.get("expression", {})
expression_type = expression.get("type")

# Action poses often have "happy" or "surprised" expressions
# Ensure expression matching preserves this
```

**Step 3: Adjust Warp Parameters**
```python
# Action poses may need different warping
# Check warp blueprint accounts for dynamic pose
blueprint = body_warper.build_warp_blueprint(
    customer_body_shape,
    template_pose
)

# Verify control points include all keypoints
```

**Step 4: Preserve Body Style in Action**
```python
# Ensure body style (running, jumping, etc.) is preserved
# Check that pose keypoints maintain action characteristics
# Verify warping doesn't flatten dynamic pose
```

**Prevention**:
- Use templates with clear action poses
- Ensure customer photos have similar action (if possible)
- Verify action pose detection is working

---

### 6. Poor Quality Overall

**Symptoms**:
- Low quality score (< 0.85)
- Multiple issues detected
- General poor appearance

**Causes**:
- Multiple issues compounding
- Input quality poor
- Refinement not applied correctly

**Solutions**:

**Step 1: Check Quality Assessment**
```python
quality = quality_control.assess_quality(
    result_image,
    customer_faces,
    template_faces,
    template_analysis,
    body_shape
)

# Review all metrics
print(f"Overall: {quality['overall_score']}")
print(f"Face similarity: {quality['face_similarity']}")
print(f"Pose accuracy: {quality['pose_accuracy']}")
print(f"Issues: {quality['issues']}")
```

**Step 2: Address Issues One by One**
```python
# Get recommended refinements
recommended = quality.get("recommended_refinements", [])

# Refine in order: problems → edges → body → face
for region in recommended:
    mask = masks.get(region)
    if mask is not None:
        refined = refiner.refine_composition(
            image,
            template_analysis,
            customer_body_shape,
            refinement_mask=mask
        )
```

**Step 3: Increase Overall Refinement**
```python
# In configs/default.yaml
processing:
  refinement_strength: 0.85  # Increase from 0.8
  num_inference_steps: 50    # Increase from 30
```

**Step 4: Check Input Quality**
- Ensure customer photos are high resolution
- Verify good lighting and clarity
- Check template quality

**Step 5: Use Combined Mask**
```python
# Refine entire image
masks = quality_control.generate_refinement_masks(...)
combined_mask = masks.get("combined")

refined = refiner.refine_composition(
    image,
    template_analysis,
    customer_body_shape,
    refinement_mask=combined_mask,
    strength=0.6
)
```

**Prevention**:
- Always use high-quality input images
- Verify all models are loaded correctly
- Check quality assessment after each refinement
- Iterate until quality score > 0.85

---

## Workflow-Specific Troubleshooting

### Face Detection Fails

**Check**:
1. Face is clearly visible and front-facing
2. Image quality is good
3. Face detector is working

**Fix**:
```python
# Try different face detector
# In configs/default.yaml
models:
  face_detector: dlib  # or opencv, insightface

# Or manually specify face region
```

### Pose Detection Fails

**Check**:
1. Full body is visible
2. Pose is not heavily occluded
3. MediaPipe is working

**Fix**:
```python
# Use face-based estimation as fallback
# Body analyzer automatically falls back
# Or manually specify pose keypoints
```

### Clothing Doesn't Fit

**Check**:
1. Body measurements are correct
2. Scale factors are reasonable
3. Clothing adaptation is applied

**Fix**:
```python
# Manually adjust clothing region
# Use body mask for targeted refinement
# Adjust scale factors if needed
```

### Expression Doesn't Match

**Check**:
1. Expression detection is working
2. Expression matching is applied
3. Face landmarks are accurate

**Fix**:
```python
# Verify expression matching
expression_match = face_processor.match_expression(
    face_identity,
    template_face,
    template_expression
)

# Check expression_type matches template
```

---

## Debugging Tips

### Enable Logging
```python
# In configs/default.yaml
logging:
  level: DEBUG  # Change from INFO to DEBUG
```

### Export Intermediate Results
```python
# Export each stage for review
quality_control.export_intermediate_results(
    pipeline_stages,
    output_dir="debug_output"
)
```

### Check Model Loading
```python
# Verify all models are loaded
from src.models import FaceDetector, PoseDetector, Segmenter, Generator

face_detector = FaceDetector()
pose_detector = PoseDetector()
segmenter = Segmenter()
generator = Generator()

# All should initialize without errors
```

### Verify Config
```python
# Check configuration is loaded correctly
from src.utils.config import get_config

config = get_config()
print(config)
```

---

## Getting Help

If issues persist:

1. **Check Logs**: Review log files for error messages
2. **Export Debug Info**: Export intermediate results for analysis
3. **Verify Inputs**: Ensure all inputs meet requirements
4. **Review Documentation**: See WORKFLOW_DOCUMENTATION.md for details
5. **Test with Simple Case**: Try with simple template first

---

## Prevention Checklist

Before processing, verify:

- [ ] Customer photos are high quality
- [ ] Faces are clearly visible
- [ ] Full body is visible (if body warping needed)
- [ ] Template is appropriate for customer
- [ ] All models are loaded correctly
- [ ] Configuration is correct
- [ ] Sufficient GPU/CPU resources available
- [ ] Output directory is writable

---

## Quick Fixes Summary

| Problem | Quick Command |
|---------|---------------|
| Reduce face refinement | Set `face: 0.5` in config |
| Fix skin tone | Check `skin_profile["tone"]` |
| Fix body size | Verify `measurements` |
| Fix blending | Use `edge_mask` refinement |
| Fix action pose | Check `is_action_pose` |
| Improve quality | Use `combined_mask` refinement |

---

This troubleshooting guide should help you resolve most issues. For workflow logic details, see WORKFLOW_DOCUMENTATION.md.

