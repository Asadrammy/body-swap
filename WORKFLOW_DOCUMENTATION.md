# Face-Body Swap Workflow Documentation

## Overview

This document explains the complete workflow logic for the face-body swap system. The workflow is designed to be **flawless, easy to use, and fully controllable** with manual touch-up capabilities when needed.

## Workflow Architecture

The pipeline consists of 9 main stages, each with specific responsibilities:

### Stage 1: Input Validation & Preprocessing
**Purpose**: Validate and prepare customer photos and templates

**Logic**:
- Validates image formats (JPEG, PNG)
- Resizes images to processing size (default: 512px, max: 1024px)
- Detects faces in customer photos using InsightFace (primary) or OpenCV (fallback)
- Detects faces in template images
- Validates that required faces are detected
- Supports 1-2 customer photos (for individuals, couples, families)

**Output**: Preprocessed images with face detection results

**Manual Control**: If face detection fails, user can manually specify face regions

---

### Stage 2: Body Shape Analysis
**Purpose**: Extract customer body proportions and classify body type

**Logic**:
- Detects body pose using MediaPipe Pose
- Extracts key measurements:
  - Shoulder width
  - Hip width
  - Waist width
  - Torso height
  - Leg length
- Classifies body type: `average`, `slim`, `athletic`, `plus_size`, `pear_shaped`
- Creates body mesh for warping
- **NEW**: Detects visible skin regions (chest, arms) for open chest shirts
- **NEW**: Estimates skin tone profile with gender and age group detection
- Builds depth map and girth profile for body conditioning

**Key Features**:
- **Body Conditioning**: Automatically detects if customer photo shows visible skin (chest, arms)
- **Multi-Subject Support**: Handles male, female, and children with appropriate body analysis
- **Size Adaptation**: Calculates scale factors to match customer body size to template

**Output**: Body shape analysis with measurements, masks, and skin profile

**Manual Control**: If pose detection fails, uses face-based estimation

---

### Stage 3: Template Analysis
**Purpose**: Analyze template image for pose, clothing, expression, and background

**Logic**:
- Detects template pose
- **NEW**: Detects if template is an action pose (running, jumping, etc.)
- Analyzes clothing:
  - Identifies clothing items (shirt, pants, sleeves)
  - **NEW**: Detects open chest shirts (visible skin)
  - Creates masks for each clothing region
- Analyzes facial expression:
  - Detects expression type: `neutral`, `happy`, `surprised`
  - Extracts expression landmarks
- Analyzes background:
  - Creates background mask
  - Estimates background complexity
- Analyzes lighting conditions

**Key Features**:
- **Action Photo Support**: Automatically detects dynamic poses for proper handling
- **Open Chest Detection**: Identifies templates with visible chest/arms for body conditioning

**Output**: Complete template analysis with pose, clothing, expression, background, and lighting

---

### Stage 4: Face Processing
**Purpose**: Extract, match, and composite customer faces into template

**Logic**:
- Extracts face identity from customer photo:
  - Aligns face to standard 112x112 format
  - Extracts 512-dim face embedding
- Matches expression:
  - Compares customer expression to template expression
  - Warps customer face landmarks to match template expression
- Composites face into template:
  - Resizes customer face to match template face size
  - Applies expression deformation if needed
  - Blends with feathering for seamless integration

**Key Features**:
- **Expression Matching**: Preserves template expression while using customer face
- **Multi-Face Support**: Handles couples, families (2 faces)

**Output**: Image with customer face composited into template

**Manual Control**: If face matching fails, user can manually adjust face position

---

### Stage 5: Body Warping
**Purpose**: Warp customer body to match template pose

**Logic**:
- Extracts corresponding keypoints between customer and template
- **Size Adjustment**: Scales template keypoints to match customer body size
  - Calculates scale factors from measurements
  - Uses median scale for robustness
- Applies Thin Plate Spline (TPS) warping:
  - Warps customer body to match template pose
  - Preserves body proportions
- Applies body mask to restrict warping to body region

**Key Features**:
- **Size Adaptation**: Automatically adjusts for different body sizes
- **Pose Matching**: Accurately matches template pose

**Output**: Warped customer body matching template pose

**Manual Control**: User can adjust warp strength or skip warping if needed

---

### Stage 6: Clothing Adaptation
**Purpose**: Adapt template clothing to customer body shape

**Logic**:
- Calculates scale map from customer and template measurements
- Adapts each clothing item:
  - Shirt/torso: Scales based on shoulder width and torso height
  - Pants/legs: Scales based on hip width and leg length
- **NEW**: Body Conditioning for Open Chest Shirts:
  - Detects if template has open chest (visible skin)
  - Synthesizes visible skin using customer's skin tone
  - Uses face texture as reference for realistic skin appearance
  - Supports male, female, and children with appropriate skin synthesis
- Generates fabric folds for realism

**Key Features**:
- **Body Conditioning**: Critical for open chest shirts - synthesizes realistic skin
- **Multi-Gender Support**: Handles male, female, and children appropriately
- **Realistic Skin**: Uses face texture to avoid flat/plastic appearance

**Output**: Image with adapted clothing and synthesized skin (if needed)

**Manual Control**: User can adjust clothing fit or manually edit skin regions

---

### Stage 7: Composition
**Purpose**: Blend warped body into template background

**Logic**:
- Extracts template background
- Blends warped body into background:
  - Uses alpha blending with feathering
  - Matches lighting conditions
  - Preserves background details
- Handles edge cases:
  - Occlusions
  - Shadows
  - Reflections

**Output**: Composed image with customer in template scene

**Manual Control**: User can adjust blending strength or manually mask regions

---

### Stage 8: Generative Refinement
**Purpose**: Refine composed image using Stable Diffusion for photorealistic results

**Logic**:
- **Face Refinement** (Enhanced to avoid plastic looks):
  - Uses detailed prompts emphasizing natural skin texture
  - Negative prompts exclude plastic, artificial, CGI terms
  - Lower strength (0.55) to preserve natural features
  - Post-processes to blend original texture back (15% original detail)
  - More inference steps (30) for better quality
  
- **Body Refinement**:
  - Refines clothing fit and fabric details
  - Adjusts body proportions if needed
  
- **Edge Refinement**:
  - Fixes blending seams
  - Removes halos and artifacts
  
- **Problem Area Refinement**:
  - Targets specific issues detected by quality control
  - Uses higher strength (0.7) for problem areas

**Key Features**:
- **No Plastic Faces**: Enhanced prompts and post-processing ensure natural appearance
- **Selective Refinement**: Different strengths for different regions
- **Quality Preservation**: Blends original texture to avoid over-processing

**Output**: Refined photorealistic image

**Manual Control**: User can skip refinement, adjust strength, or refine specific regions only

---

### Stage 9: Quality Control
**Purpose**: Assess quality and generate masks for manual touch-ups

**Logic**:
- Assesses quality metrics:
  - Face similarity (0-1)
  - Pose accuracy (0-1)
  - Clothing fit (0-1)
  - Seamless blending (0-1)
  - Sharpness (0-1)
  - Overall score (weighted average)
- Detects issues:
  - Face similarity below threshold
  - Pose alignment problems
  - Clothing fit issues
  - Blending seams
  - Soft/blurry areas
- **NEW**: Generates enhanced refinement masks:
  - Face mask (ellipse shape, expanded for blending)
  - Body mask
  - **Chest skin mask** (for open chest shirts)
  - Edge mask (for blending fixes)
  - Problem area mask (for artifacts)
  - Combined mask (for full refinement)
  - Each mask includes metadata (type, recommended strength, description)

**Key Features**:
- **Manual Touch-ups**: Generates precise masks for manual refinement
- **Troubleshooting**: Identifies specific issues with recommendations
- **Full Control**: User can refine any region independently

**Output**: Quality assessment and refinement masks

**Manual Control**: User can manually edit masks or refine specific regions

---

## Workflow Control Points

### Automatic Processing
The workflow runs automatically through all 9 stages. Each stage has error handling and fallbacks.

### Manual Intervention Points

1. **After Stage 1 (Preprocessing)**:
   - If face detection fails, manually specify face regions
   - Adjust image preprocessing settings

2. **After Stage 2 (Body Analysis)**:
   - Manually adjust body measurements if needed
   - Edit body mask if incorrect

3. **After Stage 4 (Face Processing)**:
   - Manually adjust face position
   - Edit face mask for better blending

4. **After Stage 5 (Body Warping)**:
   - Adjust warp strength
   - Skip warping if not needed

5. **After Stage 8 (Refinement)**:
   - Skip refinement for faster processing
   - Adjust refinement strength per region
   - Use generated masks for selective refinement

6. **After Stage 9 (Quality Control)**:
   - Use generated masks for manual touch-ups
   - Refine specific problem areas
   - Export intermediate results for review

---

## Manual Touch-Up Workflow

When something goes wrong, follow this process:

### Step 1: Review Quality Assessment
- Check overall score (should be > 0.85)
- Review detected issues
- Check recommended refinements

### Step 2: Export Masks
- Use `generate_refinement_masks()` to get refinement masks
- Each mask includes:
  - Type (face_refinement, body_refinement, etc.)
  - Recommended strength
  - Description

### Step 3: Selective Refinement
- Refine only problem areas using masks
- Use recommended strengths or adjust as needed
- Refine in order: problems → edges → body → face

### Step 4: Verify Results
- Re-run quality assessment
- Check if issues are resolved
- Iterate if needed

---

## Troubleshooting Guide

### Issue: Face Detection Fails
**Solution**:
- Ensure face is clearly visible and front-facing
- Try different face detector (insightface, dlib, opencv)
- Manually specify face region
- Check image quality and resolution

### Issue: Plastic-Looking Face
**Solution**:
- Reduce face refinement strength (default: 0.55)
- Check negative prompts include "plastic, artificial, CGI"
- Verify post-processing is preserving original texture
- Increase inference steps for better quality

### Issue: Body Doesn't Match Template Size
**Solution**:
- Check body measurements are accurate
- Verify pose detection is working
- Adjust scale factors manually if needed
- Check if body type classification is correct

### Issue: Open Chest Shirt Shows Wrong Skin
**Solution**:
- Verify skin profile is extracted correctly
- Check visible skin regions are detected
- Ensure face reference is available for texture
- Manually adjust skin tone if needed
- Use chest_skin mask for targeted refinement

### Issue: Action Pose Not Handled Correctly
**Solution**:
- Verify action pose detection is working
- Check pose keypoints are accurate
- Adjust warp parameters for dynamic poses
- Use body mask to restrict warping

### Issue: Blending Seams Visible
**Solution**:
- Use edge mask for refinement
- Increase edge refinement strength
- Check background segmentation
- Manually feather edges if needed

### Issue: Clothing Doesn't Fit
**Solution**:
- Check clothing adaptation scale factors
- Verify body measurements are correct
- Use body mask for targeted refinement
- Manually adjust clothing region if needed

### Issue: Poor Quality Overall
**Solution**:
- Increase refinement strength
- Use higher image resolution
- Check input image quality
- Verify all models are loaded correctly
- Use combined mask for full refinement

---

## Workflow Logic Summary

1. **Input** → Validate and preprocess
2. **Analyze** → Extract body shape and template features
3. **Match** → Align customer to template (face + body)
4. **Warp** → Transform customer to match template pose
5. **Adapt** → Adjust clothing and synthesize skin (if needed)
6. **Compose** → Blend into template background
7. **Refine** → Enhance with generative models (avoiding plastic looks)
8. **Assess** → Quality control and mask generation
9. **Touch-up** → Manual refinement using masks (if needed)

**Key Principles**:
- **Full Control**: Every stage can be manually adjusted
- **Quality First**: Multiple quality checks and refinement options
- **Natural Results**: Enhanced prompts and post-processing prevent plastic looks
- **Body Conditioning**: Automatic skin synthesis for open chest shirts
- **Action Support**: Detects and handles dynamic poses
- **Troubleshooting**: Comprehensive error handling and manual intervention points

---

## Configuration Options

Key configuration parameters in `configs/default.yaml`:

- `processing.refinement_strength`: Overall refinement strength (default: 0.8)
- `processing.region_refine_strengths`: Per-region strengths
  - `face`: 0.65 (reduced to avoid plastic looks)
  - `body`: 0.55
  - `edges`: 0.45
  - `problems`: 0.7
- `processing.quality_threshold`: Minimum quality score (default: 0.85)
- `models.face_detector`: Face detector backend (insightface, dlib, opencv)
- `models.pose_detector`: Pose detector (mediapipe)
- `models.generator.device`: Processing device (cuda, cpu)

---

## Best Practices

1. **Input Quality**: Use high-quality, well-lit customer photos
2. **Template Selection**: Choose templates with similar poses and lighting
3. **Body Conditioning**: Ensure customer photos show visible skin if template has open chest
4. **Refinement**: Start with lower strengths and increase if needed
5. **Quality Check**: Always review quality assessment before finalizing
6. **Manual Touch-ups**: Use generated masks for precise refinements
7. **Iterative Refinement**: Refine problem areas one at a time

---

## Support for Different Use Cases

### Individual Photos
- Single customer photo
- Single face processing
- Standard workflow

### Couples Photos
- Two customer photos
- Two face processing
- Body shape fusion
- Expression matching for both

### Family Photos (Father-Son, Mother-Daughter, etc.)
- Two customer photos
- Age-appropriate body analysis
- Child-specific skin tone handling
- Expression matching

### Action Photos
- Automatic action pose detection
- Dynamic expression handling
- Body style in action preservation
- Enhanced pose warping

---

## Conclusion

This workflow is designed to be **flawless and easy to use** while providing **full control** for manual touch-ups when needed. Every stage has error handling, fallbacks, and manual intervention points. The system automatically handles body conditioning for open chest shirts, avoids plastic-looking faces, and supports action photos with proper expression and body style preservation.

