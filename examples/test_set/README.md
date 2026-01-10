# Test Set for Face-Body Swap Pipeline

## Overview

This test set is designed to validate the face-body swap pipeline with various body types and scenarios.

## Directory Structure

```
test_set/
├── inputs/
│   ├── average/          # Average body type customer photos
│   └── obese/            # Plus-size/obese body type customer photos
├── templates/            # Template images for testing
├── outputs/
│   ├── average/          # Output results for average subjects
│   └── obese/            # Output results for obese subjects
├── comparisons/          # Before/after comparison images
└── manifest.json         # Test set metadata and scenarios
```

## Test Scenarios

### Scenario 001: Average Male - Casual Portrait
- **Subject**: Average body type male
- **Template**: Casual street portrait
- **Expected**: Natural face swap, body pose matching, clothing adaptation

### Scenario 002: Average Female - Action Shot
- **Subject**: Average body type female
- **Template**: Dynamic action shot with open chest
- **Expected**: Action pose handling, body conditioning, expression matching

### Scenario 003: Obese Male - Casual Portrait
- **Subject**: Plus-size male
- **Template**: Casual street portrait
- **Expected**: Body size adaptation, clothing fit, natural proportions

### Scenario 004: Obese Female - Casual Portrait
- **Subject**: Plus-size female
- **Template**: Casual street portrait
- **Expected**: Plus-size adaptation, clothing scaling, realistic proportions

### Scenario 005: Average Couple - Garden Scene
- **Subject**: Average body type couple
- **Template**: Romantic garden couple scene
- **Expected**: Multi-subject processing, couple face matching

## Usage

1. **Prepare Input Images**:
   - Place customer photos in `inputs/average/` or `inputs/obese/`
   - Ensure photos show full body or at least upper body
   - Face should be clearly visible
   - Recommended size: 512x512 to 1024x1024 pixels

2. **Run Pipeline**:
   ```bash
   python -m src.pipeline --customer inputs/average/customer_001.jpg \
                          --template examples/templates/individual_casual_001.png \
                          --output outputs/average/result_001.png
   ```

3. **Review Results**:
   - Check output images in `outputs/` directory
   - Review quality metrics in job metadata
   - Use refinement masks if needed

4. **Generate Comparisons**:
   - Create side-by-side before/after images
   - Save to `comparisons/` directory

## Quality Expectations

- **Face Similarity**: > 0.75
- **Pose Accuracy**: > 0.75
- **Clothing Fit**: > 0.70
- **Seamless Blending**: > 0.72
- **Overall Score**: > 0.85

## Notes

- Test images should be representative of real customer photos
- Include various lighting conditions and backgrounds
- Test with different clothing styles (open chest, full coverage, etc.)
- Validate action poses and dynamic expressions
- Test multi-subject scenarios (couples, families)

## Adding Test Images

To add your own test images:

1. Place customer photos in appropriate `inputs/` subdirectory
2. Name files descriptively: `{body_type}_{gender}_{id}.jpg`
3. Update `manifest.json` with new test scenarios if needed
4. Run pipeline and save outputs
5. Document any issues or edge cases found

## Troubleshooting

If test images fail:
- Check face detection: Ensure face is clearly visible
- Check pose detection: Ensure full body or upper body is visible
- Check image quality: Minimum 512x512, clear and well-lit
- Review logs for specific error messages
