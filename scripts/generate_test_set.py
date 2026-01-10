"""
Test Set Generation Script

Generates a test set with average and obese subjects for validation.
Creates before/after examples to demonstrate the pipeline capabilities.
"""

import sys
from pathlib import Path
import json
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger, get_logger
from src.utils.config import get_config
from src.utils.template_catalog import TemplateCatalog

setup_logger()
logger = get_logger(__name__)


def create_test_set_structure():
    """Create directory structure for test set"""
    examples_dir = project_root / "examples"
    test_set_dir = examples_dir / "test_set"
    
    # Create directories
    test_set_dir.mkdir(parents=True, exist_ok=True)
    (test_set_dir / "inputs" / "average").mkdir(parents=True, exist_ok=True)
    (test_set_dir / "inputs" / "obese").mkdir(parents=True, exist_ok=True)
    (test_set_dir / "templates").mkdir(parents=True, exist_ok=True)
    (test_set_dir / "outputs" / "average").mkdir(parents=True, exist_ok=True)
    (test_set_dir / "outputs" / "obese").mkdir(parents=True, exist_ok=True)
    (test_set_dir / "comparisons").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✓ Created test set structure at: {test_set_dir}")
    return test_set_dir


def create_test_manifest(test_set_dir: Path):
    """Create manifest file describing the test set"""
    catalog = TemplateCatalog()
    templates = catalog.list_templates()
    
    manifest = {
        "test_set_version": "1.0",
        "description": "Test set for face-body swap pipeline validation",
        "created_by": "generate_test_set.py",
        "test_categories": {
            "average": {
                "description": "Average body type subjects",
                "expected_subjects": [
                    "adult_male_average",
                    "adult_female_average"
                ],
                "input_dir": "inputs/average",
                "output_dir": "outputs/average"
            },
            "obese": {
                "description": "Plus-size/obese body type subjects",
                "expected_subjects": [
                    "adult_male_obese",
                    "adult_female_obese"
                ],
                "input_dir": "inputs/obese",
                "output_dir": "outputs/obese"
            }
        },
        "templates": [
            {
                "id": tpl.get("id"),
                "name": tpl.get("name"),
                "category": tpl.get("category"),
                "description": tpl.get("description"),
                "recommended_for": tpl.get("recommended_subjects", [])
            }
            for tpl in templates
        ],
        "test_scenarios": [
            {
                "id": "scenario_001",
                "name": "Average Male - Casual Portrait",
                "subject_type": "average",
                "subject_gender": "male",
                "template_id": "tpl_individual_casual_001",
                "expected_features": [
                    "natural face swap",
                    "body pose matching",
                    "clothing adaptation",
                    "seamless blending"
                ]
            },
            {
                "id": "scenario_002",
                "name": "Average Female - Action Shot",
                "subject_type": "average",
                "subject_gender": "female",
                "template_id": "tpl_individual_action_002",
                "expected_features": [
                    "action pose handling",
                    "open chest body conditioning",
                    "dynamic expression matching"
                ]
            },
            {
                "id": "scenario_003",
                "name": "Obese Male - Casual Portrait",
                "subject_type": "obese",
                "subject_gender": "male",
                "template_id": "tpl_individual_casual_001",
                "expected_features": [
                    "body size adaptation",
                    "clothing fit for plus-size",
                    "natural proportions"
                ]
            },
            {
                "id": "scenario_004",
                "name": "Obese Female - Casual Portrait",
                "subject_type": "obese",
                "subject_gender": "female",
                "template_id": "tpl_individual_casual_001",
                "expected_features": [
                    "plus-size body adaptation",
                    "clothing scaling",
                    "realistic body proportions"
                ]
            },
            {
                "id": "scenario_005",
                "name": "Average Couple - Garden Scene",
                "subject_type": "average",
                "subject_gender": "couple",
                "template_id": "tpl_couple_garden_001",
                "expected_features": [
                    "multi-subject processing",
                    "couple face matching",
                    "coordinated body warping"
                ]
            }
        ],
        "usage_instructions": {
            "step_1": "Place customer photos in inputs/average/ or inputs/obese/",
            "step_2": "Run pipeline with test templates",
            "step_3": "Compare outputs in outputs/ directory",
            "step_4": "Review quality metrics and refinement masks",
            "step_5": "Generate comparison images showing before/after"
        },
        "quality_expectations": {
            "face_similarity": "> 0.75",
            "pose_accuracy": "> 0.75",
            "clothing_fit": "> 0.70",
            "seamless_blending": "> 0.72",
            "overall_score": "> 0.85"
        }
    }
    
    manifest_path = test_set_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Created test manifest at: {manifest_path}")
    return manifest


def create_readme(test_set_dir: Path):
    """Create README for test set"""
    readme_content = """# Test Set for Face-Body Swap Pipeline

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
   python -m src.pipeline --customer inputs/average/customer_001.jpg \\
                          --template examples/templates/individual_casual_001.png \\
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
"""
    
    readme_path = test_set_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    logger.info(f"✓ Created README at: {readme_path}")


def create_sample_script(test_set_dir: Path):
    """Create sample script to run test scenarios"""
    script_content = """#!/usr/bin/env python3
\"\"\"
Sample script to run test scenarios from the test set.
This demonstrates how to use the pipeline with test images.
\"\"\"

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import run_pipeline
from src.utils.logger import setup_logger

setup_logger()

def run_test_scenario(customer_path: str, template_path: str, output_path: str):
    \"\"\"Run a single test scenario\"\"\"
    print(f"Running test scenario:")
    print(f"  Customer: {customer_path}")
    print(f"  Template: {template_path}")
    print(f"  Output: {output_path}")
    
    try:
        result = run_pipeline(
            customer_photos=[customer_path],
            template_path=template_path,
            output_path=output_path
        )
        
        if result:
            print(f"✓ Test scenario completed successfully")
            print(f"  Quality score: {result.get('quality_metrics', {}).get('overall_score', 'N/A')}")
            return True
        else:
            print(f"✗ Test scenario failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    test_set_dir = project_root / "examples" / "test_set"
    
    # Example: Run scenario 001
    customer = test_set_dir / "inputs" / "average" / "customer_001.jpg"
    template = test_set_dir / "templates" / "individual_casual_001.png"
    output = test_set_dir / "outputs" / "average" / "result_001.png"
    
    if customer.exists() and template.exists():
        run_test_scenario(str(customer), str(template), str(output))
    else:
        print("Test images not found. Please add customer photos and templates first.")
        print(f"Expected customer at: {customer}")
        print(f"Expected template at: {template}")
"""
    
    script_path = test_set_dir / "run_tests.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    script_path.chmod(0o755)
    
    logger.info(f"✓ Created test runner script at: {script_path}")


def main():
    """Main function to generate test set"""
    logger.info("Generating test set structure...")
    
    # Create directory structure
    test_set_dir = create_test_set_structure()
    
    # Create manifest
    manifest = create_test_manifest(test_set_dir)
    
    # Create README
    create_readme(test_set_dir)
    
    # Create sample script
    create_sample_script(test_set_dir)
    
    logger.info("✓ Test set generation complete!")
    logger.info(f"Test set location: {test_set_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Add customer photos to inputs/average/ and inputs/obese/")
    logger.info("2. Copy template images to templates/")
    logger.info("3. Run test scenarios using run_tests.py")
    logger.info("4. Review outputs and quality metrics")


if __name__ == "__main__":
    main()

