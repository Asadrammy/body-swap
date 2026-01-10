#!/usr/bin/env python3
"""
Sample script to run test scenarios from the test set.
This demonstrates how to use the pipeline with test images.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import run_pipeline
from src.utils.logger import setup_logger

setup_logger()

def run_test_scenario(customer_path: str, template_path: str, output_path: str):
    """Run a single test scenario"""
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
