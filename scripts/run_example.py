#!/usr/bin/env python3
"""Example script demonstrating basic usage"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def example_basic_usage():
    """Example of basic usage"""
    print("Face and Body Swap - Example Usage")
    print("=" * 60)
    
    # This is a demonstration - replace with actual paths
    print("\nExample CLI command:")
    print("""
    python -m src.api.cli swap \\
        --customer-photos examples/sample_inputs/customer.jpg \\
        --template examples/templates/template.jpg \\
        --output examples/outputs/result.jpg \\
        --export-intermediate
    """)
    
    print("\nExample Python usage:")
    print("""
    from src.pipeline.preprocessor import Preprocessor
    from src.pipeline.body_analyzer import BodyAnalyzer
    
    # Initialize components
    preprocessor = Preprocessor()
    body_analyzer = BodyAnalyzer()
    
    # Process customer photos
    customer_data = preprocessor.preprocess_customer_photos([
        "path/to/customer.jpg"
    ])
    
    # Analyze body shape
    if customer_data["faces"]:
        body_shape = body_analyzer.analyze_body_shape(
            customer_data["images"][0],
            customer_data["faces"][0]
        )
        print(f"Body type: {body_shape['body_type']}")
    """)
    
    print("\nExample API usage:")
    print("""
    # Start API server
    python -m src.api.main
    
    # Or with uvicorn
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    
    # Then use curl or visit http://localhost:8000/docs for API documentation
    """)


if __name__ == "__main__":
    example_basic_usage()

