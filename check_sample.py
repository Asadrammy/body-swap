"""Check sample output image to understand format"""

from src.pipeline.preprocessor import Preprocessor
from src.utils.image_utils import load_image
import numpy as np

# Check sample output
preprocessor = Preprocessor()
sample_data = preprocessor.preprocess_template('swap1 (1).png')
print(f"Sample image faces detected: {len(sample_data['faces'])}")

# Check new customer image
customer_data = preprocessor.preprocess_customer_photos(['IMG20251019131550.jpg'])
print(f"Customer image faces detected: {len(customer_data['faces'][0]) if customer_data['faces'] else 0}")

# Check current output
try:
    current_output = load_image('outputs/client_test_result.png')
    sample_img = load_image('swap1 (1).png')
    customer_img = load_image('IMG20251019131550.jpg')
    
    print(f"\nImage shapes:")
    print(f"Sample: {sample_img.shape}")
    print(f"Customer: {customer_img.shape}")
    print(f"Current output: {current_output.shape}")
    
    # Check if current output is same as customer
    if np.array_equal(current_output, customer_img):
        print("\n✗ Current output is IDENTICAL to customer image!")
    elif np.array_equal(current_output, sample_img):
        print("\n✗ Current output is IDENTICAL to sample image!")
    else:
        print("\n✓ Current output is different from both inputs")
        
except Exception as e:
    print(f"Error: {e}")













