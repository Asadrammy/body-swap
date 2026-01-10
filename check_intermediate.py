"""Check intermediate results to see where face swap failed"""

import numpy as np
import cv2
from src.utils.image_utils import load_image

# Load images
face_comp = load_image('outputs/new_image_conversion_result_intermediate/face_composite.png')
template = load_image('swap1 (1).png')
final = load_image('outputs/new_image_conversion_result.png')

# Resize for comparison
if face_comp.shape != template.shape:
    template_resized = cv2.resize(template, (face_comp.shape[1], face_comp.shape[0]))
else:
    template_resized = template

# Resize all to same size for comparison
target_size = (1024, 682)  # Template processed size
face_comp_resized = cv2.resize(face_comp, target_size)
final_resized = cv2.resize(final, target_size)
template_resized = cv2.resize(template, target_size)

# Calculate differences
diff_face_comp = np.mean(np.abs(face_comp_resized.astype(float) - template_resized.astype(float)))
diff_final = np.mean(np.abs(final_resized.astype(float) - template_resized.astype(float)))

print(f"Face composite difference from template: {diff_face_comp:.2f}")
print(f"Final output difference from template: {diff_final:.2f}")

if diff_face_comp > 5.0:
    print("✓ Face swap happened in face_composite step")
else:
    print("✗ Face swap did NOT happen in face_composite step")

if diff_final > diff_face_comp:
    print("✓ Final output has more changes than face composite")
elif diff_final < diff_face_comp:
    print("✗ Final output has FEWER changes than face composite - later steps overwrote the swap!")
else:
    print("⚠ Final output has same changes as face composite")

