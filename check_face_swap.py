#!/usr/bin/env python3
"""Check if face swap actually happened"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.api.cli import SwapCLI
from src.utils.logger import setup_logger
from src.models.face_detector import FaceDetector

setup_logger()

# Load images
customer_img = cv2.imread(r"D:\projects\image\face-body-swap\1760713603491 (1).jpg")
template_img = cv2.imread(r"D:\projects\image\face-body-swap\swap1 (1).png")
output_img = cv2.imread(r"D:\projects\image\face-body-swap\outputs\client_test_result.png")

print("=" * 60)
print("Face Detection Check")
print("=" * 60)

# Check face detection
face_detector = FaceDetector()
customer_faces = face_detector.detect_faces(customer_img)
template_faces = face_detector.detect_faces(template_img)

print(f"Customer faces detected: {len(customer_faces) if customer_faces else 0}")
print(f"Template faces detected: {len(template_faces) if template_faces else 0}")

if customer_faces:
    print(f"Customer face bbox: {customer_faces[0].get('bbox', 'N/A')}")
if template_faces:
    print(f"Template face bbox: {template_faces[0].get('bbox', 'N/A')}")

print("\n" + "=" * 60)
print("Image Comparison")
print("=" * 60)

# Compare template and output
if template_img is not None and output_img is not None:
    # Resize output to match template
    h, w = template_img.shape[:2]
    output_resized = cv2.resize(output_img, (w, h))
    
    # Calculate difference
    diff = cv2.absdiff(template_img, output_resized)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    print(f"Template shape: {template_img.shape}")
    print(f"Output shape: {output_img.shape}")
    print(f"Mean difference: {mean_diff:.2f}")
    print(f"Max difference: {max_diff:.2f}")
    
    # Check if they're identical (within tolerance)
    identical = np.allclose(template_img, output_resized, atol=10)
    print(f"Are images identical (within 10px tolerance)? {identical}")
    
    if identical or mean_diff < 5:
        print("\n⚠️ WARNING: Output is identical to template - face swap did NOT occur!")
    else:
        print("\n✓ Images are different - face swap may have occurred")

print("\n" + "=" * 60)
print("Face Composite Check")
print("=" * 60)

face_comp = cv2.imread(r"D:\projects\image\face-body-swap\outputs\client_test_result_intermediate\face_composite.png")
if face_comp is not None:
    # Compare face composite with template
    face_comp_resized = cv2.resize(face_comp, (w, h))
    face_diff = cv2.absdiff(template_img, face_comp_resized)
    face_mean_diff = np.mean(face_diff)
    print(f"Face composite mean difference from template: {face_mean_diff:.2f}")
    
    if face_mean_diff < 5:
        print("⚠️ WARNING: Face composite is identical to template!")
    else:
        print("✓ Face composite is different from template")

