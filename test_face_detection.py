#!/usr/bin/env python3
"""Test face detection on customer image"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
from src.models.face_detector import FaceDetector
from src.utils.logger import setup_logger

setup_logger()

def main():
    customer_image = r"D:\projects\image\face-body-swap\1760713603491 (1).jpg"
    template_image = r"D:\projects\image\face-body-swap\swap1 (1).png"
    
    print("=" * 60)
    print("Face Detection Test")
    print("=" * 60)
    
    # Load images
    customer_img = cv2.imread(customer_image)
    template_img = cv2.imread(template_image)
    
    print(f"Customer image loaded: {customer_img is not None}, shape: {customer_img.shape if customer_img is not None else None}")
    print(f"Template image loaded: {template_img is not None}, shape: {template_img.shape if template_img is not None else None}")
    
    # Detect faces
    face_detector = FaceDetector()
    
    print("\nDetecting faces in customer image...")
    customer_faces = face_detector.detect_faces(customer_img)
    print(f"Customer faces detected: {len(customer_faces) if customer_faces else 0}")
    if customer_faces:
        for i, face in enumerate(customer_faces):
            bbox = face.get("bbox", [0, 0, 0, 0])
            landmarks = face.get("landmarks", [])
            print(f"  Face {i+1}: bbox={bbox}, landmarks={len(landmarks)}")
    
    print("\nDetecting faces in template image...")
    template_faces = face_detector.detect_faces(template_img)
    print(f"Template faces detected: {len(template_faces) if template_faces else 0}")
    if template_faces:
        for i, face in enumerate(template_faces):
            bbox = face.get("bbox", [0, 0, 0, 0])
            landmarks = face.get("landmarks", [])
            print(f"  Face {i+1}: bbox={bbox}, landmarks={len(landmarks)}")
    
    # Test face alignment
    if customer_faces and customer_faces[0].get("landmarks"):
        print("\nTesting face alignment...")
        try:
            aligned = face_detector.align_face(
                customer_img,
                customer_faces[0].get("landmarks", []),
                (112, 112)
            )
            print(f"Aligned face shape: {aligned.shape if aligned is not None else None}")
            if aligned is not None:
                unique_colors = len(set(tuple(pixel) for pixel in aligned.reshape(-1, aligned.shape[-1])))
                print(f"Unique colors in aligned face: {unique_colors}")
        except Exception as e:
            print(f"Alignment failed: {e}")

if __name__ == "__main__":
    main()






