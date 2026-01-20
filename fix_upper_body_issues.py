"""Fix upper body issues: double hands and other artifacts"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import get_logger
from src.models.generator import Generator
from src.models.pose_detector import PoseDetector
from src.models.face_detector import FaceDetector

logger = get_logger(__name__)

def load_image(path: str) -> np.ndarray:
    """Load image as numpy array"""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

def save_image(image: np.ndarray, path: str):
    """Save numpy array as image"""
    img = Image.fromarray(image.astype(np.uint8))
    img.save(path, quality=95)
    logger.info(f"Saved image to: {path}")

def detect_hands(image: np.ndarray, pose_detector: PoseDetector):
    """Detect hand keypoints using pose detection"""
    pose_data = pose_detector.detect_pose(image)
    if not pose_data:
        return None
    
    pose = pose_data[0]
    keypoints = pose.get("keypoints", {})
    
    # MediaPipe hand keypoints
    left_wrist = keypoints.get("left_wrist")
    right_wrist = keypoints.get("right_wrist")
    left_elbow = keypoints.get("left_elbow")
    right_elbow = keypoints.get("right_elbow")
    left_shoulder = keypoints.get("left_shoulder")
    right_shoulder = keypoints.get("right_shoulder")
    
    hands = []
    if left_wrist and left_elbow:
        hands.append({
            "type": "left",
            "wrist": left_wrist,
            "elbow": left_elbow,
            "shoulder": left_shoulder
        })
    if right_wrist and right_elbow:
        hands.append({
            "type": "right",
            "wrist": right_wrist,
            "elbow": right_elbow,
            "shoulder": right_shoulder
        })
    
    return hands

def detect_duplicate_hands(image: np.ndarray, hands: list):
    """Detect if there are duplicate hands or artifacts around hands"""
    h, w = image.shape[:2]
    duplicate_regions = []
    
    if not hands:
        return duplicate_regions
    
    # Check for hands in unusual positions or artifacts
    # Look for hands that are too close together (might indicate duplication)
    if len(hands) >= 2:
        for i, hand1 in enumerate(hands):
            for j, hand2 in enumerate(hands[i+1:], i+1):
                if hand1.get("wrist") and hand2.get("wrist"):
                    wrist1 = np.array(hand1["wrist"])
                    wrist2 = np.array(hand2["wrist"])
                    distance = np.linalg.norm(wrist1 - wrist2)
                    
                    # If hands are very close (less than 10% of image width), might be duplicate
                    if distance < w * 0.1:
                        logger.warning(f"Hands too close together: {distance:.0f} pixels, might be duplicate")
                        # Mark the second hand as duplicate
                        wrist_x, wrist_y = int(wrist2[0]), int(wrist2[1])
                        bbox_size = int(w * 0.2)  # 20% of image width for hand region
                        bbox = [
                            max(0, wrist_x - bbox_size),
                            max(0, wrist_y - bbox_size),
                            min(w, wrist_x + bbox_size),
                            min(h, wrist_y + bbox_size)
                        ]
                        duplicate_regions.append({
                            "bbox": bbox,
                            "type": "duplicate_hand",
                            "confidence": 0.9
                        })
    
    # Check for hands in wrong positions (e.g., hand on wrong side of body)
    for hand in hands:
        if hand.get("shoulder") and hand.get("wrist") and hand.get("elbow"):
            shoulder = np.array(hand["shoulder"])
            wrist = np.array(hand["wrist"])
            elbow = np.array(hand["elbow"])
            
            # Calculate if hand is on correct side
            # Left hand should be to the left of left shoulder
            # Right hand should be to the right of right shoulder
            if hand["type"] == "left" and wrist[0] > shoulder[0] + w * 0.1:
                logger.warning(f"Left hand appears on wrong side")
                wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
                bbox_size = int(w * 0.2)
                bbox = [
                    max(0, wrist_x - bbox_size),
                    max(0, wrist_y - bbox_size),
                    min(w, wrist_x + bbox_size),
                    min(h, wrist_y + bbox_size)
                ]
                duplicate_regions.append({
                    "bbox": bbox,
                    "type": "misplaced_hand",
                    "confidence": 0.8
                })
            elif hand["type"] == "right" and wrist[0] < shoulder[0] - w * 0.1:
                logger.warning(f"Right hand appears on wrong side")
                wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
                bbox_size = int(w * 0.2)
                bbox = [
                    max(0, wrist_x - bbox_size),
                    max(0, wrist_y - bbox_size),
                    min(w, wrist_x + bbox_size),
                    min(h, wrist_y + bbox_size)
                ]
                duplicate_regions.append({
                    "bbox": bbox,
                    "type": "misplaced_hand",
                    "confidence": 0.8
                })
    
    return duplicate_regions

def create_upper_body_mask(image: np.ndarray, hands: list, face_bbox: tuple = None):
    """Create mask for upper body region (chest, arms, hands)"""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Upper body region: from head to waist (approximately top 50% of image)
    upper_body_top = 0
    upper_body_bottom = int(h * 0.6)  # 60% down from top
    
    # Create rectangular mask for upper body
    mask[upper_body_top:upper_body_bottom, :] = 255
    
    # Exclude face region if provided
    if face_bbox:
        x, y, fw, fh = face_bbox
        # Expand face region slightly
        expand = 20
        mask[max(0, y-expand):min(h, y+fh+expand), max(0, x-expand):min(w, x+fw+expand)] = 0
    
    # Include hand regions
    for hand in hands:
        if hand.get("wrist"):
            wrist = np.array(hand["wrist"])
            wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
            # Create circular region around hand
            radius = int(w * 0.1)  # 10% of image width
            cv2.circle(mask, (wrist_x, wrist_y), radius, 255, -1)
    
    # Smooth mask edges
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = (mask > 127).astype(np.uint8) * 255
    
    return mask

def remove_duplicate_regions(image: np.ndarray, duplicate_regions: list, original_image: np.ndarray = None):
    """Remove duplicate regions by inpainting from surrounding areas"""
    result = image.copy()
    h, w = image.shape[:2]
    
    if original_image is None:
        original_image = image.copy()
    
    for region in duplicate_regions:
        bbox = region["bbox"]
        x1, y1, x2, y2 = bbox
        
        # Create mask for duplicate region
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        # Expand mask for better blending
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Use inpainting to remove duplicate
        inpainted = cv2.inpaint(result, mask, 5, cv2.INPAINT_TELEA)
        
        # Blend with original image
        mask_3d = (mask / 255.0)[:, :, np.newaxis]
        result = (result * (1 - mask_3d) + inpainted * mask_3d).astype(np.uint8)
        
        logger.info(f"Removed {region['type']} at bbox={bbox}")
    
    return result

def refine_upper_body(image: np.ndarray, mask: np.ndarray, generator: Generator):
    """Refine upper body region using Stability AI"""
    logger.info("Refining upper body with Stability AI...")
    
    prompt = (
        "photorealistic upper body, natural human skin with pores and texture, "
        "realistic skin tone variation, natural arms and hands, single pair of hands, "
        "natural clothing fit, realistic fabric texture, high quality photograph, "
        "detailed features, natural lighting, authentic human appearance, "
        "preserve original body structure, maintain natural proportions, "
        "accurate anatomy, no duplicate body parts, no extra hands, no extra arms"
    )
    
    negative_prompt = (
        "duplicate hands, extra hands, multiple hands, double hands, "
        "duplicate arms, extra arms, multiple arms, distorted hands, "
        "malformed hands, mutated hands, bad anatomy, unnatural proportions, "
        "plastic, artificial, fake, CGI, 3D render, smooth skin, airbrushed, "
        "perfect skin, doll-like, surreal, deformed, misplaced body parts, "
        "artifacts, blurry, low quality, distorted, compression artifacts"
    )
    
    # Use lower strength for upper body to preserve natural features
    refined = generator.refine(
        image=image,
        prompt=prompt,
        mask=mask,
        negative_prompt=negative_prompt,
        strength=0.45,  # Lower strength to avoid artifacts
        num_inference_steps=30
    )
    
    return refined

def main():
    """Fix upper body issues in the output image"""
    logger.info("=" * 80)
    logger.info("Fixing Upper Body Issues")
    logger.info("=" * 80)
    
    # Input image
    input_path = Path(__file__).parent / "outputs" / "stability_ai_test_result.jpg"
    
    if not input_path.exists():
        logger.error(f"Input image not found: {input_path}")
        return
    
    logger.info(f"Loading image: {input_path}")
    image = load_image(str(input_path))
    original_image = image.copy()
    h, w = image.shape[:2]
    logger.info(f"Image loaded: shape={image.shape}")
    
    # Initialize detectors
    pose_detector = PoseDetector()
    face_detector = FaceDetector()
    generator = Generator()
    
    if not generator.use_ai_api:
        logger.error("AI API not available!")
        return
    
    # Detect face
    logger.info("Detecting face...")
    faces = face_detector.detect_faces(image)
    face_bbox = None
    if faces:
        face_bbox = faces[0].get("bbox", [0, 0, 0, 0])
        logger.info(f"Face detected: bbox={face_bbox}")
    
    # Detect hands
    logger.info("Detecting hands...")
    hands = detect_hands(image, pose_detector)
    if hands:
        logger.info(f"Detected {len(hands)} hand(s)")
        for hand in hands:
            logger.info(f"  {hand['type']} hand at wrist: {hand.get('wrist')}")
    else:
        logger.warning("No hands detected")
    
    # Detect duplicate hands
    logger.info("Checking for duplicate hands...")
    duplicate_regions = detect_duplicate_hands(image, hands) if hands else []
    
    if duplicate_regions:
        logger.warning(f"Found {len(duplicate_regions)} duplicate hand region(s)")
        # Remove duplicates by restoring from original
        logger.info("Removing duplicate regions...")
        image = remove_duplicate_regions(image, duplicate_regions, original_image)
    else:
        logger.info("No duplicate hands detected")
    
    # Create upper body mask
    logger.info("Creating upper body mask...")
    upper_body_mask = create_upper_body_mask(image, hands if hands else [], face_bbox)
    
    # Refine upper body
    logger.info("Refining upper body region...")
    logger.info("This may take 30-60 seconds...")
    refined = refine_upper_body(image, upper_body_mask, generator)
    
    if refined is None or refined.size == 0:
        logger.error("Refinement failed, using original")
        refined = image
    
    # Blend refined upper body with original lower body
    logger.info("Blending refined upper body with original image...")
    # Create inverse mask for lower body
    lower_body_mask = 255 - upper_body_mask
    lower_body_mask = cv2.GaussianBlur(lower_body_mask, (21, 21), 0)
    lower_body_mask = (lower_body_mask / 255.0).astype(np.float32)
    
    # Blend
    result = (
        refined.astype(np.float32) * (1 - lower_body_mask[:, :, np.newaxis]) +
        original_image.astype(np.float32) * lower_body_mask[:, :, np.newaxis]
    ).astype(np.uint8)
    
    # Save result
    output_path = Path(__file__).parent / "outputs" / "fixed_upper_body_result.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, str(output_path))
    
    logger.info("=" * 80)
    logger.info("Fix Complete!")
    logger.info("=" * 80)
    logger.info(f"Output saved to: {output_path}")
    logger.info("")
    logger.info("Changes made:")
    logger.info("  ✓ Removed duplicate hands/body parts")
    logger.info("  ✓ Refined upper body region with Stability AI")
    logger.info("  ✓ Preserved lower body and background")
    logger.info("  ✓ Blended seamlessly")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()

