"""Body shape analysis from customer photos"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..models.pose_detector import PoseDetector
from ..models.segmenter import Segmenter

logger = get_logger(__name__)


class BodyAnalyzer:
    """Analyze body shape and proportions from customer photos"""
    
    def __init__(self):
        """Initialize body analyzer"""
        self.pose_detector = PoseDetector()
        self.segmenter = Segmenter()
        self.mesh_rows = get_config("processing.mesh_rows", 24)
        self.mesh_cols = get_config("processing.mesh_cols", 12)
        self.depth_map_size = get_config("processing.depth_map_size", 128)
    
    def analyze_body_shape(self, image: np.ndarray, faces: List[Dict]) -> Dict:
        """
        Analyze body shape and proportions
        
        Args:
            image: Customer reference image
            faces: Detected faces
        
        Returns:
            Body shape analysis result
        """
        if not faces:
            raise ValueError("No faces detected in image")
        
        # Detect pose
        pose_data = self.pose_detector.detect_pose(image)
        
        if not pose_data:
            logger.warning("No pose detected, using face-based estimation")
            return self._estimate_from_face(image, faces[0])
        
        pose = pose_data[0]  # Use first detected pose
        keypoints = pose.get("keypoints", {})
        
        # Store keypoints for use in skin region detection
        self._current_keypoints = keypoints
        
        # Extract body measurements
        measurements = self._extract_measurements(keypoints, image.shape[:2])
        
        # Classify body type
        body_type = self._classify_body_type(measurements)
        
        # Extract body contour/mesh
        body_mask = self._extract_body_mask(image, keypoints)
        segment_masks = self.segmenter.segment_body_parts(image, {"keypoints": keypoints})
        if segment_masks.get("torso") is not None:
            # Use refined torso mask if available
            body_mask = np.maximum(body_mask, segment_masks["torso"])
        
        body_mesh = self._build_body_mesh(body_mask, keypoints)
        depth_map = self._estimate_depth_map(body_mask)
        girth_profile = self._compute_girth_profile(body_mask)
        skin_profile = self._estimate_skin_profile(image, body_mask, faces)
        
        result = {
            "pose_keypoints": keypoints,
            "measurements": measurements,
            "body_type": body_type,
            "body_mask": body_mask,
            "segment_masks": segment_masks,
            "body_mesh": body_mesh,
            "depth_map": depth_map,
            "girth_profile": girth_profile,
            "skin_profile": skin_profile,
            "confidence": pose.get("confidence", 0.5)
        }
        
        return result
    
    def _extract_measurements(self, keypoints: Dict, image_shape: Tuple[int, int]) -> Dict:
        """
        Extract body measurements from keypoints
        
        Args:
            keypoints: Pose keypoints
            image_shape: Image dimensions (height, width) or numpy array shape
        
        Returns:
            Measurements dictionary
        """
        # Handle both tuple and numpy array shape formats
        if isinstance(image_shape, (tuple, list)) and len(image_shape) >= 2:
            h, w = image_shape[0], image_shape[1]
        else:
            h, w = image_shape
        measurements = {}
        
        # Calculate distances in pixels
        if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
            shoulder_width = np.linalg.norm(
                np.array(keypoints["left_shoulder"]) - np.array(keypoints["right_shoulder"])
            )
            measurements["shoulder_width"] = float(shoulder_width)
        
        if "left_hip" in keypoints and "right_hip" in keypoints:
            hip_width = np.linalg.norm(
                np.array(keypoints["left_hip"]) - np.array(keypoints["right_hip"])
            )
            measurements["hip_width"] = float(hip_width)
        
        if "left_shoulder" in keypoints and "left_hip" in keypoints:
            torso_height = np.linalg.norm(
                np.array(keypoints["left_shoulder"]) - np.array(keypoints["left_hip"])
            )
            measurements["torso_height"] = float(torso_height)
        
        # Calculate waist (approximate as midpoint between shoulders and hips)
        if "left_shoulder" in keypoints and "left_hip" in keypoints:
            waist_y = (keypoints["left_shoulder"][1] + keypoints["left_hip"][1]) / 2
            measurements["waist_y"] = float(waist_y)
            
            # Estimate waist width - use more accurate calculation
            # For plus-size bodies, waist is typically wider than average of shoulders/hips
            # For slim bodies, waist is narrower
            if "shoulder_width" in measurements and "hip_width" in measurements:
                avg_width = (measurements["shoulder_width"] + measurements["hip_width"]) / 2
                # Base waist width calculation
                # For most body types, waist is 0.75-0.95 of average width
                # We'll use a more conservative estimate that works better for size differences
                waist_ratio = 0.88  # Default ratio
                
                # Adjust based on body proportions
                if "shoulder_hip_ratio" in measurements:
                    ratio = measurements.get("shoulder_hip_ratio", 1.0)
                    # If hips are wider (ratio < 1), waist might be closer to hip width
                    # If shoulders are wider (ratio > 1), waist might be closer to shoulder width
                    if ratio < 0.95:  # Pear-shaped or plus-size
                        waist_ratio = 0.92
                    elif ratio > 1.1:  # Athletic/V-shaped
                        waist_ratio = 0.85
                
                waist_width = avg_width * waist_ratio
                measurements["waist_width"] = float(waist_width)
        
        # Overall body dimensions
        if all(kp in keypoints for kp in ["left_shoulder", "left_ankle"]):
            body_height = np.linalg.norm(
                np.array(keypoints["left_shoulder"]) - np.array(keypoints["left_ankle"])
            )
            measurements["body_height"] = float(body_height)
        
        if all(kp in keypoints for kp in ["left_hip", "left_ankle"]):
            left_leg = np.linalg.norm(
                np.array(keypoints["left_hip"]) - np.array(keypoints["left_ankle"])
            )
            measurements["leg_length"] = float(left_leg)
        elif all(kp in keypoints for kp in ["right_hip", "right_ankle"]):
            right_leg = np.linalg.norm(
                np.array(keypoints["right_hip"]) - np.array(keypoints["right_ankle"])
            )
            measurements["leg_length"] = float(right_leg)
        
        # Calculate ratios
        if "shoulder_width" in measurements and "hip_width" in measurements:
            measurements["shoulder_hip_ratio"] = float(
                measurements["shoulder_width"] / measurements["hip_width"]
            )
        
        # Normalize by image size
        measurements["normalized_by"] = {"height": h, "width": w}
        
        return measurements
    
    def _classify_body_type(self, measurements: Dict) -> str:
        """
        Classify body type from measurements
        
        Args:
            measurements: Body measurements
        
        Returns:
            Body type classification
        """
        if not measurements:
            return "unknown"
        
        # Simple classification based on ratios
        if "shoulder_hip_ratio" in measurements:
            ratio = measurements["shoulder_hip_ratio"]
            
            if ratio > 1.15:
                return "athletic"  # V-shaped
            elif ratio < 0.9:
                return "pear_shaped"
            else:
                # Use waist width to distinguish
                if "waist_width" in measurements and "shoulder_width" in measurements:
                    waist_ratio = measurements["waist_width"] / measurements["shoulder_width"]
                    if waist_ratio > 1.0:
                        return "plus_size"
                    elif waist_ratio < 0.7:
                        return "slim"
        
        # Default classification
        if "waist_width" in measurements:
            if measurements["waist_width"] > measurements.get("shoulder_width", 100) * 1.1:
                return "plus_size"
            elif measurements["waist_width"] < measurements.get("shoulder_width", 100) * 0.8:
                return "slim"
        
        return "average"
    
    def _extract_body_mask(self, image: np.ndarray, keypoints: Dict) -> np.ndarray:
        """
        Extract body region mask
        
        Args:
            image: Input image
            keypoints: Pose keypoints
        
        Returns:
            Body mask
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create mask from keypoints
        if "left_shoulder" in keypoints and "right_shoulder" in keypoints and \
           "left_hip" in keypoints and "right_hip" in keypoints and \
           "left_ankle" in keypoints and "right_ankle" in keypoints:
            
            # Create polygon from body outline
            body_points = [
                keypoints["left_shoulder"],
                keypoints["right_shoulder"],
                keypoints["right_hip"],
                keypoints["right_ankle"],
                keypoints["left_ankle"],
                keypoints["left_hip"],
            ]
            
            pts = np.array(body_points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        
        return mask
    
    def _estimate_from_face(self, image: np.ndarray, face: Dict) -> Dict:
        """
        Estimate body shape from face when pose detection fails
        
        Args:
            image: Input image
            face: Face detection result
        
        Returns:
            Basic body shape estimation
        """
        h, w = image.shape[:2]
        bbox = face.get("bbox", [0, 0, w, h])
        face_x, face_y, face_w, face_h = bbox
        
        # Rough estimation based on face position and size
        # Assume face is in upper portion of body
        estimated_body_height = face_h * 7  # Typical head-to-body ratio
        estimated_body_width = face_w * 2.5
        
        measurements = {
            "estimated_body_height": float(estimated_body_height),
            "estimated_body_width": float(estimated_body_width),
            "face_bbox": bbox
        }
        
        # Create basic mask (lower portion of image from face)
        mask = np.zeros((h, w), dtype=np.uint8)
        body_start_y = face_y + face_h
        body_end_y = min(h, body_start_y + estimated_body_height)
        body_x_center = face_x + face_w // 2
        body_left = max(0, int(body_x_center - estimated_body_width / 2))
        body_right = min(w, int(body_x_center + estimated_body_width / 2))
        
        mask[body_start_y:body_end_y, body_left:body_right] = 255
        
        return {
            "pose_keypoints": {},
            "measurements": measurements,
            "body_type": "unknown",
            "body_mask": mask,
            "confidence": 0.3
        }

    def fuse_body_shapes(self, shapes: List[Dict]) -> Dict:
        """Fuse multiple body shape analyses into a consolidated profile."""
        if not shapes:
            return {}
        
        fused: Dict = {
            "source_count": len(shapes),
            "confidence": float(np.mean([s.get("confidence", 0.0) for s in shapes]))
        }
        
        # Merge keypoints
        keypoint_accumulator: Dict[str, List[List[float]]] = {}
        for shape in shapes:
            for name, value in shape.get("pose_keypoints", {}).items():
                keypoint_accumulator.setdefault(name, []).append(value)
        
        fused_keypoints = {}
        for name, points in keypoint_accumulator.items():
            arr = np.array(points)
            fused_keypoints[name] = arr.mean(axis=0).tolist()
        fused["pose_keypoints"] = fused_keypoints
        
        # Merge measurements
        measurement_accumulator: Dict[str, List[float]] = {}
        for shape in shapes:
            for name, value in shape.get("measurements", {}).items():
                if isinstance(value, (int, float)):
                    measurement_accumulator.setdefault(name, []).append(value)
        fused_measurements = {}
        for name, values in measurement_accumulator.items():
            fused_measurements[name] = float(np.median(values))
        fused["measurements"] = fused_measurements
        
        # Merge masks by taking union
        masks = [s.get("body_mask") for s in shapes if s.get("body_mask") is not None]
        if masks:
            target_h, target_w = masks[0].shape
            merged_mask = np.zeros((target_h, target_w), dtype=np.uint8)
            for mask in masks:
                resized = mask
                if resized.shape != (target_h, target_w):
                    resized = cv2.resize(resized, (target_w, target_h))
                merged_mask = np.maximum(merged_mask, resized)
            fused["body_mask"] = merged_mask
        
        # Merge girth profiles and depth maps if available
        girths = [s.get("girth_profile") for s in shapes if s.get("girth_profile")]
        if girths:
            fused["girth_profile"] = self._average_girth_profiles(girths)
        
        depth_maps = [s.get("depth_map") for s in shapes if s.get("depth_map") is not None]
        if depth_maps:
            target_h, target_w = depth_maps[0].shape[:2]
            merged_depth = np.zeros((target_h, target_w), dtype=np.float32)
            for depth in depth_maps:
                resized = depth
                if depth.shape[:2] != (target_h, target_w):
                    resized = cv2.resize(depth, (target_w, target_h))
                merged_depth += resized
            fused["depth_map"] = merged_depth / len(depth_maps)
        
        body_meshes = [s.get("body_mesh") for s in shapes if s.get("body_mesh")]
        if body_meshes:
            fused["body_mesh"] = self._merge_meshes(body_meshes)
        
        skin_profiles = [s.get("skin_profile") for s in shapes if s.get("skin_profile")]
        if skin_profiles:
            fused["skin_profile"] = self._merge_skin_profiles(skin_profiles)
        
        fused["body_type"] = self._majority_vote([s.get("body_type") for s in shapes])
        
        return fused

    # --- Internal helpers ---

    def _build_body_mesh(self, body_mask: np.ndarray, keypoints: Dict) -> Dict:
        """Create a coarse mesh representation of the body."""
        if body_mask is None or body_mask.sum() == 0:
            return {}
        
        contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {}
        
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        
        rows = max(4, int(self.mesh_rows))
        cols = max(4, int(self.mesh_cols))
        y_coords = np.linspace(y, y + h, rows)
        x_coords = np.linspace(x, x + w, cols)
        
        occupancy = []
        for cy in y_coords:
            row = []
            for cx in x_coords:
                ix = int(np.clip(cx, 0, body_mask.shape[1] - 1))
                iy = int(np.clip(cy, 0, body_mask.shape[0] - 1))
                row.append(float(body_mask[iy, ix] > 0))
            occupancy.append(row)
        
        keylines = {}
        if all(k in keypoints for k in ("neck", "mid_hip")):
            keylines["centerline"] = [
                keypoints["neck"],
                keypoints["mid_hip"] if "mid_hip" in keypoints else keypoints.get("left_hip", keypoints["neck"])
            ]
        
        return {
            "bbox": [int(x), int(y), int(w), int(h)],
            "contour": contour.squeeze().tolist() if contour is not None else [],
            "grid": occupancy,
            "rows": rows,
            "cols": cols,
            "keylines": keylines
        }

    def _estimate_depth_map(self, body_mask: np.ndarray) -> Optional[np.ndarray]:
        """Estimate relative depth from body mask using distance transform."""
        if body_mask is None or body_mask.sum() == 0:
            return None
        
        mask = (body_mask > 0).astype(np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() > 0:
            dist = dist / dist.max()
        scaled = cv2.resize(dist, (self.depth_map_size, self.depth_map_size))
        return scaled.astype(np.float32)

    def _compute_girth_profile(self, body_mask: np.ndarray, sample_points: int = 5) -> List[Dict]:
        """Compute horizontal girth profile across the torso."""
        if body_mask is None or body_mask.sum() == 0:
            return []
        
        h, w = body_mask.shape[:2]
        ratios = np.linspace(0.2, 0.85, sample_points)
        profile = []
        
        for ratio in ratios:
            y = int(np.clip(ratio * h, 0, h - 1))
            row = body_mask[y, :]
            xs = np.where(row > 0)[0]
            if len(xs) >= 2:
                width = xs[-1] - xs[0]
                profile.append({
                    "ratio": float(ratio),
                    "absolute_width": float(width),
                    "normalized_width": float(width / w)
                })
        
        return profile

    def _estimate_skin_profile(
        self,
        image: np.ndarray,
        body_mask: np.ndarray,
        faces: List[Dict]
    ) -> Dict:
        """
        Estimate skin tone and create a small reference patch.
        Enhanced to support male, female, and children with better skin tone detection.
        """
        profile: Dict = {
            "tone": None, 
            "sample_count": 0,
            "gender": None,
            "age_group": None,
            "visible_body_regions": []
        }
        
        # Detect gender and age from face if available
        if faces:
            face_data = faces[0]
            # Try to get gender/age from face detection if available
            profile["gender"] = face_data.get("gender", None)
            profile["age_group"] = self._estimate_age_group(face_data.get("age", None))
            
            face_patch = self._extract_skin_reference(image, faces[0])
            if face_patch is not None and face_patch.size > 0:
                profile["face_reference"] = face_patch
                # Use face as primary reference for skin tone
                face_tone = np.median(face_patch.reshape(-1, 3), axis=0).tolist()
                profile["tone"] = face_tone
        
        # Extract visible body regions (chest, arms, etc.)
        if body_mask is not None and body_mask.sum() > 0:
            # Check for visible skin regions in body
            visible_regions = self._detect_visible_skin_regions(image, body_mask, faces)
            profile["visible_body_regions"] = visible_regions
            
            # Sample from visible skin regions if available
            if visible_regions:
                skin_samples = []
                for region_name, region_mask in visible_regions.items():
                    if region_mask is not None and region_mask.sum() > 0:
                        region_pixels = image[region_mask > 0]
                        if region_pixels.size > 0:
                            skin_samples.append(region_pixels)
                
                if skin_samples:
                    all_samples = np.vstack(skin_samples)
                    profile["tone"] = np.median(all_samples, axis=0).tolist()
                    profile["sample_count"] = int(all_samples.shape[0])
            else:
                # Fallback to general body mask
                samples = image[body_mask > 0]
                if samples.size > 0:
                    # Filter for skin-like colors
                    skin_like = self._filter_skin_colors(samples)
                    if skin_like.size > 0:
                        profile["tone"] = np.median(skin_like, axis=0).tolist()
                        profile["sample_count"] = int(skin_like.shape[0])
        
        return profile
    
    def _estimate_age_group(self, age: Optional[float]) -> str:
        """Estimate age group from age value"""
        if age is None:
            return "adult"
        if age < 13:
            return "child"
        elif age < 18:
            return "teen"
        else:
            return "adult"
    
    def _detect_visible_skin_regions(
        self,
        image: np.ndarray,
        body_mask: np.ndarray,
        faces: List[Dict]
    ) -> Dict[str, np.ndarray]:
        """
        Detect visible skin regions (chest, arms, etc.) for body conditioning.
        Critical for open chest shirts.
        """
        regions = {}
        h, w = image.shape[:2]
        
        # Get pose keypoints if available from body analysis
        keypoints = getattr(self, '_current_keypoints', {})
        
        # Detect chest region (for open chest shirts)
        if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
            chest_mask = self._create_chest_region_mask(keypoints, h, w)
            if chest_mask is not None and chest_mask.sum() > 0:
                # Verify it's actually skin (not clothing)
                if self._verify_skin_region(image, chest_mask):
                    regions["chest"] = chest_mask
        
        # Detect arm regions (for short sleeves or sleeveless)
        for side in ["left", "right"]:
            arm_key = f"{side}_arm"
            if f"{side}_shoulder" in keypoints and f"{side}_elbow" in keypoints:
                arm_mask = self._create_arm_skin_mask(keypoints, side, h, w)
                if arm_mask is not None and arm_mask.sum() > 0:
                    if self._verify_skin_region(image, arm_mask):
                        regions[arm_key] = arm_mask
        
        return regions
    
    def _create_chest_region_mask(self, keypoints: Dict, h: int, w: int) -> Optional[np.ndarray]:
        """Create mask for chest region (between shoulders, above waist)"""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            left_shoulder = np.array(keypoints["left_shoulder"])
            right_shoulder = np.array(keypoints["right_shoulder"])
            # Ensure they're [x, y] format
            if len(left_shoulder) > 2:
                left_shoulder = left_shoulder[:2]
            if len(right_shoulder) > 2:
                right_shoulder = right_shoulder[:2]
            
            # Get neck point if available
            if "neck" in keypoints:
                neck = np.array(keypoints["neck"])
                if len(neck) > 2:
                    neck = neck[:2]
                top_y = int(neck[1])
            else:
                top_y = int(min(left_shoulder[1], right_shoulder[1]))
            
            # Get waist/hip level
            if "left_hip" in keypoints and "right_hip" in keypoints:
                left_hip = np.array(keypoints["left_hip"])
                right_hip = np.array(keypoints["right_hip"])
                if len(left_hip) > 2:
                    left_hip = left_hip[:2]
                if len(right_hip) > 2:
                    right_hip = right_hip[:2]
                bottom_y = int((left_hip[1] + right_hip[1]) / 2)
            else:
                # Estimate from shoulders
                bottom_y = int(max(left_shoulder[1], right_shoulder[1]) + 
                              np.linalg.norm(left_shoulder - right_shoulder) * 1.5)
            
            # Create chest region polygon
            center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
            chest_width = int(np.linalg.norm(left_shoulder - right_shoulder) * 0.8)
            
            left_x = max(0, center_x - chest_width // 2)
            right_x = min(w, center_x + chest_width // 2)
            
            chest_points = np.array([
                [left_x, top_y],
                [right_x, top_y],
                [right_x, bottom_y],
                [left_x, bottom_y]
            ], dtype=np.int32)
            
            cv2.fillPoly(mask, [chest_points], 255)
            
            # Smooth edges
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            mask = (mask > 127).astype(np.uint8) * 255
            
        except Exception as e:
            logger.warning(f"Chest region mask creation failed: {e}")
            return None
        
        return mask
    
    def _create_arm_skin_mask(self, keypoints: Dict, side: str, h: int, w: int) -> Optional[np.ndarray]:
        """Create mask for visible arm skin (for short sleeves)"""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            shoulder_key = f"{side}_shoulder"
            elbow_key = f"{side}_elbow"
            
            if shoulder_key not in keypoints or elbow_key not in keypoints:
                return None
            
            shoulder = np.array(keypoints[shoulder_key])
            elbow = np.array(keypoints[elbow_key])
            # Ensure they're [x, y] format
            if len(shoulder) > 2:
                shoulder = shoulder[:2]
            if len(elbow) > 2:
                elbow = elbow[:2]
            
            # Create upper arm region (shoulder to elbow)
            arm_length = np.linalg.norm(elbow - shoulder)
            arm_width = max(10, int(arm_length * 0.2))
            
            # Draw arm region
            p1 = tuple(shoulder.astype(int))
            p2 = tuple(elbow.astype(int))
            cv2.line(mask, p1, p2, 255, arm_width)
            cv2.circle(mask, p1, arm_width // 2, 255, -1)
            cv2.circle(mask, p2, arm_width // 2, 255, -1)
            
            # Smooth
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = (mask > 127).astype(np.uint8) * 255
            
        except Exception as e:
            logger.warning(f"Arm skin mask creation failed ({side}): {e}")
            return None
        
        return mask
    
    def _verify_skin_region(self, image: np.ndarray, mask: np.ndarray) -> bool:
        """Verify that a masked region actually contains skin (not clothing)"""
        if mask.sum() == 0:
            return False
        
        region_pixels = image[mask > 0]
        if region_pixels.size == 0:
            return False
        
        # Check if colors match skin tone ranges
        # Convert to RGB if needed
        if len(region_pixels.shape) == 1:
            return False
        
        # Simple skin color detection
        # Skin typically has: R > G > B, and R > 100, G > 50, B < 200
        avg_color = np.mean(region_pixels, axis=0)
        
        if len(avg_color) >= 3:
            # Handle BGR format (OpenCV default)
            b, g, r = avg_color[0], avg_color[1], avg_color[2]
            
            # Skin color heuristic
            is_skin = (r > g > b) and (r > 100) and (g > 50) and (b < 200)
            
            # Additional check: variance should be moderate (not uniform like clothing)
            color_variance = np.var(region_pixels, axis=0).mean()
            is_textured = color_variance > 200  # Not too uniform
            
            return is_skin and is_textured
        
        return False
    
    def _filter_skin_colors(self, pixels: np.ndarray) -> np.ndarray:
        """Filter pixels to keep only skin-like colors"""
        if pixels.size == 0:
            return pixels
        
        # Reshape if needed
        if len(pixels.shape) == 1:
            return pixels
        
        # Convert to RGB for analysis
        # Assuming BGR format from OpenCV
        if pixels.shape[1] >= 3:
            r = pixels[:, 2] if pixels.shape[1] >= 3 else pixels[:, 0]
            g = pixels[:, 1] if pixels.shape[1] >= 2 else pixels[:, 0]
            b = pixels[:, 0]
            
            # Skin color conditions
            skin_mask = (r > g) & (g > b) & (r > 100) & (g > 50) & (b < 200)
            
            return pixels[skin_mask]
        
        return pixels

    def _extract_skin_reference(self, image: np.ndarray, face: Dict) -> Optional[np.ndarray]:
        """Extract a downsampled patch from the face region for skin reference."""
        bbox = face.get("bbox")
        if not bbox:
            return None
        x, y, w, h = map(int, bbox)
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (64, 64))

    def _average_girth_profiles(self, profiles: List[List[Dict]]) -> List[Dict]:
        """Average multiple girth profiles."""
        if not profiles:
            return []
        
        combined = {}
        for profile in profiles:
            for entry in profile:
                key = round(entry["ratio"], 2)
                bucket = combined.setdefault(key, {"absolute": [], "normalized": []})
                bucket["absolute"].append(entry.get("absolute_width", 0.0))
                bucket["normalized"].append(entry.get("normalized_width", 0.0))
        
        averaged = []
        for ratio, values in combined.items():
            averaged.append({
                "ratio": ratio,
                "absolute_width": float(np.mean(values["absolute"])),
                "normalized_width": float(np.mean(values["normalized"]))
            })
        
        return averaged

    def _merge_meshes(self, meshes: List[Dict]) -> Dict:
        """Merge multiple mesh descriptors."""
        valid_meshes = [m for m in meshes if m]
        if not valid_meshes:
            return {}
        
        rows = int(np.median([m.get("rows", self.mesh_rows) for m in valid_meshes]))
        cols = int(np.median([m.get("cols", self.mesh_cols) for m in valid_meshes]))
        
        accum = np.zeros((rows, cols), dtype=np.float32)
        for mesh in valid_meshes:
            grid = np.array(mesh.get("grid", []), dtype=np.float32)
            if grid.size == 0:
                continue
            resized = cv2.resize(grid, (cols, rows))
            accum += resized
        
        merged = accum / len(valid_meshes)
        reference = valid_meshes[0]
        contour = reference.get("contour", [])
        bbox = reference.get("bbox", [0, 0, 0, 0])
        
        return {
            "rows": rows,
            "cols": cols,
            "grid": merged.tolist(),
            "contour": contour,
            "bbox": bbox
        }

    def _merge_skin_profiles(self, profiles: List[Dict]) -> Dict:
        """Merge multiple skin tone profiles."""
        tones = [np.array(p["tone"]) for p in profiles if p.get("tone") is not None]
        merged: Dict = {}
        if tones:
            merged["tone"] = np.mean(tones, axis=0).tolist()
            merged["sample_count"] = sum(p.get("sample_count", 0) for p in profiles)
        face_refs = [p.get("face_reference") for p in profiles if p.get("face_reference") is not None]
        if face_refs:
            merged["face_reference"] = face_refs[0]
        return merged

    def _majority_vote(self, labels: List[Optional[str]]) -> str:
        """Return the most common non-empty label."""
        filtered = [label for label in labels if label]
        if not filtered:
            return "unknown"
        unique, counts = np.unique(filtered, return_counts=True)
        return unique[np.argmax(counts)]

