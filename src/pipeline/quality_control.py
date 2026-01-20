"""Quality control and manual touch-up interface"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.config import get_config

# Optional Google AI integration
try:
    from ..models.google_ai_client import create_google_ai_client
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    create_google_ai_client = None

logger = get_logger(__name__)


class QualityControl:
    """Quality control and manual touch-up mask generation"""
    
    def __init__(self):
        """Initialize quality control"""
        cfg = get_config()
        processing_cfg = cfg.get("processing", {})
        self.quality_threshold = processing_cfg.get("quality_threshold", 0.85)
        region_strengths = processing_cfg.get("region_refine_strengths", {})
        self.region_strengths = {
            "face": region_strengths.get("face", 0.65),
            "body": region_strengths.get("body", 0.55),
            "edges": region_strengths.get("edges", 0.45),
            "problems": region_strengths.get("problems", 0.7)
        }
        
        # Initialize Google AI client if available
        self.google_ai_client = None
        if GOOGLE_AI_AVAILABLE:
            try:
                self.google_ai_client = create_google_ai_client()
                if self.google_ai_client:
                    logger.info("Google AI client initialized for quality assessment")
            except Exception as e:
                logger.warning(f"Could not initialize Google AI client: {e}")
    
    def assess_quality(
        self,
        result_image: np.ndarray,
        customer_faces: List[Dict],
        template_faces: List[Dict],
        template_analysis: Dict,
        body_shape: Optional[Dict] = None
    ) -> Dict:
        """
        Assess quality of swap result with distortion detection
        
        Args:
            result_image: Result image
            customer_faces: Customer face detections
            template_faces: Template face detections
            template_analysis: Template analysis
        
        Returns:
            Quality assessment result
        """
        face_similarity = self._compute_face_similarity(customer_faces, template_faces)
        # Ensure template_analysis is not None
        template_analysis = template_analysis or {}
        pose_accuracy = self._compute_pose_alignment(body_shape, template_analysis.get("pose"))
        clothing_fit = self._estimate_clothing_fit(body_shape)
        seamless_blending = self._check_blending_quality(result_image)
        sharpness = self._sharpness_score(result_image)
        
        # NEW: Detect face distortion (critical for client requirements)
        face_distortion = self._detect_face_distortion(result_image, template_faces)
        
        # Distortion is critical - heavily penalize if detected
        # Made more sensitive: threshold lowered to 0.25 to catch distortion earlier
        if face_distortion > 0.25:  # Significant distortion detected (lowered from 0.3)
            face_similarity = min(face_similarity, 0.3)  # Cap similarity if distorted (lowered from 0.4)
            logger.warning(f"Face distortion detected: {face_distortion:.2f} - quality will be heavily penalized")
        
        overall_score = (
            face_similarity * 0.3 +
            pose_accuracy * 0.2 +
            clothing_fit * 0.2 +
            seamless_blending * 0.2 +
            sharpness * 0.1
        )
        
        # Apply distortion penalty - made more aggressive
        if face_distortion > 0.25:  # Lowered threshold
            overall_score *= (1.0 - face_distortion * 0.6)  # Reduce score by up to 60% (increased from 50%)
        
        issues = []
        recommended = []
        
        # CRITICAL: Face distortion check (highest priority) - made more sensitive
        if face_distortion > 0.25:  # Lowered from 0.3 to catch distortion earlier
            issues.append("CRITICAL: Face distortion detected - unnatural facial features")
            recommended.append("face")
            recommended.append("problems")
        
        if face_similarity < 0.75:
            issues.append("Face similarity below target")
            recommended.append("face")
        if pose_accuracy < 0.75:
            issues.append("Pose alignment off")
            recommended.append("body")
        if clothing_fit < 0.7:
            issues.append("Clothing fit imbalance")
            recommended.append("body")
        if seamless_blending < 0.72:
            issues.append("Blending seams detected")
            recommended.append("edges")
        if sharpness < 0.6:
            issues.append("Image appears soft")
            recommended.append("problems")
        
        # Check if output should be rejected - made stricter
        should_reject = (
            overall_score < self.quality_threshold or
            face_distortion > 0.25 or  # Lowered from 0.3
            face_similarity < 0.5
        )
        
        quality_metrics = {
            "overall_score": float(np.clip(overall_score, 0.0, 1.0)),
            "face_similarity": float(face_similarity),
            "face_distortion": float(face_distortion),  # NEW
            "pose_accuracy": float(pose_accuracy),
            "clothing_fit": float(clothing_fit),
            "seamless_blending": float(seamless_blending),
            "sharpness": float(sharpness),
            "issues": issues,
            "recommended_refinements": list(dict.fromkeys(recommended)),
            "threshold": self.quality_threshold,
            "should_reject": should_reject,  # NEW: Flag for rejection
            "meets_requirements": not should_reject  # NEW: Client requirement check
        }
        
        # Enhance with Google AI analysis if available
        if self.google_ai_client:
            try:
                logger.info("Running Google AI quality analysis...")
                ai_analysis = self.google_ai_client.analyze_image_quality(result_image)
                
                # Merge AI scores (normalize to 0-1 scale)
                ai_overall = ai_analysis.get("overall_score", 7.0) / 10.0
                ai_face = ai_analysis.get("face_score", 7.0) / 10.0
                ai_body = ai_analysis.get("body_score", 7.0) / 10.0
                
                # Blend traditional and AI scores (70% traditional, 30% AI)
                quality_metrics["overall_score"] = float(np.clip(
                    overall_score * 0.7 + ai_overall * 0.3, 0.0, 1.0
                ))
                quality_metrics["face_similarity"] = float(np.clip(
                    face_similarity * 0.7 + ai_face * 0.3, 0.0, 1.0
                ))
                quality_metrics["clothing_fit"] = float(np.clip(
                    clothing_fit * 0.7 + ai_body * 0.3, 0.0, 1.0
                ))
                
                # Add AI analysis to metrics
                quality_metrics["ai_analysis"] = {
                    "overall_score": ai_overall,
                    "face_score": ai_face,
                    "body_score": ai_body,
                    "analysis_text": ai_analysis.get("analysis_text", ""),
                    "recommendations": ai_analysis.get("recommendations", [])
                }
                
                # Add AI-detected issues
                if ai_analysis.get("recommendations"):
                    quality_metrics["issues"].extend([
                        f"AI: {rec}" for rec in ai_analysis["recommendations"][:3]
                    ])
                
                logger.info(f"Google AI analysis completed: overall={ai_overall:.2f}")
            except Exception as e:
                logger.warning(f"Google AI analysis failed: {e}")
        
        return quality_metrics
    
    def generate_refinement_masks(
        self,
        result_image: np.ndarray,
        quality_assessment: Dict,
        face_regions: List[Tuple[int, int, int, int]],
        body_mask: Optional[np.ndarray] = None,
        template_analysis: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate masks for manual refinement with enhanced precision.
        Critical for manual touch-ups when something goes wrong.
        
        Args:
            result_image: Result image
            quality_assessment: Quality assessment result
            face_regions: Face bounding boxes
            body_mask: Optional body mask
            template_analysis: Optional template analysis for context
        
        Returns:
            Dictionary of refinement masks for different regions with metadata
        """
        h, w = result_image.shape[:2]
        masks = {}
        mask_metadata = {}
        
        # Face region masks - enhanced with better boundaries
        if face_regions:
            face_mask = np.zeros((h, w), dtype=np.uint8)
            for x, y, bw, bh in face_regions:
                # Use ellipse instead of rectangle for more natural shape
                center = (x + bw // 2, y + bh // 2)
                axes = (bw // 2, bh // 2)
                cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)
                # Expand slightly for blending area
                kernel = np.ones((15, 15), np.uint8)
                face_mask = cv2.dilate(face_mask, kernel, iterations=1)
            masks["face"] = face_mask
            mask_metadata["face"] = {
                "type": "face_refinement",
                "recommended_strength": 0.5,  # Reduced from 0.65 to prevent distortion
                "description": "Face region for identity preservation and expression matching"
            }
        
        # Body region mask - enhanced for open chest and visible skin
        if body_mask is not None:
            # Ensure body_mask matches result_image dimensions
            h_mask, w_mask = body_mask.shape[:2]
            if (h_mask, w_mask) != (h, w):
                body_mask = cv2.resize(body_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masks["body"] = body_mask
            mask_metadata["body"] = {
                "type": "body_refinement",
                "recommended_strength": 0.55,
                "description": "Body region for clothing fit and body conditioning"
            }
            
            # Add specific masks for visible skin regions if template has open chest
            if template_analysis:
                clothing = template_analysis.get("clothing", {})
                if clothing.get("has_open_chest"):
                    chest_mask = self._create_chest_refinement_mask(result_image, body_mask)
                    if chest_mask is not None:
                        # Ensure chest_mask matches result_image dimensions
                        h_chest, w_chest = chest_mask.shape[:2]
                        if (h_chest, w_chest) != (h, w):
                            chest_mask = cv2.resize(chest_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        masks["chest_skin"] = chest_mask
                        mask_metadata["chest_skin"] = {
                            "type": "skin_synthesis",
                            "recommended_strength": 0.5,
                            "description": "Chest region for skin tone matching (open chest shirt)"
                        }
        
        # Edge/transition regions (for blending issues) - enhanced
        edge_mask = self._detect_edge_regions(result_image)
        masks["edges"] = edge_mask
        mask_metadata["edges"] = {
            "type": "blending_fix",
            "recommended_strength": 0.45,
            "description": "Edge regions for seamless blending between subject and background"
        }
        
        # Problem areas based on quality assessment - enhanced
        if quality_assessment.get("issues"):
            problem_mask = self._detect_problem_areas(result_image, quality_assessment)
            masks["problems"] = problem_mask
            mask_metadata["problems"] = {
                "type": "artifact_removal",
                "recommended_strength": 0.7,
                "description": "Problem areas detected by quality assessment",
                "issues": quality_assessment.get("issues", [])
            }
        
        # Add combined mask for convenience
        if len(masks) > 1:
            combined = np.zeros((h, w), dtype=np.uint8)
            for key, mask in masks.items():
                # Skip metadata entry
                if key == "_metadata":
                    continue
                # Ensure mask matches result_image dimensions before combining
                h_mask, w_mask = mask.shape[:2]
                if (h_mask, w_mask) != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                combined = np.maximum(combined, mask)
            masks["combined"] = combined
            mask_metadata["combined"] = {
                "type": "full_refinement",
                "recommended_strength": 0.6,
                "description": "Combined mask for full image refinement"
            }
        
        # Store metadata with masks
        masks["_metadata"] = mask_metadata
        
        return masks
    
    def _create_chest_refinement_mask(
        self,
        image: np.ndarray,
        body_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """Create specific mask for chest region refinement (open chest shirts)"""
        if body_mask is None or body_mask.sum() == 0:
            return None
        
        # Find upper torso region (chest area)
        coords = np.column_stack(np.where(body_mask > 0))
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Chest is typically in upper 40% of body
        body_height = y_max - y_min
        chest_y_min = y_min
        chest_y_max = y_min + int(body_height * 0.4)
        
        chest_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        chest_mask[chest_y_min:chest_y_max, x_min:x_max] = body_mask[chest_y_min:chest_y_max, x_min:x_max]
        
        # Smooth edges
        kernel = np.ones((10, 10), np.uint8)
        chest_mask = cv2.morphologyEx(chest_mask, cv2.MORPH_CLOSE, kernel)
        chest_mask = cv2.GaussianBlur(chest_mask, (7, 7), 0)
        chest_mask = (chest_mask > 127).astype(np.uint8) * 255
        
        return chest_mask

    def _compute_face_similarity(
        self,
        customer_faces: List[Dict],
        template_faces: List[Dict]
    ) -> float:
        """Compute cosine similarity between customer and template face embeddings."""
        if not customer_faces or not template_faces:
            return 0.6
        
        def _embedding(face: Dict) -> Optional[np.ndarray]:
            emb = face.get("embedding_vector")
            if emb is None and face.get("embedding") is not None:
                emb = np.array(face["embedding"])
            if emb is None:
                return None
            arr = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm == 0:
                return None
            return arr / norm
        
        customer_emb = _embedding(customer_faces[0])
        template_emb = _embedding(template_faces[0])
        if customer_emb is None or template_emb is None:
            return 0.7
        
        similarity = float(np.clip(np.dot(customer_emb, template_emb), -1.0, 1.0))
        return (similarity + 1.0) / 2.0

    def _compute_pose_alignment(
        self,
        body_shape: Optional[Dict],
        template_pose: Optional[Dict]
    ) -> float:
        """Compare keypoints between customer body and template pose."""
        if not body_shape or not template_pose:
            return 0.75
        
        cust_kp = body_shape.get("pose_keypoints") or body_shape.get("keypoints")
        temp_kp = template_pose.get("keypoints")
        if not cust_kp or not temp_kp:
            return 0.75
        
        shared_keys = [k for k in cust_kp.keys() if k in temp_kp]
        if not shared_keys:
            return 0.75
        
        template_points = np.array(list(temp_kp.values()))
        diag = np.linalg.norm(template_points.max(axis=0) - template_points.min(axis=0))
        diag = diag if diag > 0 else 1.0
        
        distances = []
        for key in shared_keys:
            c_point = np.array(cust_kp[key])
            t_point = np.array(temp_kp[key])
            distances.append(np.linalg.norm(c_point - t_point) / diag)
        
        if not distances:
            return 0.75
        
        error = np.mean(distances)
        return float(np.clip(1.0 - error, 0.0, 1.0))

    def _estimate_clothing_fit(self, body_shape: Optional[Dict]) -> float:
        """Estimate clothing fit using girth profile smoothness."""
        if not body_shape:
            return 0.7
        profile = body_shape.get("girth_profile") or []
        if not profile:
            return 0.72
        widths = [entry.get("normalized_width", entry.get("absolute_width", 0)) for entry in profile]
        widths = [w for w in widths if isinstance(w, (int, float))]
        if not widths:
            return 0.72
        widths = np.array(widths)
        if widths.mean() == 0:
            return 0.72
        variation = np.std(widths) / (np.mean(widths) + 1e-8)
        return float(np.clip(1.0 - variation, 0.0, 1.0))
    
    def _check_blending_quality(self, image: np.ndarray) -> float:
        """
        Check blending quality using edge detection
        
        Args:
            image: Result image
        
        Returns:
            Blending quality score (0-1)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edge pixels
        edge_density = np.sum(edges > 0) / edges.size
        
        # Higher edge density might indicate blending artifacts
        # This is simplified - real implementation would be more sophisticated
        score = max(0, 1.0 - edge_density * 2)
        
        return float(score)

    def _sharpness_score(self, image: np.ndarray) -> float:
        """Compute sharpness via Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized = variance / (variance + 150.0)
        return float(np.clip(normalized, 0.0, 1.0))
    
    def _detect_face_distortion(
        self,
        image: np.ndarray,
        template_faces: List[Dict]
    ) -> float:
        """
        Detect face distortion - critical for "no plastic-looking faces" requirement
        
        Args:
            image: Result image
            template_faces: Template face detections with landmarks
        
        Returns:
            Distortion score (0.0 = no distortion, 1.0 = severe distortion)
        """
        if not template_faces:
            return 0.0
        
        distortion_scores = []
        
        for face in template_faces:
            bbox = face.get("bbox", [0, 0, 0, 0])
            landmarks = face.get("landmarks", [])
            
            if len(landmarks) < 5:  # Need at least 5 landmarks
                continue
            
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            if face_region.size == 0:
                continue
            
            # Check 1: Facial feature symmetry (distorted faces are asymmetric)
            symmetry_score = self._check_facial_symmetry(landmarks, w, h)
            
            # Check 2: Feature proportions (distorted faces have wrong proportions)
            proportion_score = self._check_feature_proportions(landmarks, w, h)
            
            # Check 3: Face region quality (distorted faces have artifacts)
            quality_score = self._check_face_region_quality(face_region)
            
            # Combine scores (higher = more distortion)
            distortion = (1.0 - symmetry_score) * 0.4 + (1.0 - proportion_score) * 0.3 + (1.0 - quality_score) * 0.3
            distortion_scores.append(distortion)
        
        if not distortion_scores:
            return 0.0
        
        return float(np.mean(distortion_scores))
    
    def _check_facial_symmetry(
        self,
        landmarks: List[List[float]],
        face_width: int,
        face_height: int
    ) -> float:
        """
        Check facial symmetry - distorted faces are asymmetric
        
        Returns:
            Symmetry score (1.0 = perfect symmetry, 0.0 = completely asymmetric)
        """
        if len(landmarks) < 5:
            return 0.8  # Default if insufficient landmarks
        
        landmarks = np.array(landmarks)
        
        # Find face center (typically nose or center of face)
        face_center_x = np.mean(landmarks[:, 0])
        
        # Check left-right symmetry of key features
        # For 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
        if len(landmarks) >= 5:
            # Mirror landmarks across center
            mirrored_x = 2 * face_center_x - landmarks[:, 0]
            
            # Find corresponding points (left eye <-> right eye, etc.)
            # Simplified: check if mirrored points are close to actual points
            # This is a simplified check - real implementation would match specific features
            
            # Calculate average distance from symmetry
            distances = []
            for i, (x, y) in enumerate(landmarks):
                # Find closest point on other side
                for j, (x2, y2) in enumerate(landmarks):
                    if i != j:
                        # Check if this could be symmetric pair
                        dist_to_mirror = abs((x + x2) / 2 - face_center_x)
                        if dist_to_mirror < face_width * 0.1:  # Close to center line
                            # Check vertical alignment
                            vertical_diff = abs(y - y2) / face_height
                            if vertical_diff < 0.15:  # Similar vertical position
                                distances.append(vertical_diff)
            
            if distances:
                avg_distance = np.mean(distances)
                symmetry = 1.0 - min(avg_distance * 2, 1.0)  # Normalize
            else:
                symmetry = 0.7  # Default if can't find pairs
        else:
            symmetry = 0.7
        
        return float(np.clip(symmetry, 0.0, 1.0))
    
    def _check_feature_proportions(
        self,
        landmarks: List[List[float]],
        face_width: int,
        face_height: int
    ) -> float:
        """
        Check facial feature proportions - distorted faces have wrong proportions
        
        Returns:
            Proportion score (1.0 = normal proportions, 0.0 = severely distorted)
        """
        if len(landmarks) < 5:
            return 0.8
        
        landmarks = np.array(landmarks)
        
        # Calculate key distances
        eye_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height_actual = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        
        # Normalize by face size
        if face_width > 0:
            eye_width_ratio = eye_width / face_width
        else:
            eye_width_ratio = 0.5
        
        if face_height > 0:
            face_height_ratio = face_height_actual / face_height
        else:
            face_height_ratio = 0.5
        
        # Check if proportions are reasonable
        # Normal face: eye width ~40-50% of face width, height reasonable
        eye_width_normal = 0.4 <= eye_width_ratio <= 0.6
        height_normal = 0.3 <= face_height_ratio <= 0.7
        
        # Calculate feature spread (distorted faces have features too close or too far)
        if len(landmarks) >= 3:
            feature_spread = np.std(landmarks[:, 0]) / face_width if face_width > 0 else 0.3
            spread_normal = 0.2 <= feature_spread <= 0.5
        else:
            spread_normal = True
        
        # Score based on how many checks pass
        score = 0.0
        if eye_width_normal:
            score += 0.4
        if height_normal:
            score += 0.3
        if spread_normal:
            score += 0.3
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _check_face_region_quality(
        self,
        face_region: np.ndarray
    ) -> float:
        """
        Check face region quality - distorted faces have artifacts
        
        Returns:
            Quality score (1.0 = high quality, 0.0 = severe artifacts)
        """
        if face_region.size == 0:
            return 0.0
        
        # Check 1: Color variance (distorted faces may have flat colors)
        if len(face_region.shape) == 3:
            std_dev = np.std(face_region)
            unique_colors = len(np.unique(face_region.reshape(-1, face_region.shape[-1]), axis=0))
        else:
            std_dev = np.std(face_region)
            unique_colors = len(np.unique(face_region))
        
        # Low variance or few colors indicates problems
        color_score = min(std_dev / 30.0, 1.0) if std_dev > 0 else 0.0
        color_score = min(color_score, unique_colors / 50.0)
        
        # Check 2: Sharpness (distorted faces may be blurry)
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(sharpness / 100.0, 1.0)
        
        # Check 3: Edge artifacts (distorted faces have unnatural edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        # Too many edges might indicate artifacts, but some edges are normal
        edge_score = 1.0 - min(abs(edge_density - 0.1) * 5, 1.0)  # Optimal around 10% edge density
        
        # Combine scores
        quality = color_score * 0.4 + sharpness_score * 0.4 + edge_score * 0.2
        
        return float(np.clip(quality, 0.0, 1.0))
    
    def _detect_edge_regions(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Detect edge/transition regions
        
        Args:
            image: Input image
            kernel_size: Kernel size for edge detection
        
        Returns:
            Edge region mask
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to create regions
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edge_mask = cv2.dilate(edges, kernel, iterations=2)
        
        return edge_mask
    
    def _detect_problem_areas(
        self,
        image: np.ndarray,
        quality_assessment: Dict
    ) -> np.ndarray:
        """
        Detect problem areas based on quality assessment
        
        Args:
            image: Input image
            quality_assessment: Quality assessment result
        
        Returns:
            Problem area mask
        """
        h, w = image.shape[:2]
        problem_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Simplified - detect areas with unusual colors or textures
        # Real implementation would use more sophisticated methods
        
        # Convert to LAB color space for better analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Detect areas with unusual brightness (potential blending issues)
        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)
        
        # Mark areas far from mean as potential problems
        threshold = mean_brightness + 2 * std_brightness
        problem_areas = (l_channel > threshold) | (l_channel < mean_brightness - 2 * std_brightness)
        
        problem_mask[problem_areas] = 255
        
        # Dilate to create regions
        kernel = np.ones((10, 10), np.uint8)
        problem_mask = cv2.dilate(problem_mask, kernel, iterations=1)
        
        return problem_mask
    
    def export_intermediate_results(
        self,
        pipeline_stages: Dict[str, np.ndarray],
        output_dir: str
    ):
        """
        Export intermediate results for review
        
        Args:
            pipeline_stages: Dictionary of stage names and images
            output_dir: Output directory
        """
        from pathlib import Path
        from ..utils.image_utils import save_image
        import numpy as np
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for stage_name, image in pipeline_stages.items():
            # Skip non-image items (dicts, lists, etc.)
            if not isinstance(image, np.ndarray):
                if isinstance(image, list):
                    # Handle lists of images
                    for idx, img in enumerate(image):
                        if isinstance(img, np.ndarray):
                            output_file = output_path / f"{stage_name}_{idx}.png"
                            save_image(img, output_file)
                            logger.info(f"Exported {stage_name}_{idx} to {output_file}")
                else:
                    logger.debug(f"Skipping {stage_name} (not an image: {type(image)})")
                continue
            
            output_file = output_path / f"{stage_name}.png"
            save_image(image, output_file)
            logger.info(f"Exported {stage_name} to {output_file}")

