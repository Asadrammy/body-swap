"""Quality control and manual touch-up interface"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.config import get_config

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
    
    def assess_quality(
        self,
        result_image: np.ndarray,
        customer_faces: List[Dict],
        template_faces: List[Dict],
        template_analysis: Dict,
        body_shape: Optional[Dict] = None
    ) -> Dict:
        """
        Assess quality of swap result
        
        Args:
            result_image: Result image
            customer_faces: Customer face detections
            template_faces: Template face detections
            template_analysis: Template analysis
        
        Returns:
            Quality assessment result
        """
        face_similarity = self._compute_face_similarity(customer_faces, template_faces)
        pose_accuracy = self._compute_pose_alignment(body_shape, template_analysis.get("pose"))
        clothing_fit = self._estimate_clothing_fit(body_shape)
        seamless_blending = self._check_blending_quality(result_image)
        sharpness = self._sharpness_score(result_image)
        
        overall_score = (
            face_similarity * 0.3 +
            pose_accuracy * 0.2 +
            clothing_fit * 0.2 +
            seamless_blending * 0.2 +
            sharpness * 0.1
        )
        
        issues = []
        recommended = []
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
        
        quality_metrics = {
            "overall_score": float(np.clip(overall_score, 0.0, 1.0)),
            "face_similarity": float(face_similarity),
            "pose_accuracy": float(pose_accuracy),
            "clothing_fit": float(clothing_fit),
            "seamless_blending": float(seamless_blending),
            "sharpness": float(sharpness),
            "issues": issues,
            "recommended_refinements": list(dict.fromkeys(recommended)),
            "threshold": self.quality_threshold
        }
        
        return quality_metrics
    
    def generate_refinement_masks(
        self,
        result_image: np.ndarray,
        quality_assessment: Dict,
        face_regions: List[Tuple[int, int, int, int]],
        body_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate masks for manual refinement
        
        Args:
            result_image: Result image
            quality_assessment: Quality assessment result
            face_regions: Face bounding boxes
            body_mask: Optional body mask
        
        Returns:
            Dictionary of refinement masks for different regions
        """
        h, w = result_image.shape[:2]
        masks = {}
        
        # Face region masks
        if face_regions:
            face_mask = np.zeros((h, w), dtype=np.uint8)
            for x, y, bw, bh in face_regions:
                cv2.rectangle(face_mask, (x, y), (x+bw, y+bh), 255, -1)
            masks["face"] = face_mask
        
        # Body region mask
        if body_mask is not None:
            masks["body"] = body_mask
        
        # Edge/transition regions (for blending issues)
        edge_mask = self._detect_edge_regions(result_image)
        masks["edges"] = edge_mask
        
        # Problem areas based on quality assessment
        if quality_assessment.get("issues"):
            problem_mask = self._detect_problem_areas(result_image, quality_assessment)
            masks["problems"] = problem_mask
        
        return masks

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
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for stage_name, image in pipeline_stages.items():
            output_file = output_path / f"{stage_name}.png"
            save_image(image, output_file)
            logger.info(f"Exported {stage_name} to {output_file}")

