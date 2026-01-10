"""Face processing for identity extraction and expression matching"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..models.face_detector import FaceDetector
from ..utils.image_utils import resize_image, crop_image

logger = get_logger(__name__)


class FaceProcessor:
    """Process faces for identity extraction and expression matching"""
    
    def __init__(self):
        """Initialize face processor"""
        self.face_detector = FaceDetector()
        self.alignment_size = (112, 112)  # Standard face alignment size
        self._min_expression_points = 3
        self._max_expression_points = 20
    
    def extract_face_identity(self, image: np.ndarray, face: Dict) -> Dict:
        """
        Extract face identity features
        
        Args:
            image: Customer reference image
            face: Face detection result
        
        Returns:
            Face identity information
        """
        bbox = face.get("bbox", [0, 0, 0, 0])
        landmarks = face.get("landmarks", [])
        
        # Align face and derive normalization parameters
        aligned_face = self.face_detector.align_face(image, landmarks, self.alignment_size)
        normalized_landmarks = self._normalize_landmarks(bbox, landmarks)
        
        # Extract embedding
        embedding = self.face_detector.extract_face_embedding(image, bbox)
        embedding_vector = None
        if embedding is not None:
            embedding_vector = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Extract additional features
        identity = {
            "bbox": bbox,
            "landmarks": landmarks,
            "normalized_landmarks": normalized_landmarks,
            "aligned_face": aligned_face,
            "embedding": embedding.tolist() if embedding is not None else None,
            "embedding_vector": embedding_vector,
            "face_area": float(bbox[2] * bbox[3]),
            "age": face.get("age"),
            "gender": face.get("gender")
        }
        
        return identity
    
    def match_expression(self, customer_face: Dict, template_face: Dict,
                        template_expression: Dict) -> Dict:
        """
        Match customer face to template expression
        
        Args:
            customer_face: Customer face identity
            template_face: Template face detection
            template_expression: Template expression analysis
        
        Returns:
            Expression matching result with warped landmarks
        """
        customer_landmarks = np.array(customer_face.get("landmarks", []))
        template_landmarks = np.array(template_face.get("landmarks", []))
        
        if len(customer_landmarks) == 0 or len(template_landmarks) == 0:
            logger.warning("Insufficient landmarks for expression matching")
            return {
                "warped_landmarks": customer_landmarks.tolist(),
                "expression_applied": False
            }
        
        customer_key = self._select_expression_points(customer_landmarks)
        template_key = self._select_expression_points(template_landmarks)
        
        if customer_key.size >= self._min_expression_points <= template_key.size:
            warped_landmarks = self._warp_landmarks_for_expression(
                customer_landmarks,
                customer_key,
                template_key
            )
            local_source = self._convert_to_local_coords(
                customer_landmarks,
                customer_face.get("bbox", [0, 0, 1, 1]),
                template_face.get("bbox", [0, 0, 1, 1])
            )
            local_target = self._convert_to_local_coords(
                warped_landmarks,
                template_face.get("bbox", [0, 0, 1, 1]),
                template_face.get("bbox", [0, 0, 1, 1])
            )
            result = {
                "warped_landmarks": warped_landmarks.tolist(),
                "expression_applied": True,
                "expression_type": template_expression.get("type", "neutral"),
                "source_local": local_source,
                "target_local": local_target
            }
            # Preserve full emotion data from template for Mickmumpitz workflow
            if isinstance(template_expression, dict):
                # Copy emotion data (keywords, intensity, descriptors, etc.)
                for key in ["keywords", "intensity", "descriptors", "features"]:
                    if key in template_expression:
                        result[key] = template_expression[key]
        else:
            warped_landmarks = customer_landmarks
            result = {
                "warped_landmarks": warped_landmarks.tolist(),
                "expression_applied": False
            }
        
        return result
    
    def _warp_landmarks_for_expression(
        self,
        source_landmarks: np.ndarray,
        source_keypoints: np.ndarray,
        target_keypoints: np.ndarray
    ) -> np.ndarray:
        """
        Warp landmarks to match target expression with distortion validation
        
        Args:
            source_landmarks: Source face landmarks
            source_keypoints: Key control points from source
            target_keypoints: Target key control points
        
        Returns:
            Warped landmarks (or original if distortion too high)
        """
        if len(source_landmarks) == 0:
            return source_landmarks
        
        if len(source_keypoints) < self._min_expression_points or len(target_keypoints) < self._min_expression_points:
            return source_landmarks
        
        # Check for excessive displacement before warping
        displacements = np.linalg.norm(target_keypoints - source_keypoints, axis=1)
        max_displacement = np.max(displacements)
        mean_displacement = np.mean(displacements)
        
        # Calculate face size for normalization
        face_size = np.max([
            np.max(source_keypoints[:, 0]) - np.min(source_keypoints[:, 0]),
            np.max(source_keypoints[:, 1]) - np.min(source_keypoints[:, 1])
        ])
        if face_size < 1.0:
            face_size = 1.0
        
        # Normalize displacement by face size
        normalized_max = max_displacement / face_size
        normalized_mean = mean_displacement / face_size
        
        # Reject if displacement is too large (would cause distortion)
        # Made more conservative: Max displacement < 20% of face size, mean < 10%
        # This prevents the distorted face issue seen in output images
        if normalized_max > 0.2 or normalized_mean > 0.1:
            logger.warning(f"Expression warp rejected: excessive displacement (max={normalized_max:.2f}, mean={normalized_mean:.2f}) - using original landmarks to prevent distortion")
            return source_landmarks
        
        src = source_keypoints.reshape(1, -1, 2).astype(np.float32)
        dst = target_keypoints.reshape(1, -1, 2).astype(np.float32)
        
        try:
            tps = cv2.createThinPlateSplineShapeTransformer()
            matches = [cv2.DMatch(i, i, 0) for i in range(len(source_keypoints))]
            tps.estimateTransformation(dst, src, matches)
            warped = tps.applyTransformation(source_landmarks.reshape(1, -1, 2))[1]
            
            # Validate warped landmarks for distortion
            warped_landmarks = warped[0]
            if self._validate_landmark_distortion(source_landmarks, warped_landmarks):
                return warped_landmarks
            else:
                logger.warning("Expression warp rejected: excessive distortion detected")
                return source_landmarks
        except Exception as exc:
            logger.warning(f"Expression TPS warp failed: {exc}")
            return source_landmarks
    
    def composite_face(
        self,
        customer_face: np.ndarray,
        template_image: np.ndarray,
        template_face: Dict,
        expression_match: Dict
    ) -> np.ndarray:
        """
        Composite customer face into template
        
        Args:
            customer_face: Aligned customer face image
            template_image: Template image
            template_face: Template face detection
            expression_match: Expression matching result
        
        Returns:
            Composited image with customer face
        """
        # Get template face region
        template_bbox = template_face.get("bbox", [0, 0, 0, 0])
        x, y, w, h = template_bbox
        
        # Resize customer face to match template face size
        customer_resized = resize_image(customer_face, (w, h))
        
        # Apply expression deformation in local coordinates
        if expression_match.get("expression_applied"):
            customer_resized = self._apply_expression_to_patch(
                customer_resized,
                expression_match.get("source_local"),
                expression_match.get("target_local"),
                (w, h)
            )
        
        # Create mask for blending (feathering edges)
        mask = np.ones((h, w), dtype=np.float32)
        
        # Feather edges
        feather_size = max(2, min(w, h) // 12)
        falloff = np.linspace(0, 1, feather_size, dtype=np.float32)
        for i, alpha in enumerate(falloff):
            mask[i, :] *= alpha
            mask[-i-1, :] *= alpha
            mask[:, i] *= alpha
            mask[:, -i-1] *= alpha
        
        # Validate customer face before compositing (prevent distortion)
        if not self._validate_face_patch(customer_resized):
            logger.warning("Customer face patch validation failed - using original template face")
            return template_image.copy()
        
        # Blend face into template
        result = template_image.copy()
        face_region = result[y:y+h, x:x+w]
        
        # Apply blending
        for c in range(3):
            face_region[:, :, c] = (
                face_region[:, :, c] * (1 - mask) +
                customer_resized[:, :, c] * mask
            ).astype(np.uint8)
        
        result[y:y+h, x:x+w] = face_region
        
        # Validate composited result
        if not self._validate_composited_face(result, x, y, w, h):
            logger.warning("Composited face validation failed - using original template face")
            return template_image.copy()
        
        return result
    
    def process_multiple_faces(
        self,
        customer_images: List[np.ndarray],
        customer_faces_list: List[List[Dict]],
        template_image: np.ndarray,
        template_faces: List[Dict]
    ) -> np.ndarray:
        """
        Process multiple faces (for couples, families, etc.)
        
        Args:
            customer_images: List of customer images
            customer_faces_list: List of face detections for each customer image
            template_image: Template image
            template_faces: Template face detections
        
        Returns:
            Composited image with all faces
        """
        result = template_image.copy()
        
        identities = self._prepare_customer_identities(customer_images, customer_faces_list)
        if not identities or not template_faces:
            return result
        
        assignments = self._match_customers_to_template_faces(identities, template_faces)
        
        # Apply in occlusion order (background first)
        ordered_faces = sorted(
            enumerate(template_faces),
            key=lambda item: item[1].get("bbox", [0, 0, 0, 0])[1] + item[1].get("bbox", [0, 0, 0, 0])[3]
        )
        
        for template_idx, template_face in ordered_faces:
            identity = assignments.get(template_idx)
            if not identity:
                continue
            
            expression_match = self.match_expression(
                identity,
                template_face,
                {"type": "neutral"}
            )
            
            result = self.composite_face(
                identity["aligned_face"],
                result,
                template_face,
                expression_match
            )
        
        return result

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _normalize_landmarks(self, bbox: List[int], landmarks: List[List[float]]) -> List[List[float]]:
        """Normalize landmarks into bbox-relative space."""
        if not landmarks:
            return []
        x, y, w, h = bbox
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        normalized = []
        for lx, ly in landmarks:
            normalized.append([(lx - x) / w, (ly - y) / h])
        return normalized

    def _select_expression_points(self, landmarks: np.ndarray) -> np.ndarray:
        """Select a stable subset of landmarks for expression transfer."""
        if landmarks.size == 0:
            return np.array([])
        
        total = len(landmarks)
        if total <= self._max_expression_points:
            return landmarks
        
        step = max(1, total // self._max_expression_points)
        return landmarks[::step]

    def _convert_to_local_coords(
        self,
        landmarks: np.ndarray,
        source_bbox: List[int],
        template_bbox: List[int]
    ) -> np.ndarray:
        """Convert global landmarks into template-local coordinates."""
        x_s, y_s, w_s, h_s = source_bbox
        x_t, y_t, w_t, h_t = template_bbox
        w_s = max(1.0, float(w_s))
        h_s = max(1.0, float(h_s))
        coords = []
        for (lx, ly) in landmarks:
            norm_x = (lx - x_s) / w_s
            norm_y = (ly - y_s) / h_s
            coords.append([norm_x * w_t, norm_y * h_t])
        return np.array(coords, dtype=np.float32)

    def _apply_expression_to_patch(
        self,
        patch: np.ndarray,
        src_points: Optional[np.ndarray],
        dst_points: Optional[np.ndarray],
        size: Tuple[int, int]
    ) -> np.ndarray:
        """Warp face patch to match destination expression with distortion validation."""
        if src_points is None or dst_points is None:
            return patch
        if len(src_points) < self._min_expression_points or len(dst_points) < self._min_expression_points:
            return patch
        
        # Validate displacement before warping
        displacements = np.linalg.norm(dst_points - src_points, axis=1)
        max_displacement = np.max(displacements)
        mean_displacement = np.mean(displacements)
        
        # Normalize by patch size
        patch_size = max(size[0], size[1])
        if patch_size < 1.0:
            patch_size = 1.0
        
        normalized_max = max_displacement / patch_size
        normalized_mean = mean_displacement / patch_size
        
        # Reject if displacement too large - made more conservative to prevent distortion
        # Max displacement < 20% of patch size, mean < 10%
        if normalized_max > 0.2 or normalized_mean > 0.1:
            logger.warning(f"Patch warp rejected: excessive displacement (max={normalized_max:.2f}, mean={normalized_mean:.2f}) - using original patch to prevent distortion")
            return patch
        
        try:
            tps = cv2.createThinPlateSplineShapeTransformer()
            src = src_points.reshape(1, -1, 2).astype(np.float32)
            dst = dst_points.reshape(1, -1, 2).astype(np.float32)
            matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
            tps.estimateTransformation(dst, src, matches)
            
            h, w = size[1], size[0]
            y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
            grid_points = np.dstack([x_grid.flatten(), y_grid.flatten()]).reshape(1, -1, 2)
            transformed = tps.applyTransformation(grid_points)[1]
            map_x = transformed[0, :, 0].reshape(h, w).astype(np.float32)
            map_y = transformed[0, :, 1].reshape(h, w).astype(np.float32)
            
            # Check for excessive warping in the mapping
            if self._validate_warp_mapping(map_x, map_y, w, h):
                warped = cv2.remap(patch, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                # Validate warped result
                if self._validate_warped_patch(patch, warped):
                    return warped
                else:
                    logger.warning("Patch warp rejected: warped result validation failed")
                    return patch
            else:
                logger.warning("Patch warp rejected: mapping validation failed")
                return patch
        except Exception as exc:
            logger.warning(f"Expression patch warp failed: {exc}")
            return patch

    def _prepare_customer_identities(
        self,
        customer_images: List[np.ndarray],
        customer_faces_list: List[List[Dict]]
    ) -> List[Dict]:
        """Extract identities from all customer photos."""
        identities = []
        for img, faces in zip(customer_images, customer_faces_list):
            for face in faces:
                try:
                    identity = self.extract_face_identity(img, face)
                    identity["image"] = img
                    identity["face"] = face
                    identities.append(identity)
                except Exception as exc:
                    logger.warning(f"Failed to extract customer identity: {exc}")
        return identities

    def _match_customers_to_template_faces(
        self,
        identities: List[Dict],
        template_faces: List[Dict]
    ) -> Dict[int, Dict]:
        """Match customer identities to template faces based on face size similarity."""
        remaining = identities.copy()
        assignments: Dict[int, Dict] = {}
        template_pairs = list(enumerate(template_faces))
        
        for idx, template_face in template_pairs:
            bbox = template_face.get("bbox", [0, 0, 0, 0])
            template_area = bbox[2] * bbox[3]
            best_identity = None
            best_delta = float("inf")
            
            for identity in remaining:
                delta = abs(identity.get("face_area", template_area) - template_area)
                if delta < best_delta:
                    best_delta = delta
                    best_identity = identity
            
            if best_identity:
                assignments[idx] = best_identity
                remaining.remove(best_identity)
            else:
                logger.warning("Not enough customer faces to fill template")
        
        return assignments
    
    def _validate_landmark_distortion(
        self,
        original: np.ndarray,
        warped: np.ndarray
    ) -> bool:
        """
        Validate that warped landmarks don't have excessive distortion
        
        Args:
            original: Original landmarks
            warped: Warped landmarks
        
        Returns:
            True if distortion is acceptable, False otherwise
        """
        if len(original) != len(warped):
            return False
        
        # Calculate displacement for each landmark
        displacements = np.linalg.norm(warped - original, axis=1)
        max_displacement = np.max(displacements)
        mean_displacement = np.mean(displacements)
        
        # Calculate face size for normalization
        face_size = np.max([
            np.max(original[:, 0]) - np.min(original[:, 0]),
            np.max(original[:, 1]) - np.min(original[:, 1])
        ])
        if face_size < 1.0:
            face_size = 1.0
        
        # Normalize displacement
        normalized_max = max_displacement / face_size
        normalized_mean = mean_displacement / face_size
        
        # Made more conservative: Accept if max < 20% and mean < 10%
        # This prevents the distorted face issue seen in output images
        return normalized_max < 0.2 and normalized_mean < 0.1
    
    def _validate_warp_mapping(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        width: int,
        height: int
    ) -> bool:
        """
        Validate that warp mapping doesn't have excessive distortion
        
        Args:
            map_x: X coordinate mapping
            map_y: Y coordinate mapping
            width: Image width
            height: Image height
        
        Returns:
            True if mapping is valid, False otherwise
        """
        # Check for NaN or Inf values
        if np.any(np.isnan(map_x)) or np.any(np.isnan(map_y)):
            return False
        if np.any(np.isinf(map_x)) or np.any(np.isinf(map_y)):
            return False
        
        # Check that mapping stays within reasonable bounds
        # Allow some margin for border reflection
        margin = 0.1 * min(width, height)
        if np.any(map_x < -margin) or np.any(map_x > width + margin):
            return False
        if np.any(map_y < -margin) or np.any(map_y > height + margin):
            return False
        
        # Check for excessive local distortion (gradient)
        grad_x = np.gradient(map_x)
        grad_y = np.gradient(map_y)
        max_grad = max(np.max(np.abs(grad_x[0])), np.max(np.abs(grad_x[1])),
                      np.max(np.abs(grad_y[0])), np.max(np.abs(grad_y[1])))
        
        # Gradient should not be too large (indicates excessive warping)
        if max_grad > 3.0:
            return False
        
        return True
    
    def _validate_warped_patch(
        self,
        original: np.ndarray,
        warped: np.ndarray
    ) -> bool:
        """
        Validate that warped patch doesn't have excessive artifacts
        
        Args:
            original: Original patch
            warped: Warped patch
        
        Returns:
            True if warped patch is valid, False otherwise
        """
        if original.shape != warped.shape:
            return False
        
        # Check for solid color (indicates warping failure)
        if len(warped.shape) == 3:
            unique_colors = len(np.unique(warped.reshape(-1, warped.shape[-1]), axis=0))
            std_dev = np.std(warped)
        else:
            unique_colors = len(np.unique(warped))
            std_dev = np.std(warped)
        
        # Reject if too few colors or too low variance
        if unique_colors < 10 or std_dev < 5.0:
            return False
        
        # Check for excessive blur (indicates warping artifacts)
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
        
        sharpness_original = cv2.Laplacian(gray_original, cv2.CV_64F).var()
        sharpness_warped = cv2.Laplacian(gray_warped, cv2.CV_64F).var()
        
        # Warped should not be significantly less sharp (indicates blur from warping)
        if sharpness_warped < sharpness_original * 0.3:
            return False
        
        return True
    
    def _validate_face_patch(self, patch: np.ndarray) -> bool:
        """
        Validate face patch before compositing to prevent distortion
        
        Args:
            patch: Face patch to validate
        
        Returns:
            True if patch is valid, False otherwise
        """
        if patch is None or patch.size == 0:
            return False
        
        # Check for solid color
        if len(patch.shape) == 3:
            unique_colors = len(np.unique(patch.reshape(-1, patch.shape[-1]), axis=0))
            std_dev = np.std(patch)
        else:
            unique_colors = len(np.unique(patch))
            std_dev = np.std(patch)
        
        if unique_colors < 10 or std_dev < 5.0:
            return False
        
        # Check for reasonable sharpness (too blurry indicates problems)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < 20.0:  # Too blurry
            return False
        
        return True
    
    def _validate_composited_face(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int
    ) -> bool:
        """
        Validate composited face region for distortion
        
        Args:
            image: Full image
            x, y, w, h: Face region coordinates
        
        Returns:
            True if composited face is valid, False otherwise
        """
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            return False
        
        face_region = image[y:y+h, x:x+w]
        if face_region.size == 0:
            return False
        
        # Check for artifacts (excessive edges might indicate blending issues)
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Too many edges (>30%) might indicate artifacts
        if edge_density > 0.3:
            return False
        
        return True

