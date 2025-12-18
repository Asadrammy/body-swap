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
        Warp landmarks to match target expression
        
        Args:
            source_landmarks: Source face landmarks
            source_keypoints: Key control points from source
            target_keypoints: Target key control points
        
        Returns:
            Warped landmarks
        """
        if len(source_landmarks) == 0:
            return source_landmarks
        
        if len(source_keypoints) < self._min_expression_points or len(target_keypoints) < self._min_expression_points:
            return source_landmarks
        
        src = source_keypoints.reshape(1, -1, 2).astype(np.float32)
        dst = target_keypoints.reshape(1, -1, 2).astype(np.float32)
        
        try:
            tps = cv2.createThinPlateSplineShapeTransformer()
            matches = [cv2.DMatch(i, i, 0) for i in range(len(source_keypoints))]
            tps.estimateTransformation(dst, src, matches)
            warped = tps.applyTransformation(source_landmarks.reshape(1, -1, 2))[1]
            return warped[0]
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
        """Warp face patch to match destination expression."""
        if src_points is None or dst_points is None:
            return patch
        if len(src_points) < self._min_expression_points or len(dst_points) < self._min_expression_points:
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
            warped = cv2.remap(patch, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            return warped
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

