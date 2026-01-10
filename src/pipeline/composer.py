"""Image composition and blending"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.image_utils import blend_images

logger = get_logger(__name__)


class Composer:
    """Compose warped body and face into template background"""
    
    def __init__(self):
        """Initialize composer"""
        pass
    
    def compose(
        self,
        warped_body: np.ndarray,
        template_background: np.ndarray,
        body_mask: Optional[np.ndarray] = None,
        lighting_info: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Compose warped body into template background
        
        Args:
            warped_body: Warped body image
            template_background: Template background
            body_mask: Optional body mask
            lighting_info: Lighting information for matching
        
        Returns:
            Composed image
        """
        # Match sizes
        h_bg, w_bg = template_background.shape[:2]
        h_body, w_body = warped_body.shape[:2]
        
        if (h_body, w_body) != (h_bg, w_bg):
            warped_body = cv2.resize(warped_body, (w_bg, h_bg), interpolation=cv2.INTER_LINEAR)
            if body_mask is not None:
                # Resize mask using nearest neighbor to preserve binary values
                body_mask = cv2.resize(body_mask, (w_bg, h_bg), interpolation=cv2.INTER_NEAREST)
        
        # Ensure body_mask matches warped_body dimensions (safety check)
        if body_mask is not None:
            h_mask, w_mask = body_mask.shape[:2]
            if (h_mask, w_mask) != (h_bg, w_bg):
                body_mask = cv2.resize(body_mask, (w_bg, h_bg), interpolation=cv2.INTER_NEAREST)
        
        # Match lighting
        if lighting_info:
            warped_body = self._match_lighting(warped_body, template_background, lighting_info)
        
        # Validate inputs
        if warped_body is None or warped_body.size == 0:
            logger.warning("Warped body is empty, returning template background")
            return template_background.copy()
        
        # Blend
        if body_mask is not None:
            result = blend_images(template_background, warped_body, mask=body_mask)
        else:
            # Default blending
            result = blend_images(template_background, warped_body, alpha=0.9)
        
        # Validate result
        if result is None or result.size == 0:
            logger.warning("Composition result is empty, returning template background")
            return template_background.copy()
        
        # Check if result is solid color
        if len(result.shape) == 3:
            unique_colors = len(np.unique(result.reshape(-1, result.shape[-1]), axis=0))
            std_dev = np.std(result)
        else:
            unique_colors = len(np.unique(result))
            std_dev = np.std(result)
        
        if unique_colors < 10 or std_dev < 5.0:
            logger.warning(f"Composition result is solid color (unique_colors={unique_colors}, std={std_dev:.2f}), returning template background")
            return template_background.copy()
        
        return result
    
    def match_shadows(
        self,
        composed_image: np.ndarray,
        body_mask: np.ndarray,
        lighting_info: Dict
    ) -> np.ndarray:
        """
        Add realistic shadows based on lighting
        
        Args:
            composed_image: Composed image
            body_mask: Body mask
            lighting_info: Lighting information
        
        Returns:
            Image with shadows added
        """
        # Create shadow mask
        shadow_mask = self._generate_shadow_mask(body_mask, lighting_info)
        
        # Apply shadow
        shadow = np.zeros_like(composed_image)
        shadow_intensity = 0.3
        
        result = composed_image.copy()
        shadow_3d = shadow_mask[:, :, np.newaxis] if len(shadow_mask.shape) == 2 else shadow_mask
        
        result = (result * (1 - shadow_3d * shadow_intensity)).astype(np.uint8)
        
        return result
    
    def blend_skin_tone(
        self,
        customer_face: np.ndarray,
        template_image: np.ndarray,
        face_region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Blend skin tones naturally
        
        Args:
            customer_face: Customer face image
            template_image: Template image
            face_region: Face region (x, y, w, h)
        
        Returns:
            Image with blended skin tones
        """
        x, y, w, h = face_region
        result = template_image.copy()
        
        # Extract face region
        face_crop = result[y:y+h, x:x+w]
        
        # Match color histogram
        matched_face = self._match_color_histogram(customer_face, face_crop)
        
        # Blend
        mask = np.ones((h, w), dtype=np.float32)
        feather = min(w, h) // 15
        
        for i in range(feather):
            alpha = i / feather
            mask[i, :] *= alpha
            mask[-i-1, :] *= alpha
            mask[:, i] *= alpha
            mask[:, -i-1] *= alpha
        
        mask_3d = mask[:, :, np.newaxis]
        face_region_blended = (
            face_crop * (1 - mask_3d) +
            matched_face * mask_3d
        ).astype(np.uint8)
        
        result[y:y+h, x:x+w] = face_region_blended
        
        return result
    
    def _match_lighting(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        lighting_info: Dict
    ) -> np.ndarray:
        """
        Match lighting between source and target
        
        Args:
            source_image: Source image
            target_image: Target image (template)
            lighting_info: Lighting information
        
        Returns:
            Image with matched lighting
        """
        # Calculate mean and std for color matching
        target_mean = np.mean(target_image, axis=(0, 1))
        target_std = np.std(target_image, axis=(0, 1))
        
        source_mean = np.mean(source_image, axis=(0, 1))
        source_std = np.std(source_image, axis=(0, 1))
        
        # Normalize source to match target
        result = source_image.copy().astype(np.float32)
        
        for c in range(3):
            if source_std[c] > 0:
                result[:, :, c] = (
                    (result[:, :, c] - source_mean[c]) / source_std[c] * target_std[c] +
                    target_mean[c]
                )
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _generate_shadow_mask(
        self,
        body_mask: np.ndarray,
        lighting_info: Dict
    ) -> np.ndarray:
        """
        Generate shadow mask based on body and lighting
        
        Args:
            body_mask: Body mask
            lighting_info: Lighting information
        
        Returns:
            Shadow mask
        """
        h, w = body_mask.shape
        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create shadow below body (simplified)
        # Real implementation would consider light direction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        dilated = cv2.dilate(body_mask, kernel, iterations=1)
        
        # Create shadow region (below body)
        body_coords = np.column_stack(np.where(body_mask > 0))
        if len(body_coords) > 0:
            body_bottom = body_coords[:, 0].max()
            shadow_top = min(h - 1, body_bottom + 10)
            shadow_bottom = min(h - 1, shadow_top + 50)
            
            shadow_region = dilated[shadow_top:shadow_bottom, :]
            shadow_mask[shadow_top:shadow_bottom, :] = shadow_region
        
        # Blur shadow
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        
        return shadow_mask
    
    def _match_color_histogram(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """
        Match color histogram between source and target
        
        Args:
            source: Source image
            target: Target image
        
        Returns:
            Color-matched source image
        """
        matched = source.copy()
        
        for c in range(3):
            # Calculate histograms
            source_hist, _ = np.histogram(source[:, :, c].flatten(), 256, [0, 256])
            target_hist, _ = np.histogram(target[:, :, c].flatten(), 256, [0, 256])
            
            # Calculate cumulative distributions
            source_cdf = source_hist.cumsum()
            source_cdf = 255 * source_cdf / source_cdf[-1]
            
            target_cdf = target_hist.cumsum()
            target_cdf = 255 * target_cdf / target_cdf[-1]
            
            # Create mapping
            mapping = np.interp(source_cdf, target_cdf, np.arange(256))
            
            # Apply mapping
            matched[:, :, c] = mapping[source[:, :, c]]
        
        return matched

