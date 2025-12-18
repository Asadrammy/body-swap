"""Generative refinement using Stable Diffusion"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..models.generator import Generator

logger = get_logger(__name__)


class Refiner:
    """Refine composed images using generative models"""
    
    def __init__(self):
        """Initialize refiner"""
        self.generator = Generator()
        self.config = get_config()
        processing_cfg = self.config.get("processing", {})
        region_strengths = processing_cfg.get("region_refine_strengths", {})
        self.region_strengths = {
            "face": region_strengths.get("face", 0.65),
            "body": region_strengths.get("body", 0.55),
            "edges": region_strengths.get("edges", 0.45),
            "problems": region_strengths.get("problems", 0.7)
        }
    
    def refine_composition(
        self,
        composed_image: np.ndarray,
        template_analysis: Dict,
        customer_body_shape: Dict,
        refinement_mask: Optional[np.ndarray] = None,
        strength: float = None,
        region_masks: Optional[Dict[str, np.ndarray]] = None,
        quality: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Refine composed image using generative models
        
        Args:
            composed_image: Composed image to refine
            template_analysis: Template analysis result
            customer_body_shape: Customer body shape
            refinement_mask: Optional mask for selective refinement
            strength: Refinement strength (0-1)
        
        Returns:
            Refined image
        """
        if strength is None:
            strength = self.config.get("processing", {}).get("refinement_strength", 0.8)
        
        working = composed_image.copy()
        
        # Generate prompt based on template and customer
        prompt = self._generate_refinement_prompt(template_analysis, customer_body_shape)
        negative_prompt = "blurry, low quality, distorted, deformed, plastic, artificial, fake, unnatural"
        
        # Optional global pass
        if refinement_mask is not None:
            try:
                working = self.generator.refine(
                    image=working,
                    prompt=prompt,
                    mask=refinement_mask,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    num_inference_steps=30
                )
                logger.info("Global refinement pass completed")
            except Exception as e:
                logger.error(f"Global refinement failed: {e}")
        
        # Region-specific passes
        if region_masks:
            for region_name, mask in region_masks.items():
                if mask is None or np.max(mask) == 0:
                    continue
                region_prompt = self._prompt_for_region(
                    region_name,
                    template_analysis,
                    customer_body_shape,
                    quality
                )
                region_strength = self.region_strengths.get(region_name, strength or 0.6)
                try:
                    working = self.generator.refine(
                        image=working,
                        prompt=region_prompt,
                        mask=mask,
                        negative_prompt=negative_prompt,
                        strength=region_strength,
                        num_inference_steps=25
                    )
                    logger.info(f"Refined region '{region_name}' with strength {region_strength}")
                except Exception as exc:
                    logger.warning(f"Region refinement failed for {region_name}: {exc}")
        
        return working
    
    def refine_face(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        expression_type: str = "neutral",
        expression_details: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Refine face region specifically
        
        Args:
            image: Input image
            face_bbox: Face bounding box (x, y, w, h)
            expression_type: Expression type
        
        Returns:
            Image with refined face
        """
        x, y, w, h = face_bbox
        
        # Create face mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
        
        # Expand mask slightly for blending
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Generate prompt
        prompt = f"photorealistic face, natural skin texture, {expression_type} expression, high quality, detailed"
        negative_prompt = "plastic, artificial, fake, blurred, distorted face"
        
        guided_patch = None
        if expression_details and expression_details.get("target_local") is not None:
            control_map = self._build_face_control_map((h, w), expression_details["target_local"])
            if control_map is not None:
                face_patch = image[y:y+h, x:x+w]
                try:
                    guided_patch = self.generator.guided_face_refine(
                        face_patch,
                        control_map,
                        prompt=prompt,
                        negative_prompt=negative_prompt
                    )
                except Exception as exc:
                    logger.warning(f"Guided face refinement failed: {exc}")
                    guided_patch = None
        
        if guided_patch is not None:
            result = image.copy()
            result[y:y+h, x:x+w] = guided_patch
            return result
        
        try:
            refined = self.generator.refine(
                image=image,
                prompt=prompt,
                mask=mask,
                strength=0.7,
                num_inference_steps=25
            )
            
            return refined
            
        except Exception as e:
            logger.error(f"Face refinement failed: {e}")
            return image
    
    def refine_clothing(
        self,
        image: np.ndarray,
        clothing_mask: np.ndarray,
        clothing_type: str = "clothing"
    ) -> np.ndarray:
        """
        Refine clothing region
        
        Args:
            image: Input image
            clothing_mask: Clothing mask
            clothing_type: Type of clothing
        
        Returns:
            Image with refined clothing
        """
        prompt = f"photorealistic {clothing_type}, natural fabric folds, realistic texture, high quality"
        negative_prompt = "artificial, fake, distorted clothing, plastic appearance"
        
        try:
            refined = self.generator.refine(
                image=image,
                prompt=prompt,
                mask=clothing_mask,
                strength=0.6,
                num_inference_steps=25
            )
            
            return refined
    
    def _build_face_control_map(self, size: Tuple[int, int], landmarks: np.ndarray) -> Optional[np.ndarray]:
        """Create a simple control map drawing landmark connections."""
        if landmarks is None or len(landmarks) < 2:
            return None
        
        h, w = size
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        pts = np.clip(landmarks.astype(np.int32), 0, [w - 1, h - 1])
        
        # Draw outline
        for i in range(len(pts) - 1):
            cv2.line(canvas, tuple(pts[i]), tuple(pts[i+1]), (255, 255, 255), 1)
        for pt in pts:
            cv2.circle(canvas, tuple(pt), 1, (255, 255, 255), -1)
        return canvas
            
        except Exception as e:
            logger.error(f"Clothing refinement failed: {e}")
            return image
    
    def _generate_refinement_prompt(
        self,
        template_analysis: Dict,
        customer_body_shape: Dict
    ) -> str:
        """
        Generate refinement prompt
        
        Args:
            template_analysis: Template analysis
            customer_body_shape: Customer body shape
        
        Returns:
            Refinement prompt
        """
        prompts = ["photorealistic", "high quality", "detailed"]
        
        # Add body type
        body_type = customer_body_shape.get("body_type", "average")
        if body_type != "unknown":
            prompts.append(f"realistic {body_type} body")
        
        # Add clothing info
        clothing = template_analysis.get("clothing", {})
        clothing_items = clothing.get("items", [])
        if clothing_items:
            prompts.append("natural clothing fit")
        
        # Add expression
        expression = template_analysis.get("expression", {})
        expression_type = expression.get("type", "neutral")
        prompts.append(f"{expression_type} expression")
        
        # Add background
        prompts.append("seamless background")
        
        return ", ".join(prompts)

    def _prompt_for_region(
        self,
        region: str,
        template_analysis: Dict,
        customer_body_shape: Dict,
        quality_metrics: Optional[Dict]
    ) -> str:
        """Generate region-targeted prompt."""
        base = self._generate_refinement_prompt(template_analysis, customer_body_shape)
        expression_hint = "neutral"
        if quality_metrics and quality_metrics.get("issues"):
            expression_hint = "confident"
        region_prompts = {
            "face": f"hyper-detailed face, natural skin micro-texture, {expression_hint} expression, precise features",
            "body": "tailored clothing fit, realistic fabric folding, accurate body proportions, natural shading",
            "edges": "feathered transitions, remove halos, seamless blend between subject and background",
            "problems": "clean artifacts, remove noise, fix lighting inconsistencies, photorealistic detail"
        }
        region_text = region_prompts.get(region, "high fidelity details, photorealistic finish")
        return f"{region_text}, {base}"

