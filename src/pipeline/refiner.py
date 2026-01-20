"""Generative refinement using Stable Diffusion"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..models.generator import Generator
from .emotion_handler import EmotionHandler

logger = get_logger(__name__)


class Refiner:
    """Refine composed images using generative models"""
    
    def __init__(self):
        """Initialize refiner"""
        self.generator = Generator()
        self.config = get_config()
        self.emotion_handler = EmotionHandler()
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
        quality: Optional[Dict] = None,
        custom_prompt: Optional[str] = None
    ) -> np.ndarray:
        """
        Refine composed image using generative models
        
        Args:
            composed_image: Composed image to refine
            template_analysis: Template analysis result
            customer_body_shape: Customer body shape
            refinement_mask: Optional mask for selective refinement
            strength: Refinement strength (0-1)
            custom_prompt: Optional custom prompt for AI generation
        
        Returns:
            Refined image
        """
        if strength is None:
            strength = self.config.get("processing", {}).get("refinement_strength", 0.8)
        
        working = composed_image.copy()
        
        # Ensure template_analysis and customer_body_shape are not None
        template_analysis = template_analysis or {}
        customer_body_shape = customer_body_shape or {}
        
        # Use custom prompt if provided, otherwise generate prompt based on template and customer
        if custom_prompt:
            prompt = custom_prompt
            logger.info(f"Using custom prompt: {custom_prompt[:100]}...")
        else:
            prompt = self._generate_refinement_prompt(template_analysis, customer_body_shape)
        negative_prompt = "solid color, single color, flat color, blue, pink, red, green, yellow, monochrome, uniform color, color block, blurry, low quality, distorted, deformed, plastic, artificial, fake, unnatural, CGI, 3D render, cartoon, painting, drawing"
        
        # Check if generator can be initialized (lazy loading)
        # The generator will load on first use, so we don't check here
        # Instead, we'll let it try to load and handle errors gracefully
        
        # Optional global pass
        if refinement_mask is not None:
            try:
                logger.info("=" * 80)
                logger.info("🎨 CALLING STABILITY AI API FOR IMAGE REFINEMENT")
                logger.info("=" * 80)
                logger.info(f"📝 Prompt: {prompt[:150]}..." if len(prompt) > 150 else f"📝 Prompt: {prompt}")
                logger.info(f"⚙️  Strength: {strength}")
                logger.info(f"📐 Image shape: {working.shape}")
                logger.info(f"🎭 Mask provided: Yes")
                # Use config value for inference steps (minimum 30 enforced in generator)
                inference_steps = self.config.get("processing", {}).get("num_inference_steps", 40)
                logger.info(f"⏱️  Inference steps: {inference_steps}")
                logger.info("⏳ Calling Stability AI API...")
                refined_global = self.generator.refine(
                    image=working,
                    prompt=prompt,
                    mask=refinement_mask,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    num_inference_steps=inference_steps
                )
                logger.info("✅ Stability AI API call completed")
                logger.info("=" * 80)
                # Validate refined result - if it's invalid or solid color, keep original
                if refined_global is not None and isinstance(refined_global, np.ndarray) and refined_global.size > 0:
                    if len(refined_global.shape) == 3:
                        unique_colors = len(np.unique(refined_global.reshape(-1, refined_global.shape[-1]), axis=0))
                        std_dev = np.std(refined_global)
                        channel_stds = [np.std(refined_global[:, :, c]) for c in range(refined_global.shape[2])]
                    else:
                        unique_colors = len(np.unique(refined_global))
                        std_dev = np.std(refined_global)
                        channel_stds = [std_dev]
                    
                    # Stricter validation: require more unique colors and higher variance
                    if unique_colors >= 20 and std_dev >= 8.0 and not all(std < 5.0 for std in channel_stds):
                        working = refined_global
                        logger.info("Global refinement pass completed")
                    else:
                        logger.warning(f"Global refinement returned solid color (unique_colors={unique_colors}, std={std_dev:.2f}, channel_stds={channel_stds}), keeping original")
                else:
                    logger.warning("Global refinement returned invalid result, keeping original")
            except ValueError as e:
                # Re-raise API credit/payment errors
                error_msg = str(e)
                if "credits" in error_msg.lower() or "payment" in error_msg.lower() or "API" in error_msg:
                    logger.error(f"❌ API ERROR in refinement: {error_msg}")
                    raise  # Re-raise so it's handled by the pipeline
                else:
                    logger.error(f"Global refinement failed: {e}")
            except Exception as e:
                logger.error(f"Global refinement failed: {e}", exc_info=True)
        
        # Region-specific passes
        if region_masks:
            h_working, w_working = working.shape[:2]
            for region_name, mask in region_masks.items():
                if mask is None or np.max(mask) == 0:
                    continue
                # Ensure mask matches working image dimensions
                h_mask, w_mask = mask.shape[:2]
                if (h_mask, w_mask) != (h_working, w_working):
                    mask = cv2.resize(mask, (w_working, h_working), interpolation=cv2.INTER_NEAREST)
                
                # Use custom prompt if provided for region refinement
                if custom_prompt:
                    # Enhance custom prompt with region-specific details
                    region_prompt = f"{custom_prompt}, {self._get_region_specific_text(region_name)}"
                else:
                    region_prompt = self._prompt_for_region(
                        region_name,
                        template_analysis,
                        customer_body_shape,
                        quality
                    )
                region_strength = self.region_strengths.get(region_name, strength or 0.6)
                try:
                    # Use config value (minimum 30 enforced in generator)
                    inference_steps = self.config.get("processing", {}).get("num_inference_steps", 40)
                    refined_region = self.generator.refine(
                        image=working,
                        prompt=region_prompt,
                        mask=mask,
                        negative_prompt=negative_prompt,
                        strength=region_strength,
                        num_inference_steps=inference_steps
                    )
                    # Validate refined result - if it's invalid or solid color, keep original
                    if refined_region is not None and isinstance(refined_region, np.ndarray) and refined_region.size > 0:
                        if len(refined_region.shape) == 3:
                            unique_colors = len(np.unique(refined_region.reshape(-1, refined_region.shape[-1]), axis=0))
                            std_dev = np.std(refined_region)
                            channel_stds = [np.std(refined_region[:, :, c]) for c in range(refined_region.shape[2])]
                        else:
                            unique_colors = len(np.unique(refined_region))
                            std_dev = np.std(refined_region)
                            channel_stds = [std_dev]
                        
                        # Stricter validation: require more unique colors and higher variance
                        if unique_colors >= 20 and std_dev >= 8.0 and not all(std < 5.0 for std in channel_stds):
                            working = refined_region
                            logger.info(f"Refined region '{region_name}' with strength {region_strength}")
                        else:
                            logger.warning(f"Refined region '{region_name}' is solid color (unique_colors={unique_colors}, std={std_dev:.2f}, channel_stds={channel_stds}), keeping original")
                    else:
                        logger.warning(f"Refined region '{region_name}' is invalid, keeping original")
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
        
        # Create PRECISE face mask - use ellipse, not rectangle, to avoid body regions
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        center = (x + w // 2, y + h // 2)
        axes = (int(w * 0.45), int(h * 0.45))  # Slightly smaller to avoid body
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Expand mask slightly for blending (but not too much)
        kernel = np.ones((8, 8), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # CRITICAL: Exclude body regions - ensure mask doesn't extend below neck
        h_img, w_img = image.shape[:2]
        neck_y = y + h + 20  # Neck is just below face
        mask[neck_y:, :] = 0  # Remove mask from body regions
        
        # Use Mickmumpitz workflow for emotion control
        if expression_details and isinstance(expression_details, dict) and "type" in expression_details:
            # Use emotion data from expression_details
            emotion_data = expression_details
        else:
            # Fallback to basic emotion data
            emotion_data = self.emotion_handler._get_emotion_data(expression_type)
        
        # Generate base face prompt - enhanced for better quality
        base_face_prompt = (
            "photorealistic portrait, natural human skin with pores and texture, "
            "realistic skin tone variation, high quality photograph, detailed facial features, "
            "natural lighting, authentic human appearance, subtle skin imperfections, "
            "professional photography, sharp focus, detailed eyes, natural hair, realistic clothing, "
            "preserve original facial structure, maintain natural proportions, accurate facial anatomy, "
            "single face only, one person, no duplicate faces, no extra faces, no face artifacts"
        )
        
        # Enhance with emotion using Mickmumpitz workflow
        prompt = self.emotion_handler.enhance_face_prompt_with_emotion(
            base_face_prompt,
            emotion_data
        )
        
        # Get emotion-specific negative prompt
        emotion_negative = self.emotion_handler.get_emotion_negative_prompt(emotion_data)
        
        negative_prompt = (
            "solid color, single color, flat color, blue, pink, red, green, yellow, monochrome, "
            "uniform color, color block, plastic, artificial, fake, CGI, 3D render, smooth skin, "
            "airbrushed, perfect skin, doll-like, wax figure, synthetic, "
            "blurred, distorted face, oversaturated, cartoon, painting, drawing, illustration, "
            "duplicate faces, extra faces, multiple faces, face on body, face on clothing, "
            "face on chest, face on waist, surreal, deformed face, misplaced face, floating head, "
            "disembodied head, second head, small head, miniature face"
        )
        
        # Add emotion-specific negatives
        if emotion_negative:
            negative_prompt = f"{negative_prompt}, {emotion_negative}"
        
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
            # Use config value for face refinement strength (default 0.5 for safety)
            # Lower strength preserves natural features and avoids over-processing/distortion
            face_strength = self.region_strengths.get("face", 0.5)
            # Cap at 0.6 maximum to prevent distortion
            face_strength = min(face_strength, 0.6)
            
            logger.info("=" * 80)
            logger.info("🎨 CALLING STABILITY AI API FOR FACE REFINEMENT")
            logger.info("=" * 80)
            logger.info(f"📝 Prompt: {prompt[:150]}..." if len(prompt) > 150 else f"📝 Prompt: {prompt}")
            logger.info(f"⚙️  Strength: {face_strength}")
            logger.info(f"📐 Image shape: {image.shape}")
            logger.info(f"🎭 Face mask: Yes (bbox: {face_bbox})")
            logger.info("⏳ Calling Stability AI API...")
            refined = self.generator.refine(
                image=image,
                prompt=prompt,
                mask=mask,
                strength=face_strength,  # Use config value, capped at 0.6
                num_inference_steps=30  # More steps for better quality
            )
            logger.info("✅ Stability AI API call completed")
            logger.info("=" * 80)
            
            # Validate refined result - if it's invalid or solid color, return original
            if refined is not None and isinstance(refined, np.ndarray) and refined.size > 0:
                if len(refined.shape) == 3:
                    unique_colors = len(np.unique(refined.reshape(-1, refined.shape[-1]), axis=0))
                    std_dev = np.std(refined)
                    channel_stds = [np.std(refined[:, :, c]) for c in range(refined.shape[2])]
                else:
                    unique_colors = len(np.unique(refined))
                    std_dev = np.std(refined)
                    channel_stds = [std_dev]
                
                # Stricter validation: require more unique colors and higher variance
                if unique_colors >= 20 and std_dev >= 8.0 and not all(std < 5.0 for std in channel_stds):
                    # Post-process to enhance natural appearance
                    refined = self._post_process_face(refined, mask, image)
                    return refined
                else:
                    logger.warning(f"Face refinement returned solid color (unique_colors={unique_colors}, std={std_dev:.2f}, channel_stds={channel_stds}), using original")
                    return image
            else:
                logger.warning("Face refinement returned invalid result, using original")
                return image
            
        except Exception as e:
            logger.error(f"Face refinement failed: {e}")
            return image
    
    def _post_process_face(
        self,
        refined: np.ndarray,
        mask: np.ndarray,
        original: np.ndarray
    ) -> np.ndarray:
        """
        Post-process refined face to ensure natural appearance.
        Blends original texture back to avoid over-smoothing.
        """
        # Extract face region
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return refined
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Blend original texture details back (20% original, 80% refined)
        # This helps preserve natural skin texture
        face_original = original[y_min:y_max, x_min:x_max]
        face_refined = refined[y_min:y_max, x_min:x_max]
        
        # Extract high-frequency details from original
        original_gray = cv2.cvtColor(face_original, cv2.COLOR_BGR2GRAY)
        refined_gray = cv2.cvtColor(face_refined, cv2.COLOR_BGR2GRAY)
        
        # Get detail layer (high frequency)
        original_blur = cv2.GaussianBlur(original_gray, (5, 5), 0)
        detail = original_gray.astype(np.float32) - original_blur.astype(np.float32)
        
        # Apply detail to refined (subtle)
        detail_factor = 0.15  # How much original detail to preserve
        for c in range(3):
            face_refined[:, :, c] = np.clip(
                face_refined[:, :, c].astype(np.float32) + 
                detail * detail_factor,
                0, 255
            ).astype(np.uint8)
        
        result = refined.copy()
        result[y_min:y_max, x_min:x_max] = face_refined
        
        return result
    
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
        prompt = f"photorealistic {clothing_type}, natural fabric folds, realistic texture, high quality photograph, detailed fabric, natural lighting, professional photography"
        negative_prompt = "solid color, single color, flat color, blue, pink, red, artificial, fake, distorted clothing, plastic appearance, cartoon, painting"
        
        try:
            refined = self.generator.refine(
                image=image,
                prompt=prompt,
                mask=clothing_mask,
                strength=0.6,
                num_inference_steps=25
            )
            
            return refined
        except Exception as e:
            logger.error(f"Clothing refinement failed: {e}")
            return image
    
    def _build_face_control_map(self, size: Tuple[int, int], landmarks: np.ndarray) -> Optional[np.ndarray]:
        """Create a simple control map drawing landmark connections."""
        if landmarks is None or len(landmarks) < 2:
            return None
        
        h, w = size
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        pts = np.clip(landmarks.astype(np.int32), 0, [w - 1, h - 1])
        
        # Draw outline
        try:
            for i in range(len(pts) - 1):
                cv2.line(canvas, tuple(pts[i]), tuple(pts[i+1]), (255, 255, 255), 1)
            for pt in pts:
                cv2.circle(canvas, tuple(pt), 1, (255, 255, 255), -1)
            return canvas
        except Exception as e:
            logger.error(f"Control map creation failed: {e}")
            return None
    
    def _generate_refinement_prompt(
        self,
        template_analysis: Dict,
        customer_body_shape: Dict
    ) -> str:
        """
        Generate refinement prompt with detailed realistic descriptions
        
        Args:
            template_analysis: Template analysis
            customer_body_shape: Customer body shape
        
        Returns:
            Refinement prompt
        """
        prompts = [
            "photorealistic",
            "high quality photograph",
            "detailed",
            "realistic texture",
            "natural lighting",
            "professional photography",
            "sharp focus",
            "realistic materials",
            "seamless integration",
            "natural blending",
            "convert customer to template style",
            "apply template background",
            "match template environment"
        ]
        
        # Add body type
        if customer_body_shape:
            body_type = customer_body_shape.get("body_type", "average")
            if body_type != "unknown":
                prompts.append(f"realistic {body_type} body proportions")
        
        # Add clothing info - emphasize template clothing
        if template_analysis:
            clothing = template_analysis.get("clothing", {})
            clothing_items = clothing.get("items", []) if isinstance(clothing, dict) else []
            if clothing_items:
                prompts.append("natural clothing fit and fabric texture")
                prompts.append("template clothing style")
            # Emphasize template background and environment
            prompts.append("template background and environment")
            prompts.append("match template setting")
            prompts.append("replace background with template background")
            
            # Add expression using Mickmumpitz workflow
            expression = template_analysis.get("expression", {})
            if isinstance(expression, dict) and "type" in expression:
                # Use emotion handler to generate emotion prompt
                emotion_data = expression
                emotion_prompt = self.emotion_handler.generate_emotion_prompt(emotion_data)
                prompts.append(emotion_prompt)
            else:
                expression_type = expression.get("type", "neutral") if isinstance(expression, dict) else "neutral"
                emotion_data = self.emotion_handler._get_emotion_data(expression_type)
                emotion_prompt = self.emotion_handler.generate_emotion_prompt(emotion_data)
                prompts.append(emotion_prompt)
        else:
            emotion_data = self.emotion_handler._get_emotion_data("neutral")
            emotion_prompt = self.emotion_handler.generate_emotion_prompt(emotion_data)
            prompts.append(emotion_prompt)
        
        # Add background
        prompts.append("seamless realistic background")
        
        return ", ".join(prompts)

    def _prompt_for_region(
        self,
        region: str,
        template_analysis: Dict,
        customer_body_shape: Dict,
        quality_metrics: Optional[Dict]
    ) -> str:
        """Generate region-targeted prompt."""
        # Ensure template_analysis and customer_body_shape are not None
        template_analysis = template_analysis or {}
        customer_body_shape = customer_body_shape or {}
        base = self._generate_refinement_prompt(template_analysis, customer_body_shape)
        # Get emotion from template analysis for face region
        expression = template_analysis.get("expression", {}) if template_analysis else {}
        if isinstance(expression, dict) and "type" in expression:
            emotion_data = expression
        else:
            expression_type = expression.get("type", "neutral") if isinstance(expression, dict) else "neutral"
            emotion_data = self.emotion_handler._get_emotion_data(expression_type)
        
        # Generate emotion prompt for face
        emotion_prompt = self.emotion_handler.generate_emotion_prompt(emotion_data)
        
        region_prompts = {
            "face": f"hyper-detailed face, natural skin micro-texture with pores, {emotion_prompt}, precise features, realistic skin tone, professional portrait photography, sharp focus, preserve original person's facial features, maintain identity, keep original face structure",
            "body": "tailored clothing fit, realistic fabric folding and texture, accurate body proportions, natural shading, photorealistic materials, professional photography, preserve original person's body shape, maintain natural appearance, keep original proportions",
            "edges": "feathered transitions, remove halos, seamless blend between subject and background, natural edge blending, realistic shadows, smooth integration",
            "problems": "clean artifacts, remove noise, fix lighting inconsistencies, photorealistic detail, natural appearance, professional quality, preserve original person's features"
        }
        region_text = region_prompts.get(region, "high fidelity details, photorealistic finish")
        return f"{region_text}, {base}"
    
    def _get_region_specific_text(self, region: str) -> str:
        """Get region-specific text to enhance custom prompts"""
        region_texts = {
            "face": "natural skin texture with pores, realistic facial features, professional portrait",
            "body": "realistic clothing fit, natural fabric texture, accurate body proportions",
            "edges": "seamless blending, natural edge transitions, realistic shadows",
            "problems": "clean artifacts, fix lighting issues, natural appearance"
        }
        return region_texts.get(region, "high quality, photorealistic")

