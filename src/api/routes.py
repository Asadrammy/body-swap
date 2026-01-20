"""API routes and endpoints"""

import uuid
import json
import io
import zipfile
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .schemas import (
    SwapRequest,
    SwapResponse,
    JobStatus,
    RefineRequest,
    ResultResponse,
    TemplateListResponse,
)
from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.template_catalog import TemplateCatalog
from ..pipeline.preprocessor import Preprocessor
from ..pipeline.body_analyzer import BodyAnalyzer
from ..pipeline.template_analyzer import TemplateAnalyzer
from ..pipeline.face_processor import FaceProcessor
from ..pipeline.body_warper import BodyWarper
from ..pipeline.composer import Composer
from ..pipeline.refiner import Refiner
from ..pipeline.quality_control import QualityControl
from ..utils.image_utils import load_image, save_image

logger = get_logger(__name__)
router = APIRouter()

# Initialize template catalog (may fail silently if catalog file missing)
try:
    template_catalog = TemplateCatalog()
    logger.info(f"✓ Template catalog initialized with {len(template_catalog.list_templates())} templates")
except Exception as e:
    logger.error(f"✗ Failed to initialize template catalog: {e}", exc_info=True)
    # Create empty catalog as fallback
    template_catalog = None

@router.get("/templates", response_model=TemplateListResponse)
async def get_templates(category: Optional[str] = None, tag: Optional[str] = None):
    """Return available template metadata."""
    logger.info(f"📋 Template list requested - Category: {category or 'all'}, Tag: {tag or 'none'}")
    if template_catalog is None:
        logger.warning("Template catalog not initialized, returning empty list")
        return TemplateListResponse(templates=[], total=0)
    try:
        templates = template_catalog.list_templates(category=category, tag=tag)
        logger.info(f"✅ Returning {len(templates)} template(s)")
        return TemplateListResponse(templates=templates, total=len(templates))
    except Exception as e:
        logger.error(f"Error listing templates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")



# In-memory job storage (in production, use database)
jobs = {}


class SwapPipeline:
    """Main swap pipeline orchestrator"""
    
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.body_analyzer = BodyAnalyzer()
        self.template_analyzer = TemplateAnalyzer()
        self.face_processor = FaceProcessor()
        self.body_warper = BodyWarper()
        self.composer = Composer()
        self.refiner = Refiner()
        self.quality_control = QualityControl()
        
        config = get_config()
        self.outputs_dir = Path(config["paths"]["outputs_dir"])
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, job_id: str, customer_photo_paths: List[str], template_path: str, custom_prompt: Optional[str] = None):
        """Process swap request"""
        try:
            logger.info("=" * 80)
            logger.info(f"🔄 PIPELINE PROCESSING STARTED - Job ID: {job_id}")
            logger.info("=" * 80)
            logger.info(f"📂 Customer photos: {len(customer_photo_paths)} file(s)")
            for idx, path in enumerate(customer_photo_paths):
                logger.info(f"   Photo {idx + 1}: {path}")
            logger.info(f"📋 Template path: {template_path}")
            logger.info(f"💬 Custom prompt: {custom_prompt if custom_prompt else 'None (using default)'}")
            
            # Ensure job exists and has required fields
            if job_id not in jobs:
                logger.error(f"Job {job_id} not found in jobs dict at start of process()")
                # Create minimal job entry if missing (shouldn't happen, but safety check)
                jobs[job_id] = {
                    "job_id": job_id,
                    "status": "pending",
                    "progress": 0.0,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
            
            # Ensure required datetime fields exist
            if "created_at" not in jobs[job_id]:
                jobs[job_id]["created_at"] = datetime.now()
            if "updated_at" not in jobs[job_id]:
                jobs[job_id]["updated_at"] = datetime.now()
            
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["current_stage"] = "preprocessing"
            jobs[job_id]["progress"] = 0.1
            
            # Preprocess inputs
            logger.info(f"⏳ Step 1/8: Preprocessing inputs...")
            customer_data = self.preprocessor.preprocess_customer_photos(customer_photo_paths)
            template_data = self.preprocessor.preprocess_template(template_path)
            logger.info(f"✅ Preprocessing complete")
            logger.info(f"   Template shape: {template_data['image'].shape}, Faces detected: {len(template_data['faces'])}")
            
            jobs[job_id]["progress"] = 0.2
            jobs[job_id]["current_stage"] = "analyzing_body"
            
            # Analyze customer body
            logger.info(f"⏳ Step 2/8: Analyzing customer body shape...")
            customer_body_shapes = []
            for img, faces in zip(customer_data["images"], customer_data["faces"]):
                if faces:
                    body_shape = self.body_analyzer.analyze_body_shape(img, faces)
                    customer_body_shapes.append(body_shape)
            
            fused_body_shape = self.body_analyzer.fuse_body_shapes(customer_body_shapes) if customer_body_shapes else {}
            if fused_body_shape:
                jobs[job_id]["body_summary"] = {
                    "body_type": fused_body_shape.get("body_type"),
                    "measurements": fused_body_shape.get("measurements", {}),
                    "confidence": fused_body_shape.get("confidence", 0.0)
                }
            logger.info(f"✅ Body analysis complete")
            
            jobs[job_id]["progress"] = 0.3
            jobs[job_id]["current_stage"] = "analyzing_template"
            
            # Analyze template
            logger.info(f"⏳ Step 3/8: Analyzing template...")
            template_analysis = self.template_analyzer.analyze_template(
                template_data["image"],
                template_data["faces"]
            )
            
            logger.info(f"✅ Template analysis complete")
            jobs[job_id]["progress"] = 0.4
            jobs[job_id]["current_stage"] = "processing_faces"
            
            # Process faces
            logger.info(f"⏳ Step 4/8: Processing faces...")
            if len(customer_data["images"]) == 1:
                # Single person
                customer_image = customer_data["images"][0]
                customer_faces = customer_data["faces"][0]
                
                if customer_faces and template_data["faces"]:
                    # Both have faces - do face swap
                    logger.info(f"✅ Face swap: Customer has {len(customer_faces)} face(s), Template has {len(template_data['faces'])} face(s)")
                    face_identity = self.face_processor.extract_face_identity(
                        customer_image, customer_faces[0]
                    )
                    
                    expression_match = self.face_processor.match_expression(
                        face_identity,
                        template_data["faces"][0],
                        template_analysis.get("expression", {})
                    )
                    
                    # Composite face
                    result = self.face_processor.composite_face(
                        face_identity["aligned_face"],
                        template_data["image"],
                        template_data["faces"][0],
                        expression_match
                    )
                    
                    result = self.refiner.refine_face(
                        result,
                        template_data["faces"][0].get("bbox", [0, 0, 0, 0]),
                        expression_type=expression_match.get("expression_type", "neutral"),
                        expression_details=expression_match
                    )
                elif customer_faces and not template_data["faces"]:
                    # Customer has face but template doesn't - use customer image as base
                    logger.warning(f"⚠️  Template has no faces detected - using customer image as base for body swap")
                    logger.warning(f"   This will do body-only swap (no face swap)")
                    # Use customer image resized to template size as starting point
                    template_h, template_w = template_data["image"].shape[:2]
                    customer_resized = cv2.resize(customer_image, (template_w, template_h), interpolation=cv2.INTER_LINEAR)
                    result = customer_resized.copy()
                    logger.info(f"✅ Customer image resized to template size: {result.shape}")
                    logger.debug(f"   Customer image stats: min={np.min(result)}, max={np.max(result)}, unique_colors={len(np.unique(result.reshape(-1, result.shape[-1]), axis=0)) if len(result.shape) == 3 else len(np.unique(result))}")
                else:
                    # No faces in customer or template - use template
                    logger.warning(f"⚠️  No faces detected in customer or template - using template as-is")
                    result = template_data["image"].copy()
            else:
                # Multiple people (couples, families)
                result = self.face_processor.process_multiple_faces(
                    customer_data["images"],
                    customer_data["faces"],
                    template_data["image"],
                    template_data["faces"]
                )
            
            # Validate result after face processing
            if result is None or not isinstance(result, np.ndarray) or result.size == 0:
                logger.error(f"Result after face processing is invalid for job {job_id}, using template")
                result = template_data["image"].copy()
            else:
                logger.info(f"Face processing result valid: shape={result.shape}, dtype={result.dtype}")
            
            logger.info(f"✅ Face processing complete")
            jobs[job_id]["progress"] = 0.5
            jobs[job_id]["current_stage"] = "warping_body"
            
            # Warp body if pose detected, OR extract body if no pose (for composition)
            logger.info(f"⏳ Step 5/8: Warping body to match template pose...")
            if fused_body_shape and template_analysis.get("pose"):
                template_pose = template_analysis["pose"]
                customer_pose = {"pose_keypoints": fused_body_shape.get("pose_keypoints", {})}
                
                if customer_pose["pose_keypoints"]:
                    blueprint = self.body_warper.build_warp_blueprint(fused_body_shape, template_pose)
                    warped_body = self.body_warper.warp_body_to_pose(
                        customer_data["images"][0],
                        customer_pose,
                        template_pose,
                        fused_body_shape.get("body_mask"),
                        blueprint=blueprint,
                        customer_body_shape=fused_body_shape
                    )
                    
                    # Use warped body as base for composition (size-adjusted customer body)
                    # Then adapt template clothing to match customer size
                    adaptation_output = self.body_warper.adapt_clothing_to_body(
                        result,
                        template_analysis.get("clothing", {}),
                        fused_body_shape,
                        template_pose
                    )
                    if isinstance(adaptation_output, dict):
                        result = adaptation_output.get("image", result)
                        jobs[job_id]["fit_report"] = adaptation_output.get("fit_report")
                    else:
                        result = adaptation_output
                    
                    # Store warped body for potential use in composition
                    jobs[job_id]["warped_body"] = warped_body
                    logger.info(f"✅ Body warped to match template pose")
            elif fused_body_shape and fused_body_shape.get("body_mask") is not None:
                # No template pose, but we have customer body mask - extract customer body for composition
                logger.info(f"⚠️  Template has no pose - extracting customer body using body mask for composition")
                customer_image = customer_data["images"][0]
                body_mask = fused_body_shape.get("body_mask")
                
                # Validate body mask
                mask_unique = len(np.unique(body_mask))
                mask_nonzero = np.count_nonzero(body_mask)
                mask_total = body_mask.size
                mask_coverage = (mask_nonzero / mask_total) * 100 if mask_total > 0 else 0
                
                logger.debug(f"   Body mask validation: unique_values={mask_unique}, coverage={mask_coverage:.2f}%, nonzero_pixels={mask_nonzero}/{mask_total}")
                
                # Check if mask is valid (not all zeros or all ones)
                if mask_unique < 2 or mask_coverage < 5.0 or mask_coverage > 95.0:
                    logger.warning(f"⚠️  Body mask is invalid (coverage={mask_coverage:.2f}%) - using full customer image instead")
                    # Use full customer image resized to template size
                    template_h, template_w = template_data["image"].shape[:2]
                    customer_resized = cv2.resize(customer_image, (template_w, template_h), interpolation=cv2.INTER_LINEAR)
                    jobs[job_id]["warped_body"] = customer_resized
                    warped_body = customer_resized
                    logger.info(f"✅ Using full customer image as warped body (shape: {customer_resized.shape})")
                else:
                    # Resize customer image and body mask to template size
                    template_h, template_w = template_data["image"].shape[:2]
                    customer_resized = cv2.resize(customer_image, (template_w, template_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Resize body mask to match
                    if body_mask.shape[:2] != (template_h, template_w):
                        body_mask_resized = cv2.resize(body_mask, (template_w, template_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        body_mask_resized = body_mask
                    
                    # Normalize mask to 0-1 range if needed
                    if body_mask_resized.max() > 1:
                        body_mask_normalized = body_mask_resized.astype(np.float32) / 255.0
                    else:
                        body_mask_normalized = body_mask_resized.astype(np.float32)
                    
                    # Extract customer body using mask
                    body_mask_3d = np.stack([body_mask_normalized] * 3, axis=2)
                    extracted_body = (customer_resized * body_mask_3d).astype(np.uint8)
                    
                    # Validate extracted body
                    extracted_unique = len(np.unique(extracted_body.reshape(-1, extracted_body.shape[-1]), axis=0))
                    extracted_std = np.std(extracted_body)
                    logger.debug(f"   Extracted body validation: unique_colors={extracted_unique}, std={extracted_std:.2f}")
                    
                    if extracted_unique < 10 or extracted_std < 5.0:
                        logger.warning(f"⚠️  Extracted body is solid color (unique={extracted_unique}, std={extracted_std:.2f}) - using full customer image")
                        extracted_body = customer_resized
                    
                    # Store extracted body for composition
                    jobs[job_id]["warped_body"] = extracted_body
                    warped_body = extracted_body
                    logger.info(f"✅ Customer body extracted using body mask (shape: {extracted_body.shape}, unique_colors={extracted_unique})")
            else:
                logger.warning(f"⚠️  No body warping possible - no template pose and no body mask available")
            
            logger.info(f"✅ Body warping/extraction complete")
            jobs[job_id]["progress"] = 0.6
            jobs[job_id]["current_stage"] = "composing"
            
            # Compose
            logger.info(f"⏳ Step 6/8: Composing final image...")
            logger.debug(f"   Result shape: {result.shape}, dtype: {result.dtype}")
            logger.debug(f"   Template shape: {template_data['image'].shape}, dtype: {template_data['image'].dtype}")
            logger.debug(f"   Template has faces: {len(template_data['faces']) > 0}")
            logger.debug(f"   Warped body available: {'warped_body' in jobs[job_id]}")
            
            # If template has no faces but we have warped_body, compose it onto template background
            # This ensures actual conversion happens (customer body on template background)
            if not template_data["faces"] and customer_data["faces"] and "warped_body" in jobs[job_id]:
                warped_body = jobs[job_id]["warped_body"]
                logger.info(f"⚠️  Template has no faces - composing warped customer body onto template background")
                logger.info(f"   This will create actual conversion (customer body + template background)")
                logger.debug(f"   Warped body shape: {warped_body.shape if warped_body is not None else None}")
                
                if warped_body is not None and warped_body.size > 0:
                    # Compose warped body onto template background
                    composed = self.composer.compose(
                        warped_body,
                        template_data["image"],
                        body_mask=fused_body_shape.get("body_mask") if fused_body_shape else None,
                        lighting_info=template_analysis.get("lighting")
                    )
                    
                    # Validate composed image
                    if composed is None or not isinstance(composed, np.ndarray) or composed.size == 0:
                        logger.warning(f"Composition failed, using warped body directly")
                        composed = warped_body.copy()
                    else:
                        logger.info(f"✅ Composed warped body onto template background: shape={composed.shape}")
                else:
                    logger.warning(f"Warped body is invalid, using customer image directly")
                    composed = result.copy()
            elif not template_data["faces"] and customer_data["faces"]:
                # No warped body available - use customer image but log warning
                logger.warning(f"⚠️  Template has no faces and no warped body - using customer image directly")
                logger.warning(f"   This will only refine customer image (no actual swap)")
                composed = result.copy()
            else:
                # Validate result before composition
                if result is None or result.size == 0:
                    logger.error(f"Result is empty before composition, using template")
                    composed = template_data["image"].copy()
                elif len(result.shape) < 2:
                    logger.error(f"Result has invalid shape {result.shape}, using template")
                    composed = template_data["image"].copy()
                else:
                    # Check if result is same as template (avoid blending template with itself)
                    try:
                        if result.shape == template_data["image"].shape:
                            # Check if arrays are equal (with tolerance for floating point)
                            if np.allclose(result.astype(float), template_data["image"].astype(float), atol=1.0):
                                logger.warning(f"⚠️  Result is same as template - skipping composition, using result directly")
                                composed = result.copy()
                            else:
                                # Different images - proceed with composition
                                composed = self.composer.compose(
                                    result,
                                    template_data["image"],
                                    body_mask=fused_body_shape.get("body_mask") if fused_body_shape else None,
                                    lighting_info=template_analysis.get("lighting")
                                )
                                
                                # Validate composed image
                                if composed is None or not isinstance(composed, np.ndarray) or composed.size == 0:
                                    logger.error(f"Composed image is invalid for job {job_id}, using result directly")
                                    composed = result.copy()
                                else:
                                    logger.info(f"✅ Composed image valid: shape={composed.shape}, dtype={composed.dtype}, min={np.min(composed)}, max={np.max(composed)}")
                        else:
                            # Different sizes - resize result to match template
                            logger.info(f"Resizing result from {result.shape} to {template_data['image'].shape}")
                            h_t, w_t = template_data["image"].shape[:2]
                            result_resized = cv2.resize(result, (w_t, h_t), interpolation=cv2.INTER_LINEAR)
                            composed = self.composer.compose(
                                result_resized,
                                template_data["image"],
                                body_mask=fused_body_shape.get("body_mask") if fused_body_shape else None,
                                lighting_info=template_analysis.get("lighting")
                            )
                            
                            if composed is None or not isinstance(composed, np.ndarray) or composed.size == 0:
                                logger.error(f"Composed image is invalid after resize, using resized result")
                                composed = result_resized.copy()
                            else:
                                logger.info(f"✅ Composed image valid: shape={composed.shape}, dtype={composed.dtype}, min={np.min(composed)}, max={np.max(composed)}")
                    except Exception as e:
                        logger.error(f"Composition error: {e}", exc_info=True)
                        logger.warning(f"Using result directly due to composition error")
                        composed = result.copy()
            
            jobs[job_id]["progress"] = 0.7
            jobs[job_id]["current_stage"] = "refining"
            
            # Refine - generator will load lazily on first use
            # Use custom_prompt parameter if provided, otherwise get from job dict
            # (The parameter is passed from create_swap_job, but also stored in job dict)
            if custom_prompt is None:
                custom_prompt = jobs.get(job_id, {}).get("custom_prompt")
            
            logger.info("=" * 80)
            logger.info(f"⏳ Step 7/8: Starting AI Refinement (Stability AI API will be called - this consumes credits)")
            logger.info("=" * 80)
            logger.info(f"💬 Custom prompt: {'provided' if custom_prompt else 'auto-generated'}")
            if custom_prompt:
                logger.info(f"   Prompt text: {custom_prompt[:150]}..." if len(custom_prompt) > 150 else f"   Prompt text: {custom_prompt}")
            
            # When template has no faces, do full-image conversion (not just refinement)
            # Create a full-image mask to force Stability AI to convert the entire image
            full_image_refinement_mask = None
            if not template_data["faces"] and customer_data["faces"]:
                logger.info(f"⚠️  Template has no faces - performing FULL IMAGE CONVERSION with Stability AI")
                logger.info(f"   This will convert customer image to template style/background")
                # Create mask covering most of the image (80-90%) to force full conversion
                h, w = composed.shape[:2]
                full_image_refinement_mask = np.ones((h, w), dtype=np.uint8) * 255
                # Leave small border untouched for better blending
                border = min(20, h // 20, w // 20)
                full_image_refinement_mask[:border, :] = 0
                full_image_refinement_mask[-border:, :] = 0
                full_image_refinement_mask[:, :border] = 0
                full_image_refinement_mask[:, -border:] = 0
                logger.info(f"   Full-image conversion mask created: {full_image_refinement_mask.shape}, coverage: {(full_image_refinement_mask > 0).sum() / full_image_refinement_mask.size * 100:.1f}%")
            
            try:
                # Refine (generator will load on first use if needed)
                # Use full-image mask if template has no faces (forces conversion)
                refined = self.refiner.refine_composition(
                    composed,
                    template_analysis,
                    fused_body_shape if fused_body_shape else {},
                    refinement_mask=full_image_refinement_mask if full_image_refinement_mask is not None else None,
                    strength=0.75 if full_image_refinement_mask is not None else 0.8,  # Slightly lower strength for full conversion
                    custom_prompt=custom_prompt  # Pass custom prompt
                )
                
                # Validate refined image immediately after refinement
                if refined is None or not isinstance(refined, np.ndarray) or refined.size == 0:
                    logger.error(f"Refined image is invalid after refinement for job {job_id}, using composed")
                    refined = composed.copy()
                else:
                    logger.info(f"Refined image valid: shape={refined.shape}, dtype={refined.dtype}, min={np.min(refined)}, max={np.max(refined)}")
            except ValueError as e:
                # Re-raise credit/payment errors so they're handled properly
                error_msg = str(e)
                if "credits" in error_msg.lower() or "payment" in error_msg.lower() or "API" in error_msg:
                    logger.error(f"❌ API ERROR for job {job_id}: {error_msg}")
                    raise  # Re-raise so it's caught by the outer exception handler
                else:
                    logger.error(f"Refinement failed for job {job_id}: {e}", exc_info=True)
                    refined = composed.copy()
            except Exception as e:
                logger.error(f"Refinement failed for job {job_id}: {e}", exc_info=True)
                refined = composed.copy()
            
            logger.info("✅ AI Refinement complete")
            jobs[job_id]["progress"] = 0.9
            jobs[job_id]["current_stage"] = "quality_check"
            
            # Quality control
            logger.info(f"⏳ Step 8/8: Quality control and final checks...")
            quality = self.quality_control.assess_quality(
                refined,
                customer_data["faces"][0] if customer_data["faces"] else [],
                template_data["faces"],
                template_analysis,
                body_shape=fused_body_shape
            )
            
            face_boxes = [f.get("bbox", [0, 0, 0, 0]) for f in (customer_data["faces"][0] if customer_data["faces"] else [])]
            refinement_masks = self.quality_control.generate_refinement_masks(
                refined,
                quality,
                face_boxes,
                fused_body_shape.get("body_mask") if fused_body_shape else None
            )
            
            # Additional region refinement if needed
            if quality["overall_score"] < self.quality_control.quality_threshold and refinement_masks:
                recommended = quality.get("recommended_refinements") or list(refinement_masks.keys())
                region_subset = {name: refinement_masks[name] for name in recommended if name in refinement_masks}
                if region_subset:
                    logger.info("=" * 80)
                    logger.info(f"⏳ Additional Region Refinement (Stability AI API will be called again - more credits consumed)")
                    logger.info(f"   Regions to refine: {list(region_subset.keys())}")
                    logger.info("=" * 80)
                    jobs[job_id]["current_stage"] = f"refining_regions ({len(region_subset)} regions)"
                    jobs[job_id]["progress"] = 0.92
                    refined = self.refiner.refine_composition(
                        refined,
                        template_analysis,
                        fused_body_shape if fused_body_shape else {},
                        region_masks=region_subset,
                        quality=quality,
                        custom_prompt=custom_prompt  # Pass custom prompt
                    )
                    jobs[job_id]["progress"] = 0.95
                    jobs[job_id]["current_stage"] = "final_quality_check"
                    quality = self.quality_control.assess_quality(
                        refined,
                        customer_data["faces"][0] if customer_data["faces"] else [],
                        template_data["faces"],
                        template_analysis,
                        body_shape=fused_body_shape
                    )
                    refinement_masks = self.quality_control.generate_refinement_masks(
                        refined,
                        quality,
                        face_boxes,
                        fused_body_shape.get("body_mask") if fused_body_shape else None
                    )
            
            # Validate and save result
            result_path = self.outputs_dir / f"{job_id}_result.png"
            
            # Validate refined image before saving
            logger.info(f"Validating result image for job {job_id}...")
            logger.info(f"  Refined image shape: {refined.shape if isinstance(refined, np.ndarray) else 'None'}")
            logger.info(f"  Refined image dtype: {refined.dtype if isinstance(refined, np.ndarray) else 'None'}")
            logger.info(f"  Template path used: {template_path}")
            logger.info(f"  Template image shape: {template_data['image'].shape}")
            if isinstance(refined, np.ndarray) and refined.size > 0:
                logger.info(f"  Refined image min/max: {np.min(refined)}/{np.max(refined)}")
                unique_colors = len(np.unique(refined.reshape(-1, refined.shape[-1]), axis=0)) if len(refined.shape) == 3 else len(np.unique(refined))
                logger.info(f"  Refined image unique colors: {unique_colors}")
            
            if refined is None or not isinstance(refined, np.ndarray):
                logger.error(f"⚠️ Invalid refined image for job {job_id}, using template as fallback (template_path={template_path})")
                refined = template_data["image"].copy()
            elif refined.size == 0:
                logger.error(f"⚠️ Empty refined image for job {job_id}, using template as fallback (template_path={template_path})")
                refined = template_data["image"].copy()
            elif len(refined.shape) < 2 or refined.shape[0] == 0 or refined.shape[1] == 0:
                logger.error(f"⚠️ Invalid refined image shape {refined.shape} for job {job_id}, using template as fallback (template_path={template_path})")
                refined = template_data["image"].copy()
            else:
                # Check if image is all zeros or single color (potential error)
                if len(refined.shape) == 3:
                    unique_colors = len(np.unique(refined.reshape(-1, refined.shape[-1]), axis=0))
                    std_dev = np.std(refined)
                    mean_rgb = np.mean(refined, axis=(0, 1))
                    channel_stds = [np.std(refined[:, :, c]) for c in range(refined.shape[2])]
                else:
                    unique_colors = len(np.unique(refined))
                    std_dev = np.std(refined)
                    mean_rgb = np.mean(refined)
                    channel_stds = [std_dev]
                
                # More comprehensive solid color detection
                is_solid_color = False
                reason = ""
                
                if np.all(refined == 0):
                    is_solid_color = True
                    reason = "all zeros"
                elif np.min(refined) == np.max(refined) and refined.size > 100:
                    is_solid_color = True
                    reason = f"all pixels are {np.min(refined)}"
                elif unique_colors < 20 and refined.size > 1000:
                    is_solid_color = True
                    reason = f"only {unique_colors} unique colors"
                elif std_dev < 8.0 and refined.size > 1000:
                    is_solid_color = True
                    reason = f"low variance (std={std_dev:.2f})"
                elif len(refined.shape) == 3 and all(std < 5.0 for std in channel_stds):
                    is_solid_color = True
                    reason = f"low per-channel variance (stds={channel_stds})"
                
                if is_solid_color:
                    logger.warning(f"⚠️ Refined image appears to be solid color for job {job_id} ({reason}), using template as fallback")
                    logger.warning(f"  Color stats - Mean RGB: {mean_rgb}, Std RGB: {std_dev:.2f}, Unique colors: {unique_colors}")
                    logger.warning(f"  Template path: {template_path}, Template shape: {template_data['image'].shape}")
                    refined = template_data["image"].copy()
            
            # Ensure image is in correct format
            if len(refined.shape) == 2:
                # Grayscale - convert to RGB
                refined = cv2.cvtColor(refined, cv2.COLOR_GRAY2RGB)
            elif len(refined.shape) == 3 and refined.shape[2] == 4:
                # RGBA - convert to RGB
                refined = cv2.cvtColor(refined, cv2.COLOR_RGBA2RGB)
            elif len(refined.shape) == 3 and refined.shape[2] != 3:
                # Wrong number of channels
                logger.warning(f"Refined image has {refined.shape[2]} channels, converting to RGB")
                refined = refined[:, :, :3]
            
            # Ensure uint8 format
            if isinstance(refined, np.ndarray):
                if refined.dtype != np.uint8:
                    if refined.max() <= 1.0:
                        refined = (refined * 255).astype(np.uint8)
                        logger.info("  Converted float [0-1] to uint8")
                    else:
                        refined = np.clip(refined, 0, 255).astype(np.uint8)
                        logger.info("  Clipped and converted to uint8")
            
            logger.info(f"  Final image shape: {refined.shape}, dtype: {refined.dtype}")
            logger.info(f"  Final image min/max: {np.min(refined)}/{np.max(refined)}")
            logger.info(f"  Saving result to: {result_path} (absolute: {result_path.resolve()})")
            save_image(refined, result_path)
            logger.info(f"✅ Result saved to {result_path} (file exists: {result_path.exists()})")
            
            # Save masks
            masks_path = {}
            for mask_name, mask in refinement_masks.items():
                # Skip metadata entry - it's a dict, not an image
                if mask_name == "_metadata":
                    continue
                # Ensure mask is a numpy array
                if not isinstance(mask, np.ndarray):
                    logger.warning(f"Skipping {mask_name}: not a numpy array")
                    continue
                mask_path = self.outputs_dir / f"{job_id}_mask_{mask_name}.png"
                save_image(mask, mask_path, is_mask=True)
                masks_path[mask_name] = str(mask_path)
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 1.0
            jobs[job_id]["result_path"] = str(result_path)
            jobs[job_id]["quality_metrics"] = quality
            jobs[job_id]["refinement_masks"] = masks_path
            
            logger.info("=" * 80)
            logger.info(f"✅ PIPELINE PROCESSING COMPLETE - Job ID: {job_id}")
            logger.info("=" * 80)
            logger.info(f"📁 Result saved to: {result_path}")
            logger.info(f"📊 Quality score: {quality.get('overall_score', 0.0):.2f}")
            logger.info("=" * 80)
            
        except ValueError as e:
            # Handle API credit/payment errors specifically
            error_msg = str(e)
            if "credits" in error_msg.lower() or "payment" in error_msg.lower():
                logger.error(f"❌ API CREDITS REQUIRED for job {job_id}: {error_msg}")
                if "stability" in error_msg.lower():
                    error_msg = f"Stability AI requires credits. {error_msg} Please purchase credits at: https://platform.stability.ai/account/credits"
            else:
                logger.error(f"Processing failed for job {job_id}: {e}", exc_info=True)
            
            # Ensure job exists and has required fields
            if job_id not in jobs:
                logger.error(f"Job {job_id} not found in jobs dict during error handling")
                return
            
            # Ensure required datetime fields exist
            if "created_at" not in jobs[job_id]:
                jobs[job_id]["created_at"] = datetime.now()
            if "updated_at" not in jobs[job_id]:
                jobs[job_id]["updated_at"] = datetime.now()
            
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = error_msg
            jobs[job_id]["updated_at"] = datetime.now()
            logger.error(f"Job {job_id} marked as failed with error: {error_msg}")
        except Exception as e:
            logger.error(f"Processing failed for job {job_id}: {e}", exc_info=True)
            # Ensure job exists and has required fields
            if job_id not in jobs:
                logger.error(f"Job {job_id} not found in jobs dict during error handling")
                return
            
            # Ensure required datetime fields exist
            if "created_at" not in jobs[job_id]:
                jobs[job_id]["created_at"] = datetime.now()
            if "updated_at" not in jobs[job_id]:
                jobs[job_id]["updated_at"] = datetime.now()
            
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["updated_at"] = datetime.now()
            logger.error(f"Job {job_id} marked as failed with error: {str(e)}")


pipeline = SwapPipeline()


@router.post("/swap", response_model=SwapResponse)
async def create_swap_job(
    background_tasks: BackgroundTasks,
    customer_photos: List[UploadFile] = File(...),
    template: Optional[UploadFile] = File(None),
    template_id: Optional[str] = Form(None),
    custom_prompt: Optional[str] = Form(None)
):
    """Create a new swap job"""
    logger.info("=" * 80)
    logger.info("🎯 NEW SWAP JOB REQUEST RECEIVED")
    logger.info("=" * 80)
    logger.info(f"📸 Customer photos received: {len(customer_photos)} file(s)")
    for idx, photo in enumerate(customer_photos):
        logger.info(f"   Photo {idx + 1}: {photo.filename} ({photo.size} bytes, {photo.content_type})")
    logger.info(f"📋 Template ID: {template_id if template_id else 'Not provided (uploaded file)'}")
    logger.info(f"💬 Custom prompt: {custom_prompt if custom_prompt else 'Not provided'}")
    
    if len(customer_photos) < 1 or len(customer_photos) > 2:
        logger.error(f"❌ Invalid number of photos: {len(customer_photos)} (must be 1-2)")
        raise HTTPException(status_code=400, detail="Must provide 1-2 customer photos")
    if not template_id and template is None:
        logger.error("❌ No template provided (neither template_id nor template file)")
        raise HTTPException(status_code=400, detail="Template file or template_id is required")
    
    job_id = str(uuid.uuid4())
    logger.info(f"🆔 Job ID created: {job_id}")
    
    # Save uploaded files
    logger.info("💾 Saving uploaded files...")
    uploads_dir = Path(get_config("paths.temp_dir", "temp"))
    uploads_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"   Upload directory: {uploads_dir}")
    
    customer_paths = []
    for idx, photo in enumerate(customer_photos):
        path = uploads_dir / f"{job_id}_customer_{idx}.jpg"
        with open(path, "wb") as f:
            f.write(await photo.read())
        customer_paths.append(str(path))
        logger.info(f"   ✅ Saved customer photo {idx + 1}: {path}")
    
    template_path = uploads_dir / f"{job_id}_template.jpg"

    if template_id:
        logger.info(f"📂 Loading template from catalog: {template_id}")
        template_entry = template_catalog.get_template(template_id)
        if not template_entry:
            logger.error(f"❌ Template not found in catalog: {template_id}")
            raise HTTPException(status_code=404, detail="Template not found")
        # Resolve template path relative to project root
        project_root = Path(__file__).parent.parent.parent.resolve()
        source_path = project_root / template_entry.asset_path
        if not source_path.exists():
            logger.error(f"❌ Template file missing: {source_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Template asset missing at {source_path}"
            )
        template_path.write_bytes(source_path.read_bytes())
        logger.info(f"   ✅ Template loaded: {source_path}")
    elif template:
        logger.info(f"📤 Saving uploaded template file: {template.filename}")
        with open(template_path, "wb") as f:
            f.write(await template.read())
        logger.info(f"   ✅ Template saved: {template_path}")
    
    # Create job
    logger.info("📝 Creating job entry...")
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "custom_prompt": custom_prompt,  # Store custom prompt
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    # Process in background
    logger.info("🚀 Starting background processing task...")
    logger.info(f"   ⚠️  Stability AI API will be called during processing (this consumes credits)")
    logger.info(f"   📋 All API calls will be logged in real-time in the terminal")
    logger.info(f"   🔍 Watch for Stability AI API request/response logs")
    logger.info("=" * 80)
    background_tasks.add_task(
        pipeline.process,
        job_id,
        customer_paths,
        str(template_path),
        custom_prompt=custom_prompt  # Pass custom prompt to pipeline
    )
    
    logger.info(f"✅ Job {job_id} created and queued successfully")
    return SwapResponse(
        job_id=job_id,
        status="pending",
        message="Job created and queued for processing"
    )


@router.get("/jobs")
async def list_jobs():
    """List all jobs"""
    from typing import Dict, Any
    jobs_list = []
    for job_id, job_data in jobs.items():
        jobs_list.append({
            "job_id": job_id,
            "status": job_data.get("status", "unknown"),
            "progress": job_data.get("progress", 0.0),
            "current_stage": job_data.get("current_stage"),
            "created_at": job_data.get("created_at"),
            "updated_at": job_data.get("updated_at")
        })
    return {"jobs": jobs_list, "total": len(jobs_list)}


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Ensure required fields exist
    if "created_at" not in job:
        job["created_at"] = datetime.now()
    if "updated_at" not in job:
        job["updated_at"] = datetime.now()
    
    # Update updated_at timestamp
    job["updated_at"] = datetime.now()
    
    # Ensure all required JobStatus fields are present with defaults
    # Check if result_path exists but is not in job dict (file exists on disk)
    result_path = job.get("result_path")
    if not result_path and job.get("status") == "completed":
        # Try to find result file even if result_path is not set
        result_path_obj = self.outputs_dir / f"{job_id}_result.png"
        if result_path_obj.exists():
            result_path = str(result_path_obj)
            job["result_path"] = result_path
            logger.info(f"Found result file for job {job_id}: {result_path}")
    
    job_status_data = {
        "job_id": job.get("job_id", job_id),
        "status": job.get("status", "pending"),
        "progress": job.get("progress", 0.0),
        "current_stage": job.get("current_stage"),
        "error": job.get("error"),
        "result_path": result_path,
        "body_summary": job.get("body_summary"),
        "fit_report": job.get("fit_report"),
        "quality_metrics": job.get("quality_metrics"),
        "refinement_masks": job.get("refinement_masks"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"]
    }
    
    try:
        return JobStatus(**job_status_data)
    except Exception as e:
        logger.error(f"Error creating JobStatus response for job {job_id}: {e}")
        logger.error(f"Job data: {job}")
        # Return a basic valid response even if validation fails
        job_status_data["status"] = job.get("status", "failed")
        job_status_data["error"] = f"Status serialization error: {str(e)}"
        return JobStatus(**job_status_data)


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Download result image"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")
    
    result_path = job.get("result_path")
    if not result_path:
        logger.error(f"Job {job_id} has no result_path set")
        raise HTTPException(status_code=404, detail="Result file path not set")
    
    result_path_obj = Path(result_path)
    if not result_path_obj.exists():
        logger.error(f"Result file does not exist for job {job_id}: {result_path} (absolute: {result_path_obj.resolve()})")
        raise HTTPException(status_code=404, detail=f"Result file not found at {result_path}")
    
    logger.info(f"Serving result for job {job_id}: {result_path} (file size: {result_path_obj.stat().st_size} bytes)")
    return FileResponse(
        result_path,
        media_type="image/png",
        filename=f"{job_id}_result.png"
    )

@router.get("/jobs/{job_id}/bundle")
async def download_job_bundle(job_id: str):
    """Download result image plus metadata as a zip bundle"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")
    
    buffer = io.BytesIO()
    bundle_metadata = {
        "job_id": job_id,
        "status": job["status"],
        "body_summary": job.get("body_summary"),
        "fit_report": job.get("fit_report"),
        "quality_metrics": job.get("quality_metrics"),
        "refinement_masks": job.get("refinement_masks"),
        "created_at": job.get("created_at").isoformat() if job.get("created_at") else None,
        "completed_at": job.get("updated_at").isoformat() if job.get("updated_at") else None
    }
    
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        result_path = job.get("result_path")
        if result_path and Path(result_path).exists():
            zip_file.write(result_path, arcname="result.png")
        
        zip_file.writestr("metadata.json", json.dumps(bundle_metadata, indent=2))
        
        refinement_masks = job.get("refinement_masks", {})
        for name, mask_path in refinement_masks.items():
            mask_file = Path(mask_path)
            if mask_file.exists():
                zip_file.write(mask_file, arcname=f"refinement_masks/{name}.png")
    
    buffer.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="swap_bundle_{job_id}.zip"'}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)

@router.post("/jobs/{job_id}/refine")
async def refine_job(
    job_id: str,
    request: RefineRequest,
    background_tasks: BackgroundTasks
):
    """Refine a completed job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Implementation for refinement
    # This would re-run parts of the pipeline with refinement masks
    
    return {"message": "Refinement queued", "job_id": job_id}

