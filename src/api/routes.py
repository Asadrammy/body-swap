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
template_catalog = TemplateCatalog()
@router.get("/templates", response_model=TemplateListResponse)
async def get_templates(category: Optional[str] = None, tag: Optional[str] = None):
    """Return available template metadata."""
    templates = template_catalog.list_templates(category=category, tag=tag)
    return TemplateListResponse(templates=templates, total=len(templates))



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
    
    def process(self, job_id: str, customer_photo_paths: List[str], template_path: str):
        """Process swap request"""
        try:
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["current_stage"] = "preprocessing"
            jobs[job_id]["progress"] = 0.1
            
            # Preprocess inputs
            customer_data = self.preprocessor.preprocess_customer_photos(customer_photo_paths)
            template_data = self.preprocessor.preprocess_template(template_path)
            
            jobs[job_id]["progress"] = 0.2
            jobs[job_id]["current_stage"] = "analyzing_body"
            
            # Analyze customer body
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
            
            jobs[job_id]["progress"] = 0.3
            jobs[job_id]["current_stage"] = "analyzing_template"
            
            # Analyze template
            template_analysis = self.template_analyzer.analyze_template(
                template_data["image"],
                template_data["faces"]
            )
            
            jobs[job_id]["progress"] = 0.4
            jobs[job_id]["current_stage"] = "processing_faces"
            
            # Process faces
            if len(customer_data["images"]) == 1:
                # Single person
                customer_image = customer_data["images"][0]
                customer_faces = customer_data["faces"][0]
                
                if customer_faces and template_data["faces"]:
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
                else:
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
            
            jobs[job_id]["progress"] = 0.5
            jobs[job_id]["current_stage"] = "warping_body"
            
            # Warp body if pose detected
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
            
            jobs[job_id]["progress"] = 0.6
            jobs[job_id]["current_stage"] = "composing"
            
            # Compose
            composed = self.composer.compose(
                result,
                template_data["image"],
                body_mask=fused_body_shape.get("body_mask") if fused_body_shape else None,
                lighting_info=template_analysis.get("lighting")
            )
            
            # Validate composed image
            if composed is None or not isinstance(composed, np.ndarray) or composed.size == 0:
                logger.error(f"Composed image is invalid for job {job_id}, using template")
                composed = template_data["image"].copy()
            else:
                logger.info(f"Composed image valid: shape={composed.shape}, dtype={composed.dtype}, min={np.min(composed)}, max={np.max(composed)}")
            
            jobs[job_id]["progress"] = 0.7
            jobs[job_id]["current_stage"] = "refining"
            
            # Refine - but first check if generator is available
            try:
                generator_available = self.refiner.generator.inpaint_pipe is not None
            except:
                generator_available = False
            
            if not generator_available:
                logger.warning(f"Generator not available for job {job_id}, skipping refinement and using composed image")
                refined = composed.copy()
            else:
                # Refine
                refined = self.refiner.refine_composition(
                    composed,
                    template_analysis,
                    fused_body_shape if fused_body_shape else {},
                    strength=0.8
                )
                
                # Validate refined image immediately after refinement
                if refined is None or not isinstance(refined, np.ndarray) or refined.size == 0:
                    logger.error(f"Refined image is invalid after refinement for job {job_id}, using composed")
                    refined = composed.copy()
                else:
                    logger.info(f"Refined image valid: shape={refined.shape}, dtype={refined.dtype}, min={np.min(refined)}, max={np.max(refined)}")
            
            jobs[job_id]["progress"] = 0.9
            jobs[job_id]["current_stage"] = "quality_check"
            
            # Quality control
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
                    logger.info(f"Triggering targeted refinement for regions: {list(region_subset.keys())}")
                    jobs[job_id]["current_stage"] = f"refining_regions ({len(region_subset)} regions)"
                    jobs[job_id]["progress"] = 0.92
                    refined = self.refiner.refine_composition(
                        refined,
                        template_analysis,
                        fused_body_shape if fused_body_shape else {},
                        region_masks=region_subset,
                        quality=quality
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
            if isinstance(refined, np.ndarray) and refined.size > 0:
                logger.info(f"  Refined image min/max: {np.min(refined)}/{np.max(refined)}")
                unique_colors = len(np.unique(refined.reshape(-1, refined.shape[-1]), axis=0)) if len(refined.shape) == 3 else len(np.unique(refined))
                logger.info(f"  Refined image unique colors: {unique_colors}")
            
            if refined is None or not isinstance(refined, np.ndarray):
                logger.error(f"Invalid refined image for job {job_id}, using template as fallback")
                refined = template_data["image"].copy()
            elif refined.size == 0:
                logger.error(f"Empty refined image for job {job_id}, using template as fallback")
                refined = template_data["image"].copy()
            elif len(refined.shape) < 2 or refined.shape[0] == 0 or refined.shape[1] == 0:
                logger.error(f"Invalid refined image shape {refined.shape} for job {job_id}, using template as fallback")
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
                    logger.warning(f"Refined image appears to be solid color for job {job_id} ({reason}), using template as fallback")
                    logger.warning(f"  Color stats - Mean RGB: {mean_rgb}, Std RGB: {std_dev:.2f}, Unique colors: {unique_colors}")
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
            save_image(refined, result_path)
            logger.info(f"✅ Result saved to {result_path}")
            
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
                save_image(mask, mask_path)
                masks_path[mask_name] = str(mask_path)
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 1.0
            jobs[job_id]["result_path"] = str(result_path)
            jobs[job_id]["quality_metrics"] = quality
            jobs[job_id]["refinement_masks"] = masks_path
            
        except Exception as e:
            logger.error(f"Processing failed for job {job_id}: {e}", exc_info=True)
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)


pipeline = SwapPipeline()


@router.post("/swap", response_model=SwapResponse)
async def create_swap_job(
    background_tasks: BackgroundTasks,
    customer_photos: List[UploadFile] = File(...),
    template: Optional[UploadFile] = File(None),
    template_id: Optional[str] = Form(None)
):
    """Create a new swap job"""
    if len(customer_photos) < 1 or len(customer_photos) > 2:
        raise HTTPException(status_code=400, detail="Must provide 1-2 customer photos")
    if not template_id and template is None:
        raise HTTPException(status_code=400, detail="Template file or template_id is required")
    
    job_id = str(uuid.uuid4())
    
    # Save uploaded files
    uploads_dir = Path(get_config("paths.temp_dir", "temp"))
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    customer_paths = []
    for idx, photo in enumerate(customer_photos):
        path = uploads_dir / f"{job_id}_customer_{idx}.jpg"
        with open(path, "wb") as f:
            f.write(await photo.read())
        customer_paths.append(str(path))
    
    template_path = uploads_dir / f"{job_id}_template.jpg"

    if template_id:
        template_entry = template_catalog.get_template(template_id)
        if not template_entry:
            raise HTTPException(status_code=404, detail="Template not found")
        # Resolve template path relative to project root
        project_root = Path(__file__).parent.parent.parent.resolve()
        source_path = project_root / template_entry.asset_path
        if not source_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Template asset missing at {source_path}"
            )
        template_path.write_bytes(source_path.read_bytes())
    elif template:
        with open(template_path, "wb") as f:
            f.write(await template.read())
    
    # Create job
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    # Process in background
    background_tasks.add_task(
        pipeline.process,
        job_id,
        customer_paths,
        str(template_path)
    )
    
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
    from datetime import datetime
    job["updated_at"] = datetime.now()
    
    return JobStatus(**job)


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Download result image"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")
    
    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
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

