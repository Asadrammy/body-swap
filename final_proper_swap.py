"""Final proper face swap with mask detection and replacement"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.api.cli import SwapCLI
from src.utils.image_utils import load_image, save_image
from src.pipeline.preprocessor import Preprocessor
from src.pipeline.face_processor import FaceProcessor
from src.pipeline.body_analyzer import BodyAnalyzer
from src.pipeline.template_analyzer import TemplateAnalyzer
from src.pipeline.body_warper import BodyWarper
from src.pipeline.composer import Composer
from src.pipeline.refiner import Refiner
from src.pipeline.quality_control import QualityControl

setup_logger()
logger = get_logger(__name__)


def expand_face_regions_for_masks(template_faces, expand_factor=2.5):
    """Expand face bboxes to cover entire mask regions"""
    expanded_faces = []
    for face in template_faces:
        bbox = face.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox
        
        # Expand significantly for masks
        expanded_w = int(w * expand_factor)
        expanded_h = int(h * expand_factor)
        expanded_x = max(0, x - (expanded_w - w) // 2)
        expanded_y = max(0, y - (expanded_h - h) // 2)
        
        expanded_face = face.copy()
        expanded_face["bbox"] = [expanded_x, expanded_y, expanded_w, expanded_h]
        expanded_faces.append(expanded_face)
        logger.info(f"Expanded face region: ({x}, {y}) {w}x{h} -> ({expanded_x}, {expanded_y}) {expanded_w}x{expanded_h}")
    
    return expanded_faces


def run_full_pipeline_with_mask_expansion():
    """Run full pipeline with expanded mask regions"""
    logger.info("=" * 60)
    logger.info("FINAL PROPER FACE SWAP WITH MASK EXPANSION")
    logger.info("=" * 60)
    
    # Images
    template_image_path = "IMG20251019131550.jpg"
    customer_image_path = "1760713603491 (1).jpg"
    output_path = "outputs/final_proper_swap_with_masks.png"
    
    logger.info(f"Customer Image: {customer_image_path}")
    logger.info(f"Template Image: {template_image_path}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)
    
    # Initialize components
    preprocessor = Preprocessor()
    body_analyzer = BodyAnalyzer()
    template_analyzer = TemplateAnalyzer()
    face_processor = FaceProcessor()
    body_warper = BodyWarper()
    composer = Composer()
    refiner = Refiner()
    quality_control = QualityControl()
    
    # Preprocess
    logger.info("Preprocessing inputs...")
    customer_data = preprocessor.preprocess_customer_photos([customer_image_path])
    template_data = preprocessor.preprocess_template(template_image_path)
    
    customer_faces = customer_data["faces"][0] if customer_data["faces"] else []
    template_faces = template_data["faces"] if template_data["faces"] else []
    
    logger.info(f"Customer faces: {len(customer_faces)}")
    logger.info(f"Template faces: {len(template_faces)}")
    
    if not customer_faces:
        logger.error("No customer faces!")
        return False
    
    if not template_faces:
        logger.error("No template faces!")
        return False
    
    # Expand template face regions for masks
    logger.info("Expanding template face regions to cover masks...")
    expanded_template_faces = expand_face_regions_for_masks(template_faces, expand_factor=2.5)
    
    # Analyze customer body
    logger.info("Analyzing customer body...")
    customer_body_shapes = []
    for img, faces in zip(customer_data["images"], customer_data["faces"]):
        if faces:
            body_shape = body_analyzer.analyze_body_shape(img, faces)
            customer_body_shapes.append(body_shape)
    
    fused_body_shape = body_analyzer.fuse_body_shapes(customer_body_shapes) if customer_body_shapes else {}
    
    # Analyze template
    logger.info("Analyzing template...")
    template_analysis = template_analyzer.analyze_template(
        template_data["image"],
        expanded_template_faces  # Use expanded faces
    )
    
    # Process faces with expanded regions
    logger.info("Processing faces with expanded mask regions...")
    customer_image = customer_data["images"][0]
    face_identity = face_processor.extract_face_identity(customer_image, customer_faces[0])
    
    result = template_data["image"].copy()
    
    # Swap onto all expanded template faces
    for i, template_face in enumerate(expanded_template_faces):
        logger.info(f"Swapping face {i+1}/{len(expanded_template_faces)}...")
        
        expression_match = face_processor.match_expression(
            face_identity,
            template_face,
            template_analysis.get("expression", {})
        )
        
        result = face_processor.composite_face(
            face_identity["aligned_face"],
            result,
            template_face,
            expression_match
        )
    
    # Body warping
    logger.info("Warping body...")
    if fused_body_shape and template_analysis.get("pose"):
        customer_body = fused_body_shape
        template_pose = template_analysis["pose"]
        customer_pose = {"keypoints": customer_body.get("pose_keypoints", {})}
        
        if customer_pose["keypoints"]:
            blueprint = body_warper.build_warp_blueprint(customer_body, template_pose)
            warped_body = body_warper.warp_body_to_pose(
                customer_data["images"][0],
                customer_pose,
                template_pose,
                customer_body.get("body_mask"),
                blueprint=blueprint,
                customer_body_shape=customer_body
            )
            
            adapted = body_warper.adapt_clothing_to_body(
                result,
                template_analysis.get("clothing", {}),
                customer_body,
                template_pose
            )
            if isinstance(adapted, dict):
                result = adapted.get("image", result)
            else:
                result = adapted
    
    # Compose
    logger.info("Composing final image...")
    composed = composer.compose(
        result,
        template_data["image"],
        body_mask=fused_body_shape.get("body_mask") if fused_body_shape else None,
        lighting_info=template_analysis.get("lighting")
    )
    
    # Refine
    logger.info("Refining...")
    refined = refiner.refine_composition(
        composed,
        template_analysis,
        fused_body_shape if fused_body_shape else {},
        strength=0.8
    )
    result = refined
    
    # Quality check
    logger.info("Assessing quality...")
    quality = quality_control.assess_quality(
        result,
        customer_faces,
        expanded_template_faces,
        template_analysis,
        body_shape=fused_body_shape
    )
    
    logger.info(f"Quality score: {quality['overall_score']:.2f}")
    
    # Save
    logger.info(f"Saving to {output_path}...")
    save_image(result, output_path)
    
    # Verify
    output_img = load_image(output_path)
    template_img = load_image(template_image_path)
    
    if output_img.shape != template_img.shape:
        template_resized = cv2.resize(template_img, (output_img.shape[1], output_img.shape[0]))
    else:
        template_resized = template_img
    
    diff = np.mean(np.abs(output_img.astype(float) - template_resized.astype(float)))
    logger.info(f"Mean difference from template: {diff:.2f}")
    
    # Check faces in output
    output_data = preprocessor.preprocess_template(output_path)
    output_faces = len(output_data['faces']) if output_data['faces'] else 0
    logger.info(f"Faces detected in output: {output_faces}")
    
    logger.info("=" * 60)
    logger.info("PROCESS COMPLETED")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    success = run_full_pipeline_with_mask_expansion()
    if not success:
        sys.exit(1)






