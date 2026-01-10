"""Test script to run face-body swap on new customer image with proper verification"""

import sys
from pathlib import Path
import numpy as np
import cv2
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.utils.config import get_config
from src.pipeline.preprocessor import Preprocessor
from src.pipeline.body_analyzer import BodyAnalyzer
from src.pipeline.template_analyzer import TemplateAnalyzer
from src.pipeline.face_processor import FaceProcessor
from src.pipeline.body_warper import BodyWarper
from src.pipeline.composer import Composer
from src.pipeline.refiner import Refiner
from src.pipeline.quality_control import QualityControl
from src.utils.image_utils import load_image, save_image

setup_logger()
logger = get_logger(__name__)


def verify_face_swap_happened(result_img, template_img, customer_img):
    """Verify that face swap actually occurred"""
    logger.info("Verifying face swap occurred...")
    
    # Check if result is identical to template
    if np.array_equal(result_img, template_img):
        logger.error("✗ CRITICAL: Result is IDENTICAL to template - face swap did NOT happen!")
        return False
    
    # Check if result is identical to customer
    if np.array_equal(result_img, customer_img):
        logger.error("✗ CRITICAL: Result is IDENTICAL to customer image - pipeline just copied input!")
        return False
    
    # Check if shapes match (they should be similar to template)
    if result_img.shape != template_img.shape:
        logger.warning(f"Shape mismatch: result {result_img.shape} vs template {template_img.shape}")
        # Resize for comparison
        result_resized = cv2.resize(result_img, (template_img.shape[1], template_img.shape[0]))
    else:
        result_resized = result_img
    
    # Calculate difference
    diff = np.abs(result_resized.astype(float) - template_img.astype(float))
    mean_diff = np.mean(diff)
    
    logger.info(f"Mean pixel difference from template: {mean_diff:.2f}")
    
    if mean_diff < 5.0:
        logger.error("✗ CRITICAL: Result is too similar to template - face swap likely did NOT happen!")
        return False
    
    logger.info("✓ Face swap verification passed - result is different from both inputs")
    return True


def run_full_pipeline_with_verification(customer_image_path: str, template_path: str, output_path: str):
    """Run the complete face-body swap pipeline with step-by-step verification"""
    logger.info("=" * 60)
    logger.info("STARTING FULL PIPELINE WITH VERIFICATION")
    logger.info("=" * 60)
    logger.info(f"Customer image: {customer_image_path}")
    logger.info(f"Template: {template_path}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)
    
    try:
        # Load original images for comparison
        original_customer = load_image(customer_image_path)
        original_template = load_image(template_path)
        
        # Initialize all components
        logger.info("\n[1/9] Initializing components...")
        preprocessor = Preprocessor()
        body_analyzer = BodyAnalyzer()
        template_analyzer = TemplateAnalyzer()
        face_processor = FaceProcessor()
        body_warper = BodyWarper()
        composer = Composer()
        refiner = Refiner()
        quality_control = QualityControl()
        
        # Preprocess inputs
        logger.info("\n[2/9] Preprocessing inputs...")
        customer_data = preprocessor.preprocess_customer_photos([customer_image_path])
        template_data = preprocessor.preprocess_template(template_path)
        
        logger.info(f"Customer faces detected: {len(customer_data['faces'][0]) if customer_data['faces'] and customer_data['faces'][0] else 0}")
        logger.info(f"Template faces detected: {len(template_data['faces']) if template_data['faces'] else 0}")
        
        if not customer_data['faces'] or not customer_data['faces'][0]:
            logger.error("✗ No faces detected in customer image!")
            return False
        
        if not template_data['faces']:
            logger.error("✗ No faces detected in template!")
            logger.error("This template cannot be used for face swap. Trying a different template...")
            return False
        
        # Analyze customer body
        logger.info("\n[3/9] Analyzing customer body shape...")
        customer_body_shapes = []
        for img, faces in zip(customer_data["images"], customer_data["faces"]):
            if faces:
                body_shape = body_analyzer.analyze_body_shape(img, faces)
                customer_body_shapes.append(body_shape)
                logger.info(f"Body type: {body_shape.get('body_type', 'unknown')}")
        
        fused_body_shape = body_analyzer.fuse_body_shapes(customer_body_shapes) if customer_body_shapes else {}
        
        # Analyze template
        logger.info("\n[4/9] Analyzing template...")
        template_analysis = template_analyzer.analyze_template(
            template_data["image"],
            template_data["faces"]
        )
        logger.info(f"Template pose detected: {template_analysis.get('pose') is not None}")
        logger.info(f"Template expression: {template_analysis.get('expression', {}).get('type', 'unknown')}")
        
        # Process faces - THIS IS THE CRITICAL STEP
        logger.info("\n[5/9] Processing faces (FACE SWAP) - CRITICAL STEP...")
        customer_image = customer_data["images"][0]
        customer_faces = customer_data["faces"][0]
        
        if customer_faces and template_data["faces"]:
            logger.info("Extracting face identity from customer image...")
            face_identity = face_processor.extract_face_identity(
                customer_image, customer_faces[0]
            )
            
            # Verify face identity was extracted
            if face_identity.get("aligned_face") is None:
                logger.error("✗ Failed to extract face identity!")
                return False
            
            logger.info(f"Face identity extracted: aligned_face shape {face_identity['aligned_face'].shape}")
            
            logger.info("Matching expression...")
            expression_match = face_processor.match_expression(
                face_identity,
                template_data["faces"][0],
                template_analysis.get("expression", {})
            )
            
            logger.info("Compositing face onto template...")
            result = face_processor.composite_face(
                face_identity["aligned_face"],
                template_data["image"],
                template_data["faces"][0],
                expression_match
            )
            
            # CRITICAL VERIFICATION: Check if face was actually swapped
            logger.info("\n--- VERIFYING FACE SWAP OCCURRED ---")
            if np.array_equal(result, template_data["image"]):
                logger.error("✗ CRITICAL: Face swap failed - output is identical to template!")
                logger.error("The composite_face method did not work properly")
                return False
            else:
                diff = np.mean(np.abs(result.astype(float) - template_data["image"].astype(float)))
                logger.info(f"✓ Face composite completed - mean difference from template: {diff:.2f}")
                if diff < 5.0:
                    logger.warning("⚠ Warning: Difference is very small - face swap may not have worked properly")
            
            # Early quality check
            face_bbox = template_data["faces"][0].get("bbox", [0, 0, 0, 0])
            early_quality = quality_control.assess_quality(
                result,
                [customer_faces[0]] if customer_faces else [],
                template_data["faces"],
                template_analysis,
                body_shape=fused_body_shape
            )
            
            logger.info(f"Early quality score: {early_quality.get('overall_score', 0.0):.2f}")
            
            # Refine face if no severe distortion
            if early_quality.get("face_distortion", 0.0) <= 0.4:
                logger.info("Refining face...")
                result = refiner.refine_face(
                    result,
                    face_bbox,
                    expression_type=expression_match.get("expression_type", "neutral"),
                    expression_details=expression_match
                )
            else:
                logger.warning(f"Face distortion too high ({early_quality.get('face_distortion', 0.0):.2f}), skipping refinement")
        else:
            logger.error("✗ Cannot process faces - missing face data")
            return False
        
        # Warp body
        logger.info("\n[6/9] Warping body to match pose...")
        if fused_body_shape and template_analysis.get("pose"):
            customer_body = fused_body_shape
            template_pose = template_analysis["pose"]
            customer_pose = {"keypoints": customer_body.get("pose_keypoints", {})}
            
            if customer_pose["keypoints"]:
                logger.info("Building warp blueprint...")
                blueprint = body_warper.build_warp_blueprint(customer_body, template_pose)
                
                logger.info("Warping body...")
                warped_body = body_warper.warp_body_to_pose(
                    customer_data["images"][0],
                    customer_pose,
                    template_pose,
                    customer_body.get("body_mask"),
                    blueprint=blueprint,
                    customer_body_shape=customer_body
                )
                
                logger.info("Adapting clothing to body...")
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
            else:
                logger.warning("No customer pose keypoints, skipping body warping")
        else:
            logger.warning("No body shape or template pose data, skipping body warping")
        
        # Compose
        logger.info("\n[7/9] Composing final image...")
        composed = composer.compose(
            result,
            template_data["image"],
            body_mask=fused_body_shape.get("body_mask") if fused_body_shape else None,
            lighting_info=template_analysis.get("lighting")
        )
        
        # Refine
        logger.info("\n[8/9] Refining image...")
        refined = refiner.refine_composition(
            composed,
            template_analysis,
            fused_body_shape if fused_body_shape else {},
            strength=0.8
        )
        result = refined
        
        # Quality check
        logger.info("\n[9/9] Assessing quality...")
        quality = quality_control.assess_quality(
            result,
            customer_data["faces"][0] if customer_data["faces"] else [],
            template_data["faces"],
            template_analysis,
            body_shape=fused_body_shape
        )
        
        logger.info(f"Final quality score: {quality['overall_score']:.2f}")
        if quality.get("issues"):
            logger.warning(f"Issues detected: {quality['issues']}")
        
        # FINAL VERIFICATION
        logger.info("\n" + "=" * 60)
        logger.info("FINAL VERIFICATION")
        logger.info("=" * 60)
        
        # Resize for comparison if needed
        if result.shape != original_template.shape:
            result_for_compare = cv2.resize(result, (original_template.shape[1], original_template.shape[0]))
        else:
            result_for_compare = result
        
        if not verify_face_swap_happened(result_for_compare, original_template, original_customer):
            logger.error("✗ FINAL VERIFICATION FAILED - Face swap did not occur properly!")
            return False
        
        # Save result
        logger.info(f"\nSaving result to {output_path}...")
        save_image(result, output_path)
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"✗ Pipeline failed: {e}", exc_info=True)
        return False


def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("FACE-BODY SWAP TEST - NEW IMAGE (FIXED)")
    logger.info("=" * 60)
    
    # File paths - use a proper template that has faces
    customer_image = "IMG20251019131550.jpg"
    
    # Try different templates until we find one with faces
    templates_to_try = [
        "examples/templates/individual_casual_001.png",
        "examples/templates/individual_action_002.png",
        "examples/templates/couple_garden_001.png",
    ]
    
    template_image = None
    for template in templates_to_try:
        if Path(template).exists():
            # Quick check if template has faces
            preprocessor = Preprocessor()
            try:
                template_data = preprocessor.preprocess_template(template)
                if template_data["faces"] and len(template_data["faces"]) > 0:
                    template_image = template
                    logger.info(f"Found template with faces: {template}")
                    break
            except:
                continue
    
    if not template_image:
        logger.error("No suitable template found with faces!")
        return
    
    output_image = "outputs/new_image_test_result_fixed.png"
    
    # Verify customer image exists
    if not Path(customer_image).exists():
        logger.error(f"Customer image not found: {customer_image}")
        return
    
    # Run pipeline
    success = run_full_pipeline_with_verification(customer_image, template_image, output_image)
    
    if success:
        logger.info(f"\n✓ Test completed successfully!")
        logger.info(f"Output saved to: {output_image}")
        logger.info("\nPlease verify the output image to confirm the face swap worked correctly.")
    else:
        logger.error(f"\n✗ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()






