"""Test script to run face-body swap on new customer image with full model loading"""

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


def verify_models_loaded():
    """Verify that all models are properly loaded"""
    logger.info("=" * 60)
    logger.info("VERIFYING MODEL LOADING")
    logger.info("=" * 60)
    
    try:
        preprocessor = Preprocessor()
        logger.info("✓ Preprocessor initialized")
        
        body_analyzer = BodyAnalyzer()
        logger.info("✓ BodyAnalyzer initialized")
        
        template_analyzer = TemplateAnalyzer()
        logger.info("✓ TemplateAnalyzer initialized")
        
        face_processor = FaceProcessor()
        logger.info("✓ FaceProcessor initialized")
        
        # Check if face detector has models loaded
        if hasattr(face_processor.face_detector, 'model') or hasattr(face_processor.face_detector, 'detector'):
            logger.info("✓ Face detector models loaded")
        else:
            logger.warning("⚠ Face detector models may not be loaded")
        
        body_warper = BodyWarper()
        logger.info("✓ BodyWarper initialized")
        
        composer = Composer()
        logger.info("✓ Composer initialized")
        
        refiner = Refiner()
        logger.info("✓ Refiner initialized")
        
        quality_control = QualityControl()
        logger.info("✓ QualityControl initialized")
        
        logger.info("=" * 60)
        logger.info("ALL MODELS INITIALIZED SUCCESSFULLY")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"✗ Model initialization failed: {e}", exc_info=True)
        return False


def run_full_pipeline(customer_image_path: str, template_path: str, output_path: str):
    """Run the complete face-body swap pipeline"""
    logger.info("=" * 60)
    logger.info("STARTING FULL PIPELINE EXECUTION")
    logger.info("=" * 60)
    logger.info(f"Customer image: {customer_image_path}")
    logger.info(f"Template: {template_path}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)
    
    try:
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
        
        logger.info(f"Customer faces detected: {len(customer_data['faces'][0]) if customer_data['faces'] else 0}")
        logger.info(f"Template faces detected: {len(template_data['faces']) if template_data['faces'] else 0}")
        
        if not customer_data['faces'] or not customer_data['faces'][0]:
            logger.error("✗ No faces detected in customer image!")
            return False
        
        if not template_data['faces']:
            logger.error("✗ No faces detected in template!")
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
        
        # Process faces
        logger.info("\n[5/9] Processing faces (FACE SWAP)...")
        customer_image = customer_data["images"][0]
        customer_faces = customer_data["faces"][0]
        
        if customer_faces and template_data["faces"]:
            logger.info("Extracting face identity from customer image...")
            face_identity = face_processor.extract_face_identity(
                customer_image, customer_faces[0]
            )
            
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
            
            # Verify face was actually swapped (check if result is different from template)
            if np.array_equal(result, template_data["image"]):
                logger.error("✗ CRITICAL: Face swap failed - output is identical to template!")
                logger.error("This means the face composite step did not work properly")
                return False
            else:
                logger.info("✓ Face swap completed - output differs from template")
            
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
        
        # Final verification - check if result is different from input
        customer_img = load_image(customer_image_path)
        template_img = load_image(template_path)
        
        if np.array_equal(result, template_img):
            logger.error("✗ CRITICAL ERROR: Final output is identical to template!")
            logger.error("The pipeline did not perform any transformation")
            return False
        elif np.array_equal(result, customer_img):
            logger.error("✗ CRITICAL ERROR: Final output is identical to customer image!")
            logger.error("The pipeline copied the input instead of swapping")
            return False
        else:
            logger.info("✓ Output is different from both input images - transformation occurred")
        
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
    logger.info("FACE-BODY SWAP TEST - NEW IMAGE")
    logger.info("=" * 60)
    
    # File paths
    customer_image = "IMG20251019131550.jpg"
    template_image = "examples/templates/individual_casual_001.png"  # Use a template
    output_image = "outputs/new_image_test_result.png"
    
    # Verify files exist
    if not Path(customer_image).exists():
        logger.error(f"Customer image not found: {customer_image}")
        return
    
    if not Path(template_image).exists():
        logger.error(f"Template image not found: {template_image}")
        logger.info("Available templates:")
        template_dir = Path("examples/templates")
        if template_dir.exists():
            for f in template_dir.glob("*.png"):
                logger.info(f"  - {f}")
        return
    
    # Verify models
    if not verify_models_loaded():
        logger.error("Model verification failed!")
        return
    
    # Run pipeline
    success = run_full_pipeline(customer_image, template_image, output_image)
    
    if success:
        logger.info(f"\n✓ Test completed successfully!")
        logger.info(f"Output saved to: {output_image}")
    else:
        logger.error(f"\n✗ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()






