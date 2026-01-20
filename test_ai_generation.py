"""Test script for AI image generation with client image"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
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

logger = get_logger(__name__)


def test_ai_generation_pipeline(customer_image_path: str, template_path: str = None, output_path: str = None):
    """
    Test the complete pipeline with AI image generation
    
    Args:
        customer_image_path: Path to customer image
        template_path: Optional template path (will use customer image as template if not provided)
        output_path: Output path for result
    """
    logger.info("=" * 80)
    logger.info("Testing AI Image Generation Pipeline")
    logger.info("=" * 80)
    
    # Check API keys
    logger.info("\nChecking API keys...")
    use_ai_api = os.getenv("USE_AI_API", "true").lower() == "true"
    openai_key = os.getenv("OPENAI_API_KEY")
    stability_key = os.getenv("STABILITY_API_KEY") or os.getenv("STABILITY_AI_API_KEY")
    replicate_key = os.getenv("REPLICATE_API_TOKEN")
    
    if use_ai_api:
        logger.info(f"USE_AI_API: {use_ai_api}")
        logger.info(f"OpenAI API Key: {'✓ Set' if openai_key else '✗ Not set'}")
        logger.info(f"Stability AI API Key: {'✓ Set' if stability_key else '✗ Not set'}")
        logger.info(f"Replicate API Token: {'✓ Set' if replicate_key else '✗ Not set'}")
        
        if not (openai_key or stability_key or replicate_key):
            logger.warning("\n⚠️  No AI API keys found!")
            logger.warning("Please set one of:")
            logger.warning("  - OPENAI_API_KEY (for DALL-E)")
            logger.warning("  - STABILITY_API_KEY (for Stability AI - recommended)")
            logger.warning("  - REPLICATE_API_TOKEN (for Replicate)")
            logger.warning("\nThe pipeline will fall back to local models if available.")
    else:
        logger.info("USE_AI_API is false, will use local models")
    
    # Load customer image
    logger.info(f"\nLoading customer image: {customer_image_path}")
    if not Path(customer_image_path).exists():
        logger.error(f"Customer image not found: {customer_image_path}")
        return False
    
    customer_image = load_image(customer_image_path)
    if customer_image is None:
        logger.error("Failed to load customer image")
        return False
    
    logger.info(f"Customer image loaded: {customer_image.shape}")
    
    # Use customer image as template if no template provided
    if template_path is None:
        logger.info("No template provided, using customer image as template")
        template_path = customer_image_path
        template_image = customer_image.copy()
    else:
        logger.info(f"Loading template: {template_path}")
        template_image = load_image(template_path)
        if template_image is None:
            logger.error("Failed to load template image")
            return False
        logger.info(f"Template image loaded: {template_image.shape}")
    
    # Initialize pipeline components
    logger.info("\nInitializing pipeline components...")
    preprocessor = Preprocessor()
    body_analyzer = BodyAnalyzer()
    template_analyzer = TemplateAnalyzer()
    face_processor = FaceProcessor()
    body_warper = BodyWarper()
    composer = Composer()
    refiner = Refiner()
    quality_control = QualityControl()
    
    logger.info("✓ All components initialized")
    
    try:
        # Stage 1: Preprocessing
        logger.info("\n" + "=" * 80)
        logger.info("Stage 1: Preprocessing")
        logger.info("=" * 80)
        
        customer_data = preprocessor.preprocess_customer_photos([customer_image_path])
        template_data = preprocessor.preprocess_template(template_path)
        
        logger.info(f"Customer faces detected: {len(customer_data['faces'][0]) if customer_data['faces'] else 0}")
        logger.info(f"Template faces detected: {len(template_data['faces']) if template_data['faces'] else 0}")
        
        # Stage 2: Body Analysis
        logger.info("\n" + "=" * 80)
        logger.info("Stage 2: Body Analysis")
        logger.info("=" * 80)
        
        customer_body_shapes = []
        for img, faces in zip(customer_data["images"], customer_data["faces"]):
            if faces:
                body_shape = body_analyzer.analyze_body_shape(img, faces)
                customer_body_shapes.append(body_shape)
                logger.info(f"Body type: {body_shape.get('body_type', 'unknown')}")
                logger.info(f"Measurements: {body_shape.get('measurements', {})}")
        
        fused_body_shape = body_analyzer.fuse_body_shapes(customer_body_shapes) if customer_body_shapes else {}
        
        # Stage 3: Template Analysis
        logger.info("\n" + "=" * 80)
        logger.info("Stage 3: Template Analysis")
        logger.info("=" * 80)
        
        template_analysis = template_analyzer.analyze_template(
            template_data["image"],
            template_data["faces"]
        )
        
        logger.info(f"Template expression: {template_analysis.get('expression', {}).get('type', 'neutral')}")
        logger.info(f"Action pose: {template_analysis.get('is_action_pose', False)}")
        logger.info(f"Open chest: {template_analysis.get('clothing', {}).get('has_open_chest', False)}")
        
        # Stage 4: Face Processing
        logger.info("\n" + "=" * 80)
        logger.info("Stage 4: Face Processing")
        logger.info("=" * 80)
        
        customer_faces_list = customer_data["faces"]
        template_faces = template_data["faces"]
        
        if customer_faces_list and template_faces:
            # Process faces
            result_image = face_processor.process_multiple_faces(
                customer_data["images"],
                customer_faces_list,
                template_data["image"],
                template_faces
            )
            logger.info("✓ Faces processed and composited")
        else:
            logger.warning("No faces detected, using template image as base")
            result_image = template_data["image"].copy()
        
        # Stage 5: Body Warping
        logger.info("\n" + "=" * 80)
        logger.info("Stage 5: Body Warping")
        logger.info("=" * 80)
        
        customer_pose = customer_body_shapes[0].get("pose_keypoints", {}) if customer_body_shapes else {}
        template_pose = template_analysis.get("pose_keypoints", {})
        
        if customer_pose and template_pose:
            blueprint = body_warper.build_warp_blueprint(fused_body_shape, {"keypoints": template_pose})
            warped_body = body_warper.warp_body_to_pose(
                result_image,
                customer_pose,
                {"keypoints": template_pose},
                blueprint=blueprint,
                customer_body_shape=fused_body_shape
            )
            result_image = warped_body
            logger.info("✓ Body warped to match template pose")
        else:
            logger.warning("Insufficient pose data for body warping")
        
        # Stage 6: Clothing Adaptation
        logger.info("\n" + "=" * 80)
        logger.info("Stage 6: Clothing Adaptation")
        logger.info("=" * 80)
        
        template_clothing = template_analysis.get("clothing", {})
        if template_clothing and fused_body_shape:
            clothing_result = body_warper.adapt_clothing_to_body(
                result_image,
                template_clothing,
                fused_body_shape,
                {"keypoints": template_pose}
            )
            result_image = clothing_result["image"]
            logger.info("✓ Clothing adapted to customer body")
        
        # Stage 7: Composition
        logger.info("\n" + "=" * 80)
        logger.info("Stage 7: Composition")
        logger.info("=" * 80)
        
        try:
            # Get body mask from template analysis if available
            body_mask = None
            if template_analysis and "body_mask" in template_analysis:
                body_mask = template_analysis["body_mask"]
            
            # Get lighting info from template analysis
            lighting_info = template_analysis.get("lighting") if template_analysis else None
            
            composed = composer.compose(
                result_image,
                template_data["image"],
                body_mask=body_mask,
                lighting_info=lighting_info
            )
            result_image = composed
            logger.info("✓ Image composed with background")
        except Exception as e:
            logger.warning(f"Composition failed: {e}, using current result")
        
        # Stage 8: AI Refinement
        logger.info("\n" + "=" * 80)
        logger.info("Stage 8: AI Image Generation Refinement")
        logger.info("=" * 80)
        
        if use_ai_api:
            logger.info("Using AI API for refinement (no local models, avoids distortion)")
        else:
            logger.info("Using local models for refinement")
        
        # Refine face if faces detected - with ENHANCED artifact detection
        from src.utils.artifact_detector import ArtifactDetector
        artifact_detector = ArtifactDetector()
        
        expected_face_regions = []
        if template_faces:
            for face in template_faces:
                bbox = face.get("bbox", [0, 0, 0, 0])
                if bbox[2] > 0 and bbox[3] > 0:
                    expected_face_regions.append(tuple(bbox))
                    expression = template_analysis.get("expression", {})
                    result_image = refiner.refine_face(
                        result_image,
                        tuple(bbox),
                        expression.get("type", "neutral"),
                        expression
                    )
                    logger.info("✓ Face refined")
        
        # CRITICAL: Check for duplicate faces AFTER ALL face refinements
        logger.info("Checking for duplicate faces and artifacts...")
        is_valid, duplicates = artifact_detector.validate_face_placement(
            result_image, expected_face_regions
        )
        if not is_valid:
            logger.warning(f"Found {len(duplicates)} duplicate face(s), removing...")
            result_image = artifact_detector.remove_duplicate_faces(
                result_image, duplicates, expected_face_regions, template_data["image"]
            )
            logger.info("✓ Duplicate faces removed")
        
        # Final check after composition refinement
        if customer_body_shapes:
            result_image = refiner.refine_composition(
                result_image,
                template_analysis,
                fused_body_shape
            )
            logger.info("✓ Composition refined")
            
            # Final artifact check after all processing
            is_valid, duplicates = artifact_detector.validate_face_placement(
                result_image, expected_face_regions
            )
            if not is_valid:
                logger.warning(f"Found {len(duplicates)} duplicate face(s) after composition, removing...")
                result_image = artifact_detector.remove_duplicate_faces(
                    result_image, duplicates, expected_face_regions, template_data["image"]
                )
                logger.info("✓ Final duplicate face cleanup completed")
        
        # Refine composition
        quality_metrics = None
        if customer_body_shapes:
            result_image = refiner.refine_composition(
                result_image,
                template_analysis,
                fused_body_shape
            )
            logger.info("✓ Composition refined")
        
        # Stage 9: Quality Control
        logger.info("\n" + "=" * 80)
        logger.info("Stage 9: Quality Control")
        logger.info("=" * 80)
        
        # Prepare quality assessment parameters
        customer_faces_list = customer_data.get("faces", [])
        customer_faces = customer_faces_list[0] if customer_faces_list else []
        
        quality_metrics = quality_control.assess_quality(
            result_image=result_image,
            customer_faces=customer_faces if isinstance(customer_faces, list) else [customer_faces] if customer_faces else [],
            template_faces=template_faces if isinstance(template_faces, list) else [template_faces] if template_faces else [],
            template_analysis=template_analysis,
            body_shape=fused_body_shape
        )
        
        logger.info(f"Quality score: {quality_metrics.get('overall_score', 0):.2f}")
        logger.info(f"Face similarity: {quality_metrics.get('face_similarity', 0):.2f}")
        logger.info(f"Pose accuracy: {quality_metrics.get('pose_accuracy', 0):.2f}")
        
        # Save result
        if output_path is None:
            output_path = project_root / "outputs" / f"ai_generated_result_{Path(customer_image_path).stem}.jpg"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving result to: {output_path}")
        save_image(result_image, str(output_path))
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Output saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AI image generation pipeline")
    parser.add_argument(
        "--customer-image",
        type=str,
        default=r"D:\projects\image\face-body-swap\1760713603491 (1).jpg",
        help="Path to customer image"
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Path to template image (optional, uses customer image if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for result"
    )
    
    args = parser.parse_args()
    
    # Run test
    success = test_ai_generation_pipeline(
        args.customer_image,
        args.template,
        args.output
    )
    
    if success:
        print("\n[SUCCESS] Test completed successfully!")
        sys.exit(0)
    else:
        print("\n[FAILED] Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

