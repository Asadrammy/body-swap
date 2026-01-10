"""CLI interface for face and body swap"""

import argparse
from pathlib import Path
from typing import List, Optional

from ..utils.logger import setup_logger, get_logger
from ..utils.config import get_config
from ..pipeline.preprocessor import Preprocessor
from ..pipeline.body_analyzer import BodyAnalyzer
from ..pipeline.template_analyzer import TemplateAnalyzer
from ..pipeline.face_processor import FaceProcessor
from ..pipeline.body_warper import BodyWarper
from ..pipeline.composer import Composer
from ..pipeline.refiner import Refiner
from ..pipeline.quality_control import QualityControl
from ..utils.image_utils import load_image, save_image

setup_logger()
logger = get_logger(__name__)


class SwapCLI:
    """Command-line interface for swap pipeline"""
    
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.body_analyzer = BodyAnalyzer()
        self.template_analyzer = TemplateAnalyzer()
        self.face_processor = FaceProcessor()
        self.body_warper = BodyWarper()
        self.composer = Composer()
        self.refiner = Refiner()
        self.quality_control = QualityControl()
    
    def swap(
        self,
        customer_photos: List[str],
        template: str,
        output: str,
        refine_mask: Optional[str] = None,
        no_refine: bool = False,
        export_intermediate: bool = False
    ):
        """
        Execute swap pipeline
        
        Args:
            customer_photos: List of customer photo paths
            template: Template image path
            output: Output image path
            refine_mask: Optional refinement mask path
            no_refine: Skip refinement step
            export_intermediate: Export intermediate results
        """
        logger.info(f"Starting swap: {len(customer_photos)} customer photos, template: {template}")
        
        try:
            # Preprocess inputs
            logger.info("Preprocessing inputs...")
            customer_data = self.preprocessor.preprocess_customer_photos(customer_photos)
            template_data = self.preprocessor.preprocess_template(template)
            
            intermediate_results = {}
            
            # Analyze customer body
            logger.info("Analyzing customer body shape...")
            customer_body_shapes = []
            for img, faces in zip(customer_data["images"], customer_data["faces"]):
                if faces:
                    body_shape = self.body_analyzer.analyze_body_shape(img, faces)
                    customer_body_shapes.append(body_shape)
                    if export_intermediate:
                        intermediate_results.setdefault("customer_body_samples", []).append(img)
            fused_body_shape = self.body_analyzer.fuse_body_shapes(customer_body_shapes) if customer_body_shapes else {}
            
            # Analyze template
            logger.info("Analyzing template...")
            template_analysis = self.template_analyzer.analyze_template(
                template_data["image"],
                template_data["faces"]
            )
            if export_intermediate:
                intermediate_results["template_analysis"] = template_data["image"]
            
            # Process faces
            logger.info("Processing faces...")
            if len(customer_data["images"]) == 1:
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
                    
                    result = self.face_processor.composite_face(
                        face_identity["aligned_face"],
                        template_data["image"],
                        template_data["faces"][0],
                        expression_match
                    )
                    
                    # Early quality check after face compositing to catch distortion early
                    face_bbox = template_data["faces"][0].get("bbox", [0, 0, 0, 0])
                    early_quality = self.quality_control.assess_quality(
                        result,
                        [customer_faces[0]] if customer_faces else [],
                        template_data["faces"],
                        template_analysis,
                        body_shape=fused_body_shape
                    )
                    
                    # If severe distortion detected, skip refinement and use original template face
                    if early_quality.get("face_distortion", 0.0) > 0.4:
                        logger.error(f"CRITICAL: Severe face distortion detected ({early_quality.get('face_distortion', 0.0):.2f}) - skipping face refinement to prevent further distortion")
                        # Return to template image and try again with no expression warping
                        result = template_data["image"].copy()
                        # Try compositing without expression matching
                        expression_match_no_warp = {
                            "warped_landmarks": customer_faces[0].get("landmarks", []),
                            "expression_applied": False,
                            "expression_type": template_analysis.get("expression", {}).get("type", "neutral")
                        }
                        # Preserve emotion data even without warping for Mickmumpitz workflow
                        template_expression = template_analysis.get("expression", {})
                        if isinstance(template_expression, dict):
                            for key in ["keywords", "intensity", "descriptors", "features"]:
                                if key in template_expression:
                                    expression_match_no_warp[key] = template_expression[key]
                        result = self.face_processor.composite_face(
                            face_identity["aligned_face"],
                            result,
                            template_data["faces"][0],
                            expression_match_no_warp
                        )
                    else:
                        # Only refine if no severe distortion
                        result = self.refiner.refine_face(
                            result,
                            face_bbox,
                            expression_type=expression_match.get("expression_type", "neutral"),
                            expression_details=expression_match
                        )
                    if export_intermediate:
                        intermediate_results["face_composite"] = result
                else:
                    result = template_data["image"].copy()
            else:
                # Multiple people
                result = self.face_processor.process_multiple_faces(
                    customer_data["images"],
                    customer_data["faces"],
                    template_data["image"],
                    template_data["faces"]
                )
                if export_intermediate:
                    intermediate_results["multi_face_composite"] = result
            
            # Warp body
            logger.info("Warping body to match pose...")
            if fused_body_shape and template_analysis.get("pose"):
                customer_body = fused_body_shape
                template_pose = template_analysis["pose"]
                customer_pose = {"keypoints": customer_body.get("pose_keypoints", {})}
                
                if customer_pose["keypoints"]:
                    blueprint = self.body_warper.build_warp_blueprint(customer_body, template_pose)
                    warped_body = self.body_warper.warp_body_to_pose(
                        customer_data["images"][0],
                        customer_pose,
                        template_pose,
                        customer_body.get("body_mask"),
                        blueprint=blueprint,
                        customer_body_shape=customer_body
                    )
                    
                    adapted = self.body_warper.adapt_clothing_to_body(
                        result,
                        template_analysis.get("clothing", {}),
                        customer_body,
                        template_pose
                    )
                    if isinstance(adapted, dict):
                        result = adapted.get("image", result)
                        if export_intermediate:
                            intermediate_results["fit_report"] = adapted.get("fit_report")
                    else:
                        result = adapted
                    if export_intermediate:
                        intermediate_results["warped_body"] = warped_body
                        intermediate_results["adapted_clothing"] = result
            
            # Compose
            logger.info("Composing final image...")
            composed = self.composer.compose(
                result,
                template_data["image"],
                body_mask=fused_body_shape.get("body_mask") if fused_body_shape else None,
                lighting_info=template_analysis.get("lighting")
            )
            if export_intermediate:
                intermediate_results["composed"] = composed
            
            # Refine
            if not no_refine:
                logger.info("Refining image...")
                refine_mask_img = None
                if refine_mask:
                    refine_mask_img = load_image(refine_mask)
                    if len(refine_mask_img.shape) == 3:
                        refine_mask_img = refine_mask_img[:, :, 0]  # Use first channel
                
                refined = self.refiner.refine_composition(
                    composed,
                    template_analysis,
                    fused_body_shape if fused_body_shape else {},
                    refinement_mask=refine_mask_img,
                    strength=0.8
                )
                result = refined
                if export_intermediate:
                    intermediate_results["refined"] = refined
            else:
                result = composed
            
            # Quality check
            logger.info("Assessing quality...")
            quality = self.quality_control.assess_quality(
                result,
                customer_data["faces"][0] if customer_data["faces"] else [],
                template_data["faces"],
                template_analysis,
                body_shape=fused_body_shape
            )
            
            face_boxes = [f.get("bbox", [0, 0, 0, 0]) for f in (customer_data["faces"][0] if customer_data["faces"] else [])]
            refinement_masks = self.quality_control.generate_refinement_masks(
                result,
                quality,
                face_boxes,
                fused_body_shape.get("body_mask") if fused_body_shape else None
            )
            
            if quality["overall_score"] < self.quality_control.quality_threshold and refinement_masks:
                recommended = quality.get("recommended_refinements") or list(refinement_masks.keys())
                region_subset = {name: refinement_masks[name] for name in recommended if name in refinement_masks}
                if region_subset:
                    logger.info(f"Performing targeted refinement on {list(region_subset.keys())}")
                    result = self.refiner.refine_composition(
                        result,
                        template_analysis,
                        fused_body_shape if fused_body_shape else {},
                        region_masks=region_subset,
                        quality=quality
                    )
                    quality = self.quality_control.assess_quality(
                        result,
                        customer_data["faces"][0] if customer_data["faces"] else [],
                        template_data["faces"],
                        template_analysis,
                        body_shape=fused_body_shape
                    )
            
            logger.info(f"Quality score: {quality['overall_score']:.2f}")
            if quality.get("issues"):
                logger.warning(f"Issues detected: {quality['issues']}")
            
            # Save result
            logger.info(f"Saving result to {output}...")
            save_image(result, output)
            
            # Export intermediate results if requested
            if export_intermediate:
                output_dir = Path(output).parent / f"{Path(output).stem}_intermediate"
                self.quality_control.export_intermediate_results(intermediate_results, str(output_dir))
                logger.info(f"Intermediate results exported to {output_dir}")
            
            logger.info("Swap completed successfully!")
            
        except Exception as e:
            logger.error(f"Swap failed: {e}", exc_info=True)
            raise


def main():
    """CLI main entry point"""
    parser = argparse.ArgumentParser(description="Face and Body Swap CLI")
    
    parser.add_argument(
        "command",
        choices=["swap"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--customer-photos",
        nargs="+",
        required=True,
        help="Paths to customer reference photos (1-2 images)"
    )
    
    parser.add_argument(
        "--template",
        required=True,
        help="Path to template image"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path"
    )
    
    parser.add_argument(
        "--refine-mask",
        help="Optional refinement mask path"
    )
    
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip refinement step"
    )
    
    parser.add_argument(
        "--export-intermediate",
        action="store_true",
        help="Export intermediate processing results"
    )
    
    args = parser.parse_args()
    
    if args.command == "swap":
        cli = SwapCLI()
        cli.swap(
            customer_photos=args.customer_photos,
            template=args.template,
            output=args.output,
            refine_mask=args.refine_mask,
            no_refine=args.no_refine,
            export_intermediate=args.export_intermediate
        )


if __name__ == "__main__":
    main()

