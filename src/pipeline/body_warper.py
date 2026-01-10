"""Body warping and clothing adaptation"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.warp_utils import warp_image, thin_plate_spline, mesh_warp, create_mesh
from ..utils.image_utils import resize_image

logger = get_logger(__name__)


class BodyWarper:
    """Warp customer body to match template pose and adapt clothing"""
    
    def __init__(self):
        """Initialize body warper"""
        self.warp_method = "tps"  # Thin Plate Spline
        self.mesh_rows = get_config("processing.mesh_rows", 24)
        self.mesh_cols = get_config("processing.mesh_cols", 12)
        self.skin_blend_strength = 0.7
    
    def warp_body_to_pose(
        self,
        customer_image: np.ndarray,
        customer_pose: Dict,
        template_pose: Dict,
        body_mask: Optional[np.ndarray] = None,
        blueprint: Optional[Dict] = None,
        customer_body_shape: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Warp customer body to match template pose, accounting for body size differences.
        
        If customer is larger/smaller than template, the template pose is scaled to match
        customer's body size before warping.
        
        Args:
            customer_image: Customer reference image
            customer_pose: Customer pose keypoints
            template_pose: Template pose keypoints
            body_mask: Optional body mask to restrict warping
            blueprint: Optional pre-computed warp blueprint
            customer_body_shape: Optional customer body shape analysis (for size adjustment)
        
        Returns:
            Warped customer image
        """
        if not isinstance(customer_pose, dict):
            customer_keypoints = customer_pose
        else:
            customer_keypoints = customer_pose.get("keypoints", customer_pose.get("pose_keypoints", {}))
        template_keypoints = template_pose.get("keypoints", {})
        
        if not customer_keypoints or not template_keypoints:
            logger.warning("Insufficient keypoints for body warping")
            return customer_image
        
        # Scale template keypoints to match customer body size if measurements available
        scaled_template_keypoints = template_keypoints.copy()
        if customer_body_shape and customer_body_shape.get("measurements"):
            customer_measurements = customer_body_shape.get("measurements", {})
            template_measurements = self._estimate_template_measurements(template_keypoints)
            
            if template_measurements:
                # Calculate average scale factor from multiple measurements
                scale_factors = []
                for key in ["shoulder_width", "hip_width", "waist_width", "torso_height"]:
                    if key in customer_measurements and key in template_measurements:
                        if template_measurements[key] > 1e-3:
                            scale_factors.append(customer_measurements[key] / template_measurements[key])
                
                if scale_factors:
                    # Use median scale to be robust to outliers
                    avg_scale = float(np.median(scale_factors))
                    logger.info(f"Body size adjustment: scaling template pose by {avg_scale:.2f}x")
                    
                    # Scale template keypoints relative to center
                    scaled_template_keypoints = self._scale_keypoints_to_size(
                        template_keypoints,
                        avg_scale,
                        customer_keypoints
                    )
        
        # Extract corresponding keypoints
        if blueprint and blueprint.get("control_points"):
            src_points = np.array([cp["src"] for cp in blueprint["control_points"]])
            # Use scaled destination points if size adjustment was applied
            if blueprint.get("size_adjusted_dst_points") is not None:
                dst_points = np.array(blueprint["size_adjusted_dst_points"])
            else:
                dst_points = np.array([cp["dst"] for cp in blueprint["control_points"]])
        else:
            src_points, dst_points = self._extract_corresponding_keypoints(
                customer_keypoints, scaled_template_keypoints, customer_image.shape
            )
        
        if len(src_points) < 3:
            logger.warning("Not enough corresponding keypoints for warping")
            return customer_image
        
        # Apply warping
        warped_image = warp_image(
            customer_image,
            src_points,
            dst_points,
            method=self.warp_method
        )
        
        # Apply body mask if provided
        if body_mask is not None:
            # Ensure mask matches warped_image dimensions
            h_warped, w_warped = warped_image.shape[:2]
            if body_mask.shape[:2] != (h_warped, w_warped):
                body_mask = cv2.resize(body_mask, (w_warped, h_warped), interpolation=cv2.INTER_NEAREST)
            
            # Ensure customer_image matches warped_image dimensions for blending
            h_customer, w_customer = customer_image.shape[:2]
            if (h_customer, w_customer) != (h_warped, w_warped):
                customer_image = cv2.resize(customer_image, (w_warped, h_warped))
            
            mask_3d = np.stack([body_mask] * 3, axis=2) / 255.0
            warped_image = (
                warped_image * mask_3d +
                customer_image * (1 - mask_3d)
            ).astype(np.uint8)
        
        return warped_image
    
    def adapt_clothing_to_body(
        self,
        template_image: np.ndarray,
        template_clothing: Dict,
        customer_body_shape: Dict,
        template_pose: Dict
    ) -> Dict:
        """
        Adapt template clothing to customer body shape
        
        Args:
            template_image: Template image
            template_clothing: Template clothing analysis
            customer_body_shape: Customer body shape analysis
            template_pose: Template pose keypoints
        
        Returns:
            Image with adapted clothing
        """
        result = template_image.copy()
        
        # Get customer measurements
        customer_measurements = customer_body_shape.get("measurements", {})
        template_keypoints = template_pose.get("keypoints", {})
        
        if not customer_measurements or not template_keypoints:
            return {"image": result, "fit_report": {}}
        
        template_measurements = self._estimate_template_measurements(template_keypoints)
        scale_map = self._derive_scale_map(customer_measurements, template_measurements)
        girth_profile = customer_body_shape.get("girth_profile")
        
        fit_report = {
            "scale_map": scale_map,
            "items": {}
        }
        
        # Adapt each clothing item
        clothing_masks = template_clothing.get("masks", {})
        
        for item_name, clothing_mask in clothing_masks.items():
            if item_name in ["shirt", "torso"]:
                scale_x, scale_y = self._select_scale_for_item(item_name, scale_map, girth_profile)
                result, item_report = self._warp_region_to_scale(
                    result,
                    clothing_mask,
                    scale_x,
                    scale_y,
                    item_name
                )
                fit_report["items"][item_name] = item_report
            elif "leg" in item_name or item_name in ["pants"]:
                scale_x, scale_y = self._select_scale_for_item(item_name, scale_map, girth_profile)
                result, item_report = self._warp_region_to_scale(
                    result,
                    clothing_mask,
                    scale_x,
                    scale_y,
                    item_name
                )
                fit_report["items"][item_name] = item_report
        
        if template_clothing.get("has_open_chest"):
            result = self._synthesize_visible_skin(
                result,
                template_clothing,
                customer_body_shape
            )
            fit_report["skin_synthesis_applied"] = True
        else:
            fit_report["skin_synthesis_applied"] = False
        
        return {"image": result, "fit_report": fit_report}
    
    def _extract_corresponding_keypoints(
        self,
        src_keypoints: Dict,
        dst_keypoints: Dict,
        image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract corresponding keypoints for warping
        
        Args:
            src_keypoints: Source keypoints
            dst_keypoints: Destination keypoints
            image_shape: Image dimensions
        
        Returns:
            Tuple of (source_points, destination_points)
        """
        # Key body points to use for warping
        key_point_names = [
            "nose", "neck",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]
        
        src_points = []
        dst_points = []
        
        for point_name in key_point_names:
            if point_name in src_keypoints and point_name in dst_keypoints:
                src_pt = src_keypoints[point_name]
                dst_pt = dst_keypoints[point_name]
                src_points.append(src_pt)
                dst_points.append(dst_pt)
        
        # Convert to numpy arrays
        if src_points and dst_points:
            return np.array(src_points), np.array(dst_points)
        else:
            return np.array([]), np.array([])
    
    def _adapt_torso_clothing(
        self,
        template_image: np.ndarray,
        clothing_mask: np.ndarray,
        customer_measurements: Dict,
        template_keypoints: Dict
    ) -> np.ndarray:
        """
        Adapt torso clothing to customer body shape
        
        Args:
            template_image: Template image
            clothing_mask: Clothing mask
            customer_measurements: Customer body measurements
            template_keypoints: Template keypoints
        
        Returns:
            Adapted clothing image
        """
        # Extract clothing region
        clothing_region = template_image.copy()
        clothing_region[clothing_mask == 0] = 0
        
        # Calculate scale factors based on measurements
        scale_x = 1.0
        scale_y = 1.0
        
        if "shoulder_width" in customer_measurements:
            if "left_shoulder" in template_keypoints and "right_shoulder" in template_keypoints:
                template_shoulder_width = np.linalg.norm(
                    np.array(template_keypoints["left_shoulder"]) -
                    np.array(template_keypoints["right_shoulder"])
                )
                if template_shoulder_width > 0:
                    scale_x = customer_measurements["shoulder_width"] / template_shoulder_width
        
        # Apply scaling (simplified - real implementation would use more sophisticated warping)
        h, w = clothing_region.shape[:2]
        
        # Get bounding box of clothing
        coords = np.column_stack(np.where(clothing_mask > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Crop clothing region
            clothing_crop = clothing_region[y_min:y_max, x_min:x_max]
            
            # Scale
            new_w = int(clothing_crop.shape[1] * scale_x)
            new_h = int(clothing_crop.shape[0] * scale_y)
            scaled_clothing = resize_image(clothing_crop, (new_w, new_h))
            
            # Place back (centered)
            result = template_image.copy()
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            new_x_min = center_x - new_w // 2
            new_y_min = center_y - new_h // 2
            
            # Ensure within bounds
            new_x_min = max(0, min(new_x_min, w - new_w))
            new_y_min = max(0, min(new_y_min, h - new_h))
            new_x_max = min(new_x_min + new_w, w)
            new_y_max = min(new_y_min + new_h, h)
            
            scaled_clothing_cropped = scaled_clothing[
                :new_y_max - new_y_min,
                :new_x_max - new_x_min
            ]
            
            # Blend
            mask_crop = clothing_mask[new_y_min:new_y_max, new_x_min:new_x_max]
            mask_3d = np.stack([mask_crop] * 3, axis=2) / 255.0
            result[new_y_min:new_y_max, new_x_min:new_x_max] = (
                result[new_y_min:new_y_max, new_x_min:new_x_max] * (1 - mask_3d) +
                scaled_clothing_cropped * mask_3d
            ).astype(np.uint8)
            
            return result
        
        return template_image
    
    def generate_fabric_folds(
        self,
        warped_body: np.ndarray,
        body_mask: np.ndarray,
        pose_keypoints: Dict
    ) -> np.ndarray:
        """
        Generate realistic fabric folds based on body shape and pose
        
        Args:
            warped_body: Warped body image
            body_mask: Body mask
            pose_keypoints: Pose keypoints
        
        Returns:
            Image with fabric fold details
        """
        # This is a placeholder - real implementation would use
        # more sophisticated methods like normal mapping or GANs
        # For now, apply some texture enhancement
        
        result = warped_body.copy()
        
        # Apply subtle texture to clothing regions
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Enhance edges slightly in clothing regions
        edges_3d = np.stack([edges] * 3, axis=2)
        mask_3d = np.stack([body_mask] * 3, axis=2) / 255.0
        
        # Subtle enhancement
        result = result + (edges_3d * 0.1 * mask_3d).astype(np.uint8)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

    def build_warp_blueprint(self, customer_body_shape: Dict, template_pose: Dict) -> Dict:
        """
        Create a reusable blueprint describing how to warp the body.
        Includes size-adjusted destination points to account for body size differences.
        """
        customer_keypoints = customer_body_shape.get("pose_keypoints", customer_body_shape.get("keypoints", {}))
        template_keypoints = template_pose.get("keypoints", {})
        blueprint = {
            "control_points": [],
            "scale_map": {},
            "template_measurements": {},
            "size_adjusted_dst_points": None
        }
        
        if not customer_keypoints or not template_keypoints:
            return blueprint
        
        key_order = [
            "nose", "neck",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]
        
        # Build control points
        for name in key_order:
            if name in customer_keypoints and name in template_keypoints:
                blueprint["control_points"].append({
                    "name": name,
                    "src": customer_keypoints[name],
                    "dst": template_keypoints[name]
                })
        
        # Calculate measurements and scale map
        template_measurements = self._estimate_template_measurements(template_keypoints)
        blueprint["template_measurements"] = template_measurements
        customer_measurements = customer_body_shape.get("measurements", {})
        blueprint["scale_map"] = self._derive_scale_map(
            customer_measurements,
            template_measurements
        )
        
        # Calculate size-adjusted destination points
        if customer_measurements and template_measurements:
            scale_factors = []
            for key in ["shoulder_width", "hip_width", "waist_width", "torso_height"]:
                if key in customer_measurements and key in template_measurements:
                    if template_measurements[key] > 1e-3:
                        scale_factors.append(customer_measurements[key] / template_measurements[key])
            
            if scale_factors:
                avg_scale = float(np.median(scale_factors))
                scaled_template_keypoints = self._scale_keypoints_to_size(
                    template_keypoints,
                    avg_scale,
                    customer_keypoints
                )
                
                # Extract size-adjusted destination points in same order as control_points
                size_adjusted_dst = []
                for cp in blueprint["control_points"]:
                    name = cp["name"]
                    if name in scaled_template_keypoints:
                        size_adjusted_dst.append(scaled_template_keypoints[name])
                    else:
                        size_adjusted_dst.append(cp["dst"])
                
                blueprint["size_adjusted_dst_points"] = size_adjusted_dst
                blueprint["size_scale_factor"] = avg_scale
        
        return blueprint

    def _estimate_template_measurements(self, keypoints: Dict) -> Dict:
        """Estimate template measurements using keypoints."""
        measurements: Dict[str, float] = {}
        if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
            measurements["shoulder_width"] = float(
                np.linalg.norm(np.array(keypoints["left_shoulder"]) - np.array(keypoints["right_shoulder"]))
            )
        if "left_hip" in keypoints and "right_hip" in keypoints:
            measurements["hip_width"] = float(
                np.linalg.norm(np.array(keypoints["left_hip"]) - np.array(keypoints["right_hip"]))
            )
        if "left_shoulder" in keypoints and "left_hip" in keypoints:
            measurements["torso_height"] = float(
                np.linalg.norm(np.array(keypoints["left_shoulder"]) - np.array(keypoints["left_hip"]))
            )
        if "left_hip" in keypoints and "left_ankle" in keypoints:
            measurements["leg_length"] = float(
                np.linalg.norm(np.array(keypoints["left_hip"]) - np.array(keypoints["left_ankle"]))
            )
        return measurements

    def _derive_scale_map(
        self,
        customer_measurements: Dict[str, float],
        template_measurements: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute scaling ratios between customer and template measurements."""
        scale_map: Dict[str, float] = {}
        for key, customer_value in customer_measurements.items():
            template_value = template_measurements.get(key)
            if template_value and template_value > 1e-3:
                scale_map[key] = float(customer_value / template_value)
        return scale_map

    def _select_scale_for_item(
        self,
        item_name: str,
        scale_map: Dict[str, float],
        girth_profile: Optional[List[Dict]]
    ) -> Tuple[float, float]:
        """Select horizontal and vertical scaling factors for a clothing item."""
        default_scale = scale_map.get("shoulder_width", 1.0)
        vertical_scale = scale_map.get("torso_height", scale_map.get("body_height", 1.0))
        
        if "pant" in item_name or "leg" in item_name:
            horizontal = scale_map.get("hip_width", scale_map.get("waist_width", default_scale))
            vertical = scale_map.get("leg_length", vertical_scale)
            return horizontal, vertical
        
        if girth_profile:
            waist_entry = min(girth_profile, key=lambda e: abs(e["ratio"] - 0.5))
            horizontal = max(default_scale, waist_entry.get("normalized_width", default_scale))
        else:
            horizontal = default_scale
        
        return horizontal, vertical_scale

    def _warp_region_to_scale(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        scale_x: float,
        scale_y: float,
        item_name: str
    ) -> Tuple[np.ndarray, Dict]:
        """Scale a masked clothing region and blend it back."""
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return image, {
                "item": item_name,
                "status": "no_pixels",
                "scale_x": scale_x,
                "scale_y": scale_y
            }
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        crop = image[y_min:y_max, x_min:x_max]
        mask_crop = mask[y_min:y_max, x_min:x_max]
        
        new_w = max(1, int(crop.shape[1] * scale_x))
        new_h = max(1, int(crop.shape[0] * scale_y))
        scaled_crop = cv2.resize(crop, (new_w, new_h))
        scaled_mask = cv2.resize(mask_crop, (new_w, new_h))
        
        result = image.copy()
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        new_x_min = max(0, min(center_x - new_w // 2, image.shape[1] - new_w))
        new_y_min = max(0, min(center_y - new_h // 2, image.shape[0] - new_h))
        new_x_max = new_x_min + new_w
        new_y_max = new_y_min + new_h
        
        target_region = result[new_y_min:new_y_max, new_x_min:new_x_max]
        
        # Ensure target_region, scaled_crop, and scaled_mask have matching dimensions
        h_target, w_target = target_region.shape[:2]
        h_scaled, w_scaled = scaled_crop.shape[:2]
        
        if (h_target, w_target) != (h_scaled, w_scaled):
            scaled_crop = cv2.resize(scaled_crop, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
            scaled_mask = cv2.resize(scaled_mask, (w_target, h_target), interpolation=cv2.INTER_NEAREST)
        
        mask_3d = self._prepare_mask_for_blend(scaled_mask)
        
        # Ensure mask_3d matches target_region channels
        if len(target_region.shape) == 3 and mask_3d.shape[2] == 1:
            mask_3d = np.repeat(mask_3d, target_region.shape[2], axis=2)
        
        blended = (target_region * (1 - mask_3d) + scaled_crop * mask_3d).astype(np.uint8)
        result[new_y_min:new_y_max, new_x_min:new_x_max] = blended
        
        report = {
            "item": item_name,
            "status": "scaled",
            "scale_x": round(scale_x, 3),
            "scale_y": round(scale_y, 3),
            "original_size": [int(crop.shape[1]), int(crop.shape[0])],
            "new_size": [int(new_w), int(new_h)]
        }
        return result, report

    def _prepare_mask_for_blend(self, mask: np.ndarray) -> np.ndarray:
        """Convert a single-channel mask into a feathered 3-channel alpha map."""
        if mask.max() == 0:
            return np.zeros((*mask.shape, 1), dtype=np.float32)
        blurred = cv2.GaussianBlur(mask, (11, 11), 0)
        normalized = np.clip(blurred / 255.0, 0, 1)[..., None]
        return normalized

    def _scale_keypoints_to_size(
        self,
        template_keypoints: Dict,
        scale_factor: float,
        customer_keypoints: Dict
    ) -> Dict:
        """
        Scale template keypoints to match customer body size.
        
        Args:
            template_keypoints: Original template keypoints
            scale_factor: Scale factor (customer_size / template_size)
            customer_keypoints: Customer keypoints (for reference center)
        
        Returns:
            Scaled template keypoints
        """
        if abs(scale_factor - 1.0) < 0.01:
            # No significant size difference
            return template_keypoints
        
        scaled_keypoints = {}
        
        # Find reference center point (use neck or average of shoulders)
        if "neck" in template_keypoints:
            center_ref = np.array(template_keypoints["neck"])
            # Ensure it's [x, y] format
            if len(center_ref) > 2:
                center_ref = center_ref[:2]
        elif "left_shoulder" in template_keypoints and "right_shoulder" in template_keypoints:
            left_sh = np.array(template_keypoints["left_shoulder"])
            right_sh = np.array(template_keypoints["right_shoulder"])
            # Ensure they're [x, y] format
            if len(left_sh) > 2:
                left_sh = left_sh[:2]
            if len(right_sh) > 2:
                right_sh = right_sh[:2]
            center_ref = (left_sh + right_sh) / 2
        else:
            # Use customer neck as reference if available
            if "neck" in customer_keypoints:
                center_ref = np.array(customer_keypoints["neck"])
                if len(center_ref) > 2:
                    center_ref = center_ref[:2]
            else:
                # Fallback: use template neck or first available point
                first_point = np.array(list(template_keypoints.values())[0])
                if len(first_point) > 2:
                    first_point = first_point[:2]
                center_ref = first_point
        
        # Scale each keypoint relative to center
        for name, point in template_keypoints.items():
            point_arr = np.array(point)
            # Ensure point_arr is 1D with 2 elements [x, y]
            if point_arr.ndim > 1:
                point_arr = point_arr.flatten()[:2]
            elif len(point_arr) > 2:
                point_arr = point_arr[:2]
            
            # Ensure center_ref is 1D with 2 elements
            if center_ref.ndim > 1:
                center_ref = center_ref.flatten()[:2]
            elif len(center_ref) > 2:
                center_ref = center_ref[:2]
            
            # Calculate offset from center
            offset = point_arr - center_ref
            # Scale the offset
            scaled_offset = offset * scale_factor
            # New position
            scaled_keypoints[name] = (center_ref + scaled_offset).tolist()
        
        return scaled_keypoints
    
    def _synthesize_visible_skin(
        self,
        image: np.ndarray,
        template_clothing: Dict,
        customer_body_shape: Dict
    ) -> np.ndarray:
        """
        Fill open-chest regions and visible skin areas with synthesized skin tone.
        Enhanced to support male, female, and children with realistic skin texture.
        """
        skin_profile = customer_body_shape.get("skin_profile", {})
        tone = skin_profile.get("tone")
        if tone is None:
            logger.warning("No skin tone available for synthesis")
            return image
        
        # Get visible body regions from skin profile
        visible_regions = skin_profile.get("visible_body_regions", {})
        
        result = image.copy()
        
        # Process chest region if open
        torso_mask = template_clothing.get("masks", {}).get("torso")
        visible_parts = template_clothing.get("visible_body_parts", [])
        
        if "chest" in visible_regions:
            chest_mask = visible_regions["chest"]
            result = self._apply_skin_synthesis(result, chest_mask, tone, skin_profile)
        elif torso_mask is not None and "chest" in visible_parts:
            # Use torso mask as fallback
            result = self._apply_skin_synthesis(result, torso_mask, tone, skin_profile)
        
        # Process arm regions if visible
        for side in ["left_arm", "right_arm"]:
            if side in visible_regions:
                arm_mask = visible_regions[side]
                result = self._apply_skin_synthesis(result, arm_mask, tone, skin_profile)
        
        return result
    
    def _apply_skin_synthesis(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        tone: List[float],
        skin_profile: Dict
    ) -> np.ndarray:
        """
        Apply realistic skin synthesis to a masked region.
        Uses texture from face reference if available to avoid flat/plastic look.
        """
        if mask is None or mask.sum() == 0:
            return image
        
        result = image.copy()
        
        # Get face reference for texture if available
        face_reference = skin_profile.get("face_reference")
        
        if face_reference is not None and face_reference.size > 0:
            # Use face texture as base for skin synthesis
            # Resize face reference to match mask region size
            mask_coords = np.column_stack(np.where(mask > 0))
            if len(mask_coords) > 0:
                y_min, x_min = mask_coords.min(axis=0)
                y_max, x_max = mask_coords.max(axis=0)
                region_h = y_max - y_min
                region_w = x_max - x_min
                
                if region_h > 0 and region_w > 0:
                    # Resize face reference to region size
                    face_texture = cv2.resize(face_reference, (region_w, region_h))
                    
                    # Adjust color to match skin tone while preserving texture
                    face_gray = cv2.cvtColor(face_texture, cv2.COLOR_BGR2GRAY)
                    face_gray_3d = np.stack([face_gray] * 3, axis=2)
                    
                    # Create tone layer
                    tone_array = np.array(tone, dtype=np.float32)
                    if len(tone_array) >= 3:
                        # Handle BGR format
                        tone_layer = np.ones((region_h, region_w, 3), dtype=np.float32)
                        tone_layer[..., 0] = tone_array[0]  # B
                        tone_layer[..., 1] = tone_array[1]  # G
                        tone_layer[..., 2] = tone_array[2]  # R
                        
                        # Blend texture with tone
                        # Use face texture for detail, adjust to target tone
                        texture_factor = 0.4  # How much texture to preserve
                        tone_factor = 0.6  # How much target tone to apply
                        
                        # Normalize face texture
                        face_normalized = face_texture.astype(np.float32) / 255.0
                        tone_normalized = tone_layer / 255.0
                        
                        # Blend
                        blended = (
                            face_normalized * texture_factor +
                            tone_normalized * tone_factor
                        )
                        
                        # Add subtle noise for realism
                        noise = np.random.normal(0, 0.02, blended.shape).astype(np.float32)
                        blended = np.clip(blended + noise, 0, 1)
                        
                        synthesized = (blended * 255).astype(np.uint8)
                        
                        # Apply to masked region
                        mask_crop = mask[y_min:y_max, x_min:x_max]
                        
                        # Ensure mask_crop matches synthesized dimensions
                        h_synth, w_synth = synthesized.shape[:2]
                        h_mask, w_mask = mask_crop.shape[:2]
                        if (h_mask, w_mask) != (h_synth, w_synth):
                            mask_crop = cv2.resize(mask_crop, (w_synth, h_synth), interpolation=cv2.INTER_NEAREST)
                        
                        mask_3d = self._prepare_mask_for_blend(mask_crop) * self.skin_blend_strength
                        
                        # Ensure mask_3d matches synthesized channels
                        if len(synthesized.shape) == 3 and mask_3d.shape[2] == 1:
                            mask_3d = np.repeat(mask_3d, synthesized.shape[2], axis=2)
                        
                        target_region = result[y_min:y_max, x_min:x_max]
                        
                        # Ensure target_region matches synthesized dimensions
                        h_target, w_target = target_region.shape[:2]
                        if (h_target, w_target) != (h_synth, w_synth):
                            target_region = cv2.resize(target_region, (w_synth, h_synth), interpolation=cv2.INTER_LINEAR)
                        
                        blended_region = (
                            target_region.astype(np.float32) * (1 - mask_3d) +
                            synthesized * mask_3d
                        )
                        
                        # Resize back if needed
                        if (h_target, w_target) != (h_synth, w_synth):
                            blended_region = cv2.resize(blended_region.astype(np.uint8), (w_target, h_target), interpolation=cv2.INTER_LINEAR)
                            result[y_min:y_max, x_min:x_max] = blended_region
                        else:
                            result[y_min:y_max, x_min:x_max] = blended_region.astype(np.uint8)
        
        else:
            # Fallback: simple tone application with texture
            tone_array = np.array(tone, dtype=np.float32)
            if len(tone_array) >= 3:
                tone_layer = np.ones_like(image, dtype=np.float32)
                tone_layer[..., 0] = tone_array[0]
                tone_layer[..., 1] = tone_array[1]
                tone_layer[..., 2] = tone_array[2]
                
                # Add subtle texture variation
                noise = np.random.normal(0, 5, image.shape).astype(np.float32)
                tone_layer = np.clip(tone_layer + noise, 0, 255)
                
                mask_3d = self._prepare_mask_for_blend(mask) * self.skin_blend_strength
                blended = (
                    image.astype(np.float32) * (1 - mask_3d) +
                    tone_layer * mask_3d
                )
                result = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result

