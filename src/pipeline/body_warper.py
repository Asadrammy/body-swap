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
        blueprint: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Warp customer body to match template pose
        
        Args:
            customer_image: Customer reference image
            customer_pose: Customer pose keypoints
            template_pose: Template pose keypoints
            body_mask: Optional body mask to restrict warping
        
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
        
        # Extract corresponding keypoints
        if blueprint and blueprint.get("control_points"):
            src_points = np.array([cp["src"] for cp in blueprint["control_points"]])
            dst_points = np.array([cp["dst"] for cp in blueprint["control_points"]])
        else:
            src_points, dst_points = self._extract_corresponding_keypoints(
                customer_keypoints, template_keypoints, customer_image.shape
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
        """Create a reusable blueprint describing how to warp the body."""
        customer_keypoints = customer_body_shape.get("pose_keypoints", customer_body_shape.get("keypoints", {}))
        template_keypoints = template_pose.get("keypoints", {})
        blueprint = {"control_points": [], "scale_map": {}, "template_measurements": {}}
        
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
        
        for name in key_order:
            if name in customer_keypoints and name in template_keypoints:
                blueprint["control_points"].append({
                    "name": name,
                    "src": customer_keypoints[name],
                    "dst": template_keypoints[name]
                })
        
        template_measurements = self._estimate_template_measurements(template_keypoints)
        blueprint["template_measurements"] = template_measurements
        blueprint["scale_map"] = self._derive_scale_map(
            customer_body_shape.get("measurements", {}),
            template_measurements
        )
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
        mask_3d = self._prepare_mask_for_blend(scaled_mask)
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

    def _synthesize_visible_skin(
        self,
        image: np.ndarray,
        template_clothing: Dict,
        customer_body_shape: Dict
    ) -> np.ndarray:
        """Fill open-chest regions with synthesized skin tone."""
        skin_profile = customer_body_shape.get("skin_profile", {})
        tone = skin_profile.get("tone")
        if tone is None:
            return image
        
        torso_mask = template_clothing.get("masks", {}).get("torso")
        visible_parts = template_clothing.get("visible_body_parts", [])
        if torso_mask is None or "chest" not in visible_parts:
            return image
        
        tone_layer = np.ones_like(image, dtype=np.float32)
        tone_layer[..., 0] *= tone[0]
        tone_layer[..., 1] *= tone[1]
        tone_layer[..., 2] *= tone[2]
        
        mask_3d = self._prepare_mask_for_blend(torso_mask) * self.skin_blend_strength
        blended = image.astype(np.float32) * (1 - mask_3d) + tone_layer * mask_3d
        return np.clip(blended, 0, 255).astype(np.uint8)

