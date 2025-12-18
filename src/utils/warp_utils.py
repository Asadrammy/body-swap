"""Image warping utility functions"""

import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from typing import List, Tuple, Optional


def warp_image(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray,
              method: str = "tps") -> np.ndarray:
    """
    Warp image from source points to destination points
    
    Args:
        image: Input image
        src_points: Source control points (N, 2)
        dst_points: Destination control points (N, 2)
        method: Warping method ("tps" or "affine")
    
    Returns:
        Warped image
    """
    if method == "tps":
        return thin_plate_spline(image, src_points, dst_points)
    elif method == "affine":
        return affine_warp(image, src_points, dst_points)
    else:
        raise ValueError(f"Unknown warping method: {method}")


def thin_plate_spline(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Thin Plate Spline warping
    
    Args:
        image: Input image
        src_points: Source control points
        dst_points: Destination control points
    
    Returns:
        Warped image
    """
    h, w = image.shape[:2]
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Calculate TPS transformation
    tps = cv2.createThinPlateSplineShapeTransformer()
    src_pts = src_points.reshape(1, -1, 2).astype(np.float32)
    dst_pts = dst_points.reshape(1, -1, 2).astype(np.float32)
    
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
    tps.estimateTransformation(dst_pts, src_pts, matches)
    
    # Transform all points
    grid_points = np.dstack([x_grid.flatten(), y_grid.flatten()]).reshape(1, -1, 2).astype(np.float32)
    transformed_points = tps.applyTransformation(grid_points)[1]
    
    # Reshape and create warped image
    map_x = transformed_points[0, :, 0].reshape(h, w).astype(np.float32)
    map_y = transformed_points[0, :, 1].reshape(h, w).astype(np.float32)
    
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped


def affine_warp(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Affine transformation warping
    
    Args:
        image: Input image
        src_points: Source control points (minimum 3)
        dst_points: Destination control points (minimum 3)
    
    Returns:
        Warped image
    """
    if len(src_points) < 3:
        raise ValueError("Affine warping requires at least 3 control points")
    
    # Calculate affine transformation matrix
    M = cv2.getAffineTransform(src_points[:3].astype(np.float32), dst_points[:3].astype(np.float32))
    
    h, w = image.shape[:2]
    warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped


def create_mesh(width: int, height: int, rows: int = 10, cols: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a mesh grid for warping
    
    Args:
        width: Image width
        height: Image height
        rows: Number of mesh rows
        cols: Number of mesh columns
    
    Returns:
        Mesh coordinates (x_coords, y_coords)
    """
    x = np.linspace(0, width - 1, cols)
    y = np.linspace(0, height - 1, rows)
    x_grid, y_grid = np.meshgrid(x, y)
    return x_grid, y_grid


def mesh_warp(image: np.ndarray, src_mesh: Tuple[np.ndarray, np.ndarray],
             dst_mesh: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Warp image using mesh deformation
    
    Args:
        image: Input image
        src_mesh: Source mesh (x_coords, y_coords)
        dst_mesh: Destination mesh (x_coords, y_coords)
    
    Returns:
        Warped image
    """
    h, w = image.shape[:2]
    
    # Create maps from mesh
    src_x, src_y = src_mesh
    dst_x, dst_y = dst_mesh
    
    # Create coordinate maps
    map_x = cv2.resize(dst_x.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    map_y = cv2.resize(dst_y.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped


def perspective_warp(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Perspective transformation warping
    
    Args:
        image: Input image
        src_points: Source points (4 points)
        dst_points: Destination points (4 points)
    
    Returns:
        Warped image
    """
    if len(src_points) != 4 or len(dst_points) != 4:
        raise ValueError("Perspective warping requires exactly 4 control points")
    
    M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
    h, w = image.shape[:2]
    warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped

