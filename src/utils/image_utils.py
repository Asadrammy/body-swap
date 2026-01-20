"""Image processing utility functions"""

import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple, Optional
from pathlib import Path


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image as numpy array (RGB format)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load with PIL and convert to RGB
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def save_image(image: np.ndarray, output_path: Union[str, Path], quality: int = 95, is_mask: bool = False):
    """
    Save image to file
    
    Args:
        image: Image as numpy array (RGB format)
        output_path: Output file path
        quality: JPEG quality (1-100)
        is_mask: If True, skip strict validation (masks can have few colors)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate image before saving
    if image is None or image.size == 0:
        raise ValueError(f"Cannot save empty image to {output_path}")
    
    if len(image.shape) < 2:
        raise ValueError(f"Invalid image shape: {image.shape}")
    
    # Auto-detect mask files by filename pattern
    if not is_mask:
        is_mask = "_mask_" in str(output_path) or output_path.name.startswith("mask_")
    
    # Check for solid color - skip validation for masks (they can have few colors)
    if not is_mask:
        if len(image.shape) == 3:
            unique_colors = len(np.unique(image.reshape(-1, image.shape[-1]), axis=0))
            std_dev = np.std(image)
            # Also check if all pixels are the same
            is_uniform = np.all(image == image.flat[0])
        else:
            unique_colors = len(np.unique(image))
            std_dev = np.std(image)
            is_uniform = np.all(image == image.flat[0])
        
        if is_uniform or unique_colors < 10:
            raise ValueError(f"Cannot save solid/uniform color image (unique_colors={unique_colors}, is_uniform={is_uniform}) to {output_path}. This indicates a pipeline error.")
        
        if std_dev < 5.0:
            raise ValueError(f"Cannot save low-variance image (std={std_dev:.2f}) to {output_path}. This indicates a pipeline error.")
    
    # Convert numpy array to PIL Image
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Ensure image is in correct format
    if len(image.shape) == 3 and image.shape[2] == 3:
        img = Image.fromarray(image, 'RGB')
    elif len(image.shape) == 2:
        img = Image.fromarray(image, 'L')
    else:
        img = Image.fromarray(image)
    
    # Save with appropriate format
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        img.save(output_path, 'JPEG', quality=quality)
    else:
        img.save(output_path)


def resize_image(
    image: np.ndarray,
    size: Union[int, Tuple[int, int]],
    maintain_aspect: bool = True,
    interpolation: int = cv2.INTER_LANCZOS4
) -> np.ndarray:
    """
    Resize image
    
    Args:
        image: Input image as numpy array
        size: Target size (int for square, or (width, height) tuple)
        maintain_aspect: Whether to maintain aspect ratio
        interpolation: OpenCV interpolation method
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if isinstance(size, int):
        if maintain_aspect:
            if h > w:
                new_h, new_w = size, int(w * size / h)
            else:
                new_h, new_w = int(h * size / w), size
        else:
            new_h, new_w = size, size
    else:
        new_w, new_h = size
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized


def normalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                   std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Normalize image for model input
    
    Args:
        image: Input image (0-255 range)
        mean: Mean values for normalization
        std: Standard deviation values for normalization
    
    Returns:
        Normalized image (0-1 range)
    """
    image = image.astype(np.float32) / 255.0
    
    if len(image.shape) == 3:
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        image = (image - mean) / std
    
    return image


def denormalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Denormalize image from model output
    
    Args:
        image: Normalized image
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
    
    Returns:
        Denormalized image (0-255 range)
    """
    if len(image.shape) == 3:
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        image = image * std + mean
    
    image = np.clip(image, 0, 1) * 255.0
    return image.astype(np.uint8)


def pad_image(image: np.ndarray, target_size: Tuple[int, int], fill_value: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image to target size
    
    Args:
        image: Input image
        target_size: Target (width, height)
        fill_value: Padding value
    
    Returns:
        Padded image and (pad_x, pad_y) offsets
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    pad_x = max(0, (target_w - w) // 2)
    pad_y = max(0, (target_h - h) // 2)
    
    padded = np.pad(
        image,
        ((pad_y, target_h - h - pad_y), (pad_x, target_w - w - pad_x), (0, 0)) if len(image.shape) == 3
        else ((pad_y, target_h - h - pad_y), (pad_x, target_w - w - pad_x)),
        mode='constant',
        constant_values=fill_value
    )
    
    return padded, (pad_x, pad_y)


def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop image using bounding box
    
    Args:
        image: Input image
        bbox: Bounding box (x, y, width, height)
    
    Returns:
        Cropped image
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]


def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Blend two images
    
    Args:
        img1: First image
        img2: Second image
        alpha: Blend factor (0-1)
        mask: Optional blending mask
    
    Returns:
        Blended image
    """
    # Ensure img1 and img2 have the same dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if (h1, w1) != (h2, w2):
        # Resize img2 to match img1
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_LINEAR)
    
    if mask is not None:
        # Ensure mask matches image dimensions
        h_mask, w_mask = mask.shape[:2]
        if (h_mask, w_mask) != (h1, w1):
            mask = cv2.resize(mask, (w1, h1), interpolation=cv2.INTER_NEAREST)
        
        mask = mask.astype(np.float32) / 255.0
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        
        # Ensure mask has same number of channels as images
        if len(img1.shape) == 3 and mask.shape[2] == 1:
            mask = np.repeat(mask, img1.shape[2], axis=2)
        
        blended = img1 * (1 - mask) + img2 * mask
    else:
        blended = img1 * (1 - alpha) + img2 * alpha
    
    return blended.astype(np.uint8)

