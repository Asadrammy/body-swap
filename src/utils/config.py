"""Configuration management"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables (with error handling for corrupted .env)
try:
    load_dotenv()
except Exception as e:
    # If .env has issues, try to load with override=False and ignore errors
    import warnings
    warnings.warn(f"Error loading .env file: {e}. Continuing with environment variables only.")
    try:
        load_dotenv(override=False)
    except:
        pass  # Continue without .env file

_default_config = {
    "models": {
        "face_detector": "insightface",
        "pose_detector": "mediapipe",
        "segmenter": "sam",
        "generator": {
            "base_model": "runwayml/stable-diffusion-v1-5",
            "controlnet": "lllyasviel/sd-controlnet-openpose",
            "inpaint_model": "runwayml/stable-diffusion-inpainting",
            "device": "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu",
            "lora_paths": []
        }
    },
    "processing": {
        "image_size": 512,
        "max_image_size": 1024,
        "batch_size": 1,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "refinement_strength": 0.8,
        "quality_threshold": 0.85,
        "mesh_rows": 24,
        "mesh_cols": 12,
        "depth_map_size": 128,
        "region_refine_strengths": {
            "face": 0.65,
            "body": 0.55,
            "edges": 0.45,
            "problems": 0.7
        }
    },
    "paths": {
        "models_dir": "models",
        "outputs_dir": "outputs",
        "temp_dir": "temp"
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "timeout": 300
    }
}

_config_cache: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or use defaults
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    config = _default_config.copy()
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, file_config)
    
    # Override with environment variables
    env_config = {
        "models": {
            "generator": {
                "device": os.getenv("DEVICE", config["models"]["generator"]["device"])
            }
        },
        "processing": {
            "image_size": int(os.getenv("IMAGE_SIZE", config["processing"]["image_size"])),
            "max_image_size": int(os.getenv("MAX_IMAGE_SIZE", config["processing"]["max_image_size"])),
        },
        "paths": {
            "models_dir": os.getenv("MODELS_DIR", config["paths"]["models_dir"]),
            "outputs_dir": os.getenv("OUTPUTS_DIR", config["paths"]["outputs_dir"]),
        },
        "api": {
            "host": os.getenv("API_HOST", config["api"]["host"]),
            "port": int(os.getenv("API_PORT", config["api"]["port"])),
        }
    }
    
    config = _deep_merge(config, env_config)
    _config_cache = config
    return config


def get_config(key: str = None, default: Any = None) -> Any:
    """
    Get configuration value by key (supports dot notation)
    
    Args:
        key: Configuration key (e.g., "models.generator.device")
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    """
    config = load_config()
    
    if key is None:
        return config
    
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


def _deep_merge(base: Dict, update: Dict) -> Dict:
    """Recursively merge two dictionaries"""
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

