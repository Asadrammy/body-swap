"""Google AI Studio (Gemini) integration for image analysis and quality assessment"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import io
import base64

import warnings
import os

# Suppress all warnings for google.generativeai and protobuf
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
warnings.filterwarnings("ignore", message=".*google.generativeai.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.*")
warnings.filterwarnings("ignore", message=".*GetPrototype.*")

try:
    # Try new google.genai first (recommended)
    try:
        import google.genai as genai
        GOOGLE_AI_AVAILABLE = True
    except ImportError:
        # Fallback to deprecated google.generativeai
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as genai
        GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None

from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)


class GoogleAIClient:
    """Client for Google AI Studio (Gemini) API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Google AI client
        
        Args:
            api_key: Google AI Studio API key. If None, reads from environment or config.
        """
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError(
                "google-generativeai or google.genai package not installed. "
                "Install with: pip install google-generativeai"
            )
        
        # Get API key from parameter, environment, or config
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Google AI API key not provided. "
                "Set GOOGLE_AI_API_KEY or GEMINI_API_KEY environment variable, "
                "or pass api_key parameter."
            )
        
        # Configure API
        genai.configure(api_key=self.api_key)
        
        # Initialize models
        self.text_model = None
        self.vision_model = None
        self._initialize_models()
        
        logger.info("Google AI client initialized successfully")
    
    def _initialize_models(self):
        """Initialize available Gemini models"""
        # Try to initialize text/vision model
        model_names = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash']
        
        for model_name in model_names:
            try:
                self.text_model = genai.GenerativeModel(model_name)
                self.vision_model = self.text_model  # Same model can handle vision
                logger.info(f"Initialized Google AI model: {model_name}")
                return
            except Exception as e:
                logger.debug(f"Failed to initialize {model_name}: {e}")
                continue
        
        if self.text_model is None:
            logger.warning("Could not initialize any Gemini model, some features may not work")
    
    def analyze_image_quality(
        self,
        image: np.ndarray,
        reference_image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze image quality using Google AI
        
        Args:
            image: Image to analyze (numpy array)
            reference_image: Optional reference image for comparison
        
        Returns:
            Dictionary with quality metrics and analysis
        """
        if self.vision_model is None:
            raise RuntimeError("Vision model not initialized")
        
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Prepare prompt
            prompt = """
            Analyze this image and provide a quality assessment for face and body swapping.
            Evaluate:
            1. Face similarity and natural appearance (0-10)
            2. Body proportions and realism (0-10)
            3. Overall image quality and artifacts (0-10)
            4. Lighting and color consistency (0-10)
            5. Edge blending and seam visibility (0-10)
            
            Provide scores and brief comments for each aspect.
            """
            
            # Generate analysis
            response = self.vision_model.generate_content([prompt, pil_image])
            
            # Parse response (basic parsing - can be enhanced)
            analysis_text = response.text if hasattr(response, 'text') else str(response)
            
            # Extract scores (simple regex-based extraction)
            scores = self._extract_scores(analysis_text)
            
            result = {
                "overall_score": scores.get("overall", 7.0),
                "face_score": scores.get("face", 7.0),
                "body_score": scores.get("body", 7.0),
                "quality_score": scores.get("quality", 7.0),
                "lighting_score": scores.get("lighting", 7.0),
                "blending_score": scores.get("blending", 7.0),
                "analysis_text": analysis_text,
                "recommendations": self._extract_recommendations(analysis_text)
            }
            
            logger.info(f"Image quality analysis completed: overall_score={result['overall_score']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Image quality analysis failed: {e}")
            # Return default scores on error
            return {
                "overall_score": 7.0,
                "face_score": 7.0,
                "body_score": 7.0,
                "quality_score": 7.0,
                "lighting_score": 7.0,
                "blending_score": 7.0,
                "analysis_text": f"Analysis failed: {str(e)}",
                "recommendations": []
            }
    
    def get_refinement_suggestions(
        self,
        image: np.ndarray,
        issues: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get AI-powered suggestions for image refinement
        
        Args:
            image: Image to analyze
            issues: Optional list of known issues
        
        Returns:
            Dictionary with refinement suggestions
        """
        if self.text_model is None:
            raise RuntimeError("Text model not initialized")
        
        try:
            # Convert to PIL
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            issues_text = ", ".join(issues) if issues else "none specified"
            
            prompt = f"""
            This image has been processed for face and body swapping.
            Known issues: {issues_text}
            
            Provide specific technical suggestions for improving:
            1. Face blending and natural appearance
            2. Body proportions and clothing fit
            3. Edge refinement and seam removal
            4. Color and lighting adjustments
            5. Overall realism enhancement
            
            Be specific and technical in your recommendations.
            """
            
            response = self.vision_model.generate_content([prompt, pil_image])
            suggestions_text = response.text if hasattr(response, 'text') else str(response)
            
            result = {
                "suggestions": suggestions_text,
                "priority_areas": self._extract_priority_areas(suggestions_text),
                "technical_notes": self._extract_technical_notes(suggestions_text)
            }
            
            logger.info("Refinement suggestions generated")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get refinement suggestions: {e}")
            return {
                "suggestions": f"Failed to generate suggestions: {str(e)}",
                "priority_areas": [],
                "technical_notes": []
            }
    
    def compare_images(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        description: str = "Compare these two images"
    ) -> Dict[str, Any]:
        """
        Compare two images using AI vision
        
        Args:
            image1: First image
            image2: Second image
            description: Description of what to compare
        
        Returns:
            Comparison results
        """
        if self.vision_model is None:
            raise RuntimeError("Vision model not initialized")
        
        try:
            # Convert to PIL
            def to_pil(img):
                if isinstance(img, np.ndarray):
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    return Image.fromarray(img)
                return img
            
            pil_image1 = to_pil(image1)
            pil_image2 = to_pil(image2)
            
            prompt = f"""
            {description}
            
            Analyze and compare:
            1. Similarities and differences
            2. Quality improvements or degradations
            3. Specific areas that need attention
            4. Overall assessment
            """
            
            response = self.vision_model.generate_content([prompt, pil_image1, pil_image2])
            comparison_text = response.text if hasattr(response, 'text') else str(response)
            
            result = {
                "comparison": comparison_text,
                "improvements": self._extract_improvements(comparison_text),
                "regressions": self._extract_regressions(comparison_text)
            }
            
            logger.info("Image comparison completed")
            return result
            
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            return {
                "comparison": f"Comparison failed: {str(e)}",
                "improvements": [],
                "regressions": []
            }
    
    def _extract_scores(self, text: str) -> Dict[str, float]:
        """Extract numerical scores from analysis text"""
        import re
        scores = {}
        
        # Look for patterns like "Face similarity: 8.5" or "Face: 8/10"
        patterns = [
            (r'face[^:]*:?\s*(\d+\.?\d*)', 'face'),
            (r'body[^:]*:?\s*(\d+\.?\d*)', 'body'),
            (r'quality[^:]*:?\s*(\d+\.?\d*)', 'quality'),
            (r'lighting[^:]*:?\s*(\d+\.?\d*)', 'lighting'),
            (r'blending[^:]*:?\s*(\d+\.?\d*)', 'blending'),
            (r'overall[^:]*:?\s*(\d+\.?\d*)', 'overall'),
        ]
        
        for pattern, key in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize to 0-10 scale if needed
                    if score > 10:
                        score = score / 10.0
                    scores[key] = min(10.0, max(0.0, score))
                except:
                    pass
        
        return scores
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from analysis text"""
        recommendations = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'improve', 'fix', 'adjust']):
                if len(line) > 10:  # Filter out very short lines
                    recommendations.append(line)
        
        return recommendations[:5]  # Return top 5
    
    def _extract_priority_areas(self, text: str) -> List[str]:
        """Extract priority areas from suggestions"""
        priority_keywords = ['face', 'body', 'edges', 'blending', 'lighting', 'color', 'proportions']
        areas = []
        
        for keyword in priority_keywords:
            if keyword in text.lower():
                areas.append(keyword)
        
        return areas
    
    def _extract_technical_notes(self, text: str) -> List[str]:
        """Extract technical notes from suggestions"""
        notes = []
        lines = text.split('\n')
        
        for line in lines:
            if any(term in line.lower() for term in ['mask', 'warp', 'blend', 'refine', 'inpaint', 'transform']):
                if len(line) > 15:
                    notes.append(line.strip())
        
        return notes[:3]  # Return top 3
    
    def _extract_improvements(self, text: str) -> List[str]:
        """Extract improvements from comparison"""
        improvements = []
        lines = text.split('\n')
        
        for line in lines:
            if any(word in line.lower() for word in ['better', 'improved', 'enhanced', 'superior']):
                improvements.append(line.strip())
        
        return improvements
    
    def _extract_regressions(self, text: str) -> List[str]:
        """Extract regressions from comparison"""
        regressions = []
        lines = text.split('\n')
        
        for line in lines:
            if any(word in line.lower() for word in ['worse', 'degraded', 'regressed', 'poorer']):
                regressions.append(line.strip())
        
        return regressions


def create_google_ai_client(api_key: Optional[str] = None) -> Optional[GoogleAIClient]:
    """
    Factory function to create Google AI client
    
    Args:
        api_key: Optional API key. If None, reads from environment.
    
    Returns:
        GoogleAIClient instance or None if not available
    """
    try:
        return GoogleAIClient(api_key=api_key)
    except Exception as e:
        logger.warning(f"Could not create Google AI client: {e}")
        return None

