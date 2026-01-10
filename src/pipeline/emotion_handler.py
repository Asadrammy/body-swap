"""Mickmumpitz-style emotion handler for consistent emotion control"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EmotionHandler:
    """Handle emotions using Mickmumpitz workflow approach with prompt-based control"""
    
    # Mickmumpitz-style emotion mapping with detailed descriptors
    EMOTION_MAP = {
        "neutral": {
            "keywords": ["neutral expression", "calm face", "relaxed features", "natural expression"],
            "intensity": 0.5,
            "descriptors": ["calm", "composed", "balanced", "natural"]
        },
        "happy": {
            "keywords": ["genuine smile", "happy expression", "joyful face", "bright expression"],
            "intensity": 0.8,
            "descriptors": ["joyful", "cheerful", "bright", "uplifting", "positive"]
        },
        "sad": {
            "keywords": ["sad expression", "melancholic face", "subdued expression", "somber look"],
            "intensity": 0.6,
            "descriptors": ["melancholic", "subdued", "thoughtful", "contemplative"]
        },
        "angry": {
            "keywords": ["intense expression", "focused face", "determined look", "strong expression"],
            "intensity": 0.7,
            "descriptors": ["intense", "focused", "determined", "strong", "powerful"]
        },
        "surprised": {
            "keywords": ["surprised expression", "alert face", "awakened look", "reactive expression"],
            "intensity": 0.75,
            "descriptors": ["alert", "awakened", "reactive", "engaged", "attentive"]
        },
        "fearful": {
            "keywords": ["concerned expression", "alert face", "cautious look", "watchful expression"],
            "intensity": 0.65,
            "descriptors": ["alert", "cautious", "watchful", "aware"]
        },
        "disgusted": {
            "keywords": ["displeased expression", "uncomfortable face", "distasteful look"],
            "intensity": 0.6,
            "descriptors": ["uncomfortable", "distasteful", "displeased"]
        },
        "excited": {
            "keywords": ["excited expression", "enthusiastic face", "energetic look", "vibrant expression"],
            "intensity": 0.85,
            "descriptors": ["enthusiastic", "energetic", "vibrant", "dynamic", "lively"]
        },
        "confident": {
            "keywords": ["confident expression", "assured face", "self-assured look", "poised expression"],
            "intensity": 0.7,
            "descriptors": ["assured", "self-assured", "poised", "composed", "self-confident"]
        },
        "serious": {
            "keywords": ["serious expression", "focused face", "concentrated look", "intense expression"],
            "intensity": 0.65,
            "descriptors": ["focused", "concentrated", "intense", "deliberate", "purposeful"]
        },
        "playful": {
            "keywords": ["playful expression", "mischievous smile", "lighthearted face", "cheerful look"],
            "intensity": 0.75,
            "descriptors": ["mischievous", "lighthearted", "cheerful", "fun-loving", "carefree"]
        },
        "romantic": {
            "keywords": ["warm expression", "tender smile", "affectionate look", "loving face"],
            "intensity": 0.7,
            "descriptors": ["warm", "tender", "affectionate", "loving", "gentle"]
        }
    }
    
    def __init__(self):
        """Initialize emotion handler"""
        self.emotion_cache = {}
    
    def detect_emotion_from_landmarks(
        self,
        landmarks: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Detect emotion from facial landmarks using enhanced analysis
        
        Args:
            landmarks: Facial landmarks array
            face_bbox: Face bounding box (x, y, w, h)
        
        Returns:
            Detailed emotion analysis
        """
        if landmarks is None or len(landmarks) < 5:
            return self._get_emotion_data("neutral")
        
        # Convert landmarks to numpy array if needed
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks)
        
        # Analyze facial features for emotion detection
        emotion_features = self._analyze_facial_features(landmarks, face_bbox)
        
        # Classify emotion based on features
        detected_emotion = self._classify_emotion(emotion_features)
        
        return self._get_emotion_data(detected_emotion, emotion_features)
    
    def _analyze_facial_features(
        self,
        landmarks: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Dict:
        """
        Analyze facial features for emotion detection
        
        Args:
            landmarks: Facial landmarks
            face_bbox: Face bounding box
        
        Returns:
            Feature analysis dictionary
        """
        features = {
            "mouth_open": False,
            "smile_intensity": 0.0,
            "eyebrow_raised": False,
            "eye_openness": 1.0,
            "mouth_width": 0.0,
            "mouth_height": 0.0,
            "eyebrow_angle": 0.0
        }
        
        # Normalize landmarks if bbox provided
        if face_bbox is not None:
            x, y, w, h = face_bbox
            if w > 0 and h > 0:
                normalized_landmarks = landmarks.copy()
                normalized_landmarks[:, 0] = (landmarks[:, 0] - x) / w
                normalized_landmarks[:, 1] = (landmarks[:, 1] - y) / h
            else:
                normalized_landmarks = landmarks
        else:
            normalized_landmarks = landmarks
        
        # Analyze mouth (assuming landmarks structure)
        # For standard 68-point landmarks: 48-67 are mouth points
        # For 5-point landmarks: last 2 points are mouth corners
        if len(landmarks) >= 5:
            # Try to identify mouth region
            mouth_indices = self._get_mouth_indices(len(landmarks))
            if mouth_indices:
                mouth_points = landmarks[mouth_indices]
                if len(mouth_points) >= 2:
                    # Calculate mouth width and height
                    mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0])
                    mouth_height = np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
                    
                    features["mouth_width"] = float(mouth_width)
                    features["mouth_height"] = float(mouth_height)
                    features["mouth_open"] = mouth_height > (mouth_width * 0.3)
                    
                    # Detect smile (mouth corners raised)
                    if len(mouth_points) >= 2:
                        corner_y_avg = np.mean(mouth_points[:2, 1])
                        center_y = np.mean(mouth_points[:, 1])
                        smile_intensity = max(0.0, min(1.0, (corner_y_avg - center_y) / 10.0))
                        features["smile_intensity"] = float(smile_intensity)
        
        # Analyze eyebrows (simplified - would need proper landmark indices)
        # For standard landmarks: 17-26 are eyebrow points
        
        # Analyze eyes (simplified)
        # For standard landmarks: 36-47 are eye points
        
        return features
    
    def _get_mouth_indices(self, num_landmarks: int) -> List[int]:
        """Get mouth landmark indices based on landmark count"""
        if num_landmarks == 68:
            # Standard 68-point landmarks
            return list(range(48, 68))
        elif num_landmarks == 5:
            # 5-point landmarks (last 2 are mouth corners)
            return [3, 4]
        elif num_landmarks >= 10:
            # Assume last 4-6 points are mouth
            return list(range(num_landmarks - 6, num_landmarks))
        else:
            # Fallback: use last 2 points
            return [num_landmarks - 2, num_landmarks - 1] if num_landmarks >= 2 else []
    
    def _classify_emotion(self, features: Dict) -> str:
        """
        Classify emotion from facial features
        
        Args:
            features: Feature analysis dictionary
        
        Returns:
            Detected emotion type
        """
        smile_intensity = features.get("smile_intensity", 0.0)
        mouth_open = features.get("mouth_open", False)
        eyebrow_raised = features.get("eyebrow_raised", False)
        
        # Classification logic
        if smile_intensity > 0.5:
            if smile_intensity > 0.8:
                return "excited"
            elif smile_intensity > 0.6:
                return "happy"
            else:
                return "confident"
        elif mouth_open and eyebrow_raised:
            return "surprised"
        elif mouth_open:
            return "surprised"
        elif eyebrow_raised:
            return "serious"
        else:
            return "neutral"
    
    def _get_emotion_data(self, emotion_type: str, features: Optional[Dict] = None) -> Dict:
        """
        Get detailed emotion data for Mickmumpitz workflow
        
        Args:
            emotion_type: Emotion type string
            features: Optional feature analysis
        
        Returns:
            Complete emotion data dictionary
        """
        emotion_type = emotion_type.lower()
        
        # Get base emotion data
        if emotion_type not in self.EMOTION_MAP:
            emotion_type = "neutral"
        
        base_data = self.EMOTION_MAP[emotion_type].copy()
        
        # Build emotion data
        emotion_data = {
            "type": emotion_type,
            "keywords": base_data["keywords"],
            "intensity": base_data["intensity"],
            "descriptors": base_data["descriptors"],
            "features": features or {}
        }
        
        return emotion_data
    
    def generate_emotion_prompt(
        self,
        emotion_data: Dict,
        base_prompt: str = "",
        intensity_override: Optional[float] = None
    ) -> str:
        """
        Generate Mickmumpitz-style emotion prompt
        
        Args:
            emotion_data: Emotion data dictionary
            base_prompt: Base prompt to enhance
            intensity_override: Optional intensity override
        
        Returns:
            Enhanced prompt with emotion control
        """
        emotion_type = emotion_data.get("type", "neutral")
        keywords = emotion_data.get("keywords", [])
        descriptors = emotion_data.get("descriptors", [])
        intensity = intensity_override or emotion_data.get("intensity", 0.5)
        
        # Build emotion-specific prompt components
        emotion_parts = []
        
        # Add primary emotion keyword
        if keywords:
            emotion_parts.append(keywords[0])
        
        # Add descriptors based on intensity
        if intensity > 0.7:
            # High intensity: use stronger descriptors
            emotion_parts.extend(descriptors[:3])
        elif intensity > 0.5:
            # Medium intensity: use moderate descriptors
            emotion_parts.extend(descriptors[:2])
        else:
            # Low intensity: use subtle descriptors
            if descriptors:
                emotion_parts.append(descriptors[0])
        
        # Combine emotion components
        emotion_text = ", ".join(emotion_parts)
        
        # Build final prompt
        if base_prompt:
            # Insert emotion into base prompt
            if "{emotion}" in base_prompt:
                final_prompt = base_prompt.replace("{emotion}", emotion_text)
            else:
                final_prompt = f"{base_prompt}, {emotion_text}"
        else:
            final_prompt = emotion_text
        
        return final_prompt
    
    def enhance_face_prompt_with_emotion(
        self,
        base_face_prompt: str,
        emotion_data: Dict
    ) -> str:
        """
        Enhance face refinement prompt with emotion control
        
        Args:
            base_face_prompt: Base face prompt
            emotion_data: Emotion data
        
        Returns:
            Enhanced prompt with emotion
        """
        emotion_prompt = self.generate_emotion_prompt(emotion_data)
        
        # Insert emotion into face prompt
        enhanced = f"{base_face_prompt}, {emotion_prompt} expression, natural facial expression"
        
        return enhanced
    
    def get_emotion_negative_prompt(self, emotion_data: Dict) -> str:
        """
        Get emotion-specific negative prompt
        
        Args:
            emotion_data: Emotion data
        
        Returns:
            Negative prompt excluding opposite emotions
        """
        emotion_type = emotion_data.get("type", "neutral")
        
        # Define opposite emotions to exclude
        opposite_map = {
            "happy": ["sad", "angry", "fearful"],
            "sad": ["happy", "excited", "playful"],
            "angry": ["happy", "playful", "romantic"],
            "surprised": ["neutral", "serious"],
            "excited": ["sad", "serious", "neutral"],
            "confident": ["fearful", "sad"],
            "neutral": ["excited", "angry"]
        }
        
        opposites = opposite_map.get(emotion_type, [])
        
        if opposites:
            negative_parts = [f"no {opp}" for opp in opposites]
            return ", ".join(negative_parts)
        
        return ""
    
    def merge_emotions(
        self,
        customer_emotion: Dict,
        template_emotion: Dict,
        blend_factor: float = 0.7
    ) -> Dict:
        """
        Merge customer and template emotions
        
        Args:
            customer_emotion: Customer emotion data
            template_emotion: Template emotion data
            blend_factor: How much to favor template (0-1)
        
        Returns:
            Merged emotion data
        """
        # Prefer template emotion but consider customer
        if blend_factor > 0.5:
            primary_emotion = template_emotion
            secondary_emotion = customer_emotion
        else:
            primary_emotion = customer_emotion
            secondary_emotion = template_emotion
        
        # Blend intensity
        primary_intensity = primary_emotion.get("intensity", 0.5)
        secondary_intensity = secondary_emotion.get("intensity", 0.5)
        blended_intensity = (
            primary_intensity * blend_factor +
            secondary_intensity * (1 - blend_factor)
        )
        
        # Use primary emotion type but adjust intensity
        merged = primary_emotion.copy()
        merged["intensity"] = blended_intensity
        
        return merged

