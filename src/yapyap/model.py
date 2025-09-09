#!/usr/bin/env python3
"""
Model interface and factory for different speech recognition engines.
"""
from abc import ABC, abstractmethod
from typing import List, Any
import os


class Model(ABC):
    """Base class for speech recognition models."""
    
    def __init__(self, model_name: str, language: str = "auto"):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the model to load
            language: Language for transcription (default: "auto")
        """
        self.model_name = model_name
        self.language = language
        self._model = None
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the specific model implementation."""
        pass
    
    @abstractmethod
    def transcribe(self, audio: Any) -> List[Any]:
        """
        Transcribe audio data.
        
        Args:
            audio: Audio data to transcribe
            
        Returns:
            List of segments with text attribute
        """
        pass


def create_model(model_name: str = None, language: str = "auto", engine: str = "whisper") -> Model:
    """
    Factory function to create model instances.
    
    Args:
        model_name: Name of the model to load
        language: Language for transcription
        engine: Model engine to use ("whisper", "parakeet")
        
    Returns:
        Model instance
    """
    if model_name is None:
        if engine == "whisper":
            model_name = os.getenv('MODEL', 'large-v3-turbo-q8_0')
        elif engine == "parakeet":
            model_name = os.getenv('MODEL', 'nvidia/parakeet-tdt-0.6b-v3')
        else:
            model_name = os.getenv('MODEL', 'large-v3-turbo-q8_0')
    
    if engine == "whisper":
        from .whisper_model import WhisperModel
        return WhisperModel(model_name, language)
    elif engine == "parakeet":
        from .parakeet_model import ParakeetModel
        return ParakeetModel(model_name, language)
    else:
        raise ValueError(f"Unsupported model engine: {engine}. Supported engines: 'whisper', 'parakeet'")
