#!/usr/bin/env python3
"""
Whisper-based speech recognition model implementation.
"""
import sys
from .model import Model


class WhisperModel(Model):
    """Whisper-based speech recognition model using pywhispercpp."""
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            from pywhispercpp.model import Model as WhisperCppModel
            print(f"Loading Whisper model: {self.model_name}", file=sys.stderr)
            self._model = WhisperCppModel(self.model_name, language=self.language)
        except ImportError:
            raise ImportError("pywhispercpp is required for WhisperModel. Install with: pip install pywhispercpp")
    
    def transcribe(self, audio):
        """Transcribe audio using Whisper."""
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model.transcribe(audio)
