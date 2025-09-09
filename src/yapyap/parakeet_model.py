#!/usr/bin/env python3
"""
Parakeet-based speech recognition model implementation using NVIDIA NeMo.
"""
import sys
import torch
import numpy as np
from .model import Model

class ParakeetModel(Model):
    """Parakeet-based speech recognition model using NVIDIA NeMo."""
    
    def _load_model(self):
        """Load the Parakeet model and keep it hot in memory."""
        try:
            from nemo.collections.asr.models import ASRModel
            print(f"Loading Parakeet model: {self.model_name}", file=sys.stderr)
            self._model = ASRModel.from_pretrained(model_name=self.model_name)
            self._model.eval()
            
            # Set device and move model to GPU
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device)
            self._model.to(torch.bfloat16)  # Set precision once
            
            print(f"Using device: {self.device}", file=sys.stderr)
        except ImportError:
            raise ImportError("nemo-toolkit is required for ParakeetModel. Install with: pip install nemo-toolkit[asr]")
    
    def transcribe(self, audio):
        """Transcribe audio using Parakeet."""
        if self._model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert numpy array to the format expected by Parakeet
        if isinstance(audio, np.ndarray):
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Remove extra dimensions - Parakeet expects (time,) not (time, channels)
            if len(audio.shape) > 1:
                if audio.shape[1] == 1:  # Mono audio with channel dimension
                    audio = audio.squeeze(1)  # Remove channel dimension
                else:  # Stereo audio
                    audio = np.mean(audio, axis=1)  # Convert to mono by averaging channels
            
            # Normalize audio to [-1, 1] range if needed
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / np.max(np.abs(audio))
        
        # Transcribe audio
        output = self._model.transcribe([audio])
        
        if not output or not isinstance(output, list) or not output[0]:
            raise RuntimeError("Transcription failed or produced unexpected output format")
        
        # Get the text result
        text = output[0].text if hasattr(output[0], 'text') else str(output[0])
        duration_sec = len(audio) / 16000
        
        return [Segment(text, 0.0, duration_sec)]


class Segment:
    """Simple segment class to match the expected interface."""
    
    def __init__(self, text: str, start: float, end: float):
        self.text = text
        self.start = start
        self.end = end
