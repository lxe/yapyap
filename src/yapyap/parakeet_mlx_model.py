#!/usr/bin/env python3
"""
Parakeet MLX model implementation for Apple Silicon.
"""
import sys
import numpy as np
from .model import Model

try:
    from parakeet_mlx import from_pretrained
except ImportError:
    from_pretrained = None


class ParakeetMLXModel(Model):
    """Parakeet model using MLX for Apple Silicon."""
    
    def _load_model(self):
        """Load the Parakeet MLX model."""
        if from_pretrained is None:
            raise ImportError(
                "parakeet-mlx is required for ParakeetMLXModel. "
                "Install with: uv add parakeet-mlx"
            )
        
        try:
            print(f"Loading Parakeet MLX model: {self.model_name}", file=sys.stderr)
            self._model = from_pretrained(self.model_name)
            print("Parakeet MLX model loaded successfully", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load Parakeet MLX model: {e}", file=sys.stderr)
            raise
    
    def transcribe(self, audio):
        """Transcribe audio using Parakeet MLX."""
        if self._model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert numpy array to the format expected by parakeet-mlx
            if isinstance(audio, np.ndarray):
                # Ensure audio is 1D
                if audio.ndim > 1:
                    audio = audio.flatten()
                
                # parakeet-mlx expects float32 audio data
                audio = audio.astype(np.float32)
                
                # Save audio to temporary file since parakeet-mlx expects file path
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    # Write audio to temporary file
                    sf.write(temp_file.name, audio, 16000)  # Assume 16kHz sample rate
                    temp_path = temp_file.name
                
                try:
                    # Transcribe using parakeet-mlx with file path
                    result = self._model.transcribe(temp_path)
                finally:
                    # Clean up temporary file
                    import os
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            else:
                # If it's already a file path, use it directly
                result = self._model.transcribe(audio)
            
            # Convert result to segments format expected by yapyap
            segments = []
            if hasattr(result, 'sentences') and result.sentences:
                # Use sentence-level timestamps if available
                for sentence in result.sentences:
                    segments.append(Segment(
                        text=sentence.text,
                        start=sentence.start,
                        end=sentence.end
                    ))
            else:
                # Fallback to full text without timestamps
                text = result.text if hasattr(result, 'text') else str(result)
                duration = len(audio) / 16000.0 if isinstance(audio, np.ndarray) else 0.0
                segments.append(Segment(text, 0.0, duration))
            
            return segments
            
        except Exception as e:
            print(f"Transcription error: {e}", file=sys.stderr)
            raise


class Segment:
    """Simple segment class compatible with other model implementations."""
    
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return f"Segment(text='{self.text}', start={self.start}, end={self.end})"
