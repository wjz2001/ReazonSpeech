import numpy as np
from dataclasses import dataclass

@dataclass
class AudioData:
    """Container for audio waveform"""
    waveform: np.float32
    samplerate: int

@dataclass
class Subword:
    """A subword with timestamp"""
    # Currently Subword only has a single-point timestamp.
    # Theoretically, we should be able to compute time ranges.
    seconds: float
    token_id: int
    token: str

@dataclass
class PreciseSubword(Subword):
    """继承自 Subword，包含结束时间与 VAD 限制"""
    end_seconds: float
    vad_limit: float

@dataclass
class Segment:
    """A segment of transcription with timestamps"""
    start_seconds: float
    end_seconds: float
    text: str

@dataclass
class PreciseSegment(Segment):
    """继承自 Segment，强制要求包含 vad_limit 和 subword_indices"""
    vad_limit: float
    subwords: list[PreciseSubword]

@dataclass
class TranscribeResult:
    text: str
    subwords: list[Subword]
    segments: list[Segment]
    hypothesis: object = None

@dataclass
class TranscribeConfig:
    verbose: bool = True
    raw_hypothesis: bool = False
