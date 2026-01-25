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
class Segment:
    """A segment of transcription with timestamps"""
    start_seconds: float
    end_seconds: float
    text: str

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

@dataclass(slots=True)
class ChunkInfo:
    chunk_index: int
    own_keep_start_sample: int
    own_keep_end_sample: int  # 半开 [start, end)
    vad_limit_sample: int
    keep_windows_sample: list[tuple[int, int]]  # list of 半开窗口

@dataclass(slots=True)
class SubwordInfo:
    token_id: int
    token: str
    start_sample: int
    end_sample: int
    step_index: int
    chunk_index: int
    vad_limit_sample: int

@dataclass(slots=True)
class SegmentInfo:
    start_sample: int
    end_sample: int
    text: str
    subwords: list[SubwordInfo]
    chunk_index: int
    vad_limit_sample: int

@dataclass(slots=True)
class PreciseSubword:
    seconds: float
    end_seconds: float
    token_id: int
    token: str

@dataclass(slots=True)
class PreciseSegment:
    start_seconds: float
    end_seconds: float
    text: str
    subwords: list[PreciseSubword]