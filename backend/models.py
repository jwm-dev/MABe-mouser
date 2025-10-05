"""
Pydantic models for FastAPI backend.
Converted from src/core.py MouseGroup and FramePayload.
"""
from pydantic import BaseModel
from typing import Dict, List, Optional


class MouseData(BaseModel):
    """Data for a single mouse in a frame"""
    points: List[List[float]]  # [[x, y], [x, y], ...]
    labels: List[str]  # ["nose", "left_ear", ...]


class BehaviorAnnotation(BaseModel):
    """A single behavior annotation"""
    agent_id: int
    target_id: int
    action: str
    start_frame: int
    stop_frame: int


class FrameData(BaseModel):
    """Data for a single frame"""
    frame_number: int
    mice: Dict[str, MouseData]  # {"0": MouseData, "1": MouseData, ...}


class FileMetadata(BaseModel):
    """Metadata about a loaded file"""
    total_frames: int
    num_mice: int
    fps: Optional[float] = 30.0
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    arena_width_cm: Optional[float] = None
    arena_height_cm: Optional[float] = None
    pix_per_cm: Optional[float] = None
    body_parts: Optional[List[str]] = None
    has_annotations: bool = False
    annotations: Optional[List[BehaviorAnnotation]] = None


class FileInfo(BaseModel):
    """Information about an available file"""
    name: str
    lab: str
    path: str
    size_bytes: Optional[int] = None


class FileResponse(BaseModel):
    """Response for file load request"""
    frames: List[FrameData]
    metadata: FileMetadata
