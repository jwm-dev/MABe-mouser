"""
FastAPI backend for MABe Viewer.
Serves parquet files from the dataset directory.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
from pathlib import Path
import json
import time
import asyncio
import traceback
from typing import Optional, Dict, Any
from functools import lru_cache

from backend.models import FileInfo, FileResponse, FrameData, MouseData, FileMetadata

app = FastAPI(
    title="MABe Viewer API",
    description="Backend for mouse behavior tracking visualization",
    version="1.0.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Point to data directory
BASE_DIR = Path(__file__).parent.parent / "MABe-mouse-behavior-detection"
DATA_DIR = BASE_DIR / "train_tracking"
ANNOTATION_DIR = BASE_DIR / "train_annotation"

if not DATA_DIR.exists():
    print(f"⚠️  Warning: Data directory not found: {DATA_DIR}")
    print(f"   Current directory: {Path.cwd()}")


# Load annotations for a specific video
@lru_cache(maxsize=100)
def load_annotations(video_id: str, lab: str) -> list:
    """Load behavior annotations for a video if they exist"""
    annotation_path = ANNOTATION_DIR / lab / f"{video_id}.parquet"
    
    if not annotation_path.exists():
        return []
    
    try:
        df = pd.read_parquet(annotation_path)
        annotations = []
        
        for _, row in df.iterrows():
            annotations.append({
                'agent_id': int(row['agent_id']),
                'target_id': int(row['target_id']),
                'action': str(row['action']),
                'start_frame': int(row['start_frame']),
                'stop_frame': int(row['stop_frame'])
            })
        
        return annotations
    except Exception as e:
        print(f"⚠️  Error loading annotations for {video_id}: {e}")
        return []


# Load metadata CSV files
@lru_cache(maxsize=10)
def load_csv_metadata(csv_name: str) -> Dict[str, Dict[str, Any]]:
    """Load metadata from train.csv or test.csv, indexed by video_id"""
    csv_path = BASE_DIR / csv_name
    if not csv_path.exists():
        return {}
    
    df = pd.read_csv(csv_path)
    metadata = {}
    
    for _, row in df.iterrows():
        video_id = str(row['video_id'])
        metadata[video_id] = {
            'lab_id': row.get('lab_id'),
            'fps': float(row['frames_per_second']) if pd.notna(row.get('frames_per_second')) else 30.0,
            'video_width': int(row['video_width_pix']) if pd.notna(row.get('video_width_pix')) else None,
            'video_height': int(row['video_height_pix']) if pd.notna(row.get('video_height_pix')) else None,
            'arena_width_cm': float(row['arena_width_cm']) if pd.notna(row.get('arena_width_cm')) else None,
            'arena_height_cm': float(row['arena_height_cm']) if pd.notna(row.get('arena_height_cm')) else None,
            'pix_per_cm': float(row['pix_per_cm_approx']) if pd.notna(row.get('pix_per_cm_approx')) else None,
            'body_parts': eval(row['body_parts_tracked']) if pd.notna(row.get('body_parts_tracked')) else None
        }
    
    return metadata


# Cache for file metadata (total frames, etc.)
@lru_cache(maxsize=100)
def get_file_metadata(file_path: str, lab: str = "train_tracking") -> dict:
    """Get metadata about a parquet file (cached)"""
    path = Path(file_path)
    if not path.exists():
        return {"error": "File not found"}
    
    # Extract video_id from filename (without .parquet extension)
    video_id = path.stem
    
    # Determine which CSV to use based on file location
    # Check if file is in train_tracking or test_tracking directory
    path_str = str(path)
    csv_name = "train.csv" if "/train_tracking/" in path_str else "test.csv"
    
    csv_metadata = load_csv_metadata(csv_name)
    video_meta = csv_metadata.get(video_id, {})
    
    print(f"🔍 get_file_metadata: video_id={video_id}, csv_name={csv_name}, found_in_csv={video_id in csv_metadata}")
    if video_meta:
        print(f"   video_width={video_meta.get('video_width')}, video_height={video_meta.get('video_height')}")
    
    # Read only metadata columns for efficiency
    df_meta = pd.read_parquet(path, columns=['video_frame'])
    total_frames = df_meta['video_frame'].nunique()
    min_frame = int(df_meta['video_frame'].min())
    max_frame = int(df_meta['video_frame'].max())
    
    # Check if mouse_id exists by reading a sample
    sample_df = pd.read_parquet(path, filters=[('video_frame', '==', df_meta['video_frame'].iloc[0])])
    has_mouse_id = 'mouse_id' in sample_df.columns
    num_mice = sample_df['mouse_id'].nunique() if has_mouse_id else 1
    
    # Load annotations if they exist
    annotations = load_annotations(video_id, lab)
    
    return {
        "total_frames": int(total_frames),
        "min_frame": min_frame,
        "max_frame": max_frame,
        "num_mice": int(num_mice),
        "has_mouse_id": has_mouse_id,
        "fps": video_meta.get('fps', 30.0),
        "video_width": video_meta.get('video_width'),
        "video_height": video_meta.get('video_height'),
        "arena_width_cm": video_meta.get('arena_width_cm'),
        "arena_height_cm": video_meta.get('arena_height_cm'),
        "pix_per_cm": video_meta.get('pix_per_cm'),
        "body_parts": video_meta.get('body_parts'),
        "has_annotations": len(annotations) > 0,
        "annotations": annotations
    }


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MABe Viewer API",
        "status": "running",
        "data_dir": str(DATA_DIR),
        "data_dir_exists": DATA_DIR.exists()
    }


@app.get("/api/files", response_model=dict)
async def list_files():
    """List all available parquet files organized by lab"""
    if not DATA_DIR.exists():
        raise HTTPException(status_code=500, detail=f"Data directory not found: {DATA_DIR}")
    
    # Organize files by lab
    labs = {}
    total_files = 0
    
    for lab_dir in sorted(DATA_DIR.iterdir()):
        if not lab_dir.is_dir():
            continue
        
        lab_name = lab_dir.name
        lab_files = []
        
        for parquet_file in sorted(lab_dir.glob("*.parquet")):
            lab_files.append(FileInfo(
                name=parquet_file.name,
                lab=lab_name,
                path=str(parquet_file.relative_to(DATA_DIR)),
                size_bytes=parquet_file.stat().st_size
            ))
            total_files += 1
        
        if lab_files:
            labs[lab_name] = [f.model_dump() for f in lab_files]
    
    return {
        "labs": labs,
        "total": total_files
    }


@app.get("/api/files/{lab}/{filename}")
async def get_file(
    lab: str, 
    filename: str, 
    start_frame: int = 0,
    max_frames: int = 1000
):
    """Load a parquet file with efficient frame range filtering"""
    file_path = DATA_DIR / lab / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # Time the load for performance comparison
    start_time = time.time()
    
    try:
        # OPTIMIZATION: Use parquet filter pushdown to only read needed frames
        # This is 10-20x faster than reading all data then filtering
        end_frame = start_frame + max_frames
        
        df = pd.read_parquet(
            file_path,
            filters=[
                ('video_frame', '>=', start_frame),
                ('video_frame', '<', end_frame)
            ]
        )
        
        print(f"📊 Loaded {filename}: {len(df)} rows (frames {start_frame}-{end_frame})")
        
        # Get unique columns for mouse tracking data
        # Expected columns: video_frame, mouse_id, x, y, (possibly bodypart)
        required_cols = ["video_frame", "x", "y"]
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns. Found: {list(df.columns)}"
            )
        
        # Get total frame count efficiently (cached)
        metadata_dict = get_file_metadata(str(file_path), lab)
        total_frames = metadata_dict.get("total_frames", 0)
        
        # Determine if we have mouse_id column
        has_mouse_id = "mouse_id" in df.columns
        has_bodypart = "bodypart" in df.columns
        
        # Get frames from filtered data
        unique_frames = sorted(df["video_frame"].unique())
        
        print(f"🎬 Processing {len(unique_frames)} frames (of {total_frames} total)...")
        
        # OPTIMIZATION: Use groupby instead of iterative filtering (10x faster!)
        frames = []
        
        if has_mouse_id:
            # Group by frame and mouse_id together
            grouped = df.groupby(['video_frame', 'mouse_id'])
            
            current_frame_num = None
            current_mice = {}
            
            for (frame_num, mouse_id), group in grouped:
                # Start a new frame when frame number changes
                if frame_num != current_frame_num:
                    if current_mice:
                        frames.append(FrameData(
                            frame_number=int(current_frame_num),
                            mice=current_mice
                        ))
                    current_frame_num = frame_num
                    current_mice = {}
                
                # Extract points and labels for this mouse
                if "z" in group.columns:
                    points = group[["x", "y", "z"]].values.tolist()
                else:
                    points = group[["x", "y"]].values.tolist()
                
                if has_bodypart:
                    labels = group["bodypart"].tolist()
                else:
                    labels = [f"point_{i}" for i in range(len(points))]
                
                current_mice[str(mouse_id)] = MouseData(
                    points=points,
                    labels=labels
                )
            
            # Don't forget the last frame
            if current_mice:
                frames.append(FrameData(
                    frame_number=int(current_frame_num),
                    mice=current_mice
                ))
        else:
            # Single mouse - just group by frame
            for frame_num, group in df.groupby('video_frame'):
                if "z" in group.columns:
                    points = group[["x", "y", "z"]].values.tolist()
                else:
                    points = group[["x", "y"]].values.tolist()
                
                if has_bodypart:
                    labels = group["bodypart"].tolist()
                else:
                    labels = [f"point_{i}" for i in range(len(points))]
                
                mice = {"0": MouseData(points=points, labels=labels)}
                
                frames.append(FrameData(
                    frame_number=int(frame_num),
                    mice=mice
                ))
        
        load_time = time.time() - start_time
        
        metadata = FileMetadata(
            total_frames=total_frames,  # Report total, not just loaded
            num_mice=len(df["mouse_id"].unique()) if has_mouse_id else 1
        )
        
        response = FileResponse(
            frames=frames,
            metadata=metadata
        )
        
        print(f"✅ Loaded {filename}: {len(frames)}/{total_frames} frames in {load_time*1000:.2f}ms")
        
        return response.model_dump()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading file: {str(e)}")


@app.get("/api/files/{lab}/{filename}/stream")
async def stream_file(
    request: Request,
    lab: str,
    filename: str,
    start_frame: int = 0,
    chunk_size: int = 100
):
    """Stream parquet file in chunks for progressive loading (optimized)"""
    file_path = DATA_DIR / lab / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    async def generate():
        try:
            # Get metadata (cached)
            metadata_dict = get_file_metadata(str(file_path), lab)
            total_frames = metadata_dict.get("total_frames", 0)
            min_frame = metadata_dict.get("min_frame", 0)
            max_frame = metadata_dict.get("max_frame", total_frames - 1)
            
            print(f"📡 Starting stream for {filename}: {total_frames} frames (range: {min_frame}-{max_frame})")
            
            # Send metadata first
            yield f"data: {json.dumps({'metadata': metadata_dict})}\n\n"
            
            # Ensure we flush after metadata
            await asyncio.sleep(0)
            
            # Stream in chunks using filter pushdown
            # Start from min_frame, not 0!
            current_frame = min_frame
            chunks_sent = 0
            
            while current_frame <= max_frame:
                # Check if client disconnected
                if await request.is_disconnected():
                    print(f"🔌 Client disconnected, stopping stream at frame {current_frame}/{max_frame}")
                    break
                end_frame = min(current_frame + chunk_size, max_frame + 1)
                
                # Read chunk with filter - use pyarrow engine explicitly
                try:
                    df_chunk = pd.read_parquet(
                        file_path,
                        engine='pyarrow',
                        filters=[
                            ('video_frame', '>=', current_frame),
                            ('video_frame', '<', end_frame)
                        ]
                    )
                except Exception as filter_error:
                    # Fallback: read entire file and filter in memory
                    print(f"⚠️  Filter failed ({filter_error}), reading full file and filtering...")
                    df_full = pd.read_parquet(file_path)
                    df_chunk = df_full[
                        (df_full['video_frame'] >= current_frame) &
                        (df_full['video_frame'] < end_frame)
                    ]
                
                if len(df_chunk) == 0:
                    print(f"⚠️  No data in chunk {current_frame}-{end_frame}, breaking")
                    break
                
                has_mouse_id = "mouse_id" in df_chunk.columns
                has_bodypart = "bodypart" in df_chunk.columns
                
                # Use optimized groupby
                chunk_frames = []
                
                if has_mouse_id:
                    grouped = df_chunk.groupby(['video_frame', 'mouse_id'])
                    
                    current_frame_num = None
                    current_mice = {}
                    
                    for (frame_num, mouse_id), group in grouped:
                        if frame_num != current_frame_num:
                            if current_mice:
                                chunk_frames.append({
                                    "frame_number": int(current_frame_num),
                                    "mice": current_mice
                                })
                            current_frame_num = frame_num
                            current_mice = {}
                        
                        points = group[["x", "y"]].values.tolist()
                        labels = group["bodypart"].tolist() if has_bodypart else [f"point_{i}" for i in range(len(points))]
                        
                        current_mice[str(mouse_id)] = {
                            "points": points,
                            "labels": labels
                        }
                    
                    if current_mice:
                        chunk_frames.append({
                            "frame_number": int(current_frame_num),
                            "mice": current_mice
                        })
                else:
                    for frame_num, group in df_chunk.groupby('video_frame'):
                        points = group[["x", "y"]].values.tolist()
                        labels = group["bodypart"].tolist() if has_bodypart else [f"point_{i}" for i in range(len(points))]
                        
                        chunk_frames.append({
                            "frame_number": int(frame_num),
                            "mice": {"0": {"points": points, "labels": labels}}
                        })
                
                # Send chunk
                is_complete = current_frame + chunk_size > max_frame
                chunks_sent += 1
                
                if chunks_sent % 10 == 0 or is_complete:
                    print(f"📦 Sent chunk {chunks_sent}: frames {current_frame}-{end_frame-1} ({len(chunk_frames)} frames)")
                
                yield f"data: {json.dumps({'frames': chunk_frames, 'complete': is_complete})}\n\n"
                
                current_frame = end_frame
                
            print(f"✅ Stream complete: sent {chunks_sent} chunks, {current_frame} frames")
                
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"❌ Stream error: {e}")
            print(f"📋 Traceback:\n{error_details}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting MABe Viewer API...")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📊 Visit http://localhost:8000/docs for API documentation")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
