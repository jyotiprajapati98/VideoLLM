import cv2
import os
import numpy as np
from typing import List, Tuple
import ffmpeg

def extract_frames(video_path: str, output_dir: str, sampling_rate: int = 2) -> List[str]:
    """
    Extract frames from video at specified sampling rate
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    frame_paths = []
    frame_interval = int(fps * sampling_rate)
    
    frame_num = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_num % frame_interval == 0:
            timestamp = frame_num / fps
            frame_filename = f"frame_{saved_frame_count:04d}_t{timestamp:.2f}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_frame_count += 1
            
        frame_num += 1
    
    cap.release()
    return frame_paths

def get_video_info(video_path: str) -> dict:
    """
    Get video metadata
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height
    }

def cleanup_frames(frame_paths: List[str]):
    """
    Clean up extracted frame files
    """
    for frame_path in frame_paths:
        if os.path.exists(frame_path):
            os.remove(frame_path)