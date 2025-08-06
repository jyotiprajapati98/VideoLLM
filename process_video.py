#!/usr/bin/env python3
"""
Standalone video processing script for the UI
"""
import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def process_video_file(video_path):
    """Process a video file and return JSON results"""
    try:
        from models.video_processor import VideoProcessor
        
        processor = VideoProcessor()
        result = processor.process_video(video_path)
        
        # Convert to JSON-serializable format
        return json.dumps(result, default=str, indent=2)
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "video_info": {"error": "Failed to process video"},
            "events": [],
            "summary": f"Processing failed: {str(e)}",
            "processing_stats": {}
        }
        return json.dumps(error_result, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_video.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(process_video_file(video_path))