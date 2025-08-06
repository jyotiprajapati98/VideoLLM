"""
Robust Video Processing Utilities with Enhanced Error Handling and OpenCV Integration
"""

import cv2
import os
import numpy as np
import logging
import psutil
import time
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from PIL import Image
import ffmpeg
from moviepy.editor import VideoFileClip
from skimage import filters, exposure, measure
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Comprehensive video metadata"""
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int
    bitrate: Optional[int]
    codec: Optional[str]
    format: Optional[str]
    file_size: int
    is_valid: bool
    quality_score: float
    has_audio: bool
    audio_duration: Optional[float]

@dataclass
class FrameQuality:
    """Frame quality assessment"""
    blur_score: float
    brightness: float
    contrast: float
    noise_level: float
    sharpness: float
    is_usable: bool

class RobustVideoProcessor:
    """Enhanced video processor with comprehensive error handling and OpenCV integration"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 quality_threshold: float = 0.3,
                 memory_limit_gb: float = 8.0):
        self.max_retries = max_retries
        self.quality_threshold = quality_threshold
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        
        # Initialize supported formats
        self.supported_formats = {
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'],
            'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        }
        
        # Frame extraction methods (in order of preference)
        self.extraction_methods = [
            self._extract_frames_opencv,
            self._extract_frames_ffmpeg,
            self._extract_frames_moviepy
        ]
        
        logger.info("RobustVideoProcessor initialized successfully")
    
    def validate_video_file(self, video_path: str) -> Dict[str, Any]:
        """Comprehensive video file validation"""
        try:
            if not os.path.exists(video_path):
                return {"valid": False, "error": "File does not exist"}
            
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                return {"valid": False, "error": "File is empty"}
            
            # Check file extension
            file_ext = Path(video_path).suffix.lower()
            if file_ext not in self.supported_formats['video']:
                return {"valid": False, "error": f"Unsupported format: {file_ext}"}
            
            # Try to get basic video info
            metadata = self.get_comprehensive_video_info(video_path)
            if not metadata.is_valid:
                return {"valid": False, "error": "Cannot read video metadata"}
            
            # Check duration limits (max 10 minutes for robustness)
            if metadata.duration > 600:
                return {"valid": False, "error": "Video too long (max 10 minutes)"}
            
            # Check resolution limits
            if metadata.width * metadata.height > 4096 * 2160:  # 4K limit
                return {"valid": False, "error": "Resolution too high (max 4K)"}
            
            return {
                "valid": True,
                "metadata": metadata,
                "file_size": file_size,
                "estimated_memory_usage": self._estimate_memory_usage(metadata)
            }
            
        except Exception as e:
            logger.error(f"Video validation failed: {e}")
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def get_comprehensive_video_info(self, video_path: str) -> VideoMetadata:
        """Get comprehensive video metadata with multiple fallback methods"""
        
        # Method 1: OpenCV (fastest)
        try:
            return self._get_video_info_opencv(video_path)
        except Exception as e:
            logger.warning(f"OpenCV metadata extraction failed: {e}")
        
        # Method 2: FFmpeg probe
        try:
            return self._get_video_info_ffmpeg(video_path)
        except Exception as e:
            logger.warning(f"FFmpeg metadata extraction failed: {e}")
        
        # Method 3: MoviePy (slowest but most compatible)
        try:
            return self._get_video_info_moviepy(video_path)
        except Exception as e:
            logger.error(f"All metadata extraction methods failed: {e}")
        
        # Return invalid metadata if all methods fail
        return VideoMetadata(
            fps=0, frame_count=0, duration=0, width=0, height=0,
            bitrate=None, codec=None, format=None, file_size=0,
            is_valid=False, quality_score=0.0, has_audio=False,
            audio_duration=None
        )
    
    def _get_video_info_opencv(self, video_path: str) -> VideoMetadata:
        """Get video info using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video with OpenCV")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Try to get codec info
        codec_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = chr(codec_fourcc & 0xFF) + chr((codec_fourcc >> 8) & 0xFF) + \
                chr((codec_fourcc >> 16) & 0xFF) + chr((codec_fourcc >> 24) & 0xFF)
        
        cap.release()
        
        file_size = os.path.getsize(video_path)
        quality_score = self._calculate_quality_score(width, height, fps, duration)
        
        return VideoMetadata(
            fps=fps, frame_count=frame_count, duration=duration,
            width=width, height=height, bitrate=None, codec=codec,
            format=Path(video_path).suffix, file_size=file_size,
            is_valid=True, quality_score=quality_score,
            has_audio=False, audio_duration=None
        )
    
    def _get_video_info_ffmpeg(self, video_path: str) -> VideoMetadata:
        """Get video info using FFmpeg probe"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'audio'), None)
            
            if not video_stream:
                raise Exception("No video stream found")
            
            fps = eval(video_stream['r_frame_rate'])
            duration = float(video_stream.get('duration', 0))
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            codec = video_stream.get('codec_name')
            bitrate = video_stream.get('bit_rate')
            
            frame_count = int(duration * fps) if duration > 0 else 0
            file_size = os.path.getsize(video_path)
            quality_score = self._calculate_quality_score(width, height, fps, duration)
            
            return VideoMetadata(
                fps=fps, frame_count=frame_count, duration=duration,
                width=width, height=height, 
                bitrate=int(bitrate) if bitrate else None,
                codec=codec, format=Path(video_path).suffix,
                file_size=file_size, is_valid=True,
                quality_score=quality_score,
                has_audio=audio_stream is not None,
                audio_duration=float(audio_stream.get('duration', 0)) if audio_stream else None
            )
            
        except Exception as e:
            raise Exception(f"FFmpeg probe failed: {e}")
    
    def _get_video_info_moviepy(self, video_path: str) -> VideoMetadata:
        """Get video info using MoviePy"""
        try:
            with VideoFileClip(video_path) as clip:
                fps = clip.fps
                duration = clip.duration
                width, height = clip.size
                frame_count = int(duration * fps)
                
                file_size = os.path.getsize(video_path)
                quality_score = self._calculate_quality_score(width, height, fps, duration)
                
                return VideoMetadata(
                    fps=fps, frame_count=frame_count, duration=duration,
                    width=width, height=height, bitrate=None,
                    codec=None, format=Path(video_path).suffix,
                    file_size=file_size, is_valid=True,
                    quality_score=quality_score,
                    has_audio=clip.audio is not None,
                    audio_duration=clip.audio.duration if clip.audio else None
                )
        except Exception as e:
            raise Exception(f"MoviePy extraction failed: {e}")
    
    def _calculate_quality_score(self, width: int, height: int, fps: float, duration: float) -> float:
        """Calculate video quality score (0-1)"""
        try:
            # Resolution score (higher resolution = better, max at 1080p)
            resolution_score = min(1.0, (width * height) / (1920 * 1080))
            
            # FPS score (optimal around 24-30 fps)
            fps_score = min(1.0, fps / 30.0) if fps <= 30 else max(0.5, 1.0 - (fps - 30) / 60)
            
            # Duration score (penalize very short or very long videos)
            if duration < 1:
                duration_score = duration
            elif duration <= 120:  # Optimal range 1-120 seconds
                duration_score = 1.0
            else:
                duration_score = max(0.3, 1.0 - (duration - 120) / 480)  # Decay after 2 minutes
            
            return (resolution_score * 0.4 + fps_score * 0.3 + duration_score * 0.3)
        except:
            return 0.5  # Default score if calculation fails
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def extract_frames_robust(self, 
                            video_path: str, 
                            output_dir: str, 
                            sampling_rate: float = 2.0,
                            max_frames: int = 20,
                            quality_filter: bool = True) -> List[str]:
        """
        Robust frame extraction with multiple fallback methods and quality filtering
        """
        
        # Validate video first
        validation = self.validate_video_file(video_path)
        if not validation["valid"]:
            raise Exception(f"Video validation failed: {validation['error']}")
        
        # Check memory usage
        if validation["estimated_memory_usage"] > self.memory_limit_bytes:
            raise Exception("Video requires too much memory to process safely")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Try each extraction method
        for i, extraction_method in enumerate(self.extraction_methods):
            try:
                logger.info(f"Attempting frame extraction method {i+1}/{len(self.extraction_methods)}")
                
                frame_paths = extraction_method(
                    video_path, output_dir, sampling_rate, max_frames
                )
                
                if quality_filter:
                    frame_paths = self._filter_frames_by_quality(frame_paths)
                
                if frame_paths:
                    logger.info(f"Successfully extracted {len(frame_paths)} frames")
                    return frame_paths
                else:
                    logger.warning("No usable frames extracted")
                    
            except Exception as e:
                logger.error(f"Frame extraction method {i+1} failed: {e}")
                if i == len(self.extraction_methods) - 1:  # Last method
                    raise Exception(f"All frame extraction methods failed. Last error: {e}")
                continue
        
        raise Exception("All frame extraction methods failed")
    
    def _extract_frames_opencv(self, video_path: str, output_dir: str, 
                              sampling_rate: float, max_frames: int) -> List[str]:
        """Extract frames using OpenCV with enhanced error handling"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video with OpenCV")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25  # Default fallback
            
            frame_interval = int(fps * sampling_rate)
            frame_paths = []
            frame_num = 0
            saved_frame_count = 0
            
            while saved_frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_num % frame_interval == 0:
                    # Validate frame
                    if frame is None or frame.size == 0:
                        logger.warning(f"Skipping invalid frame at position {frame_num}")
                        frame_num += 1
                        continue
                    
                    # Enhance frame if needed
                    enhanced_frame = self._enhance_frame(frame)
                    
                    timestamp = frame_num / fps
                    frame_filename = f"frame_{saved_frame_count:04d}_t{timestamp:.2f}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    # Save with error handling
                    success = cv2.imwrite(frame_path, enhanced_frame, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if success and os.path.exists(frame_path):
                        frame_paths.append(frame_path)
                        saved_frame_count += 1
                    else:
                        logger.warning(f"Failed to save frame {frame_filename}")
                
                frame_num += 1
                
                # Memory check
                if psutil.virtual_memory().percent > 85:
                    logger.warning("High memory usage detected, stopping frame extraction early")
                    break
            
            return frame_paths
            
        finally:
            cap.release()
    
    def _extract_frames_ffmpeg(self, video_path: str, output_dir: str, 
                              sampling_rate: float, max_frames: int) -> List[str]:
        """Extract frames using FFmpeg"""
        
        try:
            # Calculate frame extraction parameters
            frame_paths = []
            
            # Use ffmpeg to extract frames
            output_pattern = os.path.join(output_dir, "frame_%04d_t%f.jpg")
            
            (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=f'1/{sampling_rate}')
                .output(output_pattern, vframes=max_frames, q=2)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Collect generated files
            for file in os.listdir(output_dir):
                if file.startswith("frame_") and file.endswith(".jpg"):
                    frame_paths.append(os.path.join(output_dir, file))
            
            # Sort by frame number
            frame_paths.sort()
            return frame_paths[:max_frames]
            
        except Exception as e:
            raise Exception(f"FFmpeg frame extraction failed: {e}")
    
    def _extract_frames_moviepy(self, video_path: str, output_dir: str, 
                               sampling_rate: float, max_frames: int) -> List[str]:
        """Extract frames using MoviePy"""
        
        try:
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                frame_times = np.arange(0, min(duration, max_frames * sampling_rate), sampling_rate)
                
                frame_paths = []
                for i, t in enumerate(frame_times):
                    if i >= max_frames:
                        break
                    
                    frame = clip.get_frame(t)
                    frame_filename = f"frame_{i:04d}_t{t:.2f}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    # Convert to PIL and save
                    pil_image = Image.fromarray(frame)
                    pil_image.save(frame_path, quality=95)
                    frame_paths.append(frame_path)
                
                return frame_paths
                
        except Exception as e:
            raise Exception(f"MoviePy frame extraction failed: {e}")
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply basic enhancement to improve frame quality"""
        try:
            # Convert to float for processing
            frame_float = frame.astype(np.float32) / 255.0
            
            # Apply histogram equalization to improve contrast
            frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
            enhanced = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
            
            # Reduce noise with bilateral filter
            enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Frame enhancement failed: {e}")
            return frame  # Return original if enhancement fails
    
    def _filter_frames_by_quality(self, frame_paths: List[str]) -> List[str]:
        """Filter frames based on quality metrics"""
        quality_frames = []
        
        for frame_path in frame_paths:
            try:
                quality = self._assess_frame_quality(frame_path)
                if quality.is_usable:
                    quality_frames.append(frame_path)
                else:
                    logger.info(f"Filtering out low-quality frame: {frame_path}")
                    # Clean up low-quality frame
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                        
            except Exception as e:
                logger.warning(f"Quality assessment failed for {frame_path}: {e}")
                quality_frames.append(frame_path)  # Keep if assessment fails
        
        return quality_frames
    
    def _assess_frame_quality(self, frame_path: str) -> FrameQuality:
        """Assess frame quality using multiple metrics"""
        try:
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                return FrameQuality(0, 0, 0, 0, 0, False)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Blur detection using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Brightness assessment
            brightness = np.mean(gray)
            
            # 3. Contrast assessment
            contrast = np.std(gray)
            
            # 4. Noise level using high-pass filter
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise_level = np.var(cv2.filter2D(gray, -1, kernel))
            
            # 5. Sharpness using gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            
            # Determine if frame is usable based on thresholds
            is_usable = (
                blur_score > 100 and  # Not too blurry
                20 < brightness < 235 and  # Not too dark or bright
                contrast > 10 and  # Has sufficient contrast
                noise_level < 1000 and  # Not too noisy
                sharpness > 10  # Sufficiently sharp
            )
            
            return FrameQuality(
                blur_score=blur_score,
                brightness=brightness,
                contrast=contrast,
                noise_level=noise_level,
                sharpness=sharpness,
                is_usable=is_usable
            )
            
        except Exception as e:
            logger.error(f"Frame quality assessment failed: {e}")
            return FrameQuality(0, 0, 0, 0, 0, False)
    
    def _estimate_memory_usage(self, metadata: VideoMetadata) -> int:
        """Estimate memory usage for video processing"""
        # Rough estimation: width * height * 3 (RGB) * max_frames
        frame_size = metadata.width * metadata.height * 3
        max_frames_in_memory = 20  # Conservative estimate
        return frame_size * max_frames_in_memory
    
    def cleanup_frames_safe(self, frame_paths: List[str]) -> Dict[str, int]:
        """Safely cleanup frame files with error handling"""
        cleaned = 0
        failed = 0
        
        for frame_path in frame_paths:
            try:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to remove frame {frame_path}: {e}")
                failed += 1
        
        return {"cleaned": cleaned, "failed": failed}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "cpu_percent": cpu_percent,
                "healthy": (
                    memory.percent < 85 and 
                    disk.free > 1024**3 and  # At least 1GB free
                    cpu_percent < 90
                )
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"healthy": False, "error": str(e)}


# Convenience functions for backward compatibility
def extract_frames(video_path: str, output_dir: str, sampling_rate: float = 2.0) -> List[str]:
    """Backward compatible frame extraction function"""
    processor = RobustVideoProcessor()
    return processor.extract_frames_robust(video_path, output_dir, sampling_rate)

def get_video_info(video_path: str) -> dict:
    """Backward compatible video info function"""
    processor = RobustVideoProcessor()
    metadata = processor.get_comprehensive_video_info(video_path)
    
    # Convert to old format for compatibility
    return {
        "fps": metadata.fps,
        "frame_count": metadata.frame_count,
        "duration": metadata.duration,
        "width": metadata.width,
        "height": metadata.height
    }

def cleanup_frames(frame_paths: List[str]):
    """Backward compatible cleanup function"""
    processor = RobustVideoProcessor()
    result = processor.cleanup_frames_safe(frame_paths)
    logger.info(f"Cleanup completed: {result['cleaned']} files removed, {result['failed']} failures")