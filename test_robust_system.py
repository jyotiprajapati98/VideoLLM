#!/usr/bin/env python3
"""
Test script to verify robustness improvements in VideoLLM
"""

import os
import sys
import tempfile
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.robust_video_utils import RobustVideoProcessor
from src.models.video_processor import VideoProcessor
from src.models.chat_manager import ChatManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_robust_video_processor():
    """Test the robust video processor"""
    logger.info("Testing RobustVideoProcessor...")
    
    processor = RobustVideoProcessor()
    
    # Test system health check
    health = processor.get_system_health()
    logger.info(f"System health: {health}")
    
    # Test with a non-existent file
    validation = processor.validate_video_file("non_existent.mp4")
    logger.info(f"Non-existent file validation: {validation}")
    
    # Test with existing video files if any
    video_dirs = ["uploads", "temp"]
    for video_dir in video_dirs:
        if os.path.exists(video_dir):
            for root, dirs, files in os.walk(video_dir):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_path = os.path.join(root, file)
                        logger.info(f"Testing video: {video_path}")
                        
                        validation = processor.validate_video_file(video_path)
                        logger.info(f"Validation result: {validation['valid']}")
                        
                        if validation['valid']:
                            metadata = processor.get_comprehensive_video_info(video_path)
                            logger.info(f"Video metadata: duration={metadata.duration:.2f}s, quality={metadata.quality_score:.2f}")
                        
                        return  # Test only first video found
    
    logger.info("RobustVideoProcessor tests completed")

def test_video_processor_health():
    """Test VideoProcessor health monitoring"""
    logger.info("Testing VideoProcessor health monitoring...")
    
    try:
        processor = VideoProcessor()
        
        # Get health status
        health = processor.get_health_status()
        logger.info(f"VideoProcessor health: {health}")
        
        # Test stats reset
        processor.reset_stats()
        logger.info("Stats reset completed")
        
        logger.info("VideoProcessor health tests completed")
        
    except Exception as e:
        logger.error(f"VideoProcessor health test failed: {e}")

def test_chat_manager_health():
    """Test ChatManager health monitoring"""
    logger.info("Testing ChatManager health monitoring...")
    
    try:
        chat_manager = ChatManager()
        
        # Get health status
        health = chat_manager.get_health_status()
        logger.info(f"ChatManager health: {health}")
        
        # Test session clearing
        test_session = "test_session_123"
        result = chat_manager.clear_session(test_session)
        logger.info(f"Session clear test: {result}")
        
        logger.info("ChatManager health tests completed")
        
    except Exception as e:
        logger.error(f"ChatManager health test failed: {e}")

def test_memory_usage():
    """Test memory monitoring"""
    logger.info("Testing memory monitoring...")
    
    import psutil
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Test system memory
    system_memory = psutil.virtual_memory()
    logger.info(f"System memory: {system_memory.percent}% used, {system_memory.available / 1024**3:.1f}GB available")
    
    logger.info("Memory monitoring tests completed")

def test_error_scenarios():
    """Test various error scenarios"""
    logger.info("Testing error scenarios...")
    
    # Test with invalid video path
    processor = RobustVideoProcessor()
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # This should fail gracefully
            frames = processor.extract_frames_robust("invalid_path.mp4", temp_dir)
            logger.warning("Expected this to fail!")
    except Exception as e:
        logger.info(f"Graceful failure for invalid path: {type(e).__name__}")
    
    # Test with invalid session
    try:
        chat_manager = ChatManager()
        response = chat_manager.generate_response("", "")  # Empty inputs
        logger.info(f"Empty input handling: {len(response)} chars")
    except Exception as e:
        logger.info(f"Graceful failure for empty inputs: {type(e).__name__}")
    
    logger.info("Error scenario tests completed")

def main():
    """Run all robustness tests"""
    logger.info("=== Starting VideoLLM Robustness Tests ===")
    
    tests = [
        test_memory_usage,
        test_robust_video_processor,
        test_video_processor_health,
        test_chat_manager_health,
        test_error_scenarios
    ]
    
    for test_func in tests:
        try:
            logger.info(f"\n--- Running {test_func.__name__} ---")
            start_time = time.time()
            test_func()
            duration = time.time() - start_time
            logger.info(f"Test completed in {duration:.2f} seconds")
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n=== All tests completed ===")
    
    # Final health summary
    logger.info("\n--- Final System Summary ---")
    try:
        processor = RobustVideoProcessor()
        health = processor.get_system_health()
        logger.info(f"System is {'healthy' if health.get('healthy') else 'unhealthy'}")
        logger.info(f"Memory: {health.get('memory_percent', 0):.1f}%")
        logger.info(f"CPU: {health.get('cpu_percent', 0):.1f}%")
    except Exception as e:
        logger.error(f"Final health check failed: {e}")

if __name__ == "__main__":
    main()