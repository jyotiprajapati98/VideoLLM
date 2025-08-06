import os
import torch
import gc
import logging
import time
import psutil
from transformers import AutoProcessor, AutoModelForVision2Seq
# Note: AutoModelForVision2Seq is deprecated, but kept for compatibility with SmolVLM models
from PIL import Image
import json
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from ..utils.video_utils import extract_frames, get_video_info, cleanup_frames
from ..utils.robust_video_utils import RobustVideoProcessor
from ..utils.traffic_analyzer import TrafficAnalyzer
from ..utils.advanced_video_analyzer import AdvancedVideoAnalyzer
import traceback
from contextlib import contextmanager
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
model_load_attempts = Counter('model_load_attempts_total', 'Total model load attempts', ['model_name', 'status'])
frame_processing_time = Histogram('frame_processing_seconds', 'Frame processing time')
video_processing_time = Histogram('video_processing_seconds', 'Video processing time')
memory_usage = Gauge('memory_usage_bytes', 'Current memory usage')
model_health = Gauge('model_health_status', 'Model health status (1=healthy, 0=unhealthy)')

class VideoProcessor:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-500M-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.traffic_analyzer = TrafficAnalyzer()  # Initialize advanced traffic analyzer
        self.robust_video_processor = RobustVideoProcessor()  # Enhanced video processing
        self.advanced_analyzer = AdvancedVideoAnalyzer()  # Advanced Q&A analyzer
        
        # Enhanced fallback strategy with more models
        self.fallback_models = [
            "HuggingFaceTB/SmolVLM-500M-Instruct",
            "HuggingFaceTB/SmolVLM-256M-Instruct", 
            "HuggingFaceTB/SmolVLM2-1.7B-Instruct",
            "microsoft/Florence-2-base-ft",
            "Salesforce/blip2-opt-2.7b",
            "microsoft/git-base-vqav2",
            "nlpconnect/vit-gpt2-image-captioning"
        ]
        
        # Model health tracking
        self.model_failures = 0
        self.max_failures = 3
        self.last_health_check = time.time()
        self.health_check_interval = 300  # 5 minutes
        self.is_healthy = False
        
        # Performance tracking
        self.processing_stats = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'total_frames': 0,
            'avg_processing_time': 0
        }
        
        self._load_model_with_recovery()
    
    def _load_model_with_recovery(self):
        """Load model with comprehensive error handling and recovery"""
        models_to_try = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
        
        for attempt, model_name in enumerate(models_to_try):
            try:
                logger.info(f"Attempt {attempt + 1}/{len(models_to_try)}: Loading {model_name} on {self.device}")
                model_load_attempts.labels(model_name=model_name, status='attempt').inc()
                
                # Clear previous models and free memory
                self._cleanup_models()
                
                # Check available memory
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 85:
                    logger.warning(f"High memory usage ({memory_info.percent}%), attempting cleanup")
                    self._force_memory_cleanup()
                
                # Load processor with timeout
                with self._timeout_context(60):  # 60 second timeout
                    self.processor = AutoProcessor.from_pretrained(
                        model_name,
                        cache_dir="./model_cache",
                        local_files_only=False
                    )
                
                # Load model with optimizations
                with self._timeout_context(180):  # 3 minute timeout
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True,
                        cache_dir="./model_cache",
                        local_files_only=False,
                        low_cpu_mem_usage=True
                    )
                
                # Verify model is working
                if self._test_model_inference():
                    self.model_name = model_name
                    self.model_failures = 0
                    self.is_healthy = True
                    model_health.set(1)
                    model_load_attempts.labels(model_name=model_name, status='success').inc()
                    logger.info(f"✅ Model loaded and verified successfully: {model_name}")
                    return
                else:
                    raise Exception("Model inference test failed")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}")
                model_load_attempts.labels(model_name=model_name, status='failure').inc()
                self._cleanup_models()
                
                if attempt < len(models_to_try) - 1:
                    logger.info(f"Retrying with next model...")
                    time.sleep(2)  # Brief pause before retry
                else:
                    # Last attempt failed
                    self.is_healthy = False
                    model_health.set(0)
                    raise Exception(f"❌ Failed to load any vision model after {len(models_to_try)} attempts. Last error: {e}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def analyze_frame(self, image_path: str, prompt: str) -> str:
        """Analyze a single frame with comprehensive error handling"""
        start_time = time.time()
        
        try:
            # Health check
            if not self.is_healthy:
                self._attempt_model_recovery()
            
            # Validate image file
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load and validate image
            try:
                image = Image.open(image_path).convert("RGB")
                if image.size[0] * image.size[1] == 0:
                    raise ValueError("Invalid image dimensions")
            except Exception as e:
                raise Exception(f"Failed to load image: {e}")
            
            # Memory check before processing
            memory_info = psutil.virtual_memory()
            if memory_info.percent > 90:
                self._force_memory_cleanup()
                if psutil.virtual_memory().percent > 90:
                    raise Exception("Insufficient memory for frame analysis")
            
            # Update memory metric
            memory_usage.set(memory_info.used)
            
            # Format messages for SmolVLM chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template with error handling
            try:
                formatted_prompt = self.processor.apply_chat_template(
                    messages, 
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed, using simple prompt: {e}")
                formatted_prompt = prompt
            
            # Prepare inputs with error handling
            inputs = self.processor(
                text=formatted_prompt,
                images=[image],
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response with timeout and error handling
            with torch.no_grad():
                try:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        num_beams=1,
                        early_stopping=True,
                        timeout=30  # 30 second timeout
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error("GPU out of memory, forcing cleanup and retry")
                        self._force_memory_cleanup()
                        # Retry with smaller parameters
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            num_beams=1
                        )
                    else:
                        raise
            
            # Decode response
            response = self.processor.decode(
                generated_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Record processing time
            processing_time = time.time() - start_time
            frame_processing_time.observe(processing_time)
            
            return response.strip()
        
        except Exception as e:
            self.model_failures += 1
            processing_time = time.time() - start_time
            frame_processing_time.observe(processing_time)
            
            error_msg = f"Error analyzing frame {image_path}: {str(e)}"
            logger.error(error_msg)
            
            # Check if model needs recovery
            if self.model_failures >= self.max_failures:
                logger.warning("Model failure threshold reached, marking as unhealthy")
                self.is_healthy = False
                model_health.set(0)
            
            return f"Frame analysis failed: {str(e)}"
    
    def process_video(self, video_path: str, temp_dir: str) -> Dict[str, Any]:
        """Process entire video with robust error handling and recovery"""
        start_time = time.time()
        self.processing_stats['total_videos'] += 1
        
        try:
            # Health check and recovery
            if not self.is_healthy:
                self._attempt_model_recovery()
            
            # Comprehensive video validation using robust processor
            validation = self.robust_video_processor.validate_video_file(video_path)
            if not validation["valid"]:
                raise Exception(f"Video validation failed: {validation['error']}")
            
            # Get comprehensive video info
            video_metadata = self.robust_video_processor.get_comprehensive_video_info(video_path)
            if not video_metadata.is_valid:
                raise Exception("Cannot read video metadata")
            
            logger.info(f"Processing video: {video_metadata.duration:.2f}s, {video_metadata.fps:.2f} fps, Quality: {video_metadata.quality_score:.2f}")
            
            # Adaptive sampling based on video characteristics
            sampling_rate, max_frames = self._calculate_optimal_sampling(video_metadata)
            
            # Extract frames using robust processor with quality filtering
            try:
                frame_paths = self.robust_video_processor.extract_frames_robust(
                    video_path, temp_dir, 
                    sampling_rate=sampling_rate, 
                    max_frames=max_frames,
                    quality_filter=True
                )
            except Exception as e:
                logger.warning(f"Robust extraction failed, trying fallback: {e}")
                # Fallback to basic extraction
                frame_paths = extract_frames(video_path, temp_dir, sampling_rate=sampling_rate)
                if len(frame_paths) > max_frames:
                    step = len(frame_paths) // max_frames
                    frame_paths = frame_paths[::step][:max_frames]
            
            if not frame_paths:
                raise Exception("No frames could be extracted from video")
                
            logger.info(f"Extracted {len(frame_paths)} high-quality frames (sampling rate: {sampling_rate}s)")
            self.processing_stats['total_frames'] += len(frame_paths)
            
            # Analyze frames for events with progress tracking
            events = []
            failed_frames = 0
            max_failed_frames = len(frame_paths) // 2  # Allow up to 50% frame failures
            
            for i, frame_path in enumerate(frame_paths):
                try:
                    logger.info(f"Processing frame {i+1}/{len(frame_paths)}...")
                    
                    # Extract timestamp from filename
                    filename = os.path.basename(frame_path)
                    try:
                        timestamp = float(filename.split('_t')[1].replace('.jpg', ''))
                    except (IndexError, ValueError):
                        # Fallback timestamp calculation
                        timestamp = i * sampling_rate
                        logger.warning(f"Could not parse timestamp from filename, using calculated: {timestamp}")
                
                    # Enhanced prompt specifically designed for detailed Q&A analysis
                    event_prompt = """You are an expert video analyst specializing in detailed scene analysis for question-answering. Analyze this frame with extreme attention to detail:

1. UNEXPECTED EVENTS - Critical Detection:
   - Any SUDDEN appearances of pedestrians, people, or objects
   - Anyone RUNNING or CROSSING roads unexpectedly
   - Vehicles making SUDDEN movements (braking, swerving, accelerating)
   - Any SURPRISING or UNUSUAL activities

2. SEQUENTIAL ANALYSIS - Event Chain:
   - What happens BEFORE the main event?
   - What is the PRIMARY/MOST SIGNIFICANT event?
   - What happens IMMEDIATELY AFTER the main event?
   - How do other elements REACT to the main event?

3. CAMERA VEHICLE ANALYSIS:
   - What type of vehicle is the camera mounted on? (car/motorcycle/bicycle)
   - How does the camera vehicle BEHAVE during events?
   - Does it continue straight, swerve, brake, or change speed?
   - Any evidence of the vehicle type (handlebars, dashboard, windshield)?

4. DETAILED PARTICIPANT TRACKING:
   - Exact description of ALL people and vehicles
   - Precise colors, types, positions, and movements
   - Who is involved in the most significant event?
   - Interaction patterns between participants

5. SIGNIFICANCE RANKING:
   - What is the MOST UNEXPECTED event in this frame?
   - Rate the significance of any events (1-10)
   - Identify the main theme or story of this moment
   - Note any safety implications or violations

Be extremely specific about WHO does WHAT, WHEN, and WHERE. Focus on events that would be important for detailed video analysis questions."""
                
                    # Analyze frame with retry logic
                    event_description = self.analyze_frame(frame_path, event_prompt)
                    
                    if event_description and "error" not in event_description.lower() and len(event_description) > 10:
                        # Use advanced traffic analyzer for comprehensive analysis
                        try:
                            traffic_analysis = self.traffic_analyzer.analyze_traffic_description(
                                event_description, timestamp
                            )
                            
                            events.append({
                                "timestamp": timestamp,
                                "frame_number": i,
                                "description": event_description,
                                "frame_path": frame_path,
                                "violations_detected": [v.type for v in traffic_analysis["violations"]],
                                "violation_details": traffic_analysis["violations"],
                                "safety_score": traffic_analysis["safety_score"],
                                "vehicle_count": traffic_analysis["vehicle_count"],
                                "vehicles_detailed": traffic_analysis["vehicles"],
                                "safety_concerns": traffic_analysis["safety_concerns"],
                                "risk_level": traffic_analysis["risk_level"],
                                "context": traffic_analysis["context"],
                                "has_violations": len(traffic_analysis["violations"]) > 0
                            })
                        except Exception as analysis_error:
                            logger.warning(f"Traffic analysis failed for frame {i}: {analysis_error}")
                            # Still add basic event info
                            events.append({
                                "timestamp": timestamp,
                                "frame_number": i,
                                "description": event_description,
                                "frame_path": frame_path,
                                "violations_detected": [],
                                "violation_details": [],
                                "safety_score": 5,
                                "vehicle_count": 0,
                                "vehicles_detailed": [],
                                "safety_concerns": [],
                                "risk_level": "UNKNOWN",
                                "context": {},
                                "has_violations": False
                            })
                    else:
                        logger.warning(f"Frame {i} analysis returned insufficient data: {event_description[:100]}...")
                        failed_frames += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process frame {i}: {e}")
                    logger.debug(traceback.format_exc())
                    failed_frames += 1
                    
                    # Check if too many frames are failing
                    if failed_frames > max_failed_frames:
                        logger.error(f"Too many frame failures ({failed_frames}), aborting video processing")
                        raise Exception(f"Video processing failed: {failed_frames}/{len(frame_paths)} frames failed")
                    
                    continue
            
            # Generate comprehensive analysis using both traffic analyzer and advanced Q&A analyzer
            if events:
                # Use traffic analyzer for basic summary
                traffic_analyses = []
                for event in events:
                    analysis = {
                        "violations": event.get("violation_details", []),
                        "vehicles": event.get("vehicles_detailed", []),
                        "safety_score": event.get("safety_score", 5),
                        "context": event.get("context", {}),
                        "risk_level": event.get("risk_level", "LOW")
                    }
                    traffic_analyses.append(analysis)
                
                summary_data = self.traffic_analyzer.generate_traffic_summary(traffic_analyses)
                
                # Use advanced analyzer for detailed Q&A capabilities
                try:
                    qa_summary = self.advanced_analyzer.analyze_video_for_qa(events, video_info_dict)
                    logger.info(f"Advanced Q&A analysis completed: {len(qa_summary.key_events)} events analyzed")
                except Exception as qa_error:
                    logger.warning(f"Advanced Q&A analysis failed: {qa_error}")
                    qa_summary = None
                
                # Generate detailed summary
                total_violations = summary_data.get("total_violations", 0)
                violation_types = summary_data.get("unique_violation_types", [])
                vehicle_summary = summary_data.get("vehicle_summary", {})
                avg_safety_score = summary_data.get("average_safety_score", 5)
                overall_risk = summary_data.get("overall_risk", "LOW")
                
                # Calculate additional metrics
                total_vehicles = sum(vehicle_summary.values())
                avg_vehicles_per_frame = total_vehicles / len(events) if events else 0
                frames_with_violations = len([e for e in events if e.get('has_violations', False)])
                
                summary_parts = [
                    f"🚦 COMPREHENSIVE TRAFFIC ANALYSIS ({video_info['duration']:.1f} seconds):",
                    f"📊 STATISTICS:",
                    f"  - Frames analyzed: {len(events)}",
                    f"  - Total vehicles detected: {total_vehicles}",
                    f"  - Average vehicles per frame: {avg_vehicles_per_frame:.1f}",
                    f"  - Frames with violations: {frames_with_violations}/{len(events)}",
                    f"  - Total violations detected: {total_violations}",
                    f"  - Safety score (1-10): {avg_safety_score:.1f}",
                    f"  - Overall risk level: {overall_risk}",
                    "",
                    f"⚠️ VIOLATIONS DETECTED:"
                ]
                
                if violation_types:
                    for violation in violation_types[:10]:  # Limit to top 10
                        summary_parts.append(f"  - {violation.title()}")
                else:
                    summary_parts.append("  - No significant violations detected")
                
                summary_parts.extend([
                    "",
                    f"🚗 VEHICLE BREAKDOWN:"
                ])
                
                if vehicle_summary:
                    for vehicle_type, count in vehicle_summary.items():
                        summary_parts.append(f"  - {vehicle_type.title()}: {count}")
                else:
                    summary_parts.append("  - No vehicles specifically identified")
                
                summary_parts.extend([
                    "",
                    f"🔍 KEY OBSERVATIONS:"
                ])
                
                # Add contextual observations
                observations = set()
                risk_levels = [e.get("risk_level", "LOW") for e in events]
                high_risk_count = sum(1 for risk in risk_levels if risk in ["HIGH", "CRITICAL"])
                
                if high_risk_count > 0:
                    observations.add(f"High-risk situations detected in {high_risk_count} frames")
                
                if total_vehicles > len(events) * 3:
                    observations.add("Heavy traffic density observed")
                elif total_vehicles < len(events):
                    observations.add("Light traffic conditions")
                
                if any("intersection" in e.get("context", {}).get("location_type", "") for e in events):
                    observations.add("Intersection monitoring active")
                
                if total_violations > 0:
                    observations.add(f"Traffic enforcement attention recommended")
                else:
                    observations.add("Generally compliant traffic behavior")
                
                for obs in list(observations)[:5]:
                    summary_parts.append(f"  • {obs}")
                
                summary = "\n".join(summary_parts)
            else:
                summary = f"Video processed ({video_info['duration']:.1f} seconds) but no significant events detected."
            
            # Clean up frames safely
            cleanup_result = self.robust_video_processor.cleanup_frames_safe(frame_paths)
            logger.info(f"Cleanup completed: {cleanup_result['cleaned']} files removed, {cleanup_result['failed']} failures")
            
            # Record successful processing
            processing_time = time.time() - start_time
            video_processing_time.observe(processing_time)
            self.processing_stats['successful_videos'] += 1
            self.processing_stats['avg_processing_time'] = (
                (self.processing_stats['avg_processing_time'] * (self.processing_stats['successful_videos'] - 1) + 
                 processing_time) / self.processing_stats['successful_videos']
            )
            
            # Convert video_metadata to dict for compatibility
            video_info_dict = {
                "fps": video_metadata.fps,
                "frame_count": video_metadata.frame_count,
                "duration": video_metadata.duration,
                "width": video_metadata.width,
                "height": video_metadata.height,
                "quality_score": video_metadata.quality_score,
                "file_size": video_metadata.file_size
            }
            
            return {
                "video_info": video_info_dict,
                "events": events,
                "summary": summary,
                "total_frames_analyzed": len(events),
                "failed_frames": failed_frames,
                "processing_time": processing_time,
                "traffic_statistics": {
                    "total_violations": total_violations if events else 0,
                    "total_vehicles": total_vehicles if events else 0,
                    "avg_vehicles_per_frame": avg_vehicles_per_frame if events else 0,
                    "frames_with_violations": frames_with_violations if events else 0,
                    "safety_score": avg_safety_score if events else 5,
                    "violation_types": violation_types if events else []
                },
                "processing_stats": self.processing_stats.copy(),
                "qa_analysis": {
                    "video_summary": qa_summary.main_theme if qa_summary else "Analysis not available",
                    "most_significant_event": {
                        "description": qa_summary.significance_ranking[0][0].description if qa_summary and qa_summary.significance_ranking else "None identified",
                        "timestamp": qa_summary.significance_ranking[0][0].timestamp if qa_summary and qa_summary.significance_ranking else 0,
                        "significance_score": qa_summary.significance_ranking[0][1] if qa_summary and qa_summary.significance_ranking else 0
                    },
                    "camera_vehicle": {
                        "type": qa_summary.vehicle_analysis.camera_vehicle_type.value if qa_summary else "unknown",
                        "behavior": qa_summary.vehicle_analysis.camera_behavior if qa_summary else "unknown",
                        "confidence": qa_summary.vehicle_analysis.camera_vehicle_confidence if qa_summary else 0
                    },
                    "temporal_sequence": qa_summary.temporal_sequence if qa_summary else []
                }
            }
            
        except Exception as e:
            self.processing_stats['failed_videos'] += 1
            logger.error(f"Video processing failed: {e}")
            logger.debug(traceback.format_exc())
            
            # Cleanup any temporary files
            try:
                cleanup_result = self.robust_video_processor.cleanup_frames_safe([])
                logger.info(f"Emergency cleanup completed")
            except Exception as cleanup_error:
                logger.warning(f"Emergency cleanup failed: {cleanup_error}")
            
            # Return error result
            return {
                "error": str(e),
                "processing_time": time.time() - start_time,
                "video_info": {"error": "Failed to process video"},
                "events": [],
                "summary": f"Failed to process video: {str(e)}",
                "processing_stats": self.processing_stats.copy()
            }
    
    def _calculate_optimal_sampling(self, video_metadata) -> Tuple[float, int]:
        """Calculate optimal sampling rate and max frames based on video characteristics"""
        duration = video_metadata.duration
        quality = video_metadata.quality_score
        
        # Base sampling rate calculation
        if duration <= 10:
            base_sampling = 0.5  # Every 0.5 seconds for very short videos
        elif duration <= 30:
            base_sampling = 1.0  # Every 1 second for short videos
        elif duration <= 60:
            base_sampling = 2.0  # Every 2 seconds for medium videos
        elif duration <= 120:
            base_sampling = 2.5  # Every 2.5 seconds for longer videos
        else:
            base_sampling = 3.0  # Every 3 seconds for very long videos
        
        # Adjust based on quality
        if quality < 0.3:
            base_sampling *= 1.5  # Sample less frequently for low quality
        elif quality > 0.8:
            base_sampling *= 0.8  # Sample more frequently for high quality
        
        # Calculate max frames
        max_frames = min(25, max(10, int(duration / base_sampling)))
        
        return base_sampling, max_frames
    
    def _cleanup_models(self):
        """Cleanup loaded models to free memory"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
                self.processor = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"Model cleanup failed: {e}")
    
    def _force_memory_cleanup(self):
        """Force aggressive memory cleanup"""
        try:
            # Clear Python garbage
            gc.collect()
            
            # Clear CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Forced memory cleanup completed")
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    @contextmanager
    def _timeout_context(self, timeout_seconds: int):
        """Context manager for operation timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        # Set the timeout handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            yield
        finally:
            # Reset the alarm and handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _test_model_inference(self) -> bool:
        """Test if model can perform basic inference"""
        try:
            if self.model is None or self.processor is None:
                return False
            
            # Create a simple test image
            test_image = Image.new('RGB', (224, 224), color='white')
            test_prompt = "Describe this image briefly."
            
            # Try basic inference
            inputs = self.processor(
                text=test_prompt,
                images=[test_image],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # If we get here, model is working
            logger.info("Model inference test passed")
            return True
            
        except Exception as e:
            logger.error(f"Model inference test failed: {e}")
            return False
    
    def _attempt_model_recovery(self):
        """Attempt to recover a failed model"""
        logger.info("Attempting model recovery...")
        
        try:
            # Check if enough time has passed since last health check
            current_time = time.time()
            if current_time - self.last_health_check < self.health_check_interval:
                logger.info("Skipping recovery - too soon since last check")
                return
            
            self.last_health_check = current_time
            
            # Force cleanup and try to reload
            self._cleanup_models()
            self._force_memory_cleanup()
            
            # Reset failure counter
            self.model_failures = 0
            
            # Try to reload the model
            self._load_model_with_recovery()
            
            logger.info("Model recovery successful")
            
        except Exception as e:
            logger.error(f"Model recovery failed: {e}")
            self.is_healthy = False
            model_health.set(0)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        system_health = self.robust_video_processor.get_system_health()
        
        return {
            "model_healthy": self.is_healthy,
            "model_name": self.model_name if self.is_healthy else "None loaded",
            "model_failures": self.model_failures,
            "system_health": system_health,
            "processing_stats": self.processing_stats,
            "last_health_check": self.last_health_check,
            "device": self.device,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processing_stats = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'total_frames': 0,
            'avg_processing_time': 0
        }
        self.model_failures = 0
        logger.info("Processing statistics reset")