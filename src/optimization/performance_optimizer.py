"""
Performance Optimization Module for Real-Time Video Processing
GPU acceleration, model optimization, and resource management
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
import cv2
import numpy as np
import logging
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue, Empty
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    max_batch_size: int = 8
    model_precision: str = "fp16"  # fp32, fp16, int8
    enable_tensorrt: bool = True
    enable_model_quantization: bool = True
    max_workers: int = 4
    frame_queue_size: int = 100
    enable_frame_skipping: bool = True
    target_fps: int = 30

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    fps: float = 0.0
    processing_time: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_used: float = 0.0
    cpu_utilization: float = 0.0
    memory_used: float = 0.0
    frame_drops: int = 0
    queue_size: int = 0

class GPUAccelerator:
    """GPU acceleration utilities"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.device = self._setup_gpu()
        self.streams = []  # CUDA streams for async processing
        self.memory_pool = {}  # Pre-allocated GPU memory
        
    def _setup_gpu(self) -> torch.device:
        """Setup GPU with optimal settings"""
        
        if not self.config.use_gpu or not torch.cuda.is_available():
            logger.info("Using CPU for processing")
            return torch.device('cpu')
        
        # Select best GPU
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            logger.warning("No CUDA GPUs available")
            return torch.device('cpu')
        
        # Find GPU with most memory
        best_gpu = 0
        max_memory = 0
        
        for i in range(gpu_count):
            memory = torch.cuda.get_device_properties(i).total_memory
            if memory > max_memory:
                max_memory = memory
                best_gpu = i
        
        device = torch.device(f'cuda:{best_gpu}')
        torch.cuda.set_device(device)
        
        # Set memory management
        if self.config.gpu_memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Create CUDA streams for async processing
        for i in range(4):
            stream = torch.cuda.Stream(device=device)
            self.streams.append(stream)
        
        logger.info(f"GPU acceleration enabled on {torch.cuda.get_device_name(device)}")
        logger.info(f"GPU Memory: {max_memory / 1024**3:.1f} GB")
        
        return device
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for inference"""
        
        model = model.to(self.device)
        model.eval()
        
        # Enable inference mode
        torch.set_grad_enabled(False)
        
        # Apply precision optimization
        if self.config.model_precision == "fp16" and self.device.type == 'cuda':
            model = model.half()
            logger.info("Model converted to FP16")
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # TensorRT optimization
        if self.config.enable_tensorrt and self.device.type == 'cuda':
            model = self._convert_to_tensorrt(model)
        
        return model
    
    def _convert_to_tensorrt(self, model: torch.nn.Module) -> torch.nn.Module:
        """Convert model to TensorRT for maximum performance"""
        
        try:
            import torch_tensorrt
            
            # Example input for tracing
            example_input = torch.randn(1, 3, 640, 480).to(self.device)
            if self.config.model_precision == "fp16":
                example_input = example_input.half()
            
            # Convert to TensorRT
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[example_input],
                enabled_precisions={torch.float16 if self.config.model_precision == "fp16" else torch.float32},
                workspace_size=1 << 30,  # 1GB
                max_batch_size=self.config.max_batch_size
            )
            
            logger.info("Model converted to TensorRT")
            return trt_model
            
        except ImportError:
            logger.warning("torch_tensorrt not available, skipping TensorRT optimization")
        except Exception as e:
            logger.warning(f"TensorRT conversion failed: {e}")
        
        return model
    
    def create_cuda_stream(self) -> torch.cuda.Stream:
        """Create new CUDA stream"""
        return torch.cuda.Stream(device=self.device)
    
    def preprocess_batch_gpu(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess batch of frames on GPU"""
        
        # Convert to tensor and move to GPU
        batch_tensor = torch.from_numpy(np.stack(frames)).to(self.device)
        
        # Normalize and convert to CHW format
        batch_tensor = batch_tensor.permute(0, 3, 1, 2).float() / 255.0
        
        # Apply optimized preprocessing
        if self.config.model_precision == "fp16":
            batch_tensor = batch_tensor.half()
        
        return batch_tensor
    
    def get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU utilization metrics"""
        
        if self.device.type != 'cuda':
            return {'gpu_utilization': 0, 'gpu_memory_used': 0, 'gpu_memory_total': 0}
        
        try:
            # PyTorch GPU memory
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
            
            # Try to get GPU utilization (requires nvidia-ml-py)
            gpu_util = 0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except ImportError:
                pass
            except Exception:
                pass
            
            return {
                'gpu_utilization': gpu_util,
                'gpu_memory_used': memory_allocated,
                'gpu_memory_reserved': memory_reserved,
                'gpu_memory_total': memory_total,
                'gpu_memory_percent': (memory_allocated / memory_total) * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return {'gpu_utilization': 0, 'gpu_memory_used': 0, 'gpu_memory_total': 0}

class ModelOptimizer:
    """Model optimization and quantization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def quantize_model(self, model: torch.nn.Module, calibration_data: List[torch.Tensor]) -> torch.nn.Module:
        """Quantize model for faster inference"""
        
        if not self.config.enable_model_quantization:
            return model
        
        try:
            # Dynamic quantization (good for CPU)
            if self.config.use_gpu:
                # Static quantization for GPU (more complex setup)
                quantized_model = self._static_quantization(model, calibration_data)
            else:
                # Dynamic quantization for CPU
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
            
            logger.info(f"Model quantized successfully")
            return quantized_model
            
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            return model
    
    def _static_quantization(self, model: torch.nn.Module, calibration_data: List[torch.Tensor]) -> torch.nn.Module:
        """Apply static quantization with calibration data"""
        
        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model)
        
        # Calibrate with sample data
        model_prepared.eval()
        with torch.no_grad():
            for data in calibration_data[:100]:  # Use subset for calibration
                model_prepared(data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        return quantized_model
    
    def optimize_for_mobile(self, model: torch.nn.Module) -> torch.jit.ScriptModule:
        """Optimize model for mobile deployment"""
        
        try:
            # Convert to TorchScript
            traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
            
            # Optimize for mobile
            optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
            
            logger.info("Model optimized for mobile deployment")
            return optimized_model
            
        except Exception as e:
            logger.error(f"Mobile optimization failed: {e}")
            return model

class BatchProcessor:
    """Efficient batch processing for video frames"""
    
    def __init__(self, config: OptimizationConfig, gpu_accelerator: GPUAccelerator):
        self.config = config
        self.gpu_accelerator = gpu_accelerator
        self.batch_queue = Queue(maxsize=config.frame_queue_size)
        self.result_queue = Queue(maxsize=config.frame_queue_size)
        self.processing = False
        self.worker_threads = []
        
    def start_batch_processing(self):
        """Start batch processing threads"""
        
        self.processing = True
        
        # Start worker threads
        for i in range(self.config.max_workers):
            worker = threading.Thread(
                target=self._batch_worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info(f"Started {self.config.max_workers} batch processing workers")
    
    def _batch_worker(self, worker_id: int):
        """Batch processing worker thread"""
        
        batch_frames = []
        batch_metadata = []
        
        while self.processing:
            try:
                # Collect batch
                while len(batch_frames) < self.config.max_batch_size:
                    try:
                        frame, metadata = self.batch_queue.get(timeout=0.1)
                        batch_frames.append(frame)
                        batch_metadata.append(metadata)
                    except Empty:
                        break
                
                if not batch_frames:
                    continue
                
                # Process batch
                start_time = time.time()
                
                # GPU preprocessing
                if self.gpu_accelerator.device.type == 'cuda':
                    with torch.cuda.stream(self.gpu_accelerator.streams[worker_id % len(self.gpu_accelerator.streams)]):
                        processed_batch = self._process_batch_gpu(batch_frames, batch_metadata)
                else:
                    processed_batch = self._process_batch_cpu(batch_frames, batch_metadata)
                
                processing_time = time.time() - start_time
                
                # Queue results
                for i, result in enumerate(processed_batch):
                    result['processing_time'] = processing_time / len(processed_batch)
                    result['worker_id'] = worker_id
                    self.result_queue.put((result, batch_metadata[i]))
                
                # Clear batch
                batch_frames.clear()
                batch_metadata.clear()
                
                # Memory cleanup
                if len(batch_frames) % 10 == 0:  # Periodic cleanup
                    gc.collect()
                    if self.gpu_accelerator.device.type == 'cuda':
                        torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Batch worker {worker_id} error: {e}")
    
    def _process_batch_gpu(self, frames: List[np.ndarray], metadata: List[Dict]) -> List[Dict]:
        """Process batch on GPU"""
        
        try:
            # Convert to GPU tensor
            batch_tensor = self.gpu_accelerator.preprocess_batch_gpu(frames)
            
            # Placeholder for actual processing
            # In real implementation, this would call your optimized models
            results = []
            for i in range(len(frames)):
                result = {
                    'frame_id': metadata[i].get('frame_id', i),
                    'timestamp': metadata[i].get('timestamp', time.time()),
                    'processed_on_gpu': True,
                    'batch_size': len(frames)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"GPU batch processing failed: {e}")
            return self._process_batch_cpu(frames, metadata)
    
    def _process_batch_cpu(self, frames: List[np.ndarray], metadata: List[Dict]) -> List[Dict]:
        """Process batch on CPU"""
        
        results = []
        for i, frame in enumerate(frames):
            result = {
                'frame_id': metadata[i].get('frame_id', i),
                'timestamp': metadata[i].get('timestamp', time.time()),
                'processed_on_gpu': False,
                'batch_size': len(frames)
            }
            results.append(result)
        
        return results
    
    def add_frame(self, frame: np.ndarray, metadata: Dict) -> bool:
        """Add frame to processing queue"""
        
        try:
            self.batch_queue.put_nowait((frame, metadata))
            return True
        except:
            return False
    
    def get_result(self, timeout: float = 1.0) -> Optional[Tuple[Dict, Dict]]:
        """Get processing result"""
        
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def stop_batch_processing(self):
        """Stop batch processing"""
        
        self.processing = False
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        logger.info("Batch processing stopped")

class FrameOptimizer:
    """Optimize frame processing and memory usage"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.frame_cache = {}  # LRU cache for processed frames
        self.preprocessing_transforms = self._setup_transforms()
        
    def _setup_transforms(self):
        """Setup optimized preprocessing transforms"""
        
        transforms_list = [
            transforms.ToPILImage(),
            transforms.Resize((640, 480)),  # Standard resolution for processing
            transforms.ToTensor(),
        ]
        
        if self.config.model_precision == "fp16":
            # Add normalization for FP16
            transforms_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        return transforms.Compose(transforms_list)
    
    def optimize_frame(self, frame: np.ndarray, frame_id: str) -> np.ndarray:
        """Optimize frame for processing"""
        
        # Check cache first
        if frame_id in self.frame_cache:
            return self.frame_cache[frame_id]
        
        # Resize if too large
        height, width = frame.shape[:2]
        max_dimension = 1080
        
        if max(height, width) > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Enhance frame if needed
        frame = self._enhance_frame(frame)
        
        # Cache result (with LRU eviction)
        self._cache_frame(frame_id, frame)
        
        return frame
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply fast enhancement to frame"""
        
        # Fast contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Merge and convert back
        lab = cv2.merge((l_channel, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _cache_frame(self, frame_id: str, frame: np.ndarray, max_cache_size: int = 100):
        """Cache processed frame with LRU eviction"""
        
        # Simple LRU cache implementation
        if len(self.frame_cache) >= max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.frame_cache))
            del self.frame_cache[oldest_key]
        
        self.frame_cache[frame_id] = frame

class PerformanceMonitor:
    """Monitor and report performance metrics"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics_history = []
        self.monitoring = False
        self.monitor_thread = None
        
        # Initialize metrics
        self.current_metrics = PerformanceMetrics()
        
    def start_monitoring(self):
        """Start performance monitoring"""
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def _monitor_loop(self):
        """Performance monitoring loop"""
        
        while self.monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Update metrics
                self.current_metrics.cpu_utilization = cpu_percent
                self.current_metrics.memory_used = memory.percent
                
                # Store in history
                self.metrics_history.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'fps': self.current_metrics.fps,
                    'processing_time': self.current_metrics.processing_time
                })
                
                # Keep only last 1000 entries
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    def update_fps(self, fps: float):
        """Update FPS metric"""
        self.current_metrics.fps = fps
    
    def update_processing_time(self, processing_time: float):
        """Update processing time metric"""
        self.current_metrics.processing_time = processing_time
    
    def update_gpu_metrics(self, gpu_metrics: Dict[str, float]):
        """Update GPU metrics"""
        self.current_metrics.gpu_utilization = gpu_metrics.get('gpu_utilization', 0)
        self.current_metrics.gpu_memory_used = gpu_metrics.get('gpu_memory_used', 0)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.current_metrics
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        # Calculate statistics
        fps_values = [m['fps'] for m in self.metrics_history[-60:]]  # Last minute
        processing_times = [m['processing_time'] for m in self.metrics_history[-60:]]
        cpu_values = [m['cpu_percent'] for m in self.metrics_history[-60:]]
        memory_values = [m['memory_percent'] for m in self.metrics_history[-60:]]
        
        return {
            'current_metrics': {
                'fps': self.current_metrics.fps,
                'processing_time': self.current_metrics.processing_time,
                'cpu_utilization': self.current_metrics.cpu_utilization,
                'memory_used': self.current_metrics.memory_used,
                'gpu_utilization': self.current_metrics.gpu_utilization,
                'gpu_memory_used': self.current_metrics.gpu_memory_used
            },
            'averages_last_minute': {
                'fps': np.mean(fps_values) if fps_values else 0,
                'processing_time': np.mean(processing_times) if processing_times else 0,
                'cpu_utilization': np.mean(cpu_values) if cpu_values else 0,
                'memory_used': np.mean(memory_values) if memory_values else 0
            },
            'performance_grade': self._calculate_performance_grade(),
            'recommendations': self._generate_recommendations()
        }
    
    def _calculate_performance_grade(self) -> str:
        """Calculate overall performance grade"""
        
        metrics = self.current_metrics
        
        score = 0
        max_score = 100
        
        # FPS score (target: config.target_fps)
        if metrics.fps >= self.config.target_fps * 0.9:
            score += 30
        elif metrics.fps >= self.config.target_fps * 0.7:
            score += 20
        elif metrics.fps >= self.config.target_fps * 0.5:
            score += 10
        
        # Processing time score (target: < 33ms for 30fps)
        target_time = 1.0 / self.config.target_fps
        if metrics.processing_time <= target_time:
            score += 25
        elif metrics.processing_time <= target_time * 1.5:
            score += 15
        elif metrics.processing_time <= target_time * 2:
            score += 10
        
        # Resource utilization score
        if metrics.cpu_utilization < 80 and metrics.memory_used < 80:
            score += 25
        elif metrics.cpu_utilization < 90 and metrics.memory_used < 90:
            score += 15
        else:
            score += 5
        
        # GPU utilization (if available)
        if metrics.gpu_utilization > 0:
            if metrics.gpu_utilization < 90:
                score += 20
            else:
                score += 10
        
        # Convert to grade
        percentage = (score / max_score) * 100
        
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        metrics = self.current_metrics
        
        if metrics.fps < self.config.target_fps * 0.8:
            recommendations.append("Consider reducing video resolution or frame rate")
            recommendations.append("Enable frame skipping for non-critical analysis")
        
        if metrics.processing_time > 1.0 / self.config.target_fps:
            recommendations.append("Optimize model inference or enable GPU acceleration")
            recommendations.append("Consider using model quantization")
        
        if metrics.cpu_utilization > 85:
            recommendations.append("High CPU usage detected - consider adding more processing workers")
        
        if metrics.memory_used > 85:
            recommendations.append("High memory usage - enable memory optimization features")
        
        if metrics.gpu_utilization == 0 and self.config.use_gpu:
            recommendations.append("GPU not being utilized - check GPU acceleration settings")
        
        return recommendations
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")

# Main optimization coordinator
class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self, config: OptimizationConfig = None):
        if config is None:
            config = OptimizationConfig()
        
        self.config = config
        self.gpu_accelerator = GPUAccelerator(config)
        self.model_optimizer = ModelOptimizer(config)
        self.batch_processor = BatchProcessor(config, self.gpu_accelerator)
        self.frame_optimizer = FrameOptimizer(config)
        self.performance_monitor = PerformanceMonitor(config)
        
        logger.info("PerformanceOptimizer initialized")
    
    def optimize_system(self):
        """Apply all system optimizations"""
        
        # Start batch processing
        self.batch_processor.start_batch_processing()
        
        # Start performance monitoring  
        self.performance_monitor.start_monitoring()
        
        # Apply system-level optimizations
        self._optimize_opencv()
        self._optimize_torch()
        
        logger.info("System optimization applied")
    
    def _optimize_opencv(self):
        """Optimize OpenCV settings"""
        
        # Enable threading
        cv2.setNumThreads(self.config.max_workers)
        
        # Use optimized code paths
        cv2.setUseOptimized(True)
        
        logger.info("OpenCV optimizations applied")
    
    def _optimize_torch(self):
        """Optimize PyTorch settings"""
        
        # Set number of threads
        torch.set_num_threads(self.config.max_workers)
        
        # Enable JIT optimizations
        torch.jit.set_fusion_strategy([
            ('STATIC', 2),  # Enable static fusion
            ('DYNAMIC', 2)  # Enable dynamic fusion
        ])
        
        logger.info("PyTorch optimizations applied")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        gpu_metrics = self.gpu_accelerator.get_gpu_metrics()
        performance_report = self.performance_monitor.get_performance_report()
        
        return {
            'optimization_config': {
                'use_gpu': self.config.use_gpu,
                'model_precision': self.config.model_precision,
                'max_batch_size': self.config.max_batch_size,
                'target_fps': self.config.target_fps
            },
            'gpu_metrics': gpu_metrics,
            'performance_report': performance_report,
            'device_info': {
                'device': str(self.gpu_accelerator.device),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
    
    def shutdown(self):
        """Shutdown optimizer and cleanup resources"""
        
        self.batch_processor.stop_batch_processing()
        self.performance_monitor.stop_monitoring()
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("PerformanceOptimizer shutdown complete")