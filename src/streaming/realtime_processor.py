"""
Real-Time Processing Pipeline for Live Traffic Analysis
Handles video streams, camera feeds, and real-time analysis
"""

import cv2
import numpy as np
import asyncio
import threading
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import websockets
import redis
from datetime import datetime, timedelta
import uuid
import os

# Import our custom modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.realtime_cv_analyzer import RealtimeComputerVision, TrafficViolation, AccidentEvent, AlertLevel
from utils.advanced_video_analyzer import AdvancedVideoAnalyzer

logger = logging.getLogger(__name__)

@dataclass 
class StreamConfig:
    """Configuration for video stream processing"""
    stream_id: str
    source: Union[str, int]  # URL, file path, or camera index
    fps: int = 30
    resolution: tuple = (1920, 1080)
    roi_areas: List[np.ndarray] = field(default_factory=list)
    alert_threshold: AlertLevel = AlertLevel.MEDIUM
    record_alerts: bool = True
    enable_ai_analysis: bool = True
    processing_interval: float = 0.1  # Process every N seconds

@dataclass
class AlertMessage:
    """Real-time alert message"""
    alert_id: str
    stream_id: str
    timestamp: float
    alert_type: str  # 'violation', 'accident', 'flow'
    level: AlertLevel
    message: str
    location: tuple
    confidence: float
    requires_action: bool = False
    evidence_image: Optional[np.ndarray] = None

class RealtimeStreamProcessor:
    """Main real-time stream processing engine"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.streams = {}  # Active streams
        self.processors = {}  # CV processors per stream
        self.alert_queues = {}  # Alert queues per stream
        
        # Initialize components
        self.cv_analyzer = RealtimeComputerVision(enable_gpu=True)
        self.advanced_analyzer = AdvancedVideoAnalyzer()
        
        # Redis for real-time messaging
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis for real-time messaging")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # WebSocket server for real-time updates
        self.websocket_server = None
        self.websocket_clients = set()
        
        # Performance monitoring
        self.performance_metrics = {
            'frames_processed': 0,
            'alerts_generated': 0,
            'processing_time_avg': 0.0,
            'active_streams': 0
        }
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        logger.info("RealtimeStreamProcessor initialized")
    
    async def start_stream(self, config: StreamConfig) -> bool:
        """Start processing a new video stream"""
        
        if config.stream_id in self.streams:
            logger.warning(f"Stream {config.stream_id} already active")
            return False
        
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(config.source)
            
            if not cap.isOpened():
                logger.error(f"Failed to open stream source: {config.source}")
                return False
            
            # Configure capture properties
            cap.set(cv2.CAP_PROP_FPS, config.fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
            
            # Store stream info
            self.streams[config.stream_id] = {
                'config': config,
                'capture': cap,
                'active': True,
                'frame_count': 0,
                'start_time': time.time(),
                'last_processed': 0
            }
            
            # Initialize alert queue
            self.alert_queues[config.stream_id] = Queue(maxsize=1000)
            
            # Set ROI if specified
            if config.roi_areas:
                self.cv_analyzer.set_roi(config.roi_areas)
            
            # Start processing thread
            processing_thread = threading.Thread(
                target=self._process_stream_thread,
                args=(config.stream_id,),
                daemon=True
            )
            processing_thread.start()
            
            # Start alert handler
            alert_thread = threading.Thread(
                target=self._handle_alerts_thread,
                args=(config.stream_id,),
                daemon=True
            )
            alert_thread.start()
            
            self.performance_metrics['active_streams'] += 1
            logger.info(f"Stream {config.stream_id} started successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream {config.stream_id}: {e}")
            return False
    
    def _process_stream_thread(self, stream_id: str):
        """Process video stream in separate thread"""
        
        stream_info = self.streams[stream_id]
        config = stream_info['config']
        cap = stream_info['capture']
        
        frame_interval = 1.0 / config.fps
        last_ai_analysis = 0
        
        try:
            while stream_info['active']:
                current_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from stream {stream_id}")
                    if isinstance(config.source, str) and not os.path.exists(config.source):
                        # If it's a file and doesn't exist anymore, stop
                        break
                    time.sleep(0.1)
                    continue
                
                stream_info['frame_count'] += 1
                frame_number = stream_info['frame_count']
                
                # Process frame at configured interval
                if current_time - stream_info['last_processed'] >= config.processing_interval:
                    
                    # Real-time computer vision analysis
                    start_time = time.time()
                    
                    try:
                        # Process with CV analyzer
                        cv_results = self.cv_analyzer.process_frame_realtime(
                            frame, frame_number, current_time
                        )
                        
                        # AI analysis at lower frequency (every 2 seconds)
                        ai_results = None
                        if config.enable_ai_analysis and (current_time - last_ai_analysis) >= 2.0:
                            # Run AI analysis in thread pool to avoid blocking
                            try:
                                future = self.executor.submit(self._perform_ai_analysis_sync, frame, cv_results)
                                ai_results = future.result(timeout=1.0)  # 1 second timeout
                                last_ai_analysis = current_time
                            except Exception as e:
                                logger.warning(f"AI analysis failed: {e}")
                                ai_results = None
                        
                        # Generate alerts
                        alerts = self._generate_alerts(stream_id, cv_results, ai_results, frame)
                        
                        # Process alerts
                        for alert in alerts:
                            self._queue_alert(stream_id, alert)
                        
                        # Update metrics
                        processing_time = time.time() - start_time
                        self._update_performance_metrics(processing_time)
                        
                        stream_info['last_processed'] = current_time
                        
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_number} from stream {stream_id}: {e}")
                
                # Maintain frame rate
                elapsed = time.time() - current_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            logger.error(f"Stream processing thread error for {stream_id}: {e}")
        finally:
            self._cleanup_stream(stream_id)
    
    def _perform_ai_analysis_sync(self, frame: np.ndarray, cv_results: Dict) -> Dict[str, Any]:
        """Synchronous version of AI analysis for thread pool execution"""
        try:
            return asyncio.run(self._perform_ai_analysis(frame, cv_results))
        except Exception as e:
            logger.error(f"Sync AI analysis failed: {e}")
            return {'ai_analysis': 'analysis_failed', 'error': str(e)}
    
    async def _perform_ai_analysis(self, frame: np.ndarray, cv_results: Dict) -> Dict[str, Any]:
        """Perform AI analysis on frame (less frequent, more detailed)"""
        
        try:
            # Convert CV results to format expected by AI analyzer
            mock_events = []
            
            # Create mock events from tracked objects
            for obj_id, obj in cv_results.get('tracked_objects', {}).items():
                event = {
                    'timestamp': cv_results['timestamp'],
                    'description': f"{obj.object_type} detected at position {obj.center}",
                    'violations_detected': [],
                    'safety_score': 8  # Default
                }
                mock_events.append(event)
            
            # Add violation events
            for violation in cv_results.get('violations', []):
                event = {
                    'timestamp': violation.timestamp,
                    'description': violation.description,
                    'violations_detected': [violation.violation_type.value],
                    'safety_score': 10 - violation.severity
                }
                mock_events.append(event)
            
            # Perform advanced analysis
            if mock_events:
                video_info = {'duration': 1.0, 'fps': 30}
                summary = self.advanced_analyzer.analyze_video_for_qa(mock_events, video_info)
                
                return {
                    'ai_summary': summary.main_theme,
                    'significant_events': len(summary.key_events),
                    'camera_vehicle_type': summary.vehicle_analysis.camera_vehicle_type.value,
                    'temporal_analysis': summary.temporal_sequence[:3]  # Latest 3 events
                }
            
            return {'ai_analysis': 'no_significant_events'}
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {'ai_analysis': 'analysis_failed', 'error': str(e)}
    
    def _generate_alerts(self, stream_id: str, cv_results: Dict, 
                        ai_results: Optional[Dict], frame: np.ndarray) -> List[AlertMessage]:
        """Generate alerts based on analysis results"""
        
        alerts = []
        timestamp = time.time()
        config = self.streams[stream_id]['config']
        
        # Violation alerts
        for violation in cv_results.get('violations', []):
            if violation.alert_level.value >= config.alert_threshold.value:
                alert = AlertMessage(
                    alert_id=str(uuid.uuid4()),
                    stream_id=stream_id,
                    timestamp=timestamp,
                    alert_type='violation',
                    level=violation.alert_level,
                    message=f"Traffic Violation: {violation.description}",
                    location=violation.location,
                    confidence=violation.confidence,
                    requires_action=violation.severity >= 8,
                    evidence_image=frame.copy() if config.record_alerts else None
                )
                alerts.append(alert)
        
        # Accident alerts  
        for accident in cv_results.get('accidents', []):
            if accident.alert_level.value >= config.alert_threshold.value:
                alert = AlertMessage(
                    alert_id=str(uuid.uuid4()),
                    stream_id=stream_id,
                    timestamp=timestamp,
                    alert_type='accident',
                    level=accident.alert_level,
                    message=f"Accident Detected: {accident.description}",
                    location=accident.location,
                    confidence=accident.confidence,
                    requires_action=accident.emergency_required,
                    evidence_image=frame.copy() if config.record_alerts else None
                )
                alerts.append(alert)
        
        # Traffic flow alerts
        flow_metrics = cv_results.get('flow_metrics', {})
        if flow_metrics.get('congestion_level') == 'heavy':
            alert = AlertMessage(
                alert_id=str(uuid.uuid4()),
                stream_id=stream_id,
                timestamp=timestamp,
                alert_type='flow',
                level=AlertLevel.MEDIUM,
                message=f"Heavy traffic congestion detected. {flow_metrics.get('vehicle_count', 0)} vehicles",
                location=(0, 0),
                confidence=0.9,
                requires_action=False
            )
            alerts.append(alert)
        
        return alerts
    
    def _queue_alert(self, stream_id: str, alert: AlertMessage):
        """Queue alert for processing"""
        
        try:
            queue = self.alert_queues.get(stream_id)
            if queue:
                if queue.full():
                    # Remove oldest alert to make space
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass
                
                queue.put(alert)
                self.performance_metrics['alerts_generated'] += 1
                
        except Exception as e:
            logger.error(f"Failed to queue alert: {e}")
    
    def _handle_alerts_thread(self, stream_id: str):
        """Handle alerts in separate thread"""
        
        queue = self.alert_queues[stream_id]
        
        try:
            while stream_id in self.streams and self.streams[stream_id]['active']:
                try:
                    alert = queue.get(timeout=1.0)
                    
                    # Process alert
                    asyncio.run(self._process_alert(alert))
                    
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing alert: {e}")
                    
        except Exception as e:
            logger.error(f"Alert handling thread error for {stream_id}: {e}")
    
    async def _process_alert(self, alert: AlertMessage):
        """Process individual alert"""
        
        try:
            # Store in Redis
            if self.redis_client:
                alert_data = {
                    'alert_id': alert.alert_id,
                    'stream_id': alert.stream_id,
                    'timestamp': alert.timestamp,
                    'type': alert.alert_type,
                    'level': alert.level.value,
                    'message': alert.message,
                    'location': alert.location,
                    'confidence': alert.confidence,
                    'requires_action': alert.requires_action
                }
                
                # Store alert
                self.redis_client.hset(f"alert:{alert.alert_id}", mapping=alert_data)
                self.redis_client.expire(f"alert:{alert.alert_id}", 3600)  # 1 hour TTL
                
                # Add to stream alerts list
                self.redis_client.lpush(f"stream_alerts:{alert.stream_id}", alert.alert_id)
                self.redis_client.ltrim(f"stream_alerts:{alert.stream_id}", 0, 999)  # Keep last 1000
                
                # Publish real-time notification
                notification = {
                    'type': 'alert',
                    'data': alert_data
                }
                self.redis_client.publish('traffic_alerts', json.dumps(notification))
            
            # Send to WebSocket clients
            await self._broadcast_alert(alert)
            
            # Handle critical alerts
            if alert.level == AlertLevel.CRITICAL and alert.requires_action:
                await self._handle_critical_alert(alert)
                
        except Exception as e:
            logger.error(f"Error processing alert {alert.alert_id}: {e}")
    
    async def _broadcast_alert(self, alert: AlertMessage):
        """Broadcast alert to WebSocket clients"""
        
        if not self.websocket_clients:
            return
        
        message = {
            'type': 'alert',
            'alert_id': alert.alert_id,
            'stream_id': alert.stream_id,
            'timestamp': alert.timestamp,
            'alert_type': alert.alert_type,
            'level': alert.level.value,
            'message': alert.message,
            'location': alert.location,
            'confidence': alert.confidence,
            'requires_action': alert.requires_action
        }
        
        # Send to all connected clients
        disconnected_clients = []
        for client in self.websocket_clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
            except Exception as e:
                logger.error(f"Error sending to WebSocket client: {e}")
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.websocket_clients.discard(client)
    
    async def _handle_critical_alert(self, alert: AlertMessage):
        """Handle critical alerts requiring immediate action"""
        
        logger.critical(f"CRITICAL ALERT: {alert.message} at {alert.location}")
        
        # Save evidence image
        if alert.evidence_image is not None:
            evidence_dir = f"evidence/{alert.stream_id}"
            os.makedirs(evidence_dir, exist_ok=True)
            
            evidence_path = f"{evidence_dir}/alert_{alert.alert_id}_{int(alert.timestamp)}.jpg"
            cv2.imwrite(evidence_path, alert.evidence_image)
        
        # Could integrate with emergency services, SMS alerts, etc.
        # For now, just log and store
        
        critical_alert_data = {
            'alert': {
                'id': alert.alert_id,
                'stream_id': alert.stream_id,
                'timestamp': alert.timestamp,
                'message': alert.message,
                'location': alert.location
            },
            'status': 'pending_response',
            'created_at': datetime.now().isoformat()
        }
        
        if self.redis_client:
            self.redis_client.hset(f"critical_alert:{alert.alert_id}", mapping=critical_alert_data)
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics"""
        
        self.performance_metrics['frames_processed'] += 1
        
        # Update average processing time (sliding window)
        current_avg = self.performance_metrics['processing_time_avg']
        count = self.performance_metrics['frames_processed']
        
        # Sliding average with weight towards recent values
        alpha = 0.1  # Weight for new value
        self.performance_metrics['processing_time_avg'] = \
            alpha * processing_time + (1 - alpha) * current_avg
    
    def stop_stream(self, stream_id: str) -> bool:
        """Stop processing a video stream"""
        
        if stream_id not in self.streams:
            logger.warning(f"Stream {stream_id} not found")
            return False
        
        try:
            # Mark stream as inactive
            self.streams[stream_id]['active'] = False
            
            # Clean up
            self._cleanup_stream(stream_id)
            
            logger.info(f"Stream {stream_id} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping stream {stream_id}: {e}")
            return False
    
    def _cleanup_stream(self, stream_id: str):
        """Clean up stream resources"""
        
        if stream_id in self.streams:
            stream_info = self.streams[stream_id]
            
            # Close video capture
            if 'capture' in stream_info:
                stream_info['capture'].release()
            
            # Clear alert queue
            if stream_id in self.alert_queues:
                del self.alert_queues[stream_id]
            
            # Remove from active streams
            del self.streams[stream_id]
            self.performance_metrics['active_streams'] -= 1
    
    def get_stream_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific stream"""
        
        if stream_id not in self.streams:
            return None
        
        stream_info = self.streams[stream_id]
        current_time = time.time()
        
        return {
            'stream_id': stream_id,
            'active': stream_info['active'],
            'frame_count': stream_info['frame_count'],
            'uptime': current_time - stream_info['start_time'],
            'fps': stream_info['frame_count'] / (current_time - stream_info['start_time']),
            'source': stream_info['config'].source,
            'alert_queue_size': self.alert_queues[stream_id].qsize(),
            'processing_interval': stream_info['config'].processing_interval
        }
    
    def get_all_stream_status(self) -> Dict[str, Any]:
        """Get status of all active streams"""
        
        status = {
            'active_streams': {},
            'performance_metrics': self.performance_metrics.copy(),
            'system_health': {
                'redis_connected': self.redis_client is not None,
                'websocket_clients': len(self.websocket_clients),
                'memory_usage': self.cv_analyzer.get_performance_metrics()
            }
        }
        
        for stream_id in self.streams:
            status['active_streams'][stream_id] = self.get_stream_status(stream_id)
        
        return status
    
    async def start_websocket_server(self, host: str = 'localhost', port: int = 8765):
        """Start WebSocket server for real-time updates"""
        
        async def handle_client(websocket, path):
            """Handle new WebSocket client"""
            self.websocket_clients.add(websocket)
            logger.info(f"WebSocket client connected. Total clients: {len(self.websocket_clients)}")
            
            try:
                # Send welcome message with current status
                welcome_message = {
                    'type': 'welcome',
                    'status': self.get_all_stream_status()
                }
                await websocket.send(json.dumps(welcome_message))
                
                # Keep connection alive
                async for message in websocket:
                    # Handle client messages if needed
                    try:
                        data = json.loads(message)
                        # Could handle commands like start/stop streams
                        if data.get('command') == 'get_status':
                            status = self.get_all_stream_status()
                            await websocket.send(json.dumps({'type': 'status', 'data': status}))
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON received from WebSocket client")
                        
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.websocket_clients.discard(websocket)
                logger.info(f"WebSocket client disconnected. Total clients: {len(self.websocket_clients)}")
        
        try:
            self.websocket_server = await websockets.serve(handle_client, host, port)
            logger.info(f"WebSocket server started on ws://{host}:{port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False
    
    def get_recent_alerts(self, stream_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts for a stream"""
        
        if not self.redis_client:
            return []
        
        try:
            # Get alert IDs from Redis
            alert_ids = self.redis_client.lrange(f"stream_alerts:{stream_id}", 0, limit - 1)
            
            alerts = []
            for alert_id in alert_ids:
                alert_data = self.redis_client.hgetall(f"alert:{alert_id}")
                if alert_data:
                    alerts.append(alert_data)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []
    
    async def shutdown(self):
        """Shutdown the processor and clean up resources"""
        
        logger.info("Shutting down RealtimeStreamProcessor...")
        
        # Stop all streams
        stream_ids = list(self.streams.keys())
        for stream_id in stream_ids:
            self.stop_stream(stream_id)
        
        # Close WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("RealtimeStreamProcessor shutdown complete")


# Utility functions for integration

async def start_camera_monitoring(camera_indices: List[int], 
                                 processor: RealtimeStreamProcessor) -> List[str]:
    """Start monitoring multiple cameras"""
    
    started_streams = []
    
    for i, camera_index in enumerate(camera_indices):
        config = StreamConfig(
            stream_id=f"camera_{camera_index}",
            source=camera_index,
            fps=30,
            resolution=(1280, 720),
            alert_threshold=AlertLevel.MEDIUM,
            record_alerts=True,
            enable_ai_analysis=True,
            processing_interval=0.2
        )
        
        success = await processor.start_stream(config)
        if success:
            started_streams.append(config.stream_id)
            logger.info(f"Started monitoring camera {camera_index}")
        else:
            logger.error(f"Failed to start monitoring camera {camera_index}")
    
    return started_streams


async def start_rtsp_monitoring(rtsp_urls: List[str], 
                               processor: RealtimeStreamProcessor) -> List[str]:
    """Start monitoring RTSP streams (IP cameras)"""
    
    started_streams = []
    
    for i, rtsp_url in enumerate(rtsp_urls):
        config = StreamConfig(
            stream_id=f"rtsp_stream_{i}",
            source=rtsp_url,
            fps=25,
            resolution=(1920, 1080),
            alert_threshold=AlertLevel.LOW,  # More sensitive for IP cameras
            record_alerts=True,
            enable_ai_analysis=True,
            processing_interval=0.1
        )
        
        success = await processor.start_stream(config)
        if success:
            started_streams.append(config.stream_id)
            logger.info(f"Started monitoring RTSP stream: {rtsp_url}")
        else:
            logger.error(f"Failed to start monitoring RTSP stream: {rtsp_url}")
    
    return started_streams