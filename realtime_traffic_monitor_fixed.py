"""
Real-Time Traffic Monitoring System
Integrates advanced CV, real-time processing, and performance optimization
for live traffic analysis, accident detection, and violation monitoring
"""

import asyncio
import logging
import signal
import sys
import os
import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
import cv2
import numpy as np
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from streaming.realtime_processor import RealtimeStreamProcessor, StreamConfig, AlertLevel
from optimization.performance_optimizer import PerformanceOptimizer, OptimizationConfig
from utils.realtime_cv_analyzer import ViolationType, AccidentType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealTimeTrafficMonitor:
    """Main real-time traffic monitoring system"""
    
    def __init__(self, config_file: Optional[str] = None):
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.performance_optimizer = PerformanceOptimizer(
            OptimizationConfig(**self.config.get('optimization', {}))
        )
        
        self.stream_processor = RealtimeStreamProcessor(
            redis_host=self.config.get('redis', {}).get('host', 'localhost'),
            redis_port=self.config.get('redis', {}).get('port', 6379)
        )
        
        # System state
        self.running = False
        self.active_streams = []
        
        # Statistics
        self.stats = {
            'start_time': time.time(),
            'total_alerts': 0,
            'violations_detected': 0,
            'accidents_detected': 0,
            'streams_processed': 0
        }
        
        logger.info("RealTimeTrafficMonitor initialized")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        
        default_config = {
            'streams': [
                {
                    'id': 'demo_camera',
                    'source': 0,  # Default camera
                    'type': 'camera'
                }
            ],
            'optimization': {
                'use_gpu': True,
                'model_precision': 'fp16',
                'max_batch_size': 4,
                'target_fps': 30,
                'max_workers': 4
            },
            'monitoring': {
                'alert_threshold': 'medium',
                'record_evidence': True,
                'enable_ai_analysis': True,
                'processing_interval': 0.1
            },
            'redis': {
                'host': 'localhost',
                'port': 6379
            },
            'websocket': {
                'host': 'localhost',
                'port': 8765
            },
            'output': {
                'evidence_dir': './evidence',
                'logs_dir': './logs',
                'alerts_file': './alerts.json'
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
                logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
        
        return default_config
    
    async def start_monitoring(self):
        """Start the real-time monitoring system"""
        
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        logger.info("🚦 Starting Real-Time Traffic Monitoring System")
        
        try:
            # Apply system optimizations
            self.performance_optimizer.optimize_system()
            
            # Create output directories
            self._create_output_directories()
            
            # Start WebSocket server for real-time updates
            websocket_config = self.config.get('websocket', {})
            await self.stream_processor.start_websocket_server(
                host=websocket_config.get('host', 'localhost'),
                port=websocket_config.get('port', 8765)
            )
            
            # Start configured streams
            await self._start_configured_streams()
            
            self.running = True
            
            # Start monitoring loop
            await self._monitoring_loop()
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            await self.shutdown()
    
    def _create_output_directories(self):
        """Create necessary output directories"""
        
        output_config = self.config.get('output', {})
        
        for dir_key in ['evidence_dir', 'logs_dir']:
            dir_path = output_config.get(dir_key)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
    
    async def _start_configured_streams(self):
        """Start all configured video streams"""
        
        alert_threshold_map = {
            'low': AlertLevel.LOW,
            'medium': AlertLevel.MEDIUM,  
            'high': AlertLevel.HIGH,
            'critical': AlertLevel.CRITICAL
        }
        
        monitoring_config = self.config.get('monitoring', {})
        
        for stream_config in self.config.get('streams', []):
            try:
                # Create stream configuration
                config = StreamConfig(
                    stream_id=stream_config['id'],
                    source=stream_config['source'],
                    fps=stream_config.get('fps', 30),
                    resolution=tuple(stream_config.get('resolution', [1280, 720])),
                    alert_threshold=alert_threshold_map.get(
                        monitoring_config.get('alert_threshold', 'medium'),
                        AlertLevel.MEDIUM
                    ),
                    record_alerts=monitoring_config.get('record_evidence', True),
                    enable_ai_analysis=monitoring_config.get('enable_ai_analysis', True),
                    processing_interval=monitoring_config.get('processing_interval', 0.1)
                )
                
                # Start stream
                success = await self.stream_processor.start_stream(config)
                if success:
                    self.active_streams.append(config.stream_id)
                    self.stats['streams_processed'] += 1
                    logger.info(f"✅ Started monitoring stream: {config.stream_id}")
                else:
                    logger.error(f"❌ Failed to start stream: {config.stream_id}")
                    
            except Exception as e:
                logger.error(f"Error starting stream {stream_config.get('id', 'unknown')}: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        logger.info("🔍 Real-time monitoring active")
        
        last_stats_update = time.time()
        last_performance_check = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                # Update statistics periodically
                if current_time - last_stats_update >= 10:  # Every 10 seconds
                    await self._update_statistics()
                    last_stats_update = current_time
                
                # Performance health check
                if current_time - last_performance_check >= 30:  # Every 30 seconds
                    await self._performance_health_check()
                    last_performance_check = current_time
                
                # Process any pending alerts
                await self._process_pending_alerts()
                
                # Brief sleep to prevent busy waiting
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
        finally:
            await self.shutdown()
    
    async def _update_statistics(self):
        """Update monitoring statistics"""
        
        try:
            # Get stream statuses
            status = self.stream_processor.get_all_stream_status()
            
            # Update stats
            active_count = len([s for s in status['active_streams'].values() if s['active']])
            
            # Get recent alerts count
            total_recent_alerts = 0
            for stream_id in self.active_streams:
                recent_alerts = self.stream_processor.get_recent_alerts(stream_id, 100)
                total_recent_alerts += len(recent_alerts)
                
                # Count violations and accidents
                for alert in recent_alerts:
                    if alert.get('type') == 'violation':
                        self.stats['violations_detected'] += 1
                    elif alert.get('type') == 'accident':
                        self.stats['accidents_detected'] += 1
            
            self.stats['total_alerts'] = total_recent_alerts
            
            # Log statistics
            uptime = time.time() - self.stats['start_time']
            logger.info(f"📊 STATS - Uptime: {uptime:.0f}s, Active Streams: {active_count}, "
                       f"Total Alerts: {self.stats['total_alerts']}, "
                       f"Violations: {self.stats['violations_detected']}, "
                       f"Accidents: {self.stats['accidents_detected']}")
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    async def _performance_health_check(self):
        """Check system performance and health"""
        
        try:
            # Get performance report
            system_status = self.performance_optimizer.get_system_status()
            perf_report = system_status.get('performance_report', {})
            
            current_metrics = perf_report.get('current_metrics', {})
            grade = perf_report.get('performance_grade', 'N/A')
            
            # Log performance status
            fps = current_metrics.get('fps', 0)
            cpu = current_metrics.get('cpu_utilization', 0)
            memory = current_metrics.get('memory_used', 0)
            
            logger.info(f"🔧 PERFORMANCE - Grade: {grade}, FPS: {fps:.1f}, "
                       f"CPU: {cpu:.1f}%, Memory: {memory:.1f}%")
            
            # Check for performance issues
            recommendations = perf_report.get('recommendations', [])
            if recommendations:
                logger.warning(f"💡 RECOMMENDATIONS: {'; '.join(recommendations[:3])}")
            
            # Alert on critical performance issues
            if grade == 'F' or fps < 10:
                logger.error("🚨 CRITICAL PERFORMANCE ISSUE DETECTED")
            
        except Exception as e:
            logger.error(f"Performance health check failed: {e}")
    
    async def _process_pending_alerts(self):
        """Process any pending alerts that need action"""
        
        try:
            # This could implement custom alert processing logic
            pass
            
        except Exception as e:
            logger.error(f"Alert processing error: {e}")
    
    async def shutdown(self):
        """Shutdown the monitoring system"""
        
        logger.info("🛑 Shutting down Real-Time Traffic Monitor")
        
        self.running = False
        
        try:
            # Stop stream processing
            await self.stream_processor.shutdown()
            
            # Stop performance optimization
            self.performance_optimizer.shutdown()
            
            logger.info("✅ Real-Time Traffic Monitor shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def create_demo_config() -> Dict[str, Any]:
    """Create a demo configuration for testing"""
    
    return {
        'streams': [
            {
                'id': 'demo_camera_0',
                'source': 0,
                'type': 'camera',
                'fps': 30,
                'resolution': [1280, 720]
            }
        ],
        'optimization': {
            'use_gpu': True,
            'model_precision': 'fp16',
            'max_batch_size': 4,
            'target_fps': 30,
            'max_workers': 4
        },
        'monitoring': {
            'alert_threshold': 'medium',
            'record_evidence': True,
            'enable_ai_analysis': True,
            'processing_interval': 0.1
        },
        'output': {
            'evidence_dir': './evidence',
            'logs_dir': './logs',
            'alerts_file': './alerts.json'
        }
    }

def create_ip_camera_config(rtsp_urls: List[str]) -> Dict[str, Any]:
    """Create configuration for IP cameras"""
    
    streams = []
    for i, url in enumerate(rtsp_urls):
        streams.append({
            'id': f'ip_camera_{i}',
            'source': url,
            'type': 'rtsp',
            'fps': 25,
            'resolution': [1920, 1080]
        })
    
    return {
        'streams': streams,
        'optimization': {
            'use_gpu': True,
            'model_precision': 'fp16',
            'max_batch_size': 8,
            'target_fps': 25,
            'max_workers': 6
        },
        'monitoring': {
            'alert_threshold': 'low',
            'record_evidence': True,
            'enable_ai_analysis': True,
            'processing_interval': 0.05
        }
    }

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Real-Time Traffic Monitoring System')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with default camera')
    parser.add_argument('--cameras', type=str, nargs='+', help='Camera indices to monitor')
    parser.add_argument('--rtsp', type=str, nargs='+', help='RTSP URLs to monitor')
    parser.add_argument('--create-config', type=str, help='Create demo config file at specified path')
    
    args = parser.parse_args()
    
    # Create demo config if requested
    if args.create_config:
        config = create_demo_config()
        with open(args.create_config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Demo configuration created at {args.create_config}")
        return
    
    # Create dynamic config based on arguments
    config_file = args.config
    
    if args.demo:
        print("🎬 Starting in DEMO mode with default camera...")
        config_file = None
    
    elif args.rtsp:
        print(f"📹 Starting with RTSP streams: {args.rtsp}")
        config = create_ip_camera_config(args.rtsp)
        config_file = 'temp_rtsp_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    elif args.cameras:
        print(f"📷 Starting with cameras: {args.cameras}")
        camera_indices = [int(c) for c in args.cameras]
        config = create_demo_config()
        config['streams'] = [
            {
                'id': f'camera_{i}',
                'source': i,
                'type': 'camera'
            } for i in camera_indices
        ]
        config_file = 'temp_camera_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Initialize and start monitoring
    monitor = RealTimeTrafficMonitor(config_file)
    
    # Handle shutdown gracefully
    def signal_handler(signum, frame):
        print("\n🛑 Shutdown signal received...")
        asyncio.create_task(monitor.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n👋 Real-Time Traffic Monitor stopped")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await monitor.shutdown()

if __name__ == "__main__":
    print("🚦 Real-Time Traffic Monitoring System")
    print("=====================================\n")
    
    # Check dependencies
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - GPU Available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not found")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found")
    
    try:
        import redis
        print("✅ Redis client available")
    except ImportError:
        print("⚠️  Redis client not found - some features may be limited")
    
    print()
    
    # Run the main application
    asyncio.run(main())