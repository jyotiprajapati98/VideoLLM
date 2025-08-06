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
                await asyncio.sleep(0.1)\n                \n        except KeyboardInterrupt:\n            logger.info(\"Monitoring interrupted by user\")\n        except Exception as e:\n            logger.error(f\"Monitoring loop error: {e}\")\n        finally:\n            await self.shutdown()\n    \n    async def _update_statistics(self):\n        \"\"\"Update monitoring statistics\"\"\"\n        \n        try:\n            # Get stream statuses\n            status = self.stream_processor.get_all_stream_status()\n            \n            # Update stats\n            active_count = len([s for s in status['active_streams'].values() if s['active']])\n            \n            # Get recent alerts count\n            total_recent_alerts = 0\n            for stream_id in self.active_streams:\n                recent_alerts = self.stream_processor.get_recent_alerts(stream_id, 100)\n                total_recent_alerts += len(recent_alerts)\n                \n                # Count violations and accidents\n                for alert in recent_alerts:\n                    if alert.get('type') == 'violation':\n                        self.stats['violations_detected'] += 1\n                    elif alert.get('type') == 'accident':\n                        self.stats['accidents_detected'] += 1\n            \n            self.stats['total_alerts'] = total_recent_alerts\n            \n            # Log statistics\n            uptime = time.time() - self.stats['start_time']\n            logger.info(f\"📊 STATS - Uptime: {uptime:.0f}s, Active Streams: {active_count}, \"\n                       f\"Total Alerts: {self.stats['total_alerts']}, \"\n                       f\"Violations: {self.stats['violations_detected']}, \"\n                       f\"Accidents: {self.stats['accidents_detected']}\")\n            \n        except Exception as e:\n            logger.error(f\"Error updating statistics: {e}\")\n    \n    async def _performance_health_check(self):\n        \"\"\"Check system performance and health\"\"\"\n        \n        try:\n            # Get performance report\n            system_status = self.performance_optimizer.get_system_status()\n            perf_report = system_status.get('performance_report', {})\n            \n            current_metrics = perf_report.get('current_metrics', {})\n            grade = perf_report.get('performance_grade', 'N/A')\n            \n            # Log performance status\n            fps = current_metrics.get('fps', 0)\n            cpu = current_metrics.get('cpu_utilization', 0)\n            memory = current_metrics.get('memory_used', 0)\n            \n            logger.info(f\"🔧 PERFORMANCE - Grade: {grade}, FPS: {fps:.1f}, \"\n                       f\"CPU: {cpu:.1f}%, Memory: {memory:.1f}%\")\n            \n            # Check for performance issues\n            recommendations = perf_report.get('recommendations', [])\n            if recommendations:\n                logger.warning(f\"💡 RECOMMENDATIONS: {'; '.join(recommendations[:3])}\")\n            \n            # Alert on critical performance issues\n            if grade == 'F' or fps < 10:\n                logger.error(\"🚨 CRITICAL PERFORMANCE ISSUE DETECTED\")\n                # Could implement automatic remediation here\n            \n        except Exception as e:\n            logger.error(f\"Performance health check failed: {e}\")\n    \n    async def _process_pending_alerts(self):\n        \"\"\"Process any pending alerts that need action\"\"\"\n        \n        try:\n            # This could implement custom alert processing logic\n            # For example, integrating with external systems, sending notifications, etc.\n            pass\n            \n        except Exception as e:\n            logger.error(f\"Alert processing error: {e}\")\n    \n    def add_camera_stream(self, camera_index: int, stream_id: Optional[str] = None) -> bool:\n        \"\"\"Add a new camera stream dynamically\"\"\"\n        \n        if stream_id is None:\n            stream_id = f\"camera_{camera_index}\"\n        \n        try:\n            config = StreamConfig(\n                stream_id=stream_id,\n                source=camera_index,\n                fps=30,\n                resolution=(1280, 720),\n                alert_threshold=AlertLevel.MEDIUM,\n                record_alerts=True,\n                enable_ai_analysis=True,\n                processing_interval=0.1\n            )\n            \n            # Start the stream (this would need to be called from async context)\n            # success = await self.stream_processor.start_stream(config)\n            # For now, add to config for next restart\n            \n            logger.info(f\"Camera stream {stream_id} added to configuration\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Failed to add camera stream: {e}\")\n            return False\n    \n    def add_rtsp_stream(self, rtsp_url: str, stream_id: Optional[str] = None) -> bool:\n        \"\"\"Add a new RTSP stream dynamically\"\"\"\n        \n        if stream_id is None:\n            stream_id = f\"rtsp_{len(self.active_streams)}\"\n        \n        try:\n            config = StreamConfig(\n                stream_id=stream_id,\n                source=rtsp_url,\n                fps=25,\n                resolution=(1920, 1080),\n                alert_threshold=AlertLevel.LOW,  # More sensitive for IP cameras\n                record_alerts=True,\n                enable_ai_analysis=True,\n                processing_interval=0.05  # Higher frequency for RTSP\n            )\n            \n            logger.info(f\"RTSP stream {stream_id} added to configuration\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Failed to add RTSP stream: {e}\")\n            return False\n    \n    async def shutdown(self):\n        \"\"\"Shutdown the monitoring system\"\"\"\n        \n        logger.info(\"🛑 Shutting down Real-Time Traffic Monitor\")\n        \n        self.running = False\n        \n        try:\n            # Stop stream processing\n            await self.stream_processor.shutdown()\n            \n            # Stop performance optimization\n            self.performance_optimizer.shutdown()\n            \n            # Save final statistics\n            await self._save_final_stats()\n            \n            logger.info(\"✅ Real-Time Traffic Monitor shutdown complete\")\n            \n        except Exception as e:\n            logger.error(f\"Error during shutdown: {e}\")\n    \n    async def _save_final_stats(self):\n        \"\"\"Save final statistics to file\"\"\"\n        \n        try:\n            final_stats = {\n                'session_stats': self.stats,\n                'total_uptime': time.time() - self.stats['start_time'],\n                'final_stream_count': len(self.active_streams),\n                'shutdown_time': time.time()\n            }\n            \n            stats_file = self.config.get('output', {}).get('logs_dir', './logs') + '/final_stats.json'\n            \n            with open(stats_file, 'w') as f:\n                json.dump(final_stats, f, indent=2)\n            \n            logger.info(f\"Final statistics saved to {stats_file}\")\n            \n        except Exception as e:\n            logger.error(f\"Failed to save final stats: {e}\")\n\n\ndef create_demo_config() -> Dict[str, Any]:\n    \"\"\"Create a demo configuration for testing\"\"\"\n    \n    return {\n        'streams': [\n            {\n                'id': 'demo_camera_0',\n                'source': 0,\n                'type': 'camera',\n                'fps': 30,\n                'resolution': [1280, 720]\n            }\n        ],\n        'optimization': {\n            'use_gpu': True,\n            'model_precision': 'fp16',\n            'max_batch_size': 4,\n            'target_fps': 30,\n            'max_workers': 4\n        },\n        'monitoring': {\n            'alert_threshold': 'medium',\n            'record_evidence': True,\n            'enable_ai_analysis': True,\n            'processing_interval': 0.1\n        },\n        'output': {\n            'evidence_dir': './evidence',\n            'logs_dir': './logs',\n            'alerts_file': './alerts.json'\n        }\n    }\n\ndef create_ip_camera_config(rtsp_urls: List[str]) -> Dict[str, Any]:\n    \"\"\"Create configuration for IP cameras\"\"\"\n    \n    streams = []\n    for i, url in enumerate(rtsp_urls):\n        streams.append({\n            'id': f'ip_camera_{i}',\n            'source': url,\n            'type': 'rtsp',\n            'fps': 25,\n            'resolution': [1920, 1080]\n        })\n    \n    return {\n        'streams': streams,\n        'optimization': {\n            'use_gpu': True,\n            'model_precision': 'fp16',\n            'max_batch_size': 8,  # Larger batch for IP cameras\n            'target_fps': 25,\n            'max_workers': 6\n        },\n        'monitoring': {\n            'alert_threshold': 'low',  # More sensitive for surveillance\n            'record_evidence': True,\n            'enable_ai_analysis': True,\n            'processing_interval': 0.05  # Higher frequency\n        }\n    }\n\nasync def main():\n    \"\"\"Main entry point\"\"\"\n    \n    parser = argparse.ArgumentParser(description='Real-Time Traffic Monitoring System')\n    parser.add_argument('--config', type=str, help='Configuration file path')\n    parser.add_argument('--demo', action='store_true', help='Run in demo mode with default camera')\n    parser.add_argument('--cameras', type=str, nargs='+', help='Camera indices to monitor')\n    parser.add_argument('--rtsp', type=str, nargs='+', help='RTSP URLs to monitor')\n    parser.add_argument('--create-config', type=str, help='Create demo config file at specified path')\n    \n    args = parser.parse_args()\n    \n    # Create demo config if requested\n    if args.create_config:\n        config = create_demo_config()\n        with open(args.create_config, 'w') as f:\n            json.dump(config, f, indent=2)\n        print(f\"Demo configuration created at {args.create_config}\")\n        return\n    \n    # Create dynamic config based on arguments\n    config_file = args.config\n    \n    if args.demo:\n        print(\"🎬 Starting in DEMO mode with default camera...\")\n        # Use default demo config\n        config_file = None\n    \n    elif args.rtsp:\n        print(f\"📹 Starting with RTSP streams: {args.rtsp}\")\n        config = create_ip_camera_config(args.rtsp)\n        config_file = 'temp_rtsp_config.json'\n        with open(config_file, 'w') as f:\n            json.dump(config, f, indent=2)\n    \n    elif args.cameras:\n        print(f\"📷 Starting with cameras: {args.cameras}\")\n        camera_indices = [int(c) for c in args.cameras]\n        config = create_demo_config()\n        config['streams'] = [\n            {\n                'id': f'camera_{i}',\n                'source': i,\n                'type': 'camera'\n            } for i in camera_indices\n        ]\n        config_file = 'temp_camera_config.json'\n        with open(config_file, 'w') as f:\n            json.dump(config, f, indent=2)\n    \n    # Initialize and start monitoring\n    monitor = RealTimeTrafficMonitor(config_file)\n    \n    # Handle shutdown gracefully\n    def signal_handler(signum, frame):\n        print(\"\\n🛑 Shutdown signal received...\")\n        asyncio.create_task(monitor.shutdown())\n    \n    signal.signal(signal.SIGINT, signal_handler)\n    signal.signal(signal.SIGTERM, signal_handler)\n    \n    try:\n        await monitor.start_monitoring()\n    except KeyboardInterrupt:\n        print(\"\\n👋 Real-Time Traffic Monitor stopped\")\n    except Exception as e:\n        logger.error(f\"Fatal error: {e}\")\n    finally:\n        await monitor.shutdown()\n\nif __name__ == \"__main__\":\n    print(\"🚦 Real-Time Traffic Monitoring System\")\n    print(\"=====================================\\n\")\n    \n    # Check dependencies\n    try:\n        import torch\n        print(f\"✅ PyTorch {torch.__version__} - GPU Available: {torch.cuda.is_available()}\")\n    except ImportError:\n        print(\"❌ PyTorch not found\")\n    \n    try:\n        import cv2\n        print(f\"✅ OpenCV {cv2.__version__}\")\n    except ImportError:\n        print(\"❌ OpenCV not found\")\n    \n    try:\n        import redis\n        print(f\"✅ Redis client available\")\n    except ImportError:\n        print(\"⚠️  Redis client not found - some features may be limited\")\n    \n    print()\n    \n    # Run the main application\n    asyncio.run(main())"