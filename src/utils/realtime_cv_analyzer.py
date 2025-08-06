"""
Advanced Computer Vision Module for Real-Time Traffic Analysis
Specialized for accident detection, traffic violations, and real-time monitoring
"""

import cv2
import numpy as np
import logging
import time
import threading
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import torch
import torchvision.transforms as transforms
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

class ViolationType(Enum):
    RED_LIGHT_VIOLATION = "red_light_violation"
    SPEEDING = "speeding"
    WRONG_LANE = "wrong_lane"
    ILLEGAL_TURN = "illegal_turn"
    FOLLOWING_TOO_CLOSE = "following_too_close"
    ILLEGAL_PARKING = "illegal_parking"
    PEDESTRIAN_VIOLATION = "pedestrian_violation"
    STOP_SIGN_VIOLATION = "stop_sign_violation"

class AccidentType(Enum):
    COLLISION = "collision"
    NEAR_MISS = "near_miss"
    ROLLOVER = "rollover"
    PEDESTRIAN_ACCIDENT = "pedestrian_accident"
    REAR_END = "rear_end"
    SIDE_IMPACT = "side_impact"

class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TrackedObject:
    """Object tracking information"""
    id: int
    object_type: str  # 'car', 'truck', 'motorcycle', 'bicycle', 'pedestrian'
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]
    velocity: Tuple[float, float]  # pixels per frame
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))
    confidence: float = 0.0
    last_seen: int = 0
    color: Optional[Tuple[int, int, int]] = None
    size_category: str = "medium"  # small, medium, large
    
    def __post_init__(self):
        self.trajectory.append((self.center, time.time()))

@dataclass
class TrafficViolation:
    """Real-time traffic violation detection"""
    violation_type: ViolationType
    object_id: int
    timestamp: float
    location: Tuple[int, int]
    confidence: float
    severity: int  # 1-10 scale
    description: str
    evidence_frame: Optional[np.ndarray] = None
    alert_level: AlertLevel = AlertLevel.MEDIUM

@dataclass
class AccidentEvent:
    """Accident detection event"""
    accident_type: AccidentType
    timestamp: float
    involved_objects: List[int]
    location: Tuple[int, int]
    severity_score: float
    confidence: float
    description: str
    alert_level: AlertLevel = AlertLevel.HIGH
    emergency_required: bool = False

class RealtimeComputerVision:
    """Advanced computer vision for real-time traffic analysis"""
    
    def __init__(self, enable_gpu: bool = True):
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.enable_gpu else 'cpu')
        
        # Object tracking
        self.tracked_objects = {}
        self.next_object_id = 1
        self.max_tracking_distance = 100
        self.tracking_history_length = 30
        
        # Traffic analysis
        self.traffic_zones = {}
        self.violation_history = deque(maxlen=1000)
        self.accident_history = deque(maxlen=100)
        
        # Performance optimization
        self.frame_skip = 1  # Process every N frames
        self.processing_resolution = (640, 480)
        self.roi_areas = []  # Regions of interest
        
        # Initialize components
        self._init_object_detector()
        self._init_trackers()
        self._init_traffic_analyzers()
        
        logger.info(f"RealtimeComputerVision initialized on {self.device}")
    
    def _init_object_detector(self):
        """Initialize object detection model"""
        try:
            # Using YOLOv8 for real-time detection
            from ultralytics import YOLO
            
            # Load a model optimized for speed vs accuracy
            model_path = "yolov8n.pt"  # Nano version for speed
            self.detector = YOLO(model_path)
            
            # Move to GPU if available
            if self.enable_gpu:
                self.detector.to(self.device)
            
            # Class names for traffic objects
            self.traffic_classes = {
                0: 'person',
                2: 'car', 
                3: 'motorcycle',
                5: 'bus',
                7: 'truck',
                1: 'bicycle',
                9: 'traffic_light',
                11: 'stop_sign'
            }
            
            logger.info("Object detector initialized successfully")
            
        except ImportError:
            logger.warning("ultralytics not available, using OpenCV DNN")
            self._init_opencv_detector()
    
    def _init_opencv_detector(self):
        """Fallback OpenCV-based detection"""
        try:
            # Load pre-trained models
            self.net = cv2.dnn.readNetFromDarknet(
                "yolo/yolov4.cfg",
                "yolo/yolov4.weights"
            )
            
            if self.enable_gpu:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            self.detector = self.net
            logger.info("OpenCV DNN detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            self.detector = None
    
    def _init_trackers(self):
        """Initialize object tracking algorithms"""
        self.trackers = {}
        
        # Multiple tracking algorithms for robustness (with compatibility check)
        self.tracking_algorithms = {}
        
        # Check for modern OpenCV tracker methods
        if hasattr(cv2, 'TrackerCSRT_create'):
            self.tracking_algorithms['csrt'] = cv2.TrackerCSRT_create
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            self.tracking_algorithms['csrt'] = cv2.legacy.TrackerCSRT_create
            
        if hasattr(cv2, 'TrackerKCF_create'):
            self.tracking_algorithms['kcf'] = cv2.TrackerKCF_create
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
            self.tracking_algorithms['kcf'] = cv2.legacy.TrackerKCF_create
            
        if hasattr(cv2, 'TrackerMOSSE_create'):
            self.tracking_algorithms['mosse'] = cv2.TrackerMOSSE_create
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
            self.tracking_algorithms['mosse'] = cv2.legacy.TrackerMOSSE_create
            
        # Fallback to basic tracking if none available
        if not self.tracking_algorithms:
            logger.warning("No OpenCV trackers available, using basic tracking")
            self.use_basic_tracking = True
        else:
            self.use_basic_tracking = False
        
        # Kalman filters for prediction
        self.kalman_filters = {}
        
        # Optical flow for motion analysis
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        logger.info("Object trackers initialized")
    
    def _init_traffic_analyzers(self):
        """Initialize traffic-specific analyzers"""
        
        # Traffic light detector
        self.traffic_light_detector = self._create_traffic_light_detector()
        
        # Speed estimation
        self.speed_estimator = SpeedEstimator()
        
        # Collision predictor
        self.collision_predictor = CollisionPredictor()
        
        # Lane detector
        self.lane_detector = LaneDetector()
        
        # Traffic flow analyzer
        self.flow_analyzer = TrafficFlowAnalyzer()
        
        logger.info("Traffic analyzers initialized")
    
    def process_frame_realtime(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Dict[str, Any]:
        """Process a single frame for real-time analysis"""
        
        start_time = time.time()
        
        # Optimize frame for processing
        processed_frame = self._preprocess_frame(frame)
        
        # Detect objects
        detections = self._detect_objects(processed_frame)
        
        # Update object tracking
        tracked_objects = self._update_tracking(detections, frame_number)
        
        # Analyze traffic violations
        violations = self._detect_violations(tracked_objects, processed_frame, timestamp)
        
        # Detect accidents
        accidents = self._detect_accidents(tracked_objects, timestamp)
        
        # Analyze traffic flow
        flow_metrics = self._analyze_traffic_flow(tracked_objects)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'tracked_objects': tracked_objects,
            'violations': violations,
            'accidents': accidents,
            'flow_metrics': flow_metrics,
            'processing_time': processing_time,
            'alert_level': self._calculate_alert_level(violations, accidents)
        }
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Optimize frame for real-time processing"""
        
        # Resize for faster processing
        height, width = frame.shape[:2]
        if width > self.processing_resolution[0]:
            scale = self.processing_resolution[0] / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Enhance contrast for better detection
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        
        # Apply ROI if defined
        if self.roi_areas:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for roi in self.roi_areas:
                cv2.fillPoly(mask, [roi], 255)
            frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        return frame
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in frame using optimized detection"""
        
        if not self.detector:
            return []
        
        try:
            # Run detection
            if hasattr(self.detector, 'predict'):
                # YOLOv8
                results = self.detector.predict(frame, verbose=False, conf=0.4)
                detections = self._parse_yolo_results(results[0])
            else:
                # OpenCV DNN
                detections = self._detect_with_opencv(frame)
            
            # Filter for traffic-relevant objects
            traffic_detections = []
            for det in detections:
                if det['class_name'] in ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                    traffic_detections.append(det)
            
            return traffic_detections
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def _parse_yolo_results(self, results) -> List[Dict[str, Any]]:
        """Parse YOLOv8 detection results"""
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                center_x, center_y = x1 + w//2, y1 + h//2
                
                detection = {
                    'bbox': (x1, y1, w, h),
                    'center': (center_x, center_y),
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': self.traffic_classes.get(int(cls_id), 'unknown')
                }
                detections.append(detection)
        
        return detections
    
    def _detect_with_opencv(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects using OpenCV DNN"""
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        
        detections = []
        height, width = frame.shape[:2]
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.4:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'center': (center_x, center_y),
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.traffic_classes.get(class_id, 'unknown')
                    })
        
        return detections
    
    def _update_tracking(self, detections: List[Dict], frame_number: int) -> Dict[int, TrackedObject]:
        """Update object tracking with Hungarian algorithm for assignment"""
        
        current_time = time.time()
        
        # Predict positions for existing objects
        predicted_positions = {}
        for obj_id, obj in self.tracked_objects.items():
            if len(obj.trajectory) >= 2:
                # Simple linear prediction
                last_pos, last_time = obj.trajectory[-1]
                prev_pos, prev_time = obj.trajectory[-2]
                
                if last_time != prev_time:
                    velocity = (
                        (last_pos[0] - prev_pos[0]) / (last_time - prev_time),
                        (last_pos[1] - prev_pos[1]) / (last_time - prev_time)
                    )
                    
                    time_diff = current_time - last_time
                    predicted_x = last_pos[0] + velocity[0] * time_diff
                    predicted_y = last_pos[1] + velocity[1] * time_diff
                    predicted_positions[obj_id] = (predicted_x, predicted_y)
                else:
                    predicted_positions[obj_id] = last_pos
            else:
                predicted_positions[obj_id] = obj.center
        
        # Create cost matrix for Hungarian algorithm
        existing_objects = list(self.tracked_objects.keys())
        cost_matrix = np.zeros((len(existing_objects), len(detections)))
        
        for i, obj_id in enumerate(existing_objects):
            predicted_pos = predicted_positions[obj_id]
            for j, detection in enumerate(detections):
                distance = euclidean(predicted_pos, detection['center'])
                cost_matrix[i, j] = distance
        
        # Apply Hungarian algorithm
        if len(existing_objects) > 0 and len(detections) > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Update matched objects
            matched_detections = set()
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < self.max_tracking_distance:
                    obj_id = existing_objects[row]
                    detection = detections[col]
                    
                    self._update_tracked_object(obj_id, detection, current_time, frame_number)
                    matched_detections.add(col)
        
        # Create new objects for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                self._create_new_tracked_object(detection, current_time, frame_number)
        
        # Remove old objects
        self._cleanup_old_objects(current_time)
        
        return self.tracked_objects.copy()
    
    def _update_tracked_object(self, obj_id: int, detection: Dict, timestamp: float, frame_number: int):
        """Update an existing tracked object"""
        obj = self.tracked_objects[obj_id]
        
        # Update position
        old_center = obj.center
        obj.center = detection['center']
        obj.bbox = detection['bbox']
        obj.confidence = detection['confidence']
        obj.last_seen = frame_number
        
        # Update trajectory
        obj.trajectory.append((obj.center, timestamp))
        
        # Calculate velocity
        if len(obj.trajectory) >= 2:
            prev_pos, prev_time = obj.trajectory[-2]
            time_diff = timestamp - prev_time
            if time_diff > 0:
                obj.velocity = (
                    (obj.center[0] - prev_pos[0]) / time_diff,
                    (obj.center[1] - prev_pos[1]) / time_diff
                )
    
    def _create_new_tracked_object(self, detection: Dict, timestamp: float, frame_number: int):
        """Create a new tracked object"""
        obj = TrackedObject(
            id=self.next_object_id,
            object_type=detection['class_name'],
            bbox=detection['bbox'],
            center=detection['center'],
            velocity=(0.0, 0.0),
            confidence=detection['confidence'],
            last_seen=frame_number
        )
        
        self.tracked_objects[self.next_object_id] = obj
        self.next_object_id += 1
    
    def _cleanup_old_objects(self, current_time: float, max_age: float = 2.0):
        """Remove objects that haven't been seen recently"""
        to_remove = []
        
        for obj_id, obj in self.tracked_objects.items():
            if len(obj.trajectory) > 0:
                _, last_time = obj.trajectory[-1]
                if current_time - last_time > max_age:
                    to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
    
    def _detect_violations(self, tracked_objects: Dict[int, TrackedObject], 
                          frame: np.ndarray, timestamp: float) -> List[TrafficViolation]:
        """Detect traffic violations in real-time"""
        violations = []
        
        for obj_id, obj in tracked_objects.items():
            # Speed violation detection
            speed_violation = self._check_speed_violation(obj)
            if speed_violation:
                violations.append(speed_violation)
            
            # Lane violation detection
            lane_violation = self._check_lane_violation(obj, frame)
            if lane_violation:
                violations.append(lane_violation)
            
            # Following distance violation
            following_violation = self._check_following_distance(obj, tracked_objects)
            if following_violation:
                violations.append(following_violation)
            
            # Red light violation (if traffic lights are detected)
            red_light_violation = self._check_red_light_violation(obj, frame)
            if red_light_violation:
                violations.append(red_light_violation)
        
        return violations
    
    def _check_speed_violation(self, obj: TrackedObject) -> Optional[TrafficViolation]:
        """Check for speeding violations"""
        if len(obj.trajectory) < 3:
            return None
        
        # Calculate average speed over last few frames
        speeds = []
        for i in range(len(obj.trajectory) - 1):
            pos1, time1 = obj.trajectory[i]
            pos2, time2 = obj.trajectory[i + 1]
            
            if time2 != time1:
                distance = euclidean(pos1, pos2)
                time_diff = time2 - time1
                speed = distance / time_diff  # pixels per second
                speeds.append(speed)
        
        if speeds:
            avg_speed = np.mean(speeds)
            # Convert to approximate real-world speed (this would need calibration)
            estimated_speed_kmh = avg_speed * 0.1  # Rough conversion factor
            
            # Check against speed limit (would be configurable)
            speed_limit = 50  # km/h
            if estimated_speed_kmh > speed_limit * 1.2:  # 20% tolerance
                return TrafficViolation(
                    violation_type=ViolationType.SPEEDING,
                    object_id=obj.id,
                    timestamp=time.time(),
                    location=obj.center,
                    confidence=0.8,
                    severity=min(10, int((estimated_speed_kmh - speed_limit) / speed_limit * 10)),
                    description=f"Vehicle exceeding speed limit: {estimated_speed_kmh:.1f} km/h (limit: {speed_limit} km/h)",
                    alert_level=AlertLevel.HIGH if estimated_speed_kmh > speed_limit * 1.5 else AlertLevel.MEDIUM
                )
        
        return None
    
    def _check_lane_violation(self, obj: TrackedObject, frame: np.ndarray) -> Optional[TrafficViolation]:
        """Check for lane violations using simple lane detection"""
        # This is a simplified implementation
        # In production, you'd use more sophisticated lane detection
        
        if len(obj.trajectory) < 5:
            return None
        
        # Check if object is consistently in wrong area
        recent_positions = [pos for pos, _ in list(obj.trajectory)[-5:]]
        
        # Simple check: if object consistently in left 20% of frame (assuming right-hand traffic)
        frame_width = frame.shape[1]
        left_boundary = frame_width * 0.2
        
        if obj.object_type in ['car', 'truck', 'bus'] and all(pos[0] < left_boundary for pos in recent_positions):
            return TrafficViolation(
                violation_type=ViolationType.WRONG_LANE,
                object_id=obj.id,
                timestamp=time.time(),
                location=obj.center,
                confidence=0.6,
                severity=7,
                description="Vehicle potentially in wrong lane or oncoming traffic",
                alert_level=AlertLevel.HIGH
            )
        
        return None
    
    def _check_following_distance(self, obj: TrackedObject, 
                                 all_objects: Dict[int, TrackedObject]) -> Optional[TrafficViolation]:
        """Check for following too closely"""
        if obj.object_type not in ['car', 'truck', 'bus']:
            return None
        
        # Find objects in front
        for other_id, other_obj in all_objects.items():
            if other_id == obj.id or other_obj.object_type not in ['car', 'truck', 'bus']:
                continue
            
            # Check if other object is in front and close
            distance = euclidean(obj.center, other_obj.center)
            
            # Simple following distance check (would need calibration)
            min_distance = 50  # pixels
            if distance < min_distance:
                # Check if objects are moving in similar direction
                velocity_diff = euclidean(obj.velocity, other_obj.velocity)
                if velocity_diff < 10:  # Similar velocities
                    return TrafficViolation(
                        violation_type=ViolationType.FOLLOWING_TOO_CLOSE,
                        object_id=obj.id,
                        timestamp=time.time(),
                        location=obj.center,
                        confidence=0.7,
                        severity=6,
                        description="Vehicle following too closely",
                        alert_level=AlertLevel.MEDIUM
                    )
        
        return None
    
    def _check_red_light_violation(self, obj: TrackedObject, frame: np.ndarray) -> Optional[TrafficViolation]:
        """Check for red light violations"""
        # This would require traffic light detection
        # Placeholder implementation
        return None
    
    def _detect_accidents(self, tracked_objects: Dict[int, TrackedObject], 
                         timestamp: float) -> List[AccidentEvent]:
        """Detect potential accidents and collisions"""
        accidents = []
        
        # Check for sudden stops or direction changes
        for obj_id, obj in tracked_objects.items():
            accident = self._check_sudden_movement_accident(obj, timestamp)
            if accident:
                accidents.append(accident)
        
        # Check for collisions between objects
        collision_accidents = self._check_collision_accidents(tracked_objects, timestamp)
        accidents.extend(collision_accidents)
        
        # Check for pedestrian accidents
        pedestrian_accidents = self._check_pedestrian_accidents(tracked_objects, timestamp)
        accidents.extend(pedestrian_accidents)
        
        return accidents
    
    def _check_sudden_movement_accident(self, obj: TrackedObject, timestamp: float) -> Optional[AccidentEvent]:
        """Check for accidents based on sudden movement changes"""
        if len(obj.trajectory) < 5:
            return None
        
        # Calculate acceleration changes
        velocities = []
        for i in range(len(obj.trajectory) - 1):
            pos1, time1 = obj.trajectory[i]
            pos2, time2 = obj.trajectory[i + 1]
            
            if time2 != time1:
                velocity = euclidean(pos1, pos2) / (time2 - time1)
                velocities.append(velocity)
        
        if len(velocities) >= 3:
            # Check for sudden deceleration
            recent_velocities = velocities[-3:]
            avg_velocity = np.mean(recent_velocities)
            velocity_change = abs(recent_velocities[-1] - recent_velocities[0])
            
            # Threshold for sudden stop (would need calibration)
            if avg_velocity > 20 and velocity_change > avg_velocity * 0.8:
                return AccidentEvent(
                    accident_type=AccidentType.COLLISION,
                    timestamp=timestamp,
                    involved_objects=[obj.id],
                    location=obj.center,
                    severity_score=0.8,
                    confidence=0.7,
                    description=f"Sudden movement change detected for {obj.object_type}",
                    alert_level=AlertLevel.HIGH,
                    emergency_required=True
                )
        
        return None
    
    def _check_collision_accidents(self, tracked_objects: Dict[int, TrackedObject], 
                                  timestamp: float) -> List[AccidentEvent]:
        """Check for collisions between objects"""
        accidents = []
        checked_pairs = set()
        
        for obj1_id, obj1 in tracked_objects.items():
            for obj2_id, obj2 in tracked_objects.items():
                if obj1_id >= obj2_id:
                    continue
                
                pair = (min(obj1_id, obj2_id), max(obj1_id, obj2_id))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                
                # Check distance and collision potential
                distance = euclidean(obj1.center, obj2.center)
                
                # Calculate combined object size (rough estimate)
                size1 = math.sqrt(obj1.bbox[2] * obj1.bbox[3])
                size2 = math.sqrt(obj2.bbox[2] * obj2.bbox[3])
                min_distance = (size1 + size2) * 0.5
                
                if distance < min_distance:
                    # Potential collision or very close proximity
                    # Check if objects are moving towards each other
                    relative_velocity = euclidean(obj1.velocity, obj2.velocity)
                    
                    if relative_velocity > 5:  # Threshold for concerning interaction
                        accidents.append(AccidentEvent(
                            accident_type=AccidentType.COLLISION,
                            timestamp=timestamp,
                            involved_objects=[obj1_id, obj2_id],
                            location=((obj1.center[0] + obj2.center[0]) // 2,
                                    (obj1.center[1] + obj2.center[1]) // 2),
                            severity_score=min(1.0, relative_velocity / 50),
                            confidence=0.8,
                            description=f"Collision detected between {obj1.object_type} and {obj2.object_type}",
                            alert_level=AlertLevel.CRITICAL,
                            emergency_required=True
                        ))
        
        return accidents
    
    def _check_pedestrian_accidents(self, tracked_objects: Dict[int, TrackedObject], 
                                   timestamp: float) -> List[AccidentEvent]:
        """Check for pedestrian-vehicle accidents"""
        accidents = []
        
        pedestrians = [obj for obj in tracked_objects.values() if obj.object_type == 'person']
        vehicles = [obj for obj in tracked_objects.values() if obj.object_type in ['car', 'truck', 'bus', 'motorcycle']]
        
        for pedestrian in pedestrians:
            for vehicle in vehicles:
                distance = euclidean(pedestrian.center, vehicle.center)
                
                # Calculate danger zone around vehicle
                vehicle_size = math.sqrt(vehicle.bbox[2] * vehicle.bbox[3])
                danger_zone = vehicle_size * 2
                
                if distance < danger_zone:
                    # Check relative speeds
                    ped_speed = euclidean((0, 0), pedestrian.velocity)
                    veh_speed = euclidean((0, 0), vehicle.velocity)
                    
                    if ped_speed > 5 or veh_speed > 10:  # Moving pedestrian or vehicle
                        accidents.append(AccidentEvent(
                            accident_type=AccidentType.PEDESTRIAN_ACCIDENT,
                            timestamp=timestamp,
                            involved_objects=[pedestrian.id, vehicle.id],
                            location=pedestrian.center,
                            severity_score=0.9,
                            confidence=0.7,
                            description=f"Pedestrian-vehicle interaction detected",
                            alert_level=AlertLevel.CRITICAL,
                            emergency_required=True
                        ))
        
        return accidents
    
    def _analyze_traffic_flow(self, tracked_objects: Dict[int, TrackedObject]) -> Dict[str, Any]:
        """Analyze traffic flow metrics"""
        
        vehicles = [obj for obj in tracked_objects.values() 
                   if obj.object_type in ['car', 'truck', 'bus', 'motorcycle']]
        
        if not vehicles:
            return {
                'vehicle_count': 0,
                'average_speed': 0,
                'traffic_density': 0,
                'flow_rate': 0,
                'congestion_level': 'none'
            }
        
        # Calculate average speed
        speeds = []
        for vehicle in vehicles:
            speed = euclidean((0, 0), vehicle.velocity)
            speeds.append(speed)
        
        avg_speed = np.mean(speeds) if speeds else 0
        
        # Calculate traffic density (vehicles per area)
        frame_area = self.processing_resolution[0] * self.processing_resolution[1]
        density = len(vehicles) / (frame_area / 10000)  # vehicles per 100x100 area
        
        # Determine congestion level
        if avg_speed < 5 and density > 0.5:
            congestion = 'heavy'
        elif avg_speed < 15 and density > 0.3:
            congestion = 'moderate'
        elif avg_speed < 25:
            congestion = 'light'
        else:
            congestion = 'free_flow'
        
        return {
            'vehicle_count': len(vehicles),
            'average_speed': avg_speed,
            'traffic_density': density,
            'flow_rate': avg_speed * density,
            'congestion_level': congestion,
            'pedestrian_count': len([obj for obj in tracked_objects.values() if obj.object_type == 'person'])
        }
    
    def _calculate_alert_level(self, violations: List[TrafficViolation], 
                              accidents: List[AccidentEvent]) -> AlertLevel:
        """Calculate overall alert level for the frame"""
        
        if any(acc.alert_level == AlertLevel.CRITICAL for acc in accidents):
            return AlertLevel.CRITICAL
        
        if any(viol.alert_level == AlertLevel.HIGH for viol in violations) or \
           any(acc.alert_level == AlertLevel.HIGH for acc in accidents):
            return AlertLevel.HIGH
        
        if violations or accidents:
            return AlertLevel.MEDIUM
        
        return AlertLevel.LOW
    
    def _create_traffic_light_detector(self):
        """Create traffic light detection system"""
        # Placeholder for traffic light detection
        # Would implement color-based or ML-based traffic light detection
        return None
    
    def set_roi(self, roi_points: List[np.ndarray]):
        """Set regions of interest for focused processing"""
        self.roi_areas = roi_points
        logger.info(f"ROI areas set: {len(roi_points)} regions")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        return {
            'tracked_objects_count': len(self.tracked_objects),
            'violation_history_size': len(self.violation_history),
            'accident_history_size': len(self.accident_history),
            'device': str(self.device),
            'gpu_enabled': self.enable_gpu
        }


# Supporting classes for specialized analysis

class SpeedEstimator:
    """Estimate vehicle speeds from pixel movement"""
    
    def __init__(self):
        self.calibration_factor = 0.1  # pixels per second to km/h
    
    def estimate_speed(self, trajectory: deque) -> float:
        """Estimate speed from trajectory"""
        if len(trajectory) < 2:
            return 0.0
        
        speeds = []
        for i in range(len(trajectory) - 1):
            pos1, time1 = trajectory[i]
            pos2, time2 = trajectory[i + 1]
            
            if time2 != time1:
                distance = euclidean(pos1, pos2)
                speed = distance / (time2 - time1) * self.calibration_factor
                speeds.append(speed)
        
        return np.mean(speeds) if speeds else 0.0


class CollisionPredictor:
    """Predict potential collisions"""
    
    def __init__(self):
        self.prediction_horizon = 3.0  # seconds
    
    def predict_collision(self, obj1: TrackedObject, obj2: TrackedObject) -> Tuple[bool, float]:
        """Predict if two objects might collide"""
        
        # Simple linear prediction
        pos1 = obj1.center
        vel1 = obj1.velocity
        pos2 = obj2.center
        vel2 = obj2.velocity
        
        # Calculate closest approach
        relative_pos = (pos2[0] - pos1[0], pos2[1] - pos1[1])
        relative_vel = (vel2[0] - vel1[0], vel2[1] - vel1[1])
        
        if euclidean((0, 0), relative_vel) < 1e-6:
            # Objects moving at same speed
            return False, float('inf')
        
        # Time of closest approach
        t = -(relative_pos[0] * relative_vel[0] + relative_pos[1] * relative_vel[1]) / \
            (relative_vel[0] ** 2 + relative_vel[1] ** 2)
        
        if t < 0 or t > self.prediction_horizon:
            return False, t
        
        # Distance at closest approach
        closest_distance = euclidean(
            (pos1[0] + vel1[0] * t, pos1[1] + vel1[1] * t),
            (pos2[0] + vel2[0] * t, pos2[1] + vel2[1] * t)
        )
        
        # Object sizes (rough estimate)
        size1 = math.sqrt(obj1.bbox[2] * obj1.bbox[3])
        size2 = math.sqrt(obj2.bbox[2] * obj2.bbox[3])
        collision_threshold = (size1 + size2) * 0.7
        
        return closest_distance < collision_threshold, t


class LaneDetector:
    """Detect lane markings and boundaries"""
    
    def __init__(self):
        self.lane_cache = {}
    
    def detect_lanes(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detect lane markings using edge detection"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Hough line transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=50, maxLineGap=10)
        
        lanes = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Filter for lane-like lines (roughly horizontal)
                angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
                if 10 < angle < 80:  # Lane lines are typically not perfectly horizontal
                    lanes.append(line[0])
        
        return lanes


class TrafficFlowAnalyzer:
    """Analyze traffic flow patterns"""
    
    def __init__(self):
        self.flow_history = deque(maxlen=100)
    
    def analyze_flow(self, tracked_objects: Dict[int, TrackedObject]) -> Dict[str, Any]:
        """Analyze current traffic flow"""
        
        vehicles = [obj for obj in tracked_objects.values() 
                   if obj.object_type in ['car', 'truck', 'bus', 'motorcycle']]
        
        # Calculate flow vectors
        flow_vectors = []
        for vehicle in vehicles:
            if euclidean((0, 0), vehicle.velocity) > 1:
                flow_vectors.append(vehicle.velocity)
        
        if not flow_vectors:
            return {'flow_direction': None, 'flow_consistency': 0}
        
        # Calculate average flow direction
        avg_flow = np.mean(flow_vectors, axis=0)
        
        # Calculate flow consistency (how aligned the vectors are)
        consistency = 0
        if len(flow_vectors) > 1:
            angles = [math.atan2(v[1], v[0]) for v in flow_vectors]
            avg_angle = math.atan2(avg_flow[1], avg_flow[0])
            angle_deviations = [abs(angle - avg_angle) for angle in angles]
            consistency = 1 - (np.mean(angle_deviations) / math.pi)
        
        return {
            'flow_direction': avg_flow.tolist() if isinstance(avg_flow, np.ndarray) else list(avg_flow),
            'flow_consistency': max(0, consistency),
            'vehicle_count': len(vehicles)
        }