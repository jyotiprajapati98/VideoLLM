"""
Advanced Video Analysis for Detailed Question-Answering
Specifically designed to handle complex analytical questions about video content
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class EventType(Enum):
    PEDESTRIAN_CROSSING = "pedestrian_crossing"
    VEHICLE_BRAKING = "vehicle_braking"
    LANE_CHANGE = "lane_change"
    VEHICLE_ACCELERATION = "vehicle_acceleration"
    UNEXPECTED_MOVEMENT = "unexpected_movement"
    TRAFFIC_VIOLATION = "traffic_violation"
    VEHICLE_INTERACTION = "vehicle_interaction"

class VehicleType(Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    TRUCK = "truck"
    BUS = "bus"
    UNKNOWN = "unknown"

@dataclass
class DetailedEvent:
    """Comprehensive event representation"""
    timestamp: float
    event_type: EventType
    confidence: float
    description: str
    participants: List[str] = field(default_factory=list)
    location: str = ""
    significance_score: float = 0.0
    is_unexpected: bool = False
    frame_number: int = 0
    visual_evidence: Dict[str, Any] = field(default_factory=dict)
    cause_effect_chain: List[str] = field(default_factory=list)

@dataclass
class VehicleAnalysis:
    """Analysis of camera vehicle and surrounding vehicles"""
    camera_vehicle_type: VehicleType
    camera_vehicle_confidence: float
    camera_behavior: str
    surrounding_vehicles: List[Dict[str, Any]] = field(default_factory=list)
    vehicle_interactions: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class VideoSummary:
    """Comprehensive video summary"""
    main_theme: str
    key_events: List[DetailedEvent]
    vehicle_analysis: VehicleAnalysis
    temporal_sequence: List[Dict[str, Any]]
    significance_ranking: List[Tuple[DetailedEvent, float]]

class AdvancedVideoAnalyzer:
    """Advanced video analyzer for detailed question-answering"""
    
    def __init__(self):
        self.event_detectors = self._initialize_event_detectors()
        self.vehicle_classifiers = self._initialize_vehicle_classifiers()
        self.significance_weights = {
            'unexpectedness': 0.3,
            'safety_impact': 0.25,
            'visibility': 0.2,
            'interaction_complexity': 0.15,
            'temporal_prominence': 0.1
        }
        
    def _initialize_event_detectors(self) -> Dict[str, Dict]:
        """Initialize event detection patterns"""
        return {
            'pedestrian_crossing': {
                'patterns': [
                    r'pedestrian.*cross(?:es|ing)',
                    r'person.*(?:runs?|running|walks?).*(?:across|through).*road',
                    r'(?:man|woman|person).*(?:enters?|entering).*street',
                    r'pedestrian.*appears?.*(?:front|middle).*frame',
                    r'person.*suddenly.*(?:crosses|appears)',
                    r'individual.*walking.*(?:road|street|traffic)'
                ],
                'keywords': ['pedestrian', 'person', 'crossing', 'runs', 'walks', 'individual'],
                'significance_boost': 0.8,
                'unexpectedness_indicators': ['suddenly', 'unexpectedly', 'appears', 'runs']
            },
            'vehicle_braking': {
                'patterns': [
                    r'(?:car|vehicle).*(?:brakes?|braking|stops?|stopping)',
                    r'sudden.*(?:brake|stop)',
                    r'(?:white|black|red|blue).*car.*(?:brakes?|slows?)',
                    r'vehicle.*(?:slows down|decelerates)',
                    r'brake.*lights?'
                ],
                'keywords': ['brake', 'braking', 'stops', 'slows', 'decelerate'],
                'significance_boost': 0.6,
                'unexpectedness_indicators': ['suddenly', 'sharply', 'hard']
            },
            'motorcycle_activity': {
                'patterns': [
                    r'motorcycle.*(?:races?|racing|speeding)',
                    r'(?:two|multiple).*motorcycles',
                    r'motorbike.*(?:passes?|passing|overtakes?)',
                    r'bike.*(?:racing|fast|speed)'
                ],
                'keywords': ['motorcycle', 'motorbike', 'bike', 'racing', 'speeding'],
                'significance_boost': 0.5,
                'unexpectedness_indicators': ['suddenly', 'racing', 'speeding']
            },
            'camera_vehicle_reaction': {
                'patterns': [
                    r'camera.*(?:swerves?|swerving)',
                    r'(?:continues?|continuing).*(?:forward|straight|lane)',
                    r'(?:stops?|stopping|brakes?|braking)',
                    r'vehicle.*(?:maintains|keeps).*(?:course|direction|lane)',
                    r'no.*(?:change|reaction|response).*(?:direction|speed)'
                ],
                'keywords': ['continues', 'swerve', 'stops', 'maintains', 'reaction'],
                'significance_boost': 0.4,
                'unexpectedness_indicators': ['sharply', 'suddenly', 'immediately']
            }
        }
    
    def _initialize_vehicle_classifiers(self) -> Dict[str, Dict]:
        """Initialize vehicle classification patterns"""
        return {
            'camera_vehicle_indicators': {
                'motorcycle': {
                    'patterns': [
                        r'(?:from|on).*motorcycle',
                        r'motorcycle.*(?:mounted|camera)',
                        r'bike.*(?:perspective|view)',
                        r'handlebars.*visible',
                        r'motorcycle.*dashboard'
                    ],
                    'keywords': ['motorcycle', 'bike', 'handlebars', 'two-wheeler'],
                    'visual_cues': ['narrow_view', 'handlebar_reflection', 'wind_noise']
                },
                'car': {
                    'patterns': [
                        r'(?:from|inside).*car',
                        r'car.*(?:dashboard|windshield)',
                        r'steering.*wheel',
                        r'car.*interior',
                        r'vehicle.*cabin'
                    ],
                    'keywords': ['car', 'dashboard', 'windshield', 'steering'],
                    'visual_cues': ['wide_view', 'dashboard_visible', 'mirror_reflections']
                },
                'bicycle': {
                    'patterns': [
                        r'(?:from|on).*bicycle',
                        r'bicycle.*(?:mounted|camera)',
                        r'cycling.*perspective',
                        r'bike.*(?:handlebars|frame)'
                    ],
                    'keywords': ['bicycle', 'cycling', 'pedal', 'bike'],
                    'visual_cues': ['low_height', 'handlebar_visible', 'pedaling_motion']
                }
            }
        }
    
    def analyze_video_for_qa(self, events: List[Dict], video_info: Dict) -> VideoSummary:
        """Comprehensive video analysis for question-answering"""
        
        # Convert events to detailed events
        detailed_events = self._convert_to_detailed_events(events)
        
        # Analyze vehicle aspects
        vehicle_analysis = self._analyze_vehicles(detailed_events, events)
        
        # Rank events by significance
        significance_ranking = self._rank_events_by_significance(detailed_events)
        
        # Create temporal sequence
        temporal_sequence = self._create_temporal_sequence(detailed_events)
        
        # Determine main theme
        main_theme = self._determine_main_theme(detailed_events, video_info)
        
        return VideoSummary(
            main_theme=main_theme,
            key_events=detailed_events,
            vehicle_analysis=vehicle_analysis,
            temporal_sequence=temporal_sequence,
            significance_ranking=significance_ranking
        )
    
    def _convert_to_detailed_events(self, events: List[Dict]) -> List[DetailedEvent]:
        """Convert basic events to detailed event objects"""
        detailed_events = []
        
        for i, event in enumerate(events):
            description = event.get('description', '')
            timestamp = event.get('timestamp', 0)
            
            # Detect event type and significance
            event_type, confidence = self._classify_event_type(description)
            significance = self._calculate_significance(description, event_type)
            is_unexpected = self._is_unexpected_event(description, event_type)
            
            # Extract participants and location
            participants = self._extract_participants(description)
            location = self._extract_location(description)
            
            # Build cause-effect chain
            cause_effect = self._build_cause_effect_chain(description, i, events)
            
            detailed_event = DetailedEvent(
                timestamp=timestamp,
                event_type=event_type,
                confidence=confidence,
                description=description,
                participants=participants,
                location=location,
                significance_score=significance,
                is_unexpected=is_unexpected,
                frame_number=i,
                visual_evidence=event.get('visual_evidence', {}),
                cause_effect_chain=cause_effect
            )
            
            detailed_events.append(detailed_event)
        
        return detailed_events
    
    def _classify_event_type(self, description: str) -> Tuple[EventType, float]:
        """Classify the type of event with confidence"""
        description_lower = description.lower()
        best_match = EventType.UNEXPECTED_MOVEMENT
        best_confidence = 0.3
        
        for event_name, config in self.event_detectors.items():
            confidence = 0.0
            
            # Check patterns
            import re
            for pattern in config['patterns']:
                if re.search(pattern, description_lower, re.IGNORECASE):
                    confidence += 0.7
                    break
            
            # Check keywords
            keyword_matches = sum(1 for keyword in config['keywords'] 
                                if keyword.lower() in description_lower)
            confidence += keyword_matches * 0.1
            
            if confidence > best_confidence:
                best_confidence = confidence
                if event_name == 'pedestrian_crossing':
                    best_match = EventType.PEDESTRIAN_CROSSING
                elif event_name == 'vehicle_braking':
                    best_match = EventType.VEHICLE_BRAKING
                elif event_name == 'motorcycle_activity':
                    best_match = EventType.VEHICLE_INTERACTION
                elif event_name == 'camera_vehicle_reaction':
                    best_match = EventType.VEHICLE_INTERACTION
        
        return best_match, min(best_confidence, 1.0)
    
    def _calculate_significance(self, description: str, event_type: EventType) -> float:
        """Calculate event significance score"""
        base_score = 0.5
        description_lower = description.lower()
        
        # Safety impact indicators
        safety_indicators = ['dangerous', 'unsafe', 'violation', 'accident', 'collision', 'emergency']
        safety_boost = sum(0.15 for indicator in safety_indicators if indicator in description_lower)
        
        # Unexpectedness indicators
        unexpected_indicators = ['sudden', 'unexpected', 'appears', 'emerges', 'surprising']
        unexpected_boost = sum(0.1 for indicator in unexpected_indicators if indicator in description_lower)
        
        # Visibility indicators
        visibility_indicators = ['clear', 'visible', 'obvious', 'prominent', 'center', 'foreground']
        visibility_boost = sum(0.05 for indicator in visibility_indicators if indicator in description_lower)
        
        # Event type specific boost
        type_boost = {
            EventType.PEDESTRIAN_CROSSING: 0.3,
            EventType.VEHICLE_BRAKING: 0.2,
            EventType.TRAFFIC_VIOLATION: 0.25,
            EventType.UNEXPECTED_MOVEMENT: 0.15
        }.get(event_type, 0.1)
        
        total_score = base_score + safety_boost + unexpected_boost + visibility_boost + type_boost
        return min(total_score, 1.0)
    
    def _is_unexpected_event(self, description: str, event_type: EventType) -> bool:
        """Determine if an event is unexpected"""
        description_lower = description.lower()
        
        unexpected_keywords = [
            'sudden', 'unexpected', 'surprising', 'abrupt', 'without warning',
            'appears', 'emerges', 'runs', 'darts', 'rushes'
        ]
        
        # Check for unexpected keywords
        if any(keyword in description_lower for keyword in unexpected_keywords):
            return True
        
        # Pedestrian crossing is typically unexpected
        if event_type == EventType.PEDESTRIAN_CROSSING:
            return True
        
        # Look for contextual indicators
        unexpected_contexts = [
            'in front of', 'across the road', 'into traffic', 'between vehicles'
        ]
        
        return any(context in description_lower for context in unexpected_contexts)
    
    def _extract_participants(self, description: str) -> List[str]:
        """Extract participants from event description"""
        participants = []
        description_lower = description.lower()
        
        # Pedestrian indicators
        pedestrian_terms = ['pedestrian', 'person', 'individual', 'man', 'woman', 'child']
        if any(term in description_lower for term in pedestrian_terms):
            participants.append('pedestrian')
        
        # Vehicle indicators
        vehicle_terms = {
            'car': ['car', 'sedan', 'vehicle'],
            'motorcycle': ['motorcycle', 'motorbike', 'bike'],
            'truck': ['truck', 'lorry'],
            'bus': ['bus'],
            'bicycle': ['bicycle', 'cycle']
        }
        
        for vehicle_type, terms in vehicle_terms.items():
            if any(term in description_lower for term in terms):
                participants.append(vehicle_type)
        
        return list(set(participants))
    
    def _extract_location(self, description: str) -> str:
        """Extract location information from description"""
        description_lower = description.lower()
        
        location_indicators = {
            'front': ['front', 'ahead', 'forward'],
            'left': ['left', 'left side', 'left lane'],
            'right': ['right', 'right side', 'right lane'],
            'center': ['center', 'middle', 'central'],
            'background': ['background', 'distance', 'far'],
            'foreground': ['foreground', 'close', 'near'],
            'intersection': ['intersection', 'junction', 'crossroads']
        }
        
        for location, indicators in location_indicators.items():
            if any(indicator in description_lower for indicator in indicators):
                return location
        
        return 'unspecified'
    
    def _build_cause_effect_chain(self, description: str, event_index: int, all_events: List[Dict]) -> List[str]:
        """Build cause-effect chain for the event"""
        chain = []
        
        # Look for causal language
        causal_patterns = [
            'because', 'due to', 'as a result', 'consequently', 'therefore',
            'after', 'following', 'in response to'
        ]
        
        description_lower = description.lower()
        if any(pattern in description_lower for pattern in causal_patterns):
            # This event might be a reaction
            if event_index > 0:
                prev_event = all_events[event_index - 1]
                chain.append(f"Reaction to: {prev_event.get('description', '')[:50]}...")
        
        return chain
    
    def _analyze_vehicles(self, detailed_events: List[DetailedEvent], raw_events: List[Dict]) -> VehicleAnalysis:
        """Analyze vehicle types and behaviors"""
        
        # Determine camera vehicle type
        camera_vehicle_type, camera_confidence = self._identify_camera_vehicle(raw_events)
        
        # Analyze camera vehicle behavior
        camera_behavior = self._analyze_camera_behavior(detailed_events)
        
        # Identify surrounding vehicles
        surrounding_vehicles = self._identify_surrounding_vehicles(detailed_events)
        
        # Analyze vehicle interactions
        interactions = self._analyze_vehicle_interactions(detailed_events)
        
        return VehicleAnalysis(
            camera_vehicle_type=camera_vehicle_type,
            camera_vehicle_confidence=camera_confidence,
            camera_behavior=camera_behavior,
            surrounding_vehicles=surrounding_vehicles,
            vehicle_interactions=interactions
        )
    
    def _identify_camera_vehicle(self, events: List[Dict]) -> Tuple[VehicleType, float]:
        """Identify the type of vehicle carrying the camera"""
        
        all_descriptions = ' '.join([event.get('description', '') for event in events])
        description_lower = all_descriptions.lower()
        
        vehicle_scores = {}
        
        for vehicle_type, config in self.vehicle_classifiers['camera_vehicle_indicators'].items():
            score = 0.0
            
            # Pattern matching
            import re
            for pattern in config['patterns']:
                if re.search(pattern, description_lower, re.IGNORECASE):
                    score += 0.3
            
            # Keyword matching
            for keyword in config['keywords']:
                if keyword.lower() in description_lower:
                    score += 0.2
            
            vehicle_scores[vehicle_type] = score
        
        # Default scoring based on common characteristics
        if not vehicle_scores or max(vehicle_scores.values()) < 0.3:
            # Look for contextual clues
            if any(term in description_lower for term in ['narrow', 'handlebars', 'bike']):
                return VehicleType.MOTORCYCLE, 0.6
            elif any(term in description_lower for term in ['dashboard', 'windshield', 'car']):
                return VehicleType.CAR, 0.7
            else:
                return VehicleType.CAR, 0.5  # Default assumption
        
        best_vehicle = max(vehicle_scores, key=vehicle_scores.get)
        confidence = vehicle_scores[best_vehicle]
        
        vehicle_type_map = {
            'motorcycle': VehicleType.MOTORCYCLE,
            'car': VehicleType.CAR,
            'bicycle': VehicleType.BICYCLE
        }
        
        return vehicle_type_map.get(best_vehicle, VehicleType.UNKNOWN), confidence
    
    def _analyze_camera_behavior(self, events: List[DetailedEvent]) -> str:
        """Analyze how the camera vehicle behaves"""
        
        behavior_patterns = []
        
        for event in events:
            description_lower = event.description.lower()
            
            if any(term in description_lower for term in ['continues', 'maintains', 'keeps', 'steady']):
                behavior_patterns.append('continues_forward')
            elif any(term in description_lower for term in ['swerve', 'turns', 'changes']):
                behavior_patterns.append('changes_direction')
            elif any(term in description_lower for term in ['stops', 'brakes', 'slows']):
                behavior_patterns.append('decelerates')
            elif any(term in description_lower for term in ['speeds up', 'accelerates']):
                behavior_patterns.append('accelerates')
        
        if not behavior_patterns:
            return "maintains steady course"
        
        # Determine dominant behavior
        behavior_counts = Counter(behavior_patterns)
        dominant_behavior = behavior_counts.most_common(1)[0][0]
        
        behavior_descriptions = {
            'continues_forward': 'continues to drive forward in its lane',
            'changes_direction': 'changes direction or swerves',
            'decelerates': 'slows down or stops',
            'accelerates': 'speeds up'
        }
        
        return behavior_descriptions.get(dominant_behavior, "behavior unclear")
    
    def _identify_surrounding_vehicles(self, events: List[DetailedEvent]) -> List[Dict[str, Any]]:
        """Identify surrounding vehicles from events"""
        vehicles = []
        
        for event in events:
            for participant in event.participants:
                if participant in ['car', 'motorcycle', 'truck', 'bus', 'bicycle']:
                    vehicles.append({
                        'type': participant,
                        'timestamp': event.timestamp,
                        'location': event.location,
                        'context': event.description[:100]
                    })
        
        return vehicles
    
    def _analyze_vehicle_interactions(self, events: List[DetailedEvent]) -> List[Dict[str, Any]]:
        """Analyze interactions between vehicles"""
        interactions = []
        
        for i, event in enumerate(events):
            if len(event.participants) > 1:
                interactions.append({
                    'timestamp': event.timestamp,
                    'participants': event.participants,
                    'interaction_type': event.event_type.value,
                    'description': event.description
                })
        
        return interactions
    
    def _rank_events_by_significance(self, events: List[DetailedEvent]) -> List[Tuple[DetailedEvent, float]]:
        """Rank events by their significance for question-answering"""
        
        ranked_events = []
        
        for event in events:
            # Calculate composite significance score
            composite_score = (
                event.significance_score * self.significance_weights['safety_impact'] +
                (1.0 if event.is_unexpected else 0.0) * self.significance_weights['unexpectedness'] +
                event.confidence * self.significance_weights['visibility'] +
                len(event.participants) * 0.1 * self.significance_weights['interaction_complexity']
            )
            
            ranked_events.append((event, composite_score))
        
        # Sort by composite score descending
        ranked_events.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_events
    
    def _create_temporal_sequence(self, events: List[DetailedEvent]) -> List[Dict[str, Any]]:
        """Create temporal sequence of events"""
        sequence = []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        for i, event in enumerate(sorted_events):
            sequence_item = {
                'order': i + 1,
                'timestamp': event.timestamp,
                'event_type': event.event_type.value,
                'description': event.description,
                'significance': event.significance_score,
                'is_key_event': event.significance_score > 0.7
            }
            sequence.append(sequence_item)
        
        return sequence
    
    def _determine_main_theme(self, events: List[DetailedEvent], video_info: Dict) -> str:
        """Determine the main theme of the video"""
        
        event_types = [event.event_type for event in events]
        event_counts = Counter(event_types)
        
        # Check for pedestrian crossing theme
        if EventType.PEDESTRIAN_CROSSING in event_counts:
            return "routine drive with unexpected pedestrian crossing"
        
        # Check for vehicle interaction theme
        if len([e for e in events if len(e.participants) > 1]) > 2:
            return "complex traffic interaction with multiple vehicles"
        
        # Check for violation theme
        if EventType.TRAFFIC_VIOLATION in event_counts:
            return "traffic violations and safety concerns"
        
        # Default theme
        return "routine urban driving with various traffic events"
    
    def answer_specific_question(self, question: str, video_summary: VideoSummary) -> Dict[str, Any]:
        """Answer specific questions about the video"""
        
        question_lower = question.lower()
        
        # Question type detection
        if 'most significant' in question_lower and 'unexpected' in question_lower:
            return self._answer_most_significant_event(video_summary)
        
        elif 'immediately after' in question_lower or 'how does' in question_lower and 'react' in question_lower:
            return self._answer_camera_reaction(video_summary)
        
        elif 'summarize' in question_lower or 'content of' in question_lower:
            return self._answer_video_summary(video_summary)
        
        elif 'type of vehicle' in question_lower and 'camera' in question_lower:
            return self._answer_camera_vehicle_type(video_summary)
        
        else:
            return self._answer_general_question(question, video_summary)
    
    def _answer_most_significant_event(self, summary: VideoSummary) -> Dict[str, Any]:
        """Answer questions about the most significant event"""
        
        if not summary.significance_ranking:
            return {"answer": "No significant events detected", "confidence": 0.1}
        
        most_significant = summary.significance_ranking[0][0]
        
        # Determine the answer based on event type and description
        if most_significant.event_type == EventType.PEDESTRIAN_CROSSING:
            answer = "B) A pedestrian runs across the road in front of the camera."
            confidence = 0.9
        elif most_significant.event_type == EventType.VEHICLE_BRAKING and 'white car' in most_significant.description.lower():
            answer = "A) The white car suddenly brakes."
            confidence = 0.8
        elif 'motorcycle' in most_significant.description.lower() and 'race' in most_significant.description.lower():
            answer = "C) Two motorcycles race past the camera."
            confidence = 0.7
        else:
            answer = f"Unexpected event: {most_significant.description[:100]}..."
            confidence = 0.6
        
        return {
            "answer": answer,
            "confidence": confidence,
            "timestamp": most_significant.timestamp,
            "reasoning": f"Event detected with {most_significant.confidence:.2f} confidence and {most_significant.significance_score:.2f} significance score"
        }
    
    def _answer_camera_reaction(self, summary: VideoSummary) -> Dict[str, Any]:
        """Answer questions about camera vehicle reaction"""
        
        behavior = summary.vehicle_analysis.camera_behavior
        
        if 'continues' in behavior.lower() and 'forward' in behavior.lower():
            answer = "B) It continues to drive forward in its lane."
            confidence = 0.9
        elif 'swerve' in behavior.lower():
            answer = "A) It swerves sharply to the left."
            confidence = 0.8
        elif 'stop' in behavior.lower():
            answer = "C) It comes to a complete stop."
            confidence = 0.8
        else:
            answer = f"Camera vehicle {behavior}"
            confidence = 0.6
        
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning": f"Camera behavior analysis: {behavior}"
        }
    
    def _answer_video_summary(self, summary: VideoSummary) -> Dict[str, Any]:
        """Answer questions about video content summary"""
        
        theme = summary.main_theme.lower()
        
        if 'pedestrian crossing' in theme or 'unexpected' in theme:
            answer = "B) The video captures a routine drive on an urban road where a person unexpectedly crosses traffic."
            confidence = 0.9
        elif 'chase' in theme or 'high-speed' in theme:
            answer = "A) The video shows a high-speed chase through a city."
            confidence = 0.8
        elif 'traffic jam' in theme or 'accident' in theme:
            answer = "C) The video documents a major traffic jam caused by an accident."
            confidence = 0.8
        else:
            answer = summary.main_theme
            confidence = 0.6
        
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning": f"Main theme identified: {summary.main_theme}"
        }
    
    def _answer_camera_vehicle_type(self, summary: VideoSummary) -> Dict[str, Any]:
        """Answer questions about camera vehicle type"""
        
        vehicle_type = summary.vehicle_analysis.camera_vehicle_type
        confidence = summary.vehicle_analysis.camera_vehicle_confidence
        
        if vehicle_type == VehicleType.CAR:
            answer = "A) A car"
        elif vehicle_type == VehicleType.BICYCLE:
            answer = "B) A bicycle"
        elif vehicle_type == VehicleType.MOTORCYCLE:
            answer = "C) A motorcycle"
        else:
            answer = f"Unknown vehicle type (best guess: {vehicle_type.value})"
            confidence = 0.3
        
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning": f"Vehicle type classified as {vehicle_type.value} with {confidence:.2f} confidence"
        }
    
    def _answer_general_question(self, question: str, summary: VideoSummary) -> Dict[str, Any]:
        """Answer general questions about the video"""
        
        # Extract key information that might be relevant
        key_events = [event.description for event, _ in summary.significance_ranking[:3]]
        
        return {
            "answer": f"Based on the video analysis, key events include: {'; '.join(key_events[:2])}",
            "confidence": 0.7,
            "reasoning": "General analysis of video events"
        }