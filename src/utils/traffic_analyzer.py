"""
Advanced Traffic Analysis Utilities
Provides specialized functions for traffic violation detection and vehicle analysis
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TrafficViolation:
    """Data class for traffic violations"""
    type: str
    severity: int  # 1-10 scale
    description: str
    confidence: float  # 0-1 scale
    timestamp: Optional[float] = None

@dataclass 
class VehicleDetection:
    """Data class for vehicle detection"""
    vehicle_type: str
    count: int
    position: str  # e.g., "left lane", "intersection", "background"
    behavior: str  # e.g., "normal", "aggressive", "stopped"
    color: Optional[str] = None

class TrafficAnalyzer:
    """Advanced traffic analysis with rule-based detection"""
    
    def __init__(self):
        self.violation_patterns = self._initialize_violation_patterns()
        self.vehicle_patterns = self._initialize_vehicle_patterns()
        self.safety_keywords = self._initialize_safety_keywords()
    
    def _initialize_violation_patterns(self) -> Dict[str, Dict]:
        """Initialize comprehensive traffic violation detection patterns"""
        return {
            "red_light_violation": {
                "patterns": [
                    r"running?\s+(?:the\s+)?red\s+light",
                    r"ignore[sd]?\s+(?:the\s+)?(?:red\s+)?signal",
                    r"ran\s+(?:the\s+)?red",
                    r"drove\s+through\s+red",
                    r"violated?\s+(?:the\s+)?traffic\s+signal"
                ],
                "severity": 9,
                "keywords": ["red light", "signal", "intersection"],
                "context_required": ["intersection", "traffic light", "signal"]
            },
            "speeding": {
                "patterns": [
                    r"speeding|excessive\s+speed|driving\s+(?:too\s+)?fast",
                    r"over\s+(?:the\s+)?speed\s+limit",
                    r"reckless\s+driving",
                    r"dangerous\s+speed"
                ],
                "severity": 7,
                "keywords": ["speed", "fast", "reckless"],
                "context_required": []
            },
            "wrong_way_driving": {
                "patterns": [
                    r"wrong\s+(?:way|direction)",
                    r"driving\s+(?:in\s+)?opposite\s+direction",
                    r"against\s+(?:the\s+)?traffic(?:\s+flow)?",
                    r"head\s*-?\s*on\s+collision\s+risk"
                ],
                "severity": 10,
                "keywords": ["wrong way", "opposite", "against traffic"],
                "context_required": []
            },
            "illegal_parking": {
                "patterns": [
                    r"illegal(?:ly)?\s+park(?:ed|ing)",
                    r"block(?:ed|ing)\s+(?:the\s+)?(?:lane|road|traffic)",
                    r"double\s+park(?:ed|ing)",
                    r"park(?:ed|ing)\s+in\s+(?:no\s+parking|restricted)\s+(?:zone|area)",
                    r"obstruct(?:ed|ing)\s+traffic"
                ],
                "severity": 5,
                "keywords": ["parking", "blocked", "obstruct"],
                "context_required": []
            },
            "lane_violation": {
                "patterns": [
                    r"(?:improper|illegal|wrong)\s+lane\s+(?:change|usage)",
                    r"cut\s+off|cutting\s+off",
                    r"weaving\s+between\s+lanes",
                    r"not\s+(?:using\s+)?turn\s+signal",
                    r"illegal\s+(?:left|right)\s+turn"
                ],
                "severity": 6,
                "keywords": ["lane", "turn", "signal"],
                "context_required": []
            },
            "pedestrian_violation": {
                "patterns": [
                    r"(?:not\s+)?yield(?:ed|ing)\s+to\s+pedestrian",
                    r"hit\s+(?:a\s+)?pedestrian",
                    r"ignore[sd]?\s+crosswalk",
                    r"pedestrian\s+(?:right\s+of\s+way|safety)\s+violation"
                ],
                "severity": 9,
                "keywords": ["pedestrian", "crosswalk", "yield"],
                "context_required": ["pedestrian"]
            },
            "stop_sign_violation": {
                "patterns": [
                    r"(?:ran|running)\s+(?:the\s+)?stop\s+sign",
                    r"roll(?:ed|ing)\s+(?:through\s+)?(?:the\s+)?stop",
                    r"ignore[sd]?\s+(?:the\s+)?stop\s+sign",
                    r"fail(?:ed|ure)\s+to\s+stop"
                ],
                "severity": 8,
                "keywords": ["stop sign", "stop", "intersection"],
                "context_required": ["stop sign", "intersection"]
            },
            "aggressive_driving": {
                "patterns": [
                    r"aggressive\s+driving|road\s+rage",
                    r"tailgating|following\s+too\s+close(?:ly)?",
                    r"dangerous\s+maneuver",
                    r"erratic\s+driving",
                    r"honking\s+aggressively"
                ],
                "severity": 7,
                "keywords": ["aggressive", "dangerous", "erratic"],
                "context_required": []
            }
        }
    
    def _initialize_vehicle_patterns(self) -> Dict[str, List[str]]:
        """Initialize vehicle detection patterns"""
        return {
            "car": [r"car[s]?", r"sedan[s]?", r"hatchback[s]?", r"coupe[s]?"],
            "truck": [r"truck[s]?", r"pickup[s]?", r"lorr(?:y|ies)", r"freight"],
            "bus": [r"bus(?:es)?", r"coach(?:es)?", r"transit"],
            "motorcycle": [r"motorcycle[s]?", r"motorbike[s]?", r"bike[s]?", r"scooter[s]?"],
            "van": [r"van[s]?", r"minivan[s]?"],
            "taxi": [r"taxi[s]?", r"cab[s]?"],
            "emergency": [r"ambulance[s]?", r"fire\s+truck[s]?", r"police\s+car[s]?"],
            "bicycle": [r"bicycle[s]?", r"bike[s]?", r"cyclist[s]?"]
        }
    
    def _initialize_safety_keywords(self) -> Dict[str, int]:
        """Initialize safety concern keywords with severity scores"""
        return {
            "collision": 10, "crash": 10, "accident": 10,
            "near miss": 8, "close call": 8,
            "dangerous": 7, "hazardous": 7, "unsafe": 7,
            "emergency": 9, "siren": 6,
            "brake suddenly": 6, "hard brake": 6,
            "swerve": 6, "dodge": 6,
            "blocked": 4, "congestion": 3, "traffic jam": 3
        }
    
    def analyze_traffic_description(self, description: str, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Comprehensively analyze a traffic description for violations, vehicles, and safety
        
        Args:
            description: Text description of traffic scene
            timestamp: Optional timestamp for the analysis
            
        Returns:
            Dict containing violations, vehicles, safety score, and other metrics
        """
        description_lower = description.lower()
        
        # Detect violations
        violations = self._detect_violations(description_lower)
        
        # Count and categorize vehicles
        vehicles = self._detect_vehicles(description_lower)
        
        # Assess safety
        safety_analysis = self._assess_safety(description_lower, violations)
        
        # Extract contextual information
        context = self._extract_context(description_lower)
        
        return {
            "violations": violations,
            "vehicles": vehicles,
            "safety_score": safety_analysis["score"],
            "safety_concerns": safety_analysis["concerns"],
            "vehicle_count": sum(v.count for v in vehicles),
            "context": context,
            "timestamp": timestamp,
            "risk_level": self._calculate_risk_level(violations, safety_analysis["score"])
        }
    
    def _detect_violations(self, description: str) -> List[TrafficViolation]:
        """Detect traffic violations using pattern matching"""
        violations = []
        
        for violation_type, config in self.violation_patterns.items():
            confidence = 0.0
            matched_pattern = None
            
            # Check patterns
            for pattern in config["patterns"]:
                if re.search(pattern, description, re.IGNORECASE):
                    confidence = max(confidence, 0.8)
                    matched_pattern = pattern
                    break
            
            # Check keywords for additional confidence
            keyword_matches = sum(1 for keyword in config["keywords"] 
                                if keyword.lower() in description)
            if keyword_matches > 0:
                confidence += 0.1 * keyword_matches
            
            # Check context requirements
            if config["context_required"]:
                context_matches = sum(1 for context in config["context_required"]
                                    if context.lower() in description)
                if context_matches == 0:
                    confidence *= 0.5  # Reduce confidence if context missing
            
            # Create violation if confidence is sufficient
            if confidence >= 0.3:
                violations.append(TrafficViolation(
                    type=violation_type.replace("_", " "),
                    severity=config["severity"],
                    description=f"Detected via pattern: {matched_pattern}" if matched_pattern else "Detected via keywords",
                    confidence=min(confidence, 1.0)
                ))
        
        return violations
    
    def _detect_vehicles(self, description: str) -> List[VehicleDetection]:
        """Detect and count vehicles in the description"""
        vehicles = []
        
        for vehicle_type, patterns in self.vehicle_patterns.items():
            total_count = 0
            position = "unspecified"
            behavior = "normal"
            color = None
            
            # Count vehicles using patterns
            for pattern in patterns:
                matches = re.findall(pattern, description, re.IGNORECASE)
                total_count += len(matches)
                
                # Look for quantity descriptors
                quantity_pattern = rf"(\d+|several|many|multiple|few|some)\s+{pattern}"
                quantity_matches = re.findall(quantity_pattern, description, re.IGNORECASE)
                
                for quantity in quantity_matches:
                    if quantity.isdigit():
                        total_count += int(quantity)
                    else:
                        quantity_map = {
                            'several': 3, 'many': 5, 'multiple': 3,
                            'few': 2, 'some': 2
                        }
                        total_count += quantity_map.get(quantity.lower(), 1)
            
            # Extract position information
            position_keywords = {
                'left': ['left lane', 'left side', 'on the left'],
                'right': ['right lane', 'right side', 'on the right'],
                'center': ['center lane', 'middle lane', 'center'],
                'intersection': ['intersection', 'junction', 'crossroads'],
                'background': ['background', 'distance', 'far'],
                'foreground': ['foreground', 'front', 'close']
            }
            
            for pos, keywords in position_keywords.items():
                if any(keyword in description for keyword in keywords):
                    position = pos
                    break
            
            # Extract behavior information
            behavior_keywords = {
                'aggressive': ['aggressive', 'reckless', 'dangerous'],
                'stopped': ['stopped', 'stationary', 'parked'],
                'turning': ['turning', 'turn'],
                'speeding': ['speeding', 'fast', 'racing'],
                'normal': ['normal', 'steady', 'flowing']
            }
            
            for behav, keywords in behavior_keywords.items():
                if any(keyword in description for keyword in keywords):
                    behavior = behav
                    break
            
            # Extract color information
            colors = ['white', 'black', 'red', 'blue', 'green', 'yellow', 'silver', 'gray', 'grey']
            for color_name in colors:
                if f"{color_name} {vehicle_type}" in description:
                    color = color_name
                    break
            
            if total_count > 0:
                vehicles.append(VehicleDetection(
                    vehicle_type=vehicle_type,
                    count=min(total_count, 50),  # Cap at reasonable number
                    position=position,
                    behavior=behavior,
                    color=color
                ))
        
        return vehicles
    
    def _assess_safety(self, description: str, violations: List[TrafficViolation]) -> Dict[str, Any]:
        """Assess overall safety of the traffic situation"""
        base_score = 10
        concerns = []
        
        # Deduct points for violations
        for violation in violations:
            deduction = (violation.severity * violation.confidence) / 10
            base_score -= deduction
            concerns.append(f"{violation.type} (severity: {violation.severity})")
        
        # Check for safety keywords
        for keyword, severity in self.safety_keywords.items():
            if keyword in description:
                deduction = severity * 0.3
                base_score -= deduction
                concerns.append(f"Safety concern: {keyword}")
        
        # Ensure score stays within bounds
        safety_score = max(1, min(10, base_score))
        
        return {
            "score": round(safety_score, 1),
            "concerns": concerns[:10]  # Limit to top 10 concerns
        }
    
    def _extract_context(self, description: str) -> Dict[str, Any]:
        """Extract contextual information about the traffic scene"""
        context = {
            "time_of_day": "unknown",
            "weather": "unknown",
            "location_type": "unknown",
            "traffic_density": "unknown"
        }
        
        # Time of day
        time_keywords = {
            "morning": ["morning", "dawn", "sunrise"],
            "afternoon": ["afternoon", "midday", "noon"],
            "evening": ["evening", "dusk", "sunset"],
            "night": ["night", "dark", "nighttime"]
        }
        
        for time, keywords in time_keywords.items():
            if any(keyword in description for keyword in keywords):
                context["time_of_day"] = time
                break
        
        # Weather conditions
        weather_keywords = {
            "sunny": ["sunny", "clear", "bright"],
            "rainy": ["rain", "raining", "wet", "storm"],
            "cloudy": ["cloudy", "overcast", "gray sky"],
            "snowy": ["snow", "snowing", "winter"]
        }
        
        for weather, keywords in weather_keywords.items():
            if any(keyword in description for keyword in keywords):
                context["weather"] = weather
                break
        
        # Location type
        location_keywords = {
            "urban": ["city", "downtown", "urban", "buildings"],
            "suburban": ["suburban", "residential", "neighborhood"],
            "highway": ["highway", "freeway", "expressway", "motorway"],
            "intersection": ["intersection", "junction", "crossroads"]
        }
        
        for location, keywords in location_keywords.items():
            if any(keyword in description for keyword in keywords):
                context["location_type"] = location
                break
        
        # Traffic density
        density_keywords = {
            "heavy": ["heavy traffic", "congested", "busy", "crowded"],
            "moderate": ["moderate traffic", "steady flow"],
            "light": ["light traffic", "sparse", "few vehicles"]
        }
        
        for density, keywords in density_keywords.items():
            if any(keyword in description for keyword in keywords):
                context["traffic_density"] = density
                break
        
        return context
    
    def _calculate_risk_level(self, violations: List[TrafficViolation], safety_score: float) -> str:
        """Calculate overall risk level"""
        if safety_score >= 8 and len(violations) == 0:
            return "LOW"
        elif safety_score >= 6 and len(violations) <= 1:
            return "MODERATE"
        elif safety_score >= 4:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def generate_traffic_summary(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive summary from multiple frame analyses"""
        if not analyses:
            return {}
        
        total_violations = []
        total_vehicles = []
        safety_scores = []
        contexts = []
        
        for analysis in analyses:
            total_violations.extend(analysis.get("violations", []))
            total_vehicles.extend(analysis.get("vehicles", []))
            safety_scores.append(analysis.get("safety_score", 5))
            contexts.append(analysis.get("context", {}))
        
        # Aggregate statistics
        violation_types = list(set(v.type for v in total_violations))
        vehicle_counts = {}
        for vehicle in total_vehicles:
            vehicle_counts[vehicle.vehicle_type] = vehicle_counts.get(vehicle.vehicle_type, 0) + vehicle.count
        
        avg_safety_score = sum(safety_scores) / len(safety_scores) if safety_scores else 5
        
        return {
            "total_violations": len(total_violations),
            "unique_violation_types": violation_types,
            "vehicle_summary": vehicle_counts,
            "average_safety_score": round(avg_safety_score, 1),
            "frames_analyzed": len(analyses),
            "overall_risk": self._calculate_risk_level(total_violations, avg_safety_score)
        }