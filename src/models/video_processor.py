import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
# Note: AutoModelForVision2Seq is deprecated, but kept for compatibility with SmolVLM models
from PIL import Image
import json
from typing import List, Dict, Any
from ..utils.video_utils import extract_frames, get_video_info, cleanup_frames
from ..utils.traffic_analyzer import TrafficAnalyzer

class VideoProcessor:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-500M-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.traffic_analyzer = TrafficAnalyzer()  # Initialize advanced traffic analyzer
        self.fallback_models = [
            "HuggingFaceTB/SmolVLM-500M-Instruct",
            "HuggingFaceTB/SmolVLM-256M-Instruct", 
            "microsoft/Florence-2-base-ft",
            "Salesforce/blip2-opt-2.7b"
        ]
        self._load_model()
    
    def _load_model(self):
        """Load SmolVLM model and processor with fallbacks"""
        models_to_try = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
        
        for model_name in models_to_try:
            try:
                print(f"Trying to load {model_name} on {self.device}...")
                
                # Try to load processor first (lighter operation)
                self.processor = AutoProcessor.from_pretrained(model_name)
                
                # Then load the model
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                self.model_name = model_name  # Update to successfully loaded model
                print(f"✅ Model loaded successfully: {model_name}")
                return
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        
        # If all models fail, raise error
        raise Exception("❌ Failed to load any vision model. Please check your internet connection and try again.")
    
    def analyze_frame(self, image_path: str, prompt: str) -> str:
        """Analyze a single frame with SmolVLM2"""
        try:
            image = Image.open(image_path).convert("RGB")
            
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
            
            # Apply chat template
            formatted_prompt = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            
            # Prepare inputs with formatted prompt
            inputs = self.processor(
                text=formatted_prompt,
                images=[image],
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response with optimized parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,  # Reduced for faster processing
                    do_sample=False,     # Deterministic for speed
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    num_beams=1,         # No beam search for speed
                    early_stopping=True
                )
            
            # Decode response
            response = self.processor.decode(
                generated_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
        
        except Exception as e:
            print(f"Error analyzing frame {image_path}: {e}")
            return f"Error analyzing frame: {str(e)}"
    
    def process_video(self, video_path: str, temp_dir: str) -> Dict[str, Any]:
        """Process entire video and extract events"""
        try:
            # Get video info
            video_info = get_video_info(video_path)
            print(f"Processing video: {video_info['duration']:.2f}s, {video_info['fps']:.2f} fps")
            
            # Extract frames - optimize sampling rate based on video duration
            if video_info['duration'] <= 30:
                sampling_rate = 3  # Every 3 seconds for short videos
            elif video_info['duration'] <= 60:
                sampling_rate = 5  # Every 5 seconds for medium videos
            else:
                sampling_rate = 10  # Every 10 seconds for long videos
            
            # Cap maximum frames to prevent timeouts
            max_frames = 8
            frame_paths = extract_frames(video_path, temp_dir, sampling_rate=sampling_rate)
            
            # Limit frames if too many
            if len(frame_paths) > max_frames:
                # Keep evenly distributed frames
                step = len(frame_paths) // max_frames
                frame_paths = frame_paths[::step][:max_frames]
                
            print(f"Extracted {len(frame_paths)} frames (sampling rate: {sampling_rate}s)")
            
            # Analyze frames for events
            events = []
            for i, frame_path in enumerate(frame_paths):
                print(f"Processing frame {i+1}/{len(frame_paths)}...")
                
                # Extract timestamp from filename
                filename = os.path.basename(frame_path)
                timestamp = float(filename.split('_t')[1].replace('.jpg', ''))
                
                # Enhanced prompt for comprehensive traffic analysis
                event_prompt = """You are an expert traffic analyst. Analyze this traffic scene carefully and provide:

1. VEHICLES: List all vehicles visible (cars, trucks, motorcycles, buses, etc.) with their colors and positions
2. TRAFFIC VIOLATIONS: Identify any violations like:
   - Running red lights or stop signs
   - Wrong-way driving
   - Illegal parking or stopping
   - Speeding or reckless driving
   - Lane violations or improper turns
   - Not yielding to pedestrians
3. PEDESTRIANS: Describe pedestrian activity and any safety issues
4. TRAFFIC INFRASTRUCTURE: Note traffic lights, signs, road markings, crosswalks
5. SAFETY CONCERNS: Any potential hazards or dangerous situations

Be specific about locations (left/right side, foreground/background) and provide detailed observations."""
                
                try:
                    event_description = self.analyze_frame(frame_path, event_prompt)
                    
                    if event_description and "error" not in event_description.lower() and len(event_description) > 10:
                        # Use advanced traffic analyzer for comprehensive analysis
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
                except Exception as e:
                    print(f"Skipping frame {i} due to error: {e}")
                    continue
            
            # Generate comprehensive traffic analysis summary using advanced analyzer
            if events:
                # Use traffic analyzer to generate comprehensive summary
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
            
            # Clean up frames
            cleanup_frames(frame_paths)
            
            return {
                "video_info": video_info,
                "events": events,
                "summary": summary,
                "total_frames_analyzed": len(events),
                "traffic_statistics": {
                    "total_violations": total_violations if events else 0,
                    "total_vehicles": total_vehicles if events else 0,
                    "avg_vehicles_per_frame": avg_vehicles_per_frame if events else 0,
                    "frames_with_violations": frames_with_violations if events else 0,
                    "safety_score": avg_safety_score if events else 5,
                    "violation_types": violation_types if events else []
                }
            }
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return {
                "error": str(e),
                "video_info": {},
                "events": [],
                "summary": "Error processing video",
                "total_frames_analyzed": 0,
                "traffic_statistics": {
                    "total_violations": 0,
                    "total_vehicles": 0,
                    "avg_vehicles_per_frame": 0,
                    "frames_with_violations": 0,
                    "safety_score": 0,
                    "violation_types": []
                }
            }