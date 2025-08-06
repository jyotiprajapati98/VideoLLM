import json
import redis
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime, timedelta

class ChatManager:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.redis_client = None
        self.fallback_models = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-small",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "microsoft/phi-2",
            "distilgpt2"
        ]
        self._setup_redis()
        self._load_model()
    
    def _setup_redis(self):
        """Setup Redis connection for conversation memory"""
        try:
            self.redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                db=0, 
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            print("Connected to Redis successfully!")
        except Exception as e:
            print(f"Redis connection failed: {e}. Using in-memory storage.")
            self.redis_client = None
            self.memory_store = {}
    
    def _load_model(self):
        """Load chat model and tokenizer with fallbacks"""
        models_to_try = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
        
        for model_name in models_to_try:
            try:
                print(f"Trying to load chat model {model_name} on {self.device}...")
                
                # Load tokenizer first
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Set pad token if not exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                self.model_name = model_name  # Update to successfully loaded model
                print(f"✅ Chat model loaded successfully: {model_name}")
                return
                
            except Exception as e:
                print(f"❌ Failed to load chat model {model_name}: {e}")
                continue
        
        # If all models fail, raise error
        raise Exception("❌ Failed to load any chat model. Please check your internet connection and try again.")
    
    def _get_conversation_key(self, session_id: str) -> str:
        """Generate Redis key for conversation"""
        return f"conversation:{session_id}"
    
    def _store_conversation(self, session_id: str, conversation: List[Dict]):
        """Store conversation in Redis or memory"""
        if self.redis_client:
            try:
                key = self._get_conversation_key(session_id)
                self.redis_client.setex(
                    key, 
                    timedelta(hours=24), 
                    json.dumps(conversation)
                )
            except Exception as e:
                print(f"Redis storage error: {e}")
        else:
            self.memory_store[session_id] = conversation
    
    def _retrieve_conversation(self, session_id: str) -> List[Dict]:
        """Retrieve conversation from Redis or memory"""
        if self.redis_client:
            try:
                key = self._get_conversation_key(session_id)
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                print(f"Redis retrieval error: {e}")
        else:
            return self.memory_store.get(session_id, [])
        
        return []
    
    def _store_video_analysis(self, session_id: str, analysis: Dict[str, Any]):
        """Store video analysis results"""
        if self.redis_client:
            try:
                key = f"video_analysis:{session_id}"
                self.redis_client.setex(
                    key,
                    timedelta(hours=24),
                    json.dumps(analysis)
                )
            except Exception as e:
                print(f"Redis video storage error: {e}")
        else:
            self.memory_store[f"video_{session_id}"] = analysis
    
    def _retrieve_video_analysis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve video analysis results"""
        if self.redis_client:
            try:
                key = f"video_analysis:{session_id}"
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                print(f"Redis video retrieval error: {e}")
        else:
            return self.memory_store.get(f"video_{session_id}")
        
        return None
    
    def initialize_session(self, session_id: str, video_analysis: Dict[str, Any]):
        """Initialize a new chat session with video analysis"""
        self._store_video_analysis(session_id, video_analysis)
        
        # Create enhanced system message with traffic statistics
        traffic_stats = video_analysis.get('traffic_statistics', {})
        system_message = f"""You are an expert traffic video analysis assistant with advanced detection capabilities. You have comprehensively analyzed a traffic video with the following information:

📊 VIDEO STATISTICS:
- Duration: {video_analysis.get('video_info', {}).get('duration', 'unknown')} seconds
- Frames analyzed: {video_analysis.get('total_frames_analyzed', 0)}
- Total vehicles detected: {traffic_stats.get('total_vehicles', 0)}
- Average vehicles per frame: {traffic_stats.get('avg_vehicles_per_frame', 0):.1f}
- Safety score (1-10): {traffic_stats.get('safety_score', 5):.1f}

⚠️ VIOLATIONS & SAFETY:
- Total violations detected: {traffic_stats.get('total_violations', 0)}
- Frames with violations: {traffic_stats.get('frames_with_violations', 0)}
- Violation types: {', '.join(traffic_stats.get('violation_types', ['None detected']))}

🔍 DETAILED ANALYSIS:
{video_analysis.get('summary', 'No summary available')}

🕰️ TIMESTAMPED EVENTS:
{self._format_events_summary(video_analysis.get('events', []))}

As a traffic expert, you can provide detailed insights about:
• Specific traffic violations with timestamps and severity
• Vehicle identification, counting, and behavior analysis
• Pedestrian safety and crossing patterns
• Traffic flow efficiency and congestion points
• Infrastructure assessment (signals, signs, markings)
• Safety recommendations and risk assessment
• Accident prevention and traffic management suggestions

Always provide specific timestamps, be precise about locations (left/right, foreground/background), and give actionable insights. Use traffic terminology appropriately."""

        conversation = [{
            "role": "system",
            "content": system_message,
            "timestamp": datetime.now().isoformat()
        }]
        
        self._store_conversation(session_id, conversation)
        return "Session initialized with video analysis."
    
    def _format_events_summary(self, events: List[Dict]) -> str:
        """Format events for system prompt with enhanced traffic details"""
        if not events:
            return "No specific events detected."
        
        summary_lines = []
        for i, event in enumerate(events[:10]):  # Limit to first 10 events
            timestamp = event.get('timestamp', 0)
            description = event.get('description', '')[:300]  # Allow longer descriptions
            violations = event.get('violations_detected', [])
            vehicle_count = event.get('vehicle_count', 0)
            safety_score = event.get('safety_score', 5)
            
            # Format with enhanced details
            event_line = f"- 🕰️ {timestamp:.1f}s: {description}"
            
            if violations:
                event_line += f" [⚠️ Violations: {', '.join(violations)}]"
            
            if vehicle_count > 0:
                event_line += f" [🚗 Vehicles: {vehicle_count}]"
            
            if safety_score < 7:
                event_line += f" [🚨 Safety: {safety_score}/10]"
                
            summary_lines.append(event_line)
        
        if len(events) > 10:
            summary_lines.append(f"... and {len(events) - 10} more events with detailed analysis available")
        
        return "\n".join(summary_lines)
    
    def generate_response(self, session_id: str, user_message: str) -> str:
        """Generate enhanced traffic-focused response"""
        try:
            # Retrieve conversation history and video analysis
            conversation = self._retrieve_conversation(session_id)
            video_analysis = self._retrieve_video_analysis(session_id)
            
            # Enhance user message with traffic context if needed
            enhanced_message = self._enhance_traffic_query(user_message, video_analysis)
            
            # Add user message
            conversation.append({
                "role": "user",
                "content": enhanced_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if this is a simple traffic query that can be answered directly
            try:
                direct_response = self._get_direct_traffic_response(user_message, video_analysis)
                if direct_response:
                    conversation.append({
                        "role": "assistant",
                        "content": direct_response,
                        "timestamp": datetime.now().isoformat()
                    })
                    self._store_conversation(session_id, conversation)
                    return direct_response
            except Exception as e:
                print(f"Error in direct response: {e}")
                # Continue to regular response generation
            
            # Prepare prompt for the model
            prompt = self._build_prompt(conversation)
            
            # Generate response with traffic-optimized parameters
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=400,  # Increased for detailed traffic analysis
                    do_sample=True,
                    temperature=0.6,     # Lower for more focused responses
                    top_p=0.85,         # More focused sampling
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2  # Avoid repetition
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            response = response.strip()
            
            # Post-process response for traffic context
            response = self._enhance_traffic_response(response, video_analysis)
            
            # Add assistant response to conversation
            conversation.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Store updated conversation
            self._store_conversation(session_id, conversation)
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error analyzing the traffic data: {str(e)}. Please try asking about specific traffic violations, vehicle counts, or safety concerns."
    
    def _build_prompt(self, conversation: List[Dict]) -> str:
        """Build prompt from conversation history"""
        prompt_parts = []
        
        # Add system message
        system_msg = next((msg for msg in conversation if msg["role"] == "system"), None)
        if system_msg:
            prompt_parts.append(f"System: {system_msg['content']}\n")
        
        # Add recent conversation (last 5 exchanges to avoid token limits)
        recent_conversation = [msg for msg in conversation if msg["role"] != "system"][-10:]
        
        for msg in recent_conversation:
            role = msg["role"].title()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}\n")
        
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        return self._retrieve_conversation(session_id)
    
    def clear_session(self, session_id: str):
        """Clear session data"""
        if self.redis_client:
            try:
                self.redis_client.delete(self._get_conversation_key(session_id))
                self.redis_client.delete(f"video_analysis:{session_id}")
            except Exception as e:
                print(f"Error clearing Redis session: {e}")
        else:
            self.memory_store.pop(session_id, None)
            self.memory_store.pop(f"video_{session_id}", None)
    
    def _enhance_traffic_query(self, user_message: str, video_analysis: Optional[Dict[str, Any]]) -> str:
        """Enhance user query with relevant traffic context"""
        if not video_analysis:
            return user_message
            
        # Add context for common traffic queries
        traffic_keywords = {
            'violations': 'traffic violations, safety violations, rule violations',
            'cars': 'vehicles, automobiles, motor vehicles, cars, trucks',
            'safety': 'safety issues, hazards, dangerous situations, risk assessment',
            'traffic': 'traffic flow, vehicle movement, congestion, traffic patterns'
        }
        
        enhanced_parts = [user_message]
        
        for keyword, expansion in traffic_keywords.items():
            if keyword in user_message.lower():
                stats = video_analysis.get('traffic_statistics', {})
                if keyword == 'violations' and stats.get('total_violations', 0) > 0:
                    enhanced_parts.append(f"(Note: {stats['total_violations']} violations detected across {stats.get('frames_with_violations', 0)} frames)")
                elif keyword == 'cars' and stats.get('total_vehicles', 0) > 0:
                    enhanced_parts.append(f"(Note: {stats['total_vehicles']} total vehicles detected, averaging {stats.get('avg_vehicles_per_frame', 0):.1f} per frame)")
                elif keyword == 'safety' and stats.get('safety_score', 10) < 7:
                    enhanced_parts.append(f"(Note: Overall safety score is {stats['safety_score']:.1f}/10)")
                break
        
        return " ".join(enhanced_parts)
    
    def _get_direct_traffic_response(self, user_message: str, video_analysis: Optional[Dict[str, Any]]) -> Optional[str]:
        """Provide direct responses for simple traffic statistics queries"""
        if not video_analysis:
            return None
            
        message_lower = user_message.lower()
        stats = video_analysis.get('traffic_statistics', {})
        events = video_analysis.get('events', [])
        
        # Direct responses for common queries
        if any(word in message_lower for word in ['how many cars', 'count vehicles', 'number of vehicles', 'vehicle count']):
            total_vehicles = stats.get('total_vehicles', 0)
            avg_vehicles = stats.get('avg_vehicles_per_frame', 0)
            return f"🚗 **Vehicle Count Analysis:**\n- Total vehicles detected: **{total_vehicles}**\n- Average per frame: **{avg_vehicles:.1f}**\n- Frames analyzed: **{len(events)}**\n\nThe video shows {'heavy' if avg_vehicles > 5 else 'moderate' if avg_vehicles > 2 else 'light'} traffic density."
        
        if any(word in message_lower for word in ['violations', 'traffic violations', 'safety violations']):
            violations = stats.get('total_violations', 0)
            violation_types = stats.get('violation_types', [])
            frames_with_violations = stats.get('frames_with_violations', 0)
            
            if violations == 0:
                return "✅ **No significant traffic violations detected** in this video analysis.\n\nThe traffic appears to be flowing normally with drivers following traffic rules appropriately."
            else:
                violation_list = "\n".join([f"  • {v.title()}" for v in violation_types]) if violation_types else "  • General traffic concerns"
                return f"⚠️ **Traffic Violations Detected:**\n- Total violations: **{violations}**\n- Affected frames: **{frames_with_violations}**\n- Types of violations:\n{violation_list}\n\nThese violations represent potential safety concerns that should be addressed."
        
        if any(word in message_lower for word in ['safety score', 'safety rating', 'how safe']):
            safety_score = stats.get('safety_score', 5)
            safety_level = 'Excellent' if safety_score >= 9 else 'Good' if safety_score >= 7 else 'Fair' if safety_score >= 5 else 'Poor'
            return f"🛡️ **Safety Assessment:**\n- Overall safety score: **{safety_score:.1f}/10**\n- Safety level: **{safety_level}**\n\n{'This represents a safe traffic environment with minimal risks.' if safety_score >= 7 else 'There are some safety concerns that warrant attention.' if safety_score >= 5 else 'Significant safety issues detected that require immediate attention.'}"
        
        return None
    
    def _enhance_traffic_response(self, response: str, video_analysis: Optional[Dict[str, Any]]) -> str:
        """Enhance model response with specific traffic data"""
        if not video_analysis or not response:
            return response
        
        # Add specific timestamps if violations are mentioned
        events = video_analysis.get('events', [])
        violation_events = [e for e in events if e.get('has_violations', False)]
        
        if 'violation' in response.lower() and violation_events:
            timestamps = [f"{e['timestamp']:.1f}s" for e in violation_events[:3]]
            if timestamps:
                response += f"\n\n📍 **Specific violation timestamps:** {', '.join(timestamps)}"
        
        return response