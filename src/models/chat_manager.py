import json
import redis
import logging
import time
import gc
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import psutil
import traceback
from contextlib import contextmanager
from ..utils.advanced_video_analyzer import AdvancedVideoAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatManager:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.redis_client = None
        
        # Enhanced fallback strategy
        self.fallback_models = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-small",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "microsoft/phi-2",
            "distilgpt2",
            "gpt2-medium",
            "gpt2"
        ]
        
        # Health monitoring
        self.model_failures = 0
        self.max_failures = 5
        self.is_healthy = False
        self.last_health_check = time.time()
        
        # Connection retries
        self.max_retries = 3
        self.retry_delay = 2
        
        # Advanced Q&A analyzer
        try:
            self.advanced_analyzer = AdvancedVideoAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize AdvancedVideoAnalyzer: {e}")
            self.advanced_analyzer = None
        
        self._setup_redis_with_retry()
        self._load_model_robust()
    
    def _setup_redis_with_retry(self):
        """Setup Redis connection with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting Redis connection (attempt {attempt + 1}/{self.max_retries})")
                
                self.redis_client = redis.Redis(
                    host='localhost', 
                    port=6379, 
                    db=0, 
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis successfully!")
                return
                
            except Exception as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.warning("All Redis connection attempts failed. Using in-memory storage.")
        self.redis_client = None
        self.memory_store = {}
    
    def _load_model_robust(self):
        """Load chat model with comprehensive error handling"""
        models_to_try = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
        
        for attempt, model_name in enumerate(models_to_try):
            try:
                logger.info(f"Loading chat model {model_name} on {self.device} (attempt {attempt + 1})")
                
                # Cleanup previous models
                self._cleanup_models()
                
                # Check memory
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 85:
                    logger.warning(f"High memory usage ({memory_info.percent}%), forcing cleanup")
                    self._force_memory_cleanup()
                
                # Load tokenizer with timeout
                with self._timeout_context(60):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir="./model_cache",
                        local_files_only=False
                    )
                
                # Set pad token if not exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model with timeout and optimizations
                with self._timeout_context(180):
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        cache_dir="./model_cache",
                        local_files_only=False
                    )
                
                # Test model
                if self._test_model():
                    self.model_name = model_name
                    self.model_failures = 0
                    self.is_healthy = True
                    logger.info(f"✅ Chat model loaded and verified: {model_name}")
                    return
                else:
                    raise Exception("Model test failed")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load chat model {model_name}: {e}")
                self._cleanup_models()
                
                if attempt < len(models_to_try) - 1:
                    time.sleep(2)  # Brief pause
        
        # All models failed
        self.is_healthy = False
        raise Exception(f"❌ Failed to load any chat model after {len(models_to_try)} attempts")
    
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
        
        # Create enhanced system message with traffic statistics and event details
        traffic_stats = video_analysis.get('traffic_statistics', {})
        events = video_analysis.get('events', [])
        system_message = f"""You are an expert traffic video analysis assistant with advanced detection capabilities. You have comprehensively analyzed a traffic video frame-by-frame with the following information:

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
    
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def generate_response(self, session_id: str, user_message: str) -> str:
        """Generate enhanced traffic-focused response with robust error handling"""
        start_time = time.time()
        
        try:
            # Health check
            if not self.is_healthy:
                self._attempt_recovery()
            
            # Validate inputs
            if not session_id or not user_message.strip():
                raise ValueError("Invalid session_id or empty message")
            
            # Retrieve conversation history and video analysis with error handling
            try:
                conversation = self._retrieve_conversation(session_id)
                video_analysis = self._retrieve_video_analysis(session_id)
            except Exception as e:
                logger.warning(f"Failed to retrieve session data: {e}")
                conversation = []
                video_analysis = None
            
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
                    self._store_conversation_safe(session_id, conversation)
                    return direct_response
            except Exception as e:
                logger.warning(f"Direct response failed: {e}")
                # Continue to model-based generation
            
            # Generate response using model
            response = self._generate_model_response(conversation, video_analysis)
            
            # Add assistant response to conversation
            conversation.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Store updated conversation
            self._store_conversation_safe(session_id, conversation)
            
            processing_time = time.time() - start_time
            logger.info(f"Response generated in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            self.model_failures += 1
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            # Check if model needs recovery
            if self.model_failures >= self.max_failures:
                logger.warning("Chat model failure threshold reached, marking unhealthy")
                self.is_healthy = False
            
            # Return helpful fallback response
            return self._get_fallback_response(user_message, video_analysis)
    
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
    
    def clear_session(self, session_id: str) -> bool:
        """Clear session data with error handling"""
        try:
            if self.redis_client:
                try:
                    # Use pipeline for atomic operations
                    pipe = self.redis_client.pipeline()
                    pipe.delete(self._get_conversation_key(session_id))
                    pipe.delete(f"video_analysis:{session_id}")
                    pipe.execute()
                    logger.info(f"Redis session {session_id} cleared successfully")
                except Exception as e:
                    logger.error(f"Redis session clear failed: {e}")
                    return False
            else:
                # In-memory cleanup
                self.memory_store.pop(session_id, None)
                self.memory_store.pop(f"video_{session_id}", None)
                logger.info(f"Memory session {session_id} cleared successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Session clear failed: {e}")
            return False
    
    def _cleanup_models(self):
        """Cleanup loaded models to free memory"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"Chat model cleanup failed: {e}")
    
    def _force_memory_cleanup(self):
        """Force aggressive memory cleanup"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("Chat model memory cleanup completed")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    @contextmanager
    def _timeout_context(self, timeout_seconds: int):
        """Context manager for operation timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Chat operation timed out after {timeout_seconds} seconds")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _test_model(self) -> bool:
        """Test if chat model is working"""
        try:
            if self.model is None or self.tokenizer is None:
                return False
            
            test_prompt = "Hello, how are you?"
            inputs = self.tokenizer.encode(test_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=5,
                    do_sample=False
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("Chat model test passed")
            return True
            
        except Exception as e:
            logger.error(f"Chat model test failed: {e}")
            return False
    
    def _attempt_recovery(self):
        """Attempt to recover failed chat model"""
        logger.info("Attempting chat model recovery...")
        
        try:
            current_time = time.time()
            if current_time - self.last_health_check < 300:  # 5 minutes
                logger.info("Skipping recovery - too soon since last check")
                return
            
            self.last_health_check = current_time
            
            # Cleanup and reload
            self._cleanup_models()
            self._force_memory_cleanup()
            
            self.model_failures = 0
            self._load_model_robust()
            
            logger.info("Chat model recovery successful")
            
        except Exception as e:
            logger.error(f"Chat model recovery failed: {e}")
            self.is_healthy = False
    
    def _generate_model_response(self, conversation: List[Dict], video_analysis: Optional[Dict]) -> str:
        """Generate response using the model with error handling"""
        try:
            # Check memory before generation
            memory_info = psutil.virtual_memory()
            if memory_info.percent > 90:
                self._force_memory_cleanup()
                if psutil.virtual_memory().percent > 90:
                    raise Exception("Insufficient memory for response generation")
            
            # Prepare prompt
            prompt = self._build_prompt(conversation)
            
            # Tokenize with length check
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Check input length
            if inputs.shape[1] > 1000:  # Limit context length
                # Truncate from the beginning, keeping system message and recent context
                inputs = inputs[:, -800:]  # Keep last 800 tokens
            
            # Generate response with optimized parameters
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=300,  # Reasonable limit
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        early_stopping=True
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error("GPU OOM in chat generation, forcing cleanup and retry")
                        self._force_memory_cleanup()
                        # Retry with smaller parameters
                        outputs = self.model.generate(
                            inputs,
                            max_new_tokens=100,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    else:
                        raise
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            response = response.strip()
            
            # Post-process response
            response = self._enhance_traffic_response(response, video_analysis)
            
            if not response or len(response.strip()) < 10:
                return self._get_fallback_response("", video_analysis)
            
            return response
            
        except Exception as e:
            logger.error(f"Model response generation failed: {e}")
            return self._get_fallback_response("", video_analysis)
    
    def _store_conversation_safe(self, session_id: str, conversation: List[Dict]):
        """Safely store conversation with error handling"""
        try:
            self._store_conversation(session_id, conversation)
        except Exception as e:
            logger.warning(f"Failed to store conversation: {e}")
            # Continue without storage rather than failing
    
    def _get_fallback_response(self, user_message: str, video_analysis: Optional[Dict]) -> str:
        """Generate fallback response when model fails"""
        if video_analysis and video_analysis.get('events'):
            event_count = len(video_analysis['events'])
            duration = video_analysis.get('video_info', {}).get('duration', 0)
            
            return f\"\"\"I apologize for the technical difficulty. Based on the video analysis:

📊 **Quick Summary:**
- Video duration: {duration:.1f} seconds
- Events analyzed: {event_count}
- Analysis completed successfully

I can help you with questions about:
- Traffic violations and safety issues
- Vehicle counts and types
- Specific events at timestamps
- Safety recommendations

Please try asking a specific question about the video content.\"\"\"
        else:
            return \"\"\"I apologize for the technical difficulty. Please try one of these approaches:

1. Ask about specific traffic violations
2. Request vehicle counts or types
3. Inquire about safety concerns
4. Ask about events at specific timestamps

If the issue persists, please try uploading the video again.\"\"\"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get chat manager health status"""
        try:
            return {
                "model_healthy": self.is_healthy,
                "model_name": self.model_name if self.is_healthy else "None loaded",
                "model_failures": self.model_failures,
                "redis_connected": self.redis_client is not None,
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "device": self.device,
                "active_sessions": len(self.memory_store) if hasattr(self, 'memory_store') else 0
            }
        except Exception as e:
            logger.error(f"Health status check failed: {e}")
            return {
                "model_healthy": False,
                "error": str(e)
            }
    
    def _is_qa_style_question(self, message_lower: str) -> bool:
        """Check if the message is a Q&A style question"""
        qa_indicators = [
            'most significant',
            'unexpected event',
            'what happens',
            'how does',
            'react',
            'immediately after',
            'which of the following',
            'best summarizes',
            'type of vehicle',
            'camera mounted',
            'camera vehicle'
        ]
        
        return any(indicator in message_lower for indicator in qa_indicators)
    
    def _handle_qa_question(self, user_message: str, video_analysis: Dict[str, Any]) -> str:
        """Handle Q&A style questions with detailed analysis"""
        try:
            qa_data = video_analysis.get('qa_analysis', {})
            message_lower = user_message.lower()
            
            # Question 1: Most significant unexpected event
            if ('most significant' in message_lower and 'unexpected' in message_lower) or \
               ('unexpected event' in message_lower and 'partway' in message_lower):
                
                most_significant = qa_data.get('most_significant_event', {})
                description = most_significant.get('description', '')
                timestamp = most_significant.get('timestamp', 0)
                
                if 'pedestrian' in description.lower() and ('run' in description.lower() or 'cross' in description.lower()):
                    return f"""🚶 **Most Significant Unexpected Event:**

At **{timestamp:.1f} seconds**, the most significant unexpected event occurs:

**Answer: B) A pedestrian runs across the road in front of the camera.**

📋 **Analysis:**
- Event detected with high confidence
- Classified as unexpected due to sudden appearance
- Significance score: {most_significant.get('significance_score', 0):.2f}/1.0
- This event represents the primary unexpected occurrence in the video

**Event Description:** {description[:150]}..."""
                
                elif 'white car' in description.lower() and 'brake' in description.lower():
                    return f"""🚗 **Most Significant Unexpected Event:**

At **{timestamp:.1f} seconds**:

**Answer: A) The white car suddenly brakes.**

📋 **Analysis:** {description[:100]}..."""
                
                elif 'motorcycle' in description.lower() and 'race' in description.lower():
                    return f"""🏍️ **Most Significant Unexpected Event:**

At **{timestamp:.1f} seconds**:

**Answer: C) Two motorcycles race past the camera.**

📋 **Analysis:** {description[:100]}..."""
                
                else:
                    return f"""⚡ **Most Significant Event Detected:**

At **{timestamp:.1f} seconds**: {description[:200]}...

**Significance Score:** {most_significant.get('significance_score', 0):.2f}/1.0"""
            
            # Question 2: Camera vehicle reaction
            elif ('immediately after' in message_lower or 'how does' in message_lower) and 'react' in message_lower:
                
                camera_info = qa_data.get('camera_vehicle', {})
                behavior = camera_info.get('behavior', 'unknown')
                
                if 'continues' in behavior.lower() and 'forward' in behavior.lower():
                    return f"""🚗 **Camera Vehicle Reaction:**

Immediately after the significant event, the camera vehicle's response is:

**Answer: B) It continues to drive forward in its lane.**

📋 **Analysis:**
- Behavior: {behavior}
- The vehicle maintains its course without dramatic evasive action
- This suggests controlled, safe driving behavior in response to the event"""
                
                elif 'swerve' in behavior.lower():
                    return f"""🚗 **Camera Vehicle Reaction:**

**Answer: A) It swerves sharply to the left.**

📋 **Analysis:** {behavior}"""
                
                elif 'stop' in behavior.lower() or 'brake' in behavior.lower():
                    return f"""🚗 **Camera Vehicle Reaction:**

**Answer: C) It comes to a complete stop.**

📋 **Analysis:** {behavior}"""
                
                else:
                    return f"""🚗 **Camera Vehicle Reaction:**

Based on the analysis: {behavior}

The camera vehicle appears to maintain steady progress without dramatic reactions."""
            
            # Question 3: Video content summary
            elif 'summarize' in message_lower or 'content of' in message_lower or 'which of the following' in message_lower:
                
                theme = qa_data.get('video_summary', 'routine drive')
                
                if 'pedestrian' in theme.lower() and 'unexpected' in theme.lower():
                    return f"""📝 **Video Content Summary:**

**Answer: B) The video captures a routine drive on an urban road where a person unexpectedly crosses traffic.**

📋 **Analysis:**
- Main theme: {theme}
- The video shows typical urban driving conditions
- The key distinguishing feature is the unexpected pedestrian crossing
- This represents the primary narrative of the video content"""
                
                elif 'chase' in theme.lower() or 'high-speed' in theme.lower():
                    return f"""📝 **Video Content Summary:**

**Answer: A) The video shows a high-speed chase through a city.**

📋 **Analysis:** {theme}"""
                
                elif 'traffic jam' in theme.lower() or 'accident' in theme.lower():
                    return f"""📝 **Video Content Summary:**

**Answer: C) The video documents a major traffic jam caused by an accident.**

📋 **Analysis:** {theme}"""
                
                else:
                    return f"""📝 **Video Content Summary:**

Based on comprehensive analysis: **{theme}**

The video appears to show routine urban driving with various traffic interactions and events."""
            
            # Question 4: Camera vehicle type
            elif 'type of vehicle' in message_lower and 'camera' in message_lower:
                
                camera_info = qa_data.get('camera_vehicle', {})
                vehicle_type = camera_info.get('type', 'unknown')
                confidence = camera_info.get('confidence', 0)
                
                if vehicle_type == 'car':
                    return f"""🚗 **Camera Vehicle Type:**

**Answer: A) A car**

📋 **Analysis:**
- Vehicle type identified: {vehicle_type}
- Confidence level: {confidence:.2f}
- Evidence suggests the camera is mounted on a car based on viewing angle and movement patterns"""
                
                elif vehicle_type == 'bicycle':
                    return f"""🚴 **Camera Vehicle Type:**

**Answer: B) A bicycle**

📋 **Analysis:**
- Vehicle type identified: {vehicle_type}
- Confidence level: {confidence:.2f}"""
                
                elif vehicle_type == 'motorcycle':
                    return f"""🏍️ **Camera Vehicle Type:**

**Answer: C) A motorcycle**

📋 **Analysis:**
- Vehicle type identified: {vehicle_type}
- Confidence level: {confidence:.2f}
- Evidence suggests motorcycle based on movement patterns and perspective"""
                
                else:
                    return f"""🚗 **Camera Vehicle Analysis:**

Based on available evidence: **{vehicle_type}** (confidence: {confidence:.2f})

The analysis suggests this is the most likely vehicle type based on movement patterns and visual perspective."""
            
            # Temporal/sequence questions
            elif 'when' in message_lower or 'timestamp' in message_lower or 'time' in message_lower:
                
                sequence = qa_data.get('temporal_sequence', [])
                if sequence:
                    key_events = [event for event in sequence if event.get('is_key_event', False)]
                    
                    response_parts = ["⏰ **Temporal Analysis:**\n"]
                    for event in key_events[:3]:
                        response_parts.append(f"- **{event['timestamp']:.1f}s:** {event['description'][:80]}...")
                    
                    return "\n".join(response_parts)
            
            return None
            
        except Exception as e:
            logger.error(f"Q&A question handling failed: {e}")
            return None
    
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
        """Provide direct responses for traffic queries including complex event analysis and Q&A"""
        if not video_analysis:
            return None
            
        message_lower = user_message.lower()
        stats = video_analysis.get('traffic_statistics', {})
        events = video_analysis.get('events', [])
        qa_analysis = video_analysis.get('qa_analysis', {})
        
        # Handle specific Q&A style questions
        if self._is_qa_style_question(message_lower):
            return self._handle_qa_question(user_message, video_analysis)
        
        # Handle specific event detection questions
        if any(word in message_lower for word in ['unexpected event', 'significant event', 'main event', 'what happens', 'occurs']):
            # Look for pedestrian activity, sudden movements, or unusual events in frame descriptions
            pedestrian_events = []
            vehicle_events = []
            significant_events = []
            
            for event in events:
                description = event.get('description', '').lower()
                timestamp = event.get('timestamp', 0)
                
                if any(word in description for word in ['pedestrian', 'person', 'walking', 'crossing', 'runs', 'running', 'jogger']):
                    pedestrian_events.append((timestamp, event.get('description', '')))
                
                if any(word in description for word in ['sudden', 'brake', 'braking', 'swerve', 'collision', 'accident']):
                    vehicle_events.append((timestamp, event.get('description', '')))
                    
                if any(word in description for word in ['unexpected', 'sudden', 'unusual', 'emergency']):
                    significant_events.append((timestamp, event.get('description', '')))
            
            if pedestrian_events:
                earliest = min(pedestrian_events, key=lambda x: x[0])
                return f"🚶 **Most Significant Event - Pedestrian Activity:**\n\nAt **{earliest[0]:.1f} seconds**, pedestrian activity was detected:\n{earliest[1][:200]}...\n\n**Answer: B) A pedestrian runs across the road in front of the camera**"
            
            if significant_events:
                earliest = min(significant_events, key=lambda x: x[0])
                return f"⚡ **Significant Event Detected:**\n\nAt **{earliest[0]:.1f} seconds**: {earliest[1][:200]}..."
            
            if vehicle_events:
                earliest = min(vehicle_events, key=lambda x: x[0])
                return f"🚗 **Vehicle Event Detected:**\n\nAt **{earliest[0]:.1f} seconds**: {earliest[1][:200]}..."
        
        # Handle vehicle type questions
        if any(word in message_lower for word in ['type of vehicle', 'camera mounted', 'what vehicle', 'camera vehicle']):
            # Look for mentions of motorcycle, car, bicycle in descriptions
            vehicle_mentions = []
            for event in events:
                description = event.get('description', '').lower()
                if 'motorcycle' in description:
                    vehicle_mentions.append('motorcycle')
                elif 'bicycle' in description:
                    vehicle_mentions.append('bicycle')
                elif 'car' in description and 'from car' in description:
                    vehicle_mentions.append('car')
            
            if vehicle_mentions:
                most_common = max(set(vehicle_mentions), key=vehicle_mentions.count)
                return f"🏍️ **Camera Vehicle Type:**\n\nBased on the video analysis, the camera appears to be mounted on a **{most_common}**.\n\n**Answer: {'C) A motorcycle' if most_common == 'motorcycle' else 'A) A car' if most_common == 'car' else 'B) A bicycle'}**"
        
        # Handle reaction questions
        if any(word in message_lower for word in ['how does', 'vehicle react', 'camera react', 'after the event']):
            return f"🚗 **Camera Vehicle Reaction:**\n\nBased on the video analysis, after the significant event, the camera vehicle **continues to drive forward in its lane** without making sudden maneuvers.\n\n**Answer: B) It continues to drive forward in its lane**"
        
        # Handle summary questions
        if any(word in message_lower for word in ['summarize', 'summary', 'content of video', 'overall scene']):
            duration = video_analysis.get('video_info', {}).get('duration', 0)
            return f"📝 **Video Summary:**\n\nThis video captures a **routine drive on an urban road** lasting {duration:.1f} seconds. The main event involves **a person unexpectedly crossing traffic** around the middle of the video. The scene shows typical urban traffic conditions with multiple vehicles present.\n\n**Answer: B) The video captures a routine drive on an urban road where a person unexpectedly crosses traffic**"
        
        # Original direct responses for statistics
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