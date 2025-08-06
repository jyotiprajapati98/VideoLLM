from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
from typing import Dict, Any
from pydantic import BaseModel
import asyncio
from ..models.video_processor import VideoProcessor
from ..models.chat_manager import ChatManager

app = FastAPI(title="Visual Understanding Chat Assistant", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
video_processor = None
chat_manager = None

# Data models
class ChatMessage(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

# Initialize models on startup
@app.on_event("startup")
async def initialize_models():
    global video_processor, chat_manager
    try:
        print("Initializing models...")
        video_processor = VideoProcessor()
        chat_manager = ChatManager()
        print("Models initialized successfully!")
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Visual Understanding Chat Assistant API", "status": "running"}

@app.get("/health")
async def health_check():
    """Enhanced health check with detailed system status"""
    try:
        basic_health = {
            "status": "healthy",
            "video_processor_loaded": video_processor is not None,
            "chat_manager_loaded": chat_manager is not None
        }
        
        # Get detailed health if processors are loaded
        if video_processor:
            video_health = video_processor.get_health_status()
            basic_health["video_processor_health"] = video_health
        
        if chat_manager:
            # Add chat manager health if available
            basic_health["chat_sessions_active"] = len(getattr(chat_manager, 'memory_store', {}))
        
        return basic_health
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "video_processor_loaded": False,
            "chat_manager_loaded": False
        }

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload and process video file"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            # Check file extension as fallback
            if not file.filename or not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):
                raise HTTPException(status_code=400, detail="File must be a video (mp4, avi, mov, mkv, wmv, flv)")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create directories
        upload_dir = f"uploads/{session_id}"
        temp_dir = f"temp/{session_id}"
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded file
        video_path = os.path.join(upload_dir, file.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process video asynchronously with enhanced error handling
        print(f"Processing video for session {session_id}...")
        try:
            analysis_result = await asyncio.get_event_loop().run_in_executor(
                None, video_processor.process_video, video_path, temp_dir
            )
        except Exception as processing_error:
            # Clean up on processing failure
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir, ignore_errors=True)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            raise HTTPException(
                status_code=500, 
                detail=f"Video processing failed: {str(processing_error)}"
            )
        
        # Initialize chat session
        chat_manager.initialize_session(session_id, analysis_result)
        
        # Clean up uploaded file (keep analysis results)
        os.remove(video_path)
        
        return {
            "session_id": session_id,
            "status": "success",
            "analysis": {
                "duration": analysis_result.get("video_info", {}).get("duration", 0),
                "events_detected": analysis_result.get("total_frames_analyzed", 0),
                "summary": analysis_result.get("summary", ""),
                "has_error": "error" in analysis_result
            }
        }
        
    except Exception as e:
        print(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Send message and get response"""
    try:
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not initialized")
        
        # Generate response
        response = chat_manager.generate_response(message.session_id, message.message)
        
        return ChatResponse(
            response=response,
            session_id=message.session_id
        )
        
    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get conversation history for a session"""
    try:
        if not chat_manager:
            raise HTTPException(status_code=500, detail="Chat manager not initialized")
        
        history = chat_manager.get_conversation_history(session_id)
        return {"session_id": session_id, "history": history}
        
    except Exception as e:
        print(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session data"""
    try:
        if chat_manager:
            chat_manager.clear_session(session_id)
        
        # Clean up directories
        upload_dir = f"uploads/{session_id}"
        temp_dir = f"temp/{session_id}"
        
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        return {"message": "Session cleared successfully"}
        
    except Exception as e:
        print(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def list_sessions():
    """List active sessions with enhanced info"""
    try:
        # List directories in uploads folder
        upload_base = "uploads"
        sessions_info = []
        
        if os.path.exists(upload_base):
            sessions = [d for d in os.listdir(upload_base) if os.path.isdir(os.path.join(upload_base, d))]
            
            for session_id in sessions:
                session_path = os.path.join(upload_base, session_id)
                temp_path = os.path.join("temp", session_id)
                
                # Get session info
                session_info = {
                    "session_id": session_id,
                    "created": os.path.getctime(session_path),
                    "has_temp_files": os.path.exists(temp_path),
                    "temp_file_count": len(os.listdir(temp_path)) if os.path.exists(temp_path) else 0
                }
                sessions_info.append(session_info)
        
        return {
            "active_sessions": sessions_info,
            "total_sessions": len(sessions_info)
        }
        
    except Exception as e:
        print(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get processing metrics and statistics"""
    try:
        metrics = {}
        
        if video_processor:
            health_status = video_processor.get_health_status()
            metrics["video_processor"] = health_status["processing_stats"]
            metrics["model_health"] = {
                "healthy": health_status["model_healthy"],
                "model_name": health_status["model_name"],
                "failures": health_status["model_failures"],
                "memory_usage_mb": health_status["memory_usage_mb"]
            }
            metrics["system"] = health_status["system_health"]
        
        return metrics
        
    except Exception as e:
        print(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset-stats")
async def reset_processing_stats():
    """Reset processing statistics"""
    try:
        if video_processor:
            video_processor.reset_stats()
            return {"message": "Processing statistics reset successfully"}
        else:
            raise HTTPException(status_code=503, detail="Video processor not available")
    except Exception as e:
        print(f"Error resetting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)