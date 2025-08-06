#!/usr/bin/env python3
"""
Run the FastAPI server for the Visual Understanding Chat Assistant
"""
import uvicorn
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("model_cache", exist_ok=True)
    
    print("Starting Visual Understanding Chat Assistant API...")
    print("API will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )