import streamlit as st
import requests
import json
import time
import os
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="Visual Understanding Chat Assistant",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Visual Understanding Chat Assistant")
    st.markdown("Upload a video and chat about its content with AI!")
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    
    # Sidebar for video upload
    with st.sidebar:
        st.header("📁 Video Upload")
        
        # Check API health
        health_status = check_api_health()
        if health_status:
            st.success("✅ API is running")
            st.json(health_status)
        else:
            st.error("❌ API is not responding. Make sure to start the FastAPI server first!")
            st.code("python -m uvicorn src.api.main:app --reload", language="bash")
            return
        
        # Video upload
        uploaded_file = st.file_uploader(
            "Choose a video file (max 2 minutes)",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Video", type="primary"):
                process_video(uploaded_file)
        
        # Session controls
        if st.session_state.video_processed:
            st.divider()
            if st.button("🗑️ Clear Session"):
                clear_session()
            
            if st.button("📊 Show Analysis"):
                if st.session_state.analysis_result:
                    show_analysis_details()
    
    # Main chat interface
    if st.session_state.video_processed:
        st.header("💬 Chat Interface")
        
        # Display video analysis summary
        if st.session_state.analysis_result:
            with st.expander("📋 Video Analysis Summary", expanded=False):
                result = st.session_state.analysis_result
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Duration", f"{result.get('duration', 0):.1f}s")
                with col2:
                    st.metric("Events Detected", result.get('events_detected', 0))
                with col3:
                    status = "❌ Error" if result.get('has_error') else "✅ Success"
                    st.metric("Status", status)
                
                if result.get('summary'):
                    st.markdown("**Summary:**")
                    st.markdown(result['summary'])
        
        # Chat messages display
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about the video..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = send_chat_message(prompt)
                    if response:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        error_msg = "Sorry, I couldn't process your message. Please try again."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Visual Understanding Chat Assistant! 🎬
        
        This AI-powered assistant can:
        - 🎯 **Analyze video content** and detect events
        - 🚦 **Identify traffic violations** and safety issues  
        - 📝 **Summarize key activities** with timestamps
        - 💬 **Answer questions** about the video content
        - 🔄 **Maintain conversation context** for follow-up questions
        
        ### How to use:
        1. **Upload a video** (max 2 minutes) using the sidebar
        2. **Wait for processing** - the AI will analyze the content
        3. **Start chatting** - ask questions about what happened in the video
        
        ### Example questions you can ask:
        - "What traffic violations did you detect?"
        - "Tell me about the events at timestamp 30 seconds"
        - "How many vehicles were involved?"
        - "What safety issues did you observe?"
        
        👈 **Start by uploading a video in the sidebar!**
        """)
        
        # Show some example videos or demo content
        st.markdown("---")
        st.markdown("### 🎥 Supported Video Formats")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**MP4**\n📹 Most common")
        with col2:
            st.markdown("**AVI**\n🎬 Classic format")
        with col3:
            st.markdown("**MOV**\n🍎 Apple format")
        with col4:
            st.markdown("**MKV**\n📦 Open format")

def check_api_health() -> Dict[str, Any]:
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"API connection error: {e}")
    return None

def process_video(uploaded_file):
    """Process uploaded video"""
    try:
        with st.spinner("Processing video... This may take a few minutes."):
            # Upload file to API
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{API_BASE_URL}/upload-video", files=files, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.session_id = result["session_id"]
                st.session_state.analysis_result = result["analysis"]
                st.session_state.video_processed = True
                st.session_state.messages = []  # Clear previous messages
                
                st.success("✅ Video processed successfully!")
                st.json(result["analysis"])
                st.rerun()
            else:
                st.error(f"❌ Error processing video: {response.text}")
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def send_chat_message(message: str) -> str:
    """Send chat message to API"""
    try:
        payload = {
            "session_id": st.session_state.session_id,
            "message": message
        }
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return result["response"]
        else:
            st.error(f"Chat error: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return None

def clear_session():
    """Clear current session"""
    try:
        if st.session_state.session_id:
            response = requests.delete(f"{API_BASE_URL}/session/{st.session_state.session_id}")
            if response.status_code == 200:
                st.success("Session cleared successfully!")
            else:
                st.warning("Session may not have been fully cleared")
        
        # Reset session state
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.video_processed = False
        st.session_state.analysis_result = None
        st.rerun()
        
    except Exception as e:
        st.error(f"Error clearing session: {str(e)}")

def show_analysis_details():
    """Show detailed analysis results"""
    if not st.session_state.analysis_result:
        st.error("No analysis data available")
        return
    
    st.subheader("📊 Detailed Analysis")
    result = st.session_state.analysis_result
    
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Video Duration", f"{result.get('duration', 0):.1f} seconds")
    with col2:
        st.metric("Events Detected", result.get('events_detected', 0))
    
    # Summary
    if result.get('summary'):
        st.subheader("📝 Summary")
        st.markdown(result['summary'])
    
    # Error status
    if result.get('has_error'):
        st.error("⚠️ Some errors occurred during processing")

if __name__ == "__main__":
    main()