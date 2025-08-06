# 🎥 Traffic Video Analysis - Visual Understanding Chat Assistant

A sophisticated AI-powered chat assistant that can analyze video content, detect traffic violations, identify safety issues, count vehicles, and engage in multi-turn conversations about traffic and visual content analysis.

## 🌟 Features

- **🎯 Video Event Recognition**: Automatically detect and identify events in video streams
- **🚦 Traffic Violation Detection**: Specialized in identifying traffic violations and safety issues
- **📝 Intelligent Summarization**: Generate comprehensive summaries with timestamps
- **💬 Multi-Turn Conversations**: Contextual chat about video content with memory retention
- **🔄 Temporal Analysis**: Frame-by-frame analysis with precise timestamp mapping
- **🎨 Interactive UI**: User-friendly Streamlit interface for testing and interaction

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   Streamlit UI  │────│   FastAPI       │────│   Video         │
│                 │    │   Backend       │    │   Processor     │
│                 │    │                 │    │   (SmolVLM2)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   User          │    │   Chat Manager  │    │   Frame         │
│   Interface     │    │   (Llama 3)     │    │   Extraction    │
│                 │    │                 │    │   (OpenCV)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                     ┌─────────────────┐
                     │                 │
                     │   Redis         │
                     │   Memory Store  │
                     │                 │
                     └─────────────────┘
```

### Component Breakdown

1. **Frontend Layer**
   - **Streamlit UI**: Interactive web interface for video upload and chat
   - **Real-time Communication**: WebSocket-like updates for processing status

2. **API Layer**
   - **FastAPI Backend**: RESTful API with async support
   - **Session Management**: Multi-user session handling
   - **File Management**: Secure video upload and temporary storage

3. **AI Models Layer**
   - **SmolVLM2**: Vision-language model for video frame analysis
   - **Llama 3**: Large language model for conversational AI
   - **Model Integration**: Seamless communication between vision and language models

4. **Data Processing Layer**
   - **Video Processing**: Frame extraction and temporal analysis
   - **Event Detection**: Intelligent event recognition and classification
   - **Memory Management**: Conversation context and video analysis storage

## 🛠️ Tech Stack Justification

### Backend Technologies

- **FastAPI**: Chosen for its high performance, automatic API documentation, and excellent async support - crucial for handling video processing workloads
- **Python**: Optimal for AI/ML workloads with extensive library ecosystem
- **Redis**: High-performance in-memory storage for conversation context and session management
- **OpenCV**: Industry-standard computer vision library for video processing

### AI Models

- **SmolVLM2-1.7B-Instruct**: Efficient vision-language model that balances performance with resource requirements, specifically designed for visual understanding tasks
- **Llama 3.2-3B-Instruct**: Powerful yet efficient language model for conversational AI with excellent instruction-following capabilities
- **Transformers Library**: Hugging Face ecosystem for easy model loading and inference

### Infrastructure

- **Virtual Environment**: Isolated Python environment for dependency management
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Async Processing**: Non-blocking I/O for handling multiple concurrent users

## 📋 Prerequisites

- Python 3.8+ (tested with 3.12)
- CUDA-compatible GPU (recommended for optimal performance)
- At least 8GB RAM (16GB recommended)
- Redis server (for production deployment)
- FFmpeg (for video processing)

## 🚀 Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd VideoLLM
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server ffmpeg
```

**macOS (with Homebrew):**
```bash
brew install redis ffmpeg
```

**Windows:**
- Install Redis from https://redis.io/download
- Install FFmpeg from https://ffmpeg.org/download.html

### 5. Configure Environment

```bash
# Copy and modify the environment file
cp .env.example .env

# Edit .env with your preferred settings
nano .env
```

### 6. Start Redis Server

```bash
# Ubuntu/Debian/macOS
redis-server

# Or as a service
sudo systemctl start redis-server
```

## 📖 Usage Instructions

### Starting the Application

#### Method 1: Using Run Scripts (Recommended)

**Terminal 1 - Start API Server:**
```bash
python run_server.py
```

**Terminal 2 - Start UI:**
```bash
python run_ui.py
```

#### Method 2: Manual Start

**Terminal 1 - API Server:**
```bash
source venv/bin/activate
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Streamlit UI:**
```bash
source venv/bin/activate
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Using the Application

1. **Access the UI**: Open http://localhost:8501 in your browser
2. **Upload Video**: Use the sidebar to upload a video file (max 2 minutes)
3. **Wait for Processing**: The system will analyze the video content
4. **Start Chatting**: Ask questions about the video content

### Example Interactions

#### Video Upload
- Supported formats: MP4, AVI, MOV, MKV
- Maximum duration: 2 minutes
- Recommended resolution: 720p or higher

#### Sample Questions
```
"What traffic violations did you detect?"
"Tell me about the events between 30-60 seconds"
"How many vehicles were involved in violations?"
"What safety issues did you observe?"
"Describe the pedestrian activity in the video"
"Were there any red light violations?"
```

### API Endpoints

The FastAPI backend provides the following endpoints:

- `GET /`: Health check
- `POST /upload-video`: Upload and process video
- `POST /chat`: Send chat message
- `GET /chat/history/{session_id}`: Get conversation history
- `DELETE /session/{session_id}`: Clear session data
- `GET /docs`: Interactive API documentation

### Example API Usage

```python
import requests

# Upload video
with open('traffic_video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload-video',
        files={'file': f}
    )
session_data = response.json()

# Chat about the video
chat_response = requests.post(
    'http://localhost:8000/chat',
    json={
        'session_id': session_data['session_id'],
        'message': 'What violations did you detect?'
    }
)
print(chat_response.json()['response'])
```

## 🔧 Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Video Processing
MAX_VIDEO_DURATION=120  # seconds
FRAME_SAMPLING_RATE=2   # seconds between frames

# Model Configuration
MODEL_CACHE_DIR=./model_cache
```

### Model Configuration

The system automatically downloads and caches models on first use:

- **SmolVLM2**: ~7GB download
- **Llama 3.2-3B**: ~6GB download

Ensure sufficient disk space and stable internet connection for initial setup.

## 🧪 Testing

### Manual Testing

1. Start both services (API and UI)
2. Upload a test video through the UI
3. Verify video processing completes successfully
4. Test conversation functionality with sample questions

### API Testing

Use the interactive API documentation at http://localhost:8000/docs to test endpoints directly.

### Sample Test Videos

For testing purposes, you can use:
- Traffic scene videos from YouTube (download with appropriate tools)
- Dashboard camera footage
- Surveillance camera feeds
- Any video with detectable events or violations

## 🚨 Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Solution: Ensure sufficient RAM and CUDA compatibility
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
```

**2. Redis Connection Failed**
```bash
# Solution: Start Redis server
sudo systemctl start redis-server
# Or install Redis if not available
```

**3. Video Processing Timeout**
```bash
# Solution: Reduce video duration or increase timeout
# Edit .env file: MAX_VIDEO_DURATION=60
```

**4. Out of Memory Errors**
```bash
# Solution: Use CPU-only mode for smaller models
# Edit model loading code to use device="cpu"
```

### Performance Optimization

- **GPU Usage**: Ensure CUDA is properly installed for GPU acceleration
- **Memory Management**: Close unused applications when processing large videos
- **Batch Processing**: Process multiple videos sequentially rather than concurrently

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For providing the Transformers library and model hosting
- **SmolVLM2 Team**: For the excellent vision-language model
- **Meta AI**: For the Llama 3 language model
- **FastAPI Team**: For the outstanding web framework
- **Streamlit Team**: For the intuitive UI framework

## 📞 Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review API documentation at `/docs` endpoint

---

**🎬 Ready to analyze your videos? Start the application and upload your first video!**