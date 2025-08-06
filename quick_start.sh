#!/bin/bash

# Quick Start Script for Visual Understanding Chat Assistant
echo "🎥 Visual Understanding Chat Assistant - Quick Start"
echo "======================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if Redis is running (optional)
redis_running=false
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis is running"
        redis_running=true
    else
        echo "⚠️ Redis is not running - using in-memory storage"
        echo "   To start Redis: redis-server"
    fi
else
    echo "⚠️ Redis not installed - using in-memory storage"
fi

# Run system test
echo ""
echo "🧪 Running system test..."
if ./venv/bin/python test_system.py > /dev/null 2>&1; then
    echo "✅ System test passed"
else
    echo "❌ System test failed. Please check the installation."
    exit 1
fi

echo ""
echo "🚀 Starting the application..."
echo ""

# Start API server in the background
echo "📡 Starting API server..."
./venv/bin/python run_server.py &
api_pid=$!

# Wait a moment for the API to start
sleep 3

# Check if API started successfully
if kill -0 $api_pid 2>/dev/null; then
    echo "✅ API server started (PID: $api_pid)"
    echo "   API URL: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
else
    echo "❌ Failed to start API server"
    exit 1
fi

# Start UI
echo ""
echo "🖥️ Starting UI..."
echo "   UI URL: http://localhost:8501"
echo ""
echo "🎉 Application is starting up!"
echo "   Please wait a moment for the UI to load..."
echo ""
echo "📝 To stop the application:"
echo "   Press Ctrl+C to stop the UI"
echo "   Then run: kill $api_pid"
echo ""

# Start UI (this will block)
./venv/bin/python run_ui.py

# Cleanup when UI stops
echo ""
echo "🛑 Stopping API server..."
kill $api_pid 2>/dev/null
echo "✅ Application stopped successfully"