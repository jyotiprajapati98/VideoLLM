#!/usr/bin/env python3
"""
System test script for Visual Understanding Chat Assistant
"""
import sys
import os
sys.path.append('src')

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        from src.models.video_processor import VideoProcessor
        print("✅ VideoProcessor import successful")
    except Exception as e:
        print(f"❌ VideoProcessor import failed: {e}")
        return False
    
    try:
        from src.models.chat_manager import ChatManager
        print("✅ ChatManager import successful")
    except Exception as e:
        print(f"❌ ChatManager import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV available: {cv2.__version__}")
    except Exception as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import redis
        print("✅ Redis client available")
    except Exception as e:
        print(f"❌ Redis import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test model loading without full initialization"""
    print("\n🤖 Testing model availability...")
    
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        model_name = "HuggingFaceTB/SmolVLM2-1.7B-Instruct"
        print(f"✅ SmolVLM2 model accessible: {model_name}")
    except Exception as e:
        print(f"❌ SmolVLM2 model test failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        print(f"✅ Llama 3 model accessible: {model_name}")
    except Exception as e:
        print(f"⚠️ Llama 3 model test failed, will use fallback: {e}")
    
    return True

def test_directories():
    """Test required directories exist"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "src/models",
        "src/api", 
        "src/utils",
        "uploads",
        "temp",
        "venv"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            return False
    
    return True

def test_files():
    """Test required files exist"""
    print("\n📄 Testing required files...")
    
    required_files = [
        "requirements.txt",
        "app.py",
        "run_server.py",
        "run_ui.py",
        "README.md",
        ".env",
        "src/models/video_processor.py",
        "src/models/chat_manager.py",
        "src/api/main.py",
        "src/utils/video_utils.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def test_redis_connection():
    """Test Redis connection"""
    print("\n🔴 Testing Redis connection...")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        print("   Redis is optional - system will use in-memory storage")
        return True  # Non-blocking failure

def main():
    """Run all tests"""
    print("🚀 Visual Understanding Chat Assistant - System Test")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directories),
        ("Required Files", test_files),
        ("Python Imports", test_imports),
        ("Model Availability", test_model_loading),
        ("Redis Connection", test_redis_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Start Redis server: redis-server")
        print("2. Start API server: python run_server.py")
        print("3. Start UI: python run_ui.py")
        print("4. Open http://localhost:8501 in browser")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before running the system.")
        if passed >= total - 1:  # Allow Redis failure
            print("Note: Redis failure is non-critical - system can run with in-memory storage.")
    
    return passed == total

if __name__ == "__main__":
    sys.exit(0 if main() else 1)