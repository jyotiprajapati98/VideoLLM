#!/usr/bin/env python3
"""
Simple server startup without complex dependencies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Try to start with basic imports
    import uvicorn
    from src.api.main import app
    
    print("🚀 Starting VideoLLM API Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    
    # Create directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Starting basic file server instead...")
    
    # Fallback to simple HTTP server for file uploads
    import http.server
    import socketserver
    
    PORT = 8000
    
    class SimpleHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                html = """
                <!DOCTYPE html>
                <html>
                <head><title>VideoLLM Upload</title></head>
                <body>
                    <h1>🚦 VideoLLM System</h1>
                    <h2>Upload Video for Analysis</h2>
                    <form action="/upload" method="post" enctype="multipart/form-data">
                        <input type="file" name="video" accept="video/*" required>
                        <br><br>
                        <input type="submit" value="Upload and Analyze">
                    </form>
                    <hr>
                    <p><strong>Note:</strong> Use the real-time traffic monitor for live analysis:</p>
                    <code>python realtime_traffic_monitor_fixed.py --demo</code>
                </body>
                </html>
                """
                self.wfile.write(html.encode())
            else:
                super().do_GET()
    
    with socketserver.TCPServer(("", PORT), SimpleHandler) as httpd:
        print(f"📡 Basic server running at http://localhost:{PORT}")
        httpd.serve_forever()
        
except Exception as e:
    print(f"❌ Server startup failed: {e}")
    print("\n🎯 Alternative: Use the real-time traffic monitoring system:")
    print("   python realtime_traffic_monitor_fixed.py --demo")