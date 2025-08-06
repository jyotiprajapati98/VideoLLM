#!/usr/bin/env python3
"""
Run the Streamlit UI for the Visual Understanding Chat Assistant
"""
import streamlit.web.cli as stcli
import sys
import os

if __name__ == "__main__":
    print("Starting Visual Understanding Chat Assistant UI...")
    print("Make sure the API server is running first!")
    print("UI will be available at: http://localhost:8501")
    
    sys.argv = ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
    sys.exit(stcli.main())