#!/usr/bin/env python3
"""
Quick run script for the AIkaBoOM web application.
Usage: python run.py
"""
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the app
from aikaboom.web.app import app

if __name__ == '__main__':
    host = os.getenv('BOM_HOST', '127.0.0.1')
    port = int(os.getenv('BOM_PORT', '5000'))
    print("\n" + "="*70)
    print("💥 AIkaBoOM - Web Interface")
    print("="*70)
    print(f"\nServer starting at http://{host}:{port}")
    print("\n⚠️  To stop the server, press Ctrl+C")
    print("="*70 + "\n")

    app.run(host=host, port=port, debug=False, threaded=True)
