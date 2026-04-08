#!/usr/bin/env python3
"""
Quick run script for BOM Tools web application
Usage: python run.py
"""
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the app
from bom_tools.web.app import app

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 Unified BOM Generator - WEB INTERFACE")
    print("="*70)
    print("\nServer starting...")
    print("Access the web interface at: http://localhost:5000")
    print("Or from another machine: http://<server-ip>:5000")
    print("\n⚠️  To stop the server, press Ctrl+C")
    print("="*70 + "\n")
    
    # Run Flask app
    host = os.getenv('BOM_HOST', '127.0.0.1')
    port = int(os.getenv('BOM_PORT', '5000'))
    app.run(host=host, port=port, debug=False, threaded=True)
