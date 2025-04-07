#!/usr/bin/env python
"""
Run the web interface for neural style transfer application.
"""

import os
import sys
import argparse
from neural_style.web.app import run_app
from neural_style.utils.logging import get_logger

# Setup logger
logger = get_logger(__name__)

def main():
    """Run the web interface."""
    parser = argparse.ArgumentParser(description='Run the web interface for neural style transfer')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the web server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the web server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs('neural_style/web/static/uploads', exist_ok=True)
    os.makedirs('neural_style/web/static/results', exist_ok=True)
    os.makedirs('neural_style/web/static/styles', exist_ok=True)
    
    # Run web app
    logger.info(f"Starting web UI on http://{args.host}:{args.port}")
    run_app(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
