"""
Main entry point for neural style transfer application.
"""

import sys
import argparse

from .cli.main import cli
from .web.app import run_app
from .utils.logging import setup_global_logger


def main():
    """Main entry point for the application."""
    # Setup global logger
    logger = setup_global_logger()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Style Transfer Application')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Web UI command
    web_parser = subparsers.add_parser('web', help='Start the web UI')
    web_parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the web server on')
    web_parser.add_argument('--port', type=int, default=8000, help='Port to run the web server on')
    web_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    # CLI command
    cli_parser = subparsers.add_parser('cli', help='Run the command-line interface')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == 'web':
        logger.info(f"Starting web UI on http://{args.host}:{args.port}")
        run_app(host=args.host, port=args.port, debug=args.debug)
    elif args.command == 'cli':
        # Pass remaining arguments to click CLI
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        cli()
    else:
        # Default to web UI if no command is specified
        logger.info("Starting web UI on http://127.0.0.1:8000")
        run_app(host='127.0.0.1', port=8000, debug=False)


if __name__ == '__main__':
    main()
