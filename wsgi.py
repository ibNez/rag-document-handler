#!/usr/bin/env python3
"""
WSGI entry point for RAG Knowledgebase Manager.

This module provides the WSGI application instance for production deployment
using servers like Gunicorn, uWSGI, or mod_wsgi.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_manager.app import RAGKnowledgebaseManager


def create_app():
    """
    Create and configure the Flask application for WSGI deployment.
    
    Returns:
        Flask: Configured Flask application instance
    """
    # Ensure production mode
    os.environ['FLASK_DEBUG'] = 'False'
    os.environ['FLASK_ENV'] = 'production'
    
    # Initialize the application manager
    app_manager = RAGKnowledgebaseManager()
    
    # Start the background scheduler (for production, this runs in the same process)
    app_manager.scheduler_manager.start_scheduler()
    
    return app_manager.app


# Create the WSGI application instance
application = create_app()

if __name__ == "__main__":
    # This allows the file to be run directly for testing
    application.run(host='0.0.0.0', port=3000)