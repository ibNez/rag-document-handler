"""
Centralized logging configuration for RAG Document Handler.

This module provides logging configuration and utilities for the entire application,
ensuring consistent logging behavior across all components.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_log_dir() -> str:
    """
    Get the configured log directory from environment variables.
    
    Returns:
        str: The log directory path (defaults to "logs" if not configured)
    """
    return os.getenv("LOG_DIR", "logs")


def setup_logging(log_dir: Optional[str] = None) -> None:
    """
    Set up centralized logging configuration for the application.
    
    Args:
        log_dir: Log directory override (defaults to configured LOG_DIR)
    """
    if log_dir is None:
        log_dir = get_log_dir()
    
    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path / 'rag_knowledge_base_manager.log')
        ]
    )
    
    # Suppress noisy external library logs
    logging.getLogger('unstructured').setLevel(logging.WARNING)
    logging.getLogger('pdfminer').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Log that logging was configured
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - log directory: {log_path.absolute()}")


def setup_script_logging(
    script_name: Optional[str] = None,
    log_level: int = logging.INFO,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for standalone scripts using the configured LOG_DIR.
    
    Args:
        script_name: Name of the script (defaults to __main__)
        log_level: Logging level (defaults to INFO)
        log_dir: Log directory override (defaults to configured LOG_DIR)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_dir is None:
        log_dir = get_log_dir()
    
    if script_name is None:
        script_name = "script"
    
    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path / f'{script_name}.log')
        ]
    )
    
    # Suppress noisy external library logs
    logging.getLogger('unstructured').setLevel(logging.WARNING)
    logging.getLogger('pdfminer').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logger = logging.getLogger(script_name)
    logger.info(f"Logging configured for {script_name} - log directory: {log_path.absolute()}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
