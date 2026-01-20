"""Logging setup and configuration"""

import sys
from loguru import logger
from pathlib import Path


def setup_logger(log_level: str = "DEBUG", log_file: str = None):
    """
    Setup logger with console and file handlers
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with enhanced color and formatting for live logs
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
    
    return logger


def get_logger(name: str = None):
    """
    Get logger instance
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger

