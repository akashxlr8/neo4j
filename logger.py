import logging
from datetime import datetime
import os
from logging.handlers import RotatingFileHandler

def setup_logger():
    """Setup a single logger for the entire application"""
    
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create formatter with component name
    formatter = logging.Formatter(
        '%(asctime)s | %(component)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create handler
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'northwind_app.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger('northwind')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    return logger

# Create the logger instance
logger = setup_logger()

# Utility function to add component context
def log(component):
    """Returns a logger that automatically adds component information"""
    return logging.LoggerAdapter(logger, {'component': component}) 