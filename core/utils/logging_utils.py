"""
Simple logging utility for data actions
Replaces the Flask log_data_action function
"""
import logging
from typing import Optional

# Get logger for data actions
data_logger = logging.getLogger("data_actions")


def log_data_action(action: str, success: bool = True, details: Optional[str] = None):
    """
    Log data-related actions for monitoring and debugging
    
    Args:
        action: The action being performed
        success: Whether the action succeeded
        details: Additional details about the action
    """
    level = logging.INFO if success else logging.ERROR
    message = f"Action: {action}"
    
    if details:
        message += f" | Details: {details}"
    
    if not success:
        message += " | Status: FAILED"
    else:
        message += " | Status: SUCCESS"
    
    data_logger.log(level, message)