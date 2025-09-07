"""
Input validation utilities for web routes.

Following DEVELOPMENT_RULES.md for all development requirements.
Provides secure input validation and sanitization for all user inputs.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class InputValidator:
    """
    Secure input validation and sanitization utilities.
    
    Implements all security requirements from DEVELOPMENT_RULES.md:
    - Input type validation
    - Length limits enforcement
    - Pattern validation
    - XSS prevention
    - Path traversal prevention
    """
    
    # Security patterns
    SAFE_TEXT_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-_\.\@\(\)\[\]]+$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9.-]*local[a-zA-Z0-9.-]*)$')
    DOMAIN_PATTERN = re.compile(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    @staticmethod
    def validate_string(value: Any, field_name: str, max_length: int = 255, 
                       required: bool = True, pattern: Optional[re.Pattern] = None) -> str:
        """
        Validate and sanitize string input.
        
        Args:
            value: Input value to validate
            field_name: Name of field for error messages
            max_length: Maximum allowed length
            required: Whether field is required
            pattern: Optional regex pattern to match
            
        Returns:
            Validated and sanitized string
            
        Raises:
            ValidationError: If validation fails
        """
        if value is None or value == '':
            if required:
                raise ValidationError(f"{field_name} is required")
            return ''
        
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string")
            
        # Strip and sanitize
        cleaned = str(value).strip()
        
        # Length validation
        if len(cleaned) > max_length:
            raise ValidationError(f"{field_name} exceeds maximum length of {max_length}")
            
        # Pattern validation
        if pattern and not pattern.match(cleaned):
            raise ValidationError(f"{field_name} contains invalid characters")
            
        # XSS prevention - remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '\\']
        for char in dangerous_chars:
            if char in cleaned:
                logger.warning(f"Dangerous character '{char}' removed from {field_name}")
                cleaned = cleaned.replace(char, '')
                
        return cleaned
    
    @staticmethod
    def validate_integer(value: Any, field_name: str, min_val: int = 0, 
                        max_val: int = 2147483647, required: bool = True) -> int:
        """
        Validate integer input with bounds checking.
        
        Args:
            value: Input value to validate
            field_name: Name of field for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            required: Whether field is required
            
        Returns:
            Validated integer
            
        Raises:
            ValidationError: If validation fails
        """
        if value is None or value == '':
            if required:
                raise ValidationError(f"{field_name} is required")
            return min_val
            
        try:
            int_val = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a valid integer")
            
        if int_val < min_val:
            raise ValidationError(f"{field_name} must be at least {min_val}")
            
        if int_val > max_val:
            raise ValidationError(f"{field_name} must not exceed {max_val}")
            
        return int_val
    
    @staticmethod
    def validate_email(value: Any, field_name: str, required: bool = True) -> str:
        """
        Validate email address format.
        
        Args:
            value: Input value to validate
            field_name: Name of field for error messages
            required: Whether field is required
            
        Returns:
            Validated email address
            
        Raises:
            ValidationError: If validation fails
        """
        email = InputValidator.validate_string(value, field_name, max_length=254, required=required)
        
        if email and not InputValidator.EMAIL_PATTERN.match(email):
            raise ValidationError(f"{field_name} is not a valid email address")
            
        return email
    
    @staticmethod
    def validate_port(value: Any, field_name: str, required: bool = True) -> int:
        """
        Validate network port number.
        
        Args:
            value: Input value to validate
            field_name: Name of field for error messages
            required: Whether field is required
            
        Returns:
            Validated port number
            
        Raises:
            ValidationError: If validation fails
        """
        return InputValidator.validate_integer(value, field_name, min_val=1, max_val=65535, required=required)
    
    @staticmethod
    def validate_search_query(value: Any, max_length: int = 500) -> str:
        """
        Validate search query with security considerations.
        
        Args:
            value: Input value to validate
            max_length: Maximum query length
            
        Returns:
            Validated search query
            
        Raises:
            ValidationError: If validation fails
        """
        if not value or not str(value).strip():
            raise ValidationError("Search query cannot be empty")
            
        query = InputValidator.validate_string(value, "Search query", max_length=max_length, required=True)
        
        # Additional search-specific validation
        if len(query) < 2:
            raise ValidationError("Search query must be at least 2 characters")
            
        return query
    
    @staticmethod
    def validate_filename(value: Any, field_name: str) -> str:
        """
        Validate filename with path traversal prevention.
        
        Args:
            value: Input value to validate
            field_name: Name of field for error messages
            
        Returns:
            Validated filename
            
        Raises:
            ValidationError: If validation fails
        """
        filename = InputValidator.validate_string(value, field_name, max_length=255, required=True)
        
        # Path traversal prevention
        if '..' in filename or filename.startswith('/') or '\\' in filename:
            raise ValidationError(f"{field_name} contains invalid path characters")
            
        # Check for dangerous characters
        dangerous_patterns = ['<', '>', ':', '"', '|', '?', '*']
        for pattern in dangerous_patterns:
            if pattern in filename:
                raise ValidationError(f"{field_name} contains invalid characters")
                
        return filename
