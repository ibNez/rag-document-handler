# Alfi Voice Assistant - Development Rules & Standards

## 📋 Overview
This document contains the development standards, patterns, and requirements for the Alfi Voice Assistant project. These rules ensure consistency, maintainability, and proper functionality across all components.

---

## 🐍 Python Development Best Practices (MANDATORY)

### 1. Code Quality Standards
- **ALWAYS follow Python development best practices**
- **PEP 8 compliance**: Use consistent formatting and style
- **Type hints**: All functions MUST have proper type annotations
- **Docstrings**: All modules, classes, and functions MUST have descriptive docstrings
- **Error handling**: Use proper exception handling with specific exception types
- **Code organization**: Follow single responsibility principle
- **Code cleanliness**: Always leave code tidied up with no unused code or test variables

### 2. Project Structure Standards
- **ALWAYS use Python development best practices for codebase folder structure**
- **Standard structure**: Follow Python packaging conventions
- **Module organization**: Logical grouping of related functionality
- **Import hierarchy**: Clear dependency management
- **Configuration**: Centralized configuration management

### 3. Package Management Standards
- **ALWAYS keep package requirements up to date in the pyproject.toml file**
- **NEVER create new requirements*.txt files** - All dependencies must be managed in pyproject.toml
- **Dependency categories**: Use appropriate sections (dependencies, dev, optional-dependencies)
- **Version constraints**: Use appropriate version specifiers (>=, ~=, ==)
- **Regular updates**: Keep dependencies current and secure

### 4. Documentation Requirements
- **ALWAYS update relevant documentation for code being changed**
- **Inline documentation**: Update docstrings for modified functions
- **README updates**: Keep project documentation current
- **API documentation**: Document all public interfaces
- **Change logs**: Document significant changes

### 5. Code Annotation Standards
- **ALWAYS annotate the code properly**
- **Type hints**: Use typing module for complex types
- **Function signatures**: Clear parameter and return types
- **Variable annotations**: For complex data structures
- **Class annotations**: Properties and methods must be typed

### 6. Logging Standards
- **ALWAYS maintain consistent logging of process steps**
- **Log levels**: Use appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **Structured logging**: Include relevant context and metadata
- **Process tracking**: Log key steps in complex operations
- **Error logging**: Include full error context and stack traces

### 7. Configuration Management Standards (MANDATORY)
- **ALWAYS use configuration variables when they are available**
- **Import from config**: Use `from config import VARIABLE_NAME` instead of hardcoding values
- **Environment variables**: All configuration should be externally configurable via environment variables
- **Default values**: Provide sensible defaults in config.py for all configuration options
- **Centralized configuration**: Keep all configuration in config.py, never scatter across multiple files
- **Type conversion**: Use proper type conversion (int(), float(), bool()) for environment variables
- **Configuration documentation**: Document all configuration options with comments
- **Configuration validation**: Validate configuration values at startup

**Examples of proper configuration usage:**
```python
# ✅ CORRECT - Use configuration variables
from config import WHISPER_MODEL_SIZE, VOICE_VERIFICATION_THRESHOLD, SERVER_PORT

# Use the configured values
model = whisper.load_model(WHISPER_MODEL_SIZE)
threshold = VOICE_VERIFICATION_THRESHOLD
server.run(port=SERVER_PORT)

# ❌ INCORRECT - Hardcoded values
model = whisper.load_model("large")  # Should use WHISPER_MODEL_SIZE
threshold = 0.45  # Should use VOICE_VERIFICATION_THRESHOLD
server.run(port=8000)  # Should use SERVER_PORT
```

### 8. Security Best Practices (MANDATORY)
- **ALWAYS follow security best practices in all code**
- **Input validation**: Validate and sanitize all user inputs
- **Environment variables**: Use .env files for sensitive configuration (never hardcode secrets)
- **Authentication**: Implement proper authentication and authorization
- **Data protection**: Encrypt sensitive data in transit and at rest
- **Error handling**: Never expose sensitive information in error messages
- **Dependencies**: Regularly audit and update dependencies for security vulnerabilities
- **File permissions**: Use appropriate file permissions and access controls
- **API security**: Implement rate limiting, CORS, and proper HTTP headers
- **Logging security**: Never log sensitive information (passwords, tokens, PII)

### 9. Code Examples
```python
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

def process_audio_data(
    audio_data: bytes,
    transcript: Optional[str] = None,
    speaker_id: Optional[str] = None
) -> Dict[str, Union[str, int, List[Dict]]]:
    """
    Process audio data with speaker diarization.
    
    Args:
        audio_data: Raw audio bytes to process
        transcript: Optional transcript text
        speaker_id: Optional known speaker identifier
        
    Returns:
        Dictionary containing processing results with segments and metadata
        
    Raises:
        ValueError: If audio_data is empty or invalid
        ProcessingError: If audio processing fails
    """
    logger.info(f"Starting audio processing for {len(audio_data)} bytes")
    
    try:
        # Process audio logic here
        result = {"success": True, "segments": []}
        logger.info("Audio processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}", exc_info=True)
        raise
```

---

## 🔒 Security Best Practices (MANDATORY)

### 1. Input Validation & Sanitization
- **ALL user inputs MUST be validated and sanitized**
- **Type checking**: Validate input types and formats
- **Length limits**: Enforce maximum input lengths
- **Pattern validation**: Use regex or validation libraries
- **SQL injection prevention**: Use parameterized queries
- **XSS prevention**: Sanitize HTML and JavaScript inputs
- **Path traversal prevention**: Validate file paths and names

### 2. Authentication & Authorization
- **Implement proper authentication mechanisms**
- **Session management**: Use secure session handling
- **Token validation**: Properly validate API tokens and JWTs
- **Role-based access**: Implement proper authorization controls
- **Multi-factor authentication**: Consider MFA for sensitive operations
- **Password security**: Use proper password hashing (bcrypt, scrypt, etc.)

### 3. Environment & Configuration Security
- **NEVER hardcode secrets in source code**
- **Environment variables**: Use .env files for sensitive configuration
- **Secret management**: Use proper secret management systems
- **Configuration validation**: Validate all configuration parameters
- **Default security**: Ensure secure defaults for all settings
- **Production vs development**: Separate configurations for different environments

### 4. Data Protection
- **Encryption in transit**: Use HTTPS/TLS for all communications
- **Encryption at rest**: Encrypt sensitive data in databases
- **PII handling**: Implement proper handling of personally identifiable information
- **Data retention**: Implement proper data retention policies
- **Backup security**: Secure backup data with encryption
- **Data anonymization**: Anonymize data when possible

### 5. API Security
- **Rate limiting**: Implement proper rate limiting for all endpoints
- **CORS configuration**: Configure Cross-Origin Resource Sharing properly
- **HTTP headers**: Use security headers (HSTS, CSP, X-Frame-Options)
- **Request validation**: Validate all API requests
- **Response security**: Ensure responses don't leak sensitive information
- **API versioning**: Implement proper API versioning for security updates

### 6. Error Handling & Logging Security
- **Secure error messages**: Never expose sensitive information in errors
- **Log security**: Never log passwords, tokens, or sensitive data
- **Error logging**: Log security events for monitoring
- **Stack traces**: Don't expose stack traces in production
- **Audit logging**: Implement audit trails for security events

### 7. Dependency Security
- **Regular audits**: Regularly audit dependencies for vulnerabilities
- **Automated scanning**: Use tools like safety, bandit, or dependabot
- **Minimal dependencies**: Only include necessary dependencies
- **Version pinning**: Pin dependency versions for security
- **Security updates**: Promptly update dependencies with security fixes

### 8. Code Security Examples
```python
import os
import hashlib
import secrets
from typing import Optional
import logging
from functools import wraps

# Secure logging - never log sensitive data
security_logger = logging.getLogger('security')

def validate_input(input_value: str, max_length: int = 255) -> str:
    """
    Validate and sanitize user input.
    
    Args:
        input_value: Raw input from user
        max_length: Maximum allowed length
        
    Returns:
        Sanitized input string
        
    Raises:
        ValueError: If input is invalid or too long
    """
    if not isinstance(input_value, str):
        raise ValueError("Input must be a string")
    
    if len(input_value) > max_length:
        raise ValueError(f"Input too long (max {max_length} characters)")
    
    # Remove potentially dangerous characters
    sanitized = input_value.strip()
    # Additional sanitization based on context
    return sanitized

def secure_hash_password(password: str) -> str:
    """
    Securely hash a password using a strong algorithm.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    # Use a strong, slow hashing algorithm
    salt = secrets.token_hex(32)
    password_hash = hashlib.pbkdf2_hmac('sha256', 
                                       password.encode('utf-8'), 
                                       salt.encode('utf-8'), 
                                       100000)  # 100,000 iterations
    return f"{salt}:{password_hash.hex()}"

def require_auth(f):
    """
    Decorator to require authentication for API endpoints.
    """
    @wraps(f)
    async def decorated_function(request, *args, **kwargs):
        # Check authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not validate_token(auth_header):
            security_logger.warning(f"Unauthorized access attempt to {request.url}")
            return JSONResponse(
                {"error": "Unauthorized"}, 
                status_code=401
            )
        return await f(request, *args, **kwargs)
    return decorated_function

def get_secure_config(key: str, default: Optional[str] = None) -> str:
    """
    Securely retrieve configuration values.
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value
        
    Raises:
        ValueError: If required config is missing
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Required configuration '{key}' not found")
    return value
```

---

## 🛠️ API Development Standards

### 1. Performance Monitoring (MANDATORY)
- **ALL API endpoints MUST use the `@api_performance_monitor` decorator**
- **Pattern**: `@api_performance_monitor("endpoint_name")`
- **Purpose**: Automatic runtime tracking, logging, and metrics storage
- **Example**:
  ```python
  @api_performance_monitor("config")
  async def config_api(request: Request) -> JSONResponse:
      # API implementation
  ```

### 2. API Response Standards
- **ALL APIs MUST return runtime information in responses where applicable**
- **Success responses**: Include `success: true` field
- **Error responses**: Include `success: false` and `error` field
- **Timing**: Performance metrics automatically logged via decorator
- **Status codes**: Use appropriate HTTP status codes

### 3. Error Handling Pattern
```python
try:
    # API logic
    return JSONResponse({"success": True, "data": result})
except Exception as e:
    logger.error(f"API endpoint_name error: {str(e)}")
    return JSONResponse(
        {"success": False, "error": str(e)},
        status_code=500
    )
```

### 4. Logging Requirements
- **ALL API calls MUST be logged with INFO level**
- **Pattern**: `logger.info(f"API {name} completed in {duration:.2f}ms")`
- **Errors**: Use `logger.error()` with full exception details
- **Success**: Use `logger.info()` for successful operations

---

## 🗄️ Database & Metrics Standards

### 1. Metrics Storage
- **Location**: `src/server/logs/metrics.db` (active server metrics)
- **Legacy**: `logs/metrics.db` (not used by running server)
- **Pattern**: All performance metrics MUST be stored via `MetricsStore`
- **Timing**: Use `store_step_timing(step_id, duration_ms)` for all operations

### 2. Database Schema Requirements
- **Timestamps**: All metrics MUST include ISO format timestamps
- **Duration**: Store in milliseconds (float precision)
- **Success tracking**: Include success/failure status
- **Request ID**: Include where applicable for tracing

### 3. Performance Data Structure
```python
{
    "step_id": "api_endpoint_name",
    "duration_ms": 123.45,
    "timestamp": "2025-07-08T20:51:22.960839",
    "success": true,
    "error_message": null,
    "metadata": {}
}
```

---

## 🎨 Frontend Development Standards

### 1. Diagnostics Dashboard Requirements
- **Real-time updates**: Auto-refresh every 10 seconds
- **Performance tracking**: Track frontend rendering times
- **Error handling**: Graceful degradation on API failures
- **Responsive design**: Support for different screen sizes

### 2. Performance Visualization
- **Color coding**: Green (<30%), Yellow (30-70%), Red (>70%)
- **Progress bars**: Visual indicators for all metrics
- **Tooltips**: Detailed information on hover
- **Units**: Consistent units (ms for time, % for rates)

### 3. Log Display Standards
- **Filtering**: Support for all log levels (DEBUG, INFO, WARNING, ERROR, SUCCESS)
- **Real-time**: Live log streaming with minimal visual disruption
- **Optimization**: Change detection to avoid unnecessary DOM updates
- **Limits**: Maximum 100 entries displayed, auto-scroll to bottom

---

## 📁 File Organization Standards

### 2. Python Import Standards
- **Relative imports**: Use `from .module import item` for local modules within same package
- **Absolute imports**: Use for external libraries and cross-package imports
- **Import order**: Standard library, third-party, local imports (follow PEP 8)
- **Type imports**: Use `from typing import` for type hints

### 3. Python Code Organization
- **Module structure**: Each module should have a clear, single purpose
- **Class organization**: Related functionality grouped in classes
- **Function organization**: Pure functions separate from stateful operations
- **Constants**: All caps, defined in config.py or module-level
- **Decorators**: Define before the functions they'll be used on

---

## 🔧 Development Workflow Standards

### 1. Testing Requirements
- **API testing**: Use `curl` with `jq` for JSON parsing
- **Performance validation**: Check metrics storage in database
- **Dashboard testing**: Verify visual updates and functionality
- **Error scenarios**: Test failure cases and error handling

### 2. Server Management
- **Port management**: Use `lsof -i :8000` to check port usage
- **Process cleanup**: Use `pkill -f "python.*run_alfi.py"` to stop servers
- **Restart procedure**: Kill existing processes before starting new ones

### 3. Code Changes
- **Decorator application**: Apply `@api_performance_monitor` to ALL new API endpoints
- **Metrics storage**: Ensure all new operations store timing metrics
- **Dashboard updates**: Add new metrics to frontend visualization
- **Documentation**: Update this file with new standards

---

## 🧪 Testing Standards (MANDATORY)

### 1. Test File Organization
- **ALL test files MUST be created in the `/tests` folder**
- **Naming convention**: `test_*.py` (e.g., `test_api_monitor.py`, `test_metrics_store.py`)
- **Test assets**: Store test fixtures, audio files, and data in `/tests` folder
- **No exceptions**: ALL testing code must be in the designated `/tests` directory

### 2. Test File Structure
```python
# tests/test_example.py
"""
Test module for [component being tested]
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'server'))

# Import components to test
from logging_utils import MetricsStore
from app import api_performance_monitor

def test_component_functionality():
    """Test description"""
    # Test implementation
    pass
```

### 3. Test Categories
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **API Tests**: Test endpoint functionality and performance
- **Performance Tests**: Validate timing and metrics storage
- **Error Handling Tests**: Test failure scenarios
- **Security Tests**: Test input validation, authentication, and authorization

### 4. Test File Requirements
- **Docstrings**: ALL test files MUST have module-level docstrings
- **Function docs**: Each test function MUST have descriptive docstring
- **Path setup**: Include proper path configuration for imports
- **Cleanup**: Tests MUST clean up after themselves (temp files, database entries)

### 5. Test Execution Standards
- **Run from project root**: Tests should be runnable from `/Users/tonyphilip/Code/Alfi-web`
- **Independent**: Each test MUST be able to run independently
- **Deterministic**: Tests MUST produce consistent results
- **Fast**: Unit tests should complete in < 1 second

### 6. Test Data Management
- **Test database**: Use separate database for testing (not production metrics.db)
- **Test fixtures**: Store in `/tests` folder with descriptive names
- **Mock data**: Use realistic but safe test data
- **Cleanup**: Remove test data after test completion

### 7. Unit Testing Requirements (MANDATORY)
- **EVERY new method, function, or class MUST have corresponding unit tests**
- **Test coverage**: Minimum 80% code coverage for all new code
- **Test-driven development**: Write tests BEFORE implementing new functionality when possible
- **Regression prevention**: Tests MUST catch API breaking changes, import failures, and core functionality issues
- **Mock external dependencies**: Use mocks for file system, network calls, and external services
- **Edge cases**: Test error conditions, boundary values, and exceptional scenarios
- **Integration validation**: Test that components work together correctly after refactoring
- **Automated testing**: All tests MUST be runnable via `pytest` command
- **CI/CD ready**: Tests should be suitable for continuous integration pipelines

**Examples of required unit tests:**
```python
# For every new API endpoint
def test_text_chat_api_success():
    """Test successful text chat with LLM integration"""
    pass

def test_text_chat_api_missing_message():
    """Test text chat with missing message parameter"""
    pass

def test_text_chat_api_llm_failure():
    """Test text chat when LLM service is unavailable"""
    pass

# For every new class/module
def test_hybrid_speaker_identifier_initialization():
    """Test HybridSpeakerIdentifier initializes correctly with all strategies"""
    pass

def test_rolling_buffer_add_segment():
    """Test adding audio segments to rolling buffer"""
    pass

def test_voice_cache_manager_speaker_storage():
    """Test voice cache stores and retrieves speaker data correctly"""
    pass
```

### 8. Testing Framework Requirements
- **Framework**: Use `pytest` for all unit and integration tests
- **Assertions**: Use `pytest` assertions and descriptive error messages
- **Fixtures**: Create reusable test fixtures for common test data
- **Parametrized tests**: Use `@pytest.mark.parametrize` for testing multiple scenarios
- **Test organization**: Group related tests in test classes
- **Continuous monitoring**: Run tests after every significant code change

### 9. Code Restoration Standards (MANDATORY)
- **NEVER create stubs when restoring functionality from backup**
- **ALWAYS restore the original implementation from backup files when available**
- **Verify restoration**: Check that restored code uses the same services and methods as the original
- **Source verification**: Use `inspect.getsource()` to verify restored code matches expected patterns
- **Integration testing**: Test that restored functionality works with existing components
- **Frontend compatibility**: Ensure restored APIs maintain the expected response format for frontend consumption
- **Regression prevention**: Create specific tests to prevent future accidental stubbing of restored functionality

---
