<!-- 
STRICT RULES:
1. Do not include design instructions in the code or documentation only describe the functionality.
2. Do not use backward compatibility code. This is a new application.
3. Do not use stubbed in data, place holder code, or fake wireframes.
4. Do not use fallbacks for failures, use exceptions with logging to expose the problem.
5. Do not modify data at the database level
6. Do not run commands in the terminal with more than 5 lines.
7. Source the python environment before running code in the terminal. source .venv/bin/activate
8. Move depricated files to backup/ folder -->
# RAG Knowledge Base Manager - Development Rules & Standards

## 📋 Overview
This document contains the development standards, patterns, and requirements for a python project. These rules ensure consistency, maintainability, and proper functionality across all components.

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
- **Database Schema**: Document all database schemas
- **README updates**: Keep project documentation current
- **Configuration documentation**: Document all configuration options with comments

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
- **Environment variables**: All configuration should be externally configurable via environment variables
- **Default values**: Provide sensible defaults for all configuration options
- **Type conversion**: Use proper type conversion (int(), float(), bool()) for environment variables
- **Configuration validation**: Validate configuration values at startup

### 8. Security Best Practices (MANDATORY)
- **ALWAYS follow security best practices in all code**
- **Input validation**: Validate and sanitize all user inputs
- **Environment variables**: Use .env files for sensitive configuration (never hardcode secrets)
- **Authentication**: Implement proper authentication and authorization
- **Password storage**: Any passwords stored to a database must be encrypted
- **Data protection**: Encrypt sensitive data in transit and at rest
- **Error handling**: Never expose sensitive information in error messages
- **Dependencies**: Regularly audit and update dependencies for security vulnerabilities
- **File permissions**: Use appropriate file permissions and access controls
- **Logging security**: Never log sensitive information (passwords, tokens, PII)

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

### 5. Error Handling & Logging Security
- **Secure error messages**: Never expose sensitive information in errors
- **Log security**: Never log passwords, tokens, or sensitive data
- **Error logging**: Log security events for monitoring
- **Stack traces**: Don't expose stack traces in production
- **Audit logging**: Implement audit trails for security events

### 6. Dependency Security
- **Regular audits**: Regularly audit dependencies for vulnerabilities
- **Automated scanning**: Use tools like safety, bandit, or dependabot
- **Minimal dependencies**: Only include necessary dependencies
- **Version pinning**: Pin dependency versions for security
- **Security updates**: Promptly update dependencies with security fixes

---

## 🗄️ Database & Metrics Standards

### 1. Metrics Storage
- **Location**: `databases/metrics.db` (active server metrics)
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
- **Performance validation**: Check metrics storage in database
- **Dashboard testing**: Verify visual updates and functionality
- **Error scenarios**: Test failure cases and error handling

---

## 🧪 Testing Standards (MANDATORY)

### 1. Test File Organization
- **ALL test files MUST be created in the `/tests` folder**
- **No fake or stubbed in data/methods/functions**: Test must test actual implemented functions and not create fake or stubbed in data or method/processes.
- **Naming convention**: `test_*.py` (e.g., `test_api_monitor.py`, `test_metrics_store.py`)
- **No exceptions**: ALL testing code must be in the designated `/tests` directory

### 2. Test Categories
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
- **Run from project root**: Tests should be runnable from `/Users/tonyphilip/Code/rag-document-handler/tests`
- **Independent**: Each test MUST be able to run independently
- **Deterministic**: Tests MUST produce consistent results
- **Fast**: Unit tests should complete in < 1 second

### 6. Test Data Management
- **Test database**: Use real data when testing database functions
- **Cleanup**: Remove test data after test completion

### 7. Unit Testing Requirements (MANDATORY)
- **EVERY new method, function, or class MUST have corresponding unit tests**
- **Test coverage**: Minimum 80% code coverage for all new code
- **Test-driven development**: Write tests BEFORE implementing new functionality when possible
- **Edge cases**: Test error conditions, boundary values, and exceptional scenarios
- **Integration validation**: Test that components work together correctly after refactoring
- **Automated testing**: All tests MUST be runnable via `pytest` command
- **CI/CD ready**: Tests should be suitable for continuous integration pipelines

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
- **Regression prevention**: Create specific tests to prevent future accidental stubbing of restored functionality

---
