"""
Ollama service health checking utilities.

This module provides functionality to check the health status of Ollama services
for embeddings, chat, and classification. Follows development rules with proper
error handling, logging, and type hints.
"""

import requests
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class OllamaServiceStatus:
    """
    Data class representing the status of an Ollama service.
    
    Attributes:
        connected: Whether the service is reachable
        version: Ollama version if available
        model_loaded: Whether the required model is loaded
        model_name: Name of the model that should be loaded
        error_message: Error details if connection failed
        response_time_ms: Response time in milliseconds
    """
    connected: bool = False
    version: Optional[str] = None
    model_loaded: bool = False
    model_name: Optional[str] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None


class OllamaHealthChecker:
    """
    Health checker for Ollama services.
    
    Provides methods to check the health and availability of Ollama services
    for embeddings, chat, and classification with proper timeout handling
    and error reporting.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the Ollama health checker.
        
        Args:
            config: Application configuration containing Ollama service details
        """
        self.config = config
        self.timeout = 5  # 5 second timeout for health checks
    
    def check_ollama_service(self, host: str, port: int, model_name: str) -> OllamaServiceStatus:
        """
        Check the health of a specific Ollama service.
        
        Args:
            host: Ollama service host
            port: Ollama service port
            model_name: Expected model name to check
            
        Returns:
            OllamaServiceStatus with connection and model status
        """
        base_url = f"http://{host}:{port}"
        status = OllamaServiceStatus(model_name=model_name)
        
        try:
            import time
            start_time = time.time()
            
            # Check if Ollama is running by hitting the /api/version endpoint
            version_response = requests.get(
                f"{base_url}/api/version",
                timeout=self.timeout
            )
            
            if version_response.status_code == 200:
                status.connected = True
                status.response_time_ms = (time.time() - start_time) * 1000
                
                try:
                    version_data = version_response.json()
                    status.version = version_data.get('version', 'unknown')
                except Exception:
                    status.version = 'unknown'
                
                # Check if the specific model is available
                try:
                    models_response = requests.get(
                        f"{base_url}/api/tags",
                        timeout=self.timeout
                    )
                    
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        models = models_data.get('models', [])
                        
                        # Check if our required model is in the list
                        model_names = [model.get('name', '').split(':')[0] for model in models]
                        base_model_name = model_name.split(':')[0]
                        
                        if base_model_name in model_names:
                            status.model_loaded = True
                        else:
                            status.error_message = f"Model '{model_name}' not found. Available models: {', '.join(model_names) if model_names else 'none'}"
                    else:
                        status.error_message = f"Failed to fetch models list: HTTP {models_response.status_code}"
                        
                except Exception as e:
                    status.error_message = f"Error checking models: {str(e)}"
            else:
                status.error_message = f"Ollama service returned HTTP {version_response.status_code}"
                
        except requests.exceptions.Timeout:
            status.error_message = f"Connection timeout to {base_url}"
            logger.warning(f"Ollama service timeout: {base_url}")
            
        except requests.exceptions.ConnectionError:
            status.error_message = f"Connection refused to {base_url}"
            logger.warning(f"Ollama service connection refused: {base_url}")
            
        except Exception as e:
            status.error_message = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error checking Ollama service {base_url}: {e}")
        
        return status
    
    def check_embedding_service(self) -> OllamaServiceStatus:
        """
        Check the health of the Ollama embedding service.
        
        Returns:
            OllamaServiceStatus for the embedding service
        """
        return self.check_ollama_service(
            self.config.OLLAMA_EMBEDDING_HOST,
            self.config.OLLAMA_EMBEDDING_PORT,
            self.config.EMBEDDING_MODEL
        )
    
    def check_chat_service(self) -> OllamaServiceStatus:
        """
        Check the health of the Ollama chat service.
        
        Returns:
            OllamaServiceStatus for the chat service
        """
        return self.check_ollama_service(
            self.config.OLLAMA_CHAT_HOST,
            self.config.OLLAMA_CHAT_PORT,
            self.config.CHAT_MODEL
        )
    
    def check_classification_service(self) -> OllamaServiceStatus:
        """
        Check the health of the Ollama classification service.
        
        Returns:
            OllamaServiceStatus for the classification service
        """
        return self.check_ollama_service(
            self.config.OLLAMA_CLASSIFICATION_HOST,
            self.config.OLLAMA_CLASSIFICATION_PORT,
            self.config.CLASSIFICATION_MODEL
        )
    
    def check_all_services(self) -> Dict[str, OllamaServiceStatus]:
        """
        Check the health of all Ollama services.
        
        Returns:
            Dictionary mapping service names to their status
        """
        return {
            'embedding': self.check_embedding_service(),
            'chat': self.check_chat_service(),
            'classification': self.check_classification_service()
        }
    
    def get_overall_status(self) -> Dict[str, Any]:
        """
        Get an overall health summary of all Ollama services.
        
        Returns:
            Dictionary with overall status summary
        """
        services = self.check_all_services()
        
        total_services = len(services)
        connected_services = sum(1 for status in services.values() if status.connected)
        models_loaded = sum(1 for status in services.values() if status.model_loaded)
        
        return {
            'services': services,
            'total_services': total_services,
            'connected_services': connected_services,
            'models_loaded': models_loaded,
            'all_healthy': connected_services == total_services and models_loaded == total_services,
            'any_connected': connected_services > 0
        }
