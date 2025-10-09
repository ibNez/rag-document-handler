"""
Domain Detector using proper one-shot LLM calls with structured output.

This module provides domain detection using single, complete LLM requests
that return all needed information in one response with proper error handling.
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class DomainDetector:
    """Domain detector using one-shot LLM calls with structured output."""

    def __init__(self, config: Any) -> None:
        """
        Initialize the domain detector.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        from langchain_ollama import ChatOllama
        self.llm = ChatOllama(
            model=config.CLASSIFICATION_MODEL,
            base_url=config.CLASSIFICATION_BASE_URL,
            temperature=0.0  # Deterministic for classification
        )

    def detect_domain(self, query: str) -> Dict[str, Any]:
        """
        One-shot domain detection with complete structured response.
        
        Args:
            query: User's search query
            
        Returns:
            Dict containing complete domain information
        """
        # Single prompt requesting complete domain analysis
        prompt = f"""Analyze this query and detect its professional domain. Respond with exactly this JSON structure:

Query: "{query}"

{{
    "domain": {{
        "domain": "<domain_name>",
        "confidence": <number 1-10>,
        "secondary": "<secondary_domain_or_null>",
        "reasoning": "<explanation>",
        "keywords": ["<relevant>", "<domain>", "<keywords>"]
    }}
}}

Supported domains: legal, medical, technical, business, academic, scientific, financial, history, general

Rules:
- domain: Most likely domain (use "general" if unclear)
- confidence: 1-10 (10 = very certain)
- secondary: Second most likely domain or null
- reasoning: Brief explanation for the classification
- keywords: 3-5 domain-specific terms found in query

Response:"""
        
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Single LLM call requesting complete structured response
                from langchain_core.messages import HumanMessage
                response = self.llm.invoke([HumanMessage(content=prompt)])
                response_text = response.content.strip()
                
                # Check for empty response
                if not response_text:
                    raise ValueError("LLM returned empty response")
                
                # Clean JSON response
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                
                # Check again after cleaning
                if not response_text:
                    raise ValueError("LLM response was empty after cleaning")
                
                result = json.loads(response_text)
                
                # Validate complete structure
                if not self._validate_domain_structure(result):
                    raise ValueError("Invalid domain structure returned by LLM")
                
                domain_info = result['domain']
                logger.info(f"Domain detected: {domain_info['domain']} "
                           f"(confidence: {domain_info['confidence']}/10)")
                
                return result
                
            except json.JSONDecodeError as e:
                last_error = f"JSON parsing failed: {e}"
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed with JSON error, retrying...")
                    continue
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    continue
                
        # All attempts failed - return fallback to "general" domain
        logger.error(f"One-shot domain detection failed after {max_retries + 1} attempts: {last_error}")
        logger.warning("Falling back to 'general' domain classification")
        
        # Return a safe fallback domain classification
        fallback_result = {
            "domain": {
                "domain": "general",
                "confidence": 1,  # Low confidence since detection failed
                "secondary": None,
                "reasoning": f"Domain detection failed, defaulting to general. Error: {last_error}",
                "keywords": []  # Empty keywords since we couldn't extract any
            }
        }
        
        logger.info("Using fallback domain classification: general (confidence: 1/10)")
        return fallback_result
    
    def _validate_domain_structure(self, result: dict) -> bool:
        """
        Validate complete domain response structure.
        
        Args:
            result: Parsed JSON result from LLM
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            domain = result['domain']
            confidence = domain.get('confidence')
            
            # Convert float confidence to int if needed
            if isinstance(confidence, float):
                confidence = int(confidence)
                domain['confidence'] = confidence
                
            valid_domains = ['legal', 'medical', 'technical', 'business', 
                           'academic', 'scientific', 'financial', 'history', 'general']
            return (
                domain['domain'] in valid_domains and
                isinstance(confidence, (int, float)) and
                1 <= confidence <= 10 and
                isinstance(domain['reasoning'], str) and
                len(domain['reasoning']) > 0 and
                isinstance(domain['keywords'], list) and
                len(domain['keywords']) > 0
            )
        except (KeyError, TypeError):
            return False