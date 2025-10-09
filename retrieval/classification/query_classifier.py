"""
Query Classifier using proper one-shot LLM calls with structured output.

This module provides query classification using single, complete LLM requests
that return all needed information in one response with proper error handling.
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class QueryClassifier:
    """Query classifier using one-shot LLM calls with structured output."""

    def __init__(self, config: Any) -> None:
        """
        Initialize the query classifier.
        
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

    def classify_query_intent(self, query: str) -> str:
        """
        Simple classification returning just the type for backward compatibility.
        
        Args:
            query: User's search query
            
        Returns:
            'email' or 'general'
        """
        result = self.classify_with_confidence(query)
        return result['classification']['type']

    def classify_with_confidence(self, query: str) -> Dict[str, Any]:
        """
        One-shot classification with complete structured response.
        
        Args:
            query: User's search query
            
        Returns:
            Dict containing complete classification information
        """
        # Single, complete prompt requesting all information at once
        prompt = f"""Analyze this user query and provide a complete classification response in valid JSON format.

Query: "{query}"

Respond with exactly this JSON structure (no additional text):
{{
    "classification": {{
        "type": "general" or "email",
        "confidence": <number 1-10>,
        "reasoning": "<brief explanation>"
    }}
}}

Classification Rules:
- "email" if query asks about emails, messages, communications, senders, recipients, email addresses, email subjects, email dates, email folders, email attachments
- "general" for all other queries about documents, knowledge, facts, uploaded files, URLs, web content

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
                
                # Clean response (remove markdown formatting if present)
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                
                # Check again after cleaning
                if not response_text:
                    raise ValueError("LLM response was empty after cleaning")
                
                # Parse complete JSON response
                result = json.loads(response_text)
                
                # Validate structure in one shot
                if not self._validate_classification_structure(result):
                    raise ValueError("Invalid classification structure returned by LLM")
                    
                logger.info(f"Query classified as '{result['classification']['type']}' "
                           f"(confidence: {result['classification']['confidence']}/10)")
                
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
        
        # All attempts failed
        logger.error(f"One-shot query classification failed after {max_retries + 1} attempts: {last_error}")
        # Clean failure - no partial parsing attempts
        raise RuntimeError(f"Query classification failed: {last_error}")
    
    def _validate_classification_structure(self, result: dict) -> bool:
        """
        Validate the complete structure was returned properly.
        
        Args:
            result: Parsed JSON result from LLM
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            classification = result['classification']
            confidence = classification.get('confidence')
            
            # Convert float confidence to int if needed
            if isinstance(confidence, float):
                confidence = int(confidence)
                classification['confidence'] = confidence
            
            return (
                classification['type'] in ['general', 'email'] and
                isinstance(confidence, (int, float)) and
                1 <= confidence <= 10 and
                isinstance(classification['reasoning'], str) and
                len(classification['reasoning']) > 0
            )
        except (KeyError, TypeError):
            return False