"""
Synonym Expander for generating query variants with alternative terminology.

This module provides synonym expansion capabilities to improve search recall
by generating alternative phrasings and terminology for user queries.
"""

import logging
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

class SynonymExpander:
    """Generates synonym variants of user queries for improved search recall."""
    
    def __init__(self, config):
        """
        Initialize the synonym expander.
        
        Args:
            config: Configuration object containing model settings
        """
        self.config = config
        self.expander = OllamaLLM(
            model=config.CLASSIFICATION_MODEL,  # Use fast model for expansion
            base_url=config.CLASSIFICATION_BASE_URL,
            temperature=0.4  # Moderate creativity for synonym generation
        )
        
        # Cache for common expansions
        self._cache = {}
    
    def generate_synonym_variants(self, query: str, max_variants: int = 5) -> List[str]:
        """
        Generate synonym-based query variants.
        
        Args:
            query: Original user query
            max_variants: Maximum number of variants to generate
            
        Returns:
            List of synonym variants
        """
        # Check cache first
        cache_key = f"{query.lower()}:{max_variants}"
        if cache_key in self._cache:
            logger.debug(f"Using cached synonym variants for: {query}")
            return self._cache[cache_key]
        
        synonym_prompt = """Generate {max_variants} alternative ways to ask the same question using different 
terminology, synonyms, and phrasing. Each variant should preserve the 
original meaning while using different vocabulary.

Original: {user_prompt}

Return only the variants, one per line, numbered:
1. [variant 1]
2. [variant 2]
3. [variant 3]
4. [variant 4]
5. [variant 5]"""

        try:
            response = self.expander.invoke(synonym_prompt.format(
                user_prompt=query,
                max_variants=max_variants
            ))
            result_text = str(response).strip()
            
            # Parse numbered variants
            variants = self._parse_variants(result_text, query)
            
            # Cache the result
            self._cache[cache_key] = variants
            
            logger.debug(f"Generated {len(variants)} synonym variants for: {query}")
            return variants
            
        except Exception as e:
            logger.error(f"Synonym expansion failed: {e}")
            return []
    
    def _parse_variants(self, response_text: str, original_query: str) -> List[str]:
        """
        Parse variants from LLM response.
        
        Args:
            response_text: Raw LLM response
            original_query: Original query for fallback
            
        Returns:
            List of parsed variants
        """
        variants = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle numbered format: "1. variant text"
            if line[0].isdigit() and '. ' in line:
                variant = line.split('. ', 1)[1].strip()
            # Handle bulleted format: "- variant text"
            elif line.startswith('- '):
                variant = line[2:].strip()
            # Handle direct variant text
            else:
                variant = line.strip()
            
            # Clean up quotes and ensure it's different from original
            variant = variant.strip('"\'')
            if (len(variant) > 10 and 
                variant.lower() != original_query.lower() and
                variant not in variants):
                variants.append(variant)
        
        return variants[:5]  # Maximum 5 variants
    
    def generate_domain_aware_variants(self, query: str, domain: str, 
                                     max_variants: int = 3) -> List[str]:
        """
        Generate domain-specific synonym variants.
        
        Args:
            query: Original user query
            domain: Professional domain
            max_variants: Maximum number of variants to generate
            
        Returns:
            List of domain-specific variants
        """
        domain_synonym_prompts = {
            'legal': """Generate {max_variants} alternative legal terminology variants for this question:
            - Use legal synonyms (case law vs. jurisprudence vs. precedent)
            - Include procedural variations (motion vs. petition vs. application)
            - Consider different legal contexts (civil vs. criminal vs. regulatory)
            
            Original: {user_prompt}
            
            Legal variants:""",
            
            'medical': """Generate {max_variants} alternative medical terminology variants for this question:
            - Use medical synonyms (disease vs. condition vs. disorder)
            - Include clinical variations (treatment vs. therapy vs. intervention)
            - Consider different medical contexts (diagnosis vs. prognosis vs. management)
            
            Original: {user_prompt}
            
            Medical variants:""",
            
            'technical': """Generate {max_variants} alternative technical terminology variants for this question:
            - Use technical synonyms (function vs. method vs. procedure)
            - Include implementation variations (database vs. datastore vs. repository)
            - Consider different technical contexts (API vs. interface vs. endpoint)
            
            Original: {user_prompt}
            
            Technical variants:""",
            
            'business': """Generate {max_variants} alternative business terminology variants for this question:
            - Use business synonyms (revenue vs. income vs. sales)
            - Include strategic variations (plan vs. strategy vs. approach)
            - Consider different business contexts (customer vs. client vs. stakeholder)
            
            Original: {user_prompt}
            
            Business variants:"""
        }
        
        prompt_template = domain_synonym_prompts.get(domain)
        if not prompt_template:
            # Fall back to general synonym expansion
            return self.generate_synonym_variants(query, max_variants)
        
        try:
            response = self.expander.invoke(prompt_template.format(
                user_prompt=query,
                max_variants=max_variants
            ))
            result_text = str(response).strip()
            
            variants = self._parse_variants(result_text, query)
            
            logger.debug(f"Generated {len(variants)} domain-aware ({domain}) variants for: {query}")
            return variants
            
        except Exception as e:
            logger.error(f"Domain-aware synonym expansion failed for {domain}: {e}")
            # Fall back to general synonym expansion
            return self.generate_synonym_variants(query, max_variants)
    
    def expand_for_email_context(self, query: str) -> List[str]:
        """
        Generate email-specific variants.
        
        Args:
            query: Original user query
            
        Returns:
            List of email-specific variants
        """
        email_prompt = """Generate 3 alternative ways to phrase this email-related question:
        - Use email-specific terminology (message vs. email vs. communication)
        - Include sender/recipient variations (from vs. sent by vs. authored by)
        - Consider email metadata (subject vs. title vs. regarding)
        
        Original: {user_prompt}
        
        Email variants:"""

        try:
            response = self.expander.invoke(email_prompt.format(user_prompt=query))
            result_text = str(response).strip()
            
            variants = self._parse_variants(result_text, query)
            
            logger.debug(f"Generated {len(variants)} email-specific variants for: {query}")
            return variants
            
        except Exception as e:
            logger.error(f"Email-specific synonym expansion failed: {e}")
            return []
    
    def clear_cache(self):
        """Clear the synonym expansion cache."""
        self._cache.clear()
        logger.info("Synonym expansion cache cleared")