"""
Query Reframer for generating multiple perspectives of the same query.

This module provides professor, child, and domain-specific reframing capabilities
to improve search recall by generating different perspectives of user queries.
"""

import logging
from typing import Dict, Any, Optional
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

class QueryReframer:
    """Generates multiple perspectives of user queries for improved search recall."""
    
    def __init__(self, config):
        """
        Initialize the query reframer.
        
        Args:
            config: Configuration object containing model settings
        """
        self.config = config
        self.reframer = OllamaLLM(
            model=config.CLASSIFICATION_MODEL,  # Use fast model for reframing
            base_url=config.CLASSIFICATION_BASE_URL,
            temperature=0.3  # Slight creativity for reframing
        )
    
    def reframe_as_professor(self, query: str) -> str:
        """
        Reframe query as if a professor was asking it.
        
        Args:
            query: Original user query
            
        Returns:
            Professor-style reframed query
        """
        professor_prompt = """Rewrite this question using sophisticated academic language and scholarly terminology. 
Respond with ONLY the rewritten question - no preamble, no explanations, no introductory text.

Original question: {user_prompt}

Academic version:"""

        try:
            response = self.reframer.invoke(professor_prompt.format(user_prompt=query))
            result = str(response).strip()
            
            # Clean up common LLM response patterns and unwanted preamble
            unwanted_phrases = [
                "I'd be delighted to", "Here's my attempt", "I would rephrase", 
                "Allow me to", "Certainly", "Here's a rewritten version",
                "I can help you rephrase", "Let me rewrite", "Here's the academic version",
                "Academic version:", "Professor version:", "Scholarly version:"
            ]
            
            # Remove unwanted preamble text
            for phrase in unwanted_phrases:
                if result.lower().startswith(phrase.lower()):
                    # Find where the actual question starts
                    colon_index = result.find(':')
                    if colon_index != -1:
                        result = result[colon_index + 1:].strip()
                    else:
                        # Try to find quote marks or other indicators
                        quote_start = result.find('"')
                        if quote_start != -1:
                            quote_end = result.rfind('"')
                            if quote_end > quote_start:
                                result = result[quote_start + 1:quote_end]
                        else:
                            # Just remove the first sentence if it contains the phrase
                            sentences = result.split('. ')
                            if len(sentences) > 1:
                                result = '. '.join(sentences[1:])
                    break
            
            # Clean up quotes and extra whitespace
            result = result.strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            
            # Ensure we have a meaningful reframe
            if len(result) > 10 and result.lower() != query.lower():
                logger.debug(f"Professor reframe: '{query}' -> '{result}'")
                return result
            else:
                logger.warning(f"Professor reframing failed, using original: {query}")
                return query
                
        except Exception as e:
            logger.error(f"Professor reframing failed: {e}")
            return query
    
    def reframe_as_child(self, query: str) -> str:
        """
        Reframe query as if a child was asking it.
        
        Args:
            query: Original user query
            
        Returns:
            Child-style reframed query
        """
        child_prompt = """Rewrite this question as if a curious 5-year-old child was asking it. 
Use simple language and basic concepts. Respond with ONLY the rewritten question - no explanations or extra text.

Original question: {user_prompt}

Child version:"""

        try:
            response = self.reframer.invoke(child_prompt.format(user_prompt=query))
            result = str(response).strip()
            
            # Clean up common LLM response patterns and unwanted preamble
            unwanted_phrases = [
                "Here's a rewritten version", "Here's how a child might ask",
                "A child might say", "Child version:", "Simple version:",
                "Here's the question", "A 5-year-old might ask"
            ]
            
            # Remove unwanted preamble text
            for phrase in unwanted_phrases:
                if phrase.lower() in result.lower():
                    # Find where the actual question starts
                    colon_index = result.find(':')
                    if colon_index != -1:
                        result = result[colon_index + 1:].strip()
                    break
            
            # Clean up quotes and extra whitespace
            result = result.strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            
            # Ensure we have a meaningful reframe
            if len(result) > 10 and result.lower() != query.lower():
                logger.debug(f"Child reframe: '{query}' -> '{result}'")
                return result
            else:
                logger.warning(f"Child reframing failed, using original: {query}")
                return query
                
        except Exception as e:
            logger.error(f"Child reframing failed: {e}")
            return query
            logger.error(f"Child reframing failed: {e}")
            return query
    
    def reframe_for_domain(self, query: str, domain: str, domain_confidence: int) -> Optional[str]:
        """
        Reframe query for a specific professional domain with expert perspectives.
        
        Args:
            query: Original user query
            domain: Professional domain (legal, medical, technical, etc.)
            domain_confidence: Confidence score (1-10) for domain detection
            
        Returns:
            Domain-specific reframed query, or None if confidence is too low
        """
        # Only reframe if we have reasonable confidence in domain detection
        if domain_confidence < 6:
            logger.debug(f"Domain confidence too low ({domain_confidence}) for reframing")
            return None

        # Domain-specific expert prompts
        domain_prompts = {
            'legal': """Rewrite this question as if a senior attorney or legal expert was asking it. 
Use legal terminology, cite relevant legal concepts, and frame it with jurisprudential precision.

Original: {user_prompt}
Legal expert version:""",
            
            'medical': """Rewrite this question as if a physician or medical researcher was asking it.
Use medical terminology, clinical concepts, and healthcare professional language.

Original: {user_prompt}
Medical professional version:""",
            
            'technical': """Rewrite this question as if a senior software engineer or technical architect was asking it.
Use technical jargon, programming concepts, system design terminology, and engineering precision.

Original: {user_prompt}
Technical expert version:""",
            
            'business': """Rewrite this question as if a business executive or management consultant was asking it.
Use business terminology, strategic concepts, and corporate language with focus on ROI and outcomes.

Original: {user_prompt}
Business executive version:""",
            
            'academic': """Rewrite this question as if a university professor or research scholar was asking it.
Use academic terminology, scholarly concepts, and research-oriented language with theoretical depth.

Original: {user_prompt}
Academic researcher version:""",
            
            'scientific': """Rewrite this question as if a research scientist or principal investigator was asking it.
Use scientific methodology, research terminology, and empirical analysis concepts.

Original: {user_prompt}
Research scientist version:""",
            
            'financial': """Rewrite this question as if a financial analyst or investment professional was asking it.
Use financial terminology, market concepts, and quantitative analysis language.

Original: {user_prompt}
Financial expert version:""",
            
            'history': """Rewrite this question as if a history professor or historical researcher was asking it.
Use historical methodology, chronological context, primary source concepts, and historiographical terminology.
Frame it with attention to historical periods, causation, and documentary evidence.

Original: {user_prompt}
History professor version:""",
            
            'general': """Rewrite this question as if a knowledgeable professional with deep 
expertise was asking it. Use more precise terminology and professional language while preserving the core intent.

Original: {user_prompt}
Professional version:"""
        }

        # Get the appropriate prompt for the domain
        domain_prompt = domain_prompts.get(domain, domain_prompts['general'])

        try:
            response = self.reframer.invoke(domain_prompt.format(user_prompt=query))
            result = str(response).strip()
            
            # Clean up common LLM response patterns
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            
            # Ensure we have a meaningful reframe
            if len(result) > 10 and result.lower() != query.lower():
                logger.debug(f"Domain ({domain}) reframe: '{query}' -> '{result}'")
                return result
            else:
                logger.warning(f"Domain reframing failed for {domain}, using original: {query}")
                return None
                
        except Exception as e:
            logger.error(f"Domain reframing failed for {domain}: {e}")
            return None
    
    def reframe_for_email_domain(self, query: str, domain: str) -> str:
        """
        Reframe query for email-specific domain context.
        
        Args:
            query: Original user query
            domain: Professional domain
            
        Returns:
            Email-domain-specific reframed query
        """
        email_domain_prompt = """Rewrite this email-related question as if a {domain} professional was 
searching through their email communications. Focus on {domain}-specific 
email terminology, communication patterns, and relevant topics.

Original: {user_prompt}
Domain: {domain}

Email search version for {domain} professional:"""

        try:
            response = self.reframer.invoke(email_domain_prompt.format(
                user_prompt=query,
                domain=domain
            ))
            result = str(response).strip()
            
            # Clean up common LLM response patterns
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            
            # Ensure we have a meaningful reframe
            if len(result) > 10 and result.lower() != query.lower():
                logger.debug(f"Email domain ({domain}) reframe: '{query}' -> '{result}'")
                return result
            else:
                logger.warning(f"Email domain reframing failed, using original: {query}")
                return query
                
        except Exception as e:
            logger.error(f"Email domain reframing failed: {e}")
            return query
    
    def generate_all_perspectives(self, query: str, domain_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all perspective reframes for a query.
        
        Args:
            query: Original user query
            domain_info: Domain detection information
            
        Returns:
            Dict containing all reframed perspectives
        """
        perspectives = {
            'original': query,
            'professor': self.reframe_as_professor(query),
            'child': self.reframe_as_child(query)
        }
        
        # Add domain-specific reframe if confidence is high enough
        domain = domain_info.get('domain', 'general')
        confidence = domain_info.get('confidence', 5)
        
        domain_reframe = self.reframe_for_domain(query, domain, confidence)
        if domain_reframe:
            perspectives['domain'] = domain_reframe
        
        return perspectives