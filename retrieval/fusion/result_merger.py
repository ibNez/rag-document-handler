"""
Result Merger for combining and weighting multi-perspective search results.

This module implements sophisticated result fusion algorithms that combine
results from different search perspectives with appropriate weighting and
ranking strategies.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

class ResultMerger:
    """Merges and weights results from multiple search perspectives."""
    
    def __init__(self, config):
        """
        Initialize the result merger.
        
        Args:
            config: Configuration object containing fusion settings
        """
        self.config = config
        
        # Default weights for different perspectives
        self.default_weights = {
            'original': 1.0,
            'professor': 0.8,
            'child': 0.6,
            'domain': 0.7,
            'synonym': 0.5
        }
        
        # Default rank boost multipliers
        self.rank_boost_multipliers = {
            'multiple_methods': 0.1,
            'domain_match': 0.15,
            'synonym_match': 0.05
        }
    
    def merge_weighted_results(self, perspective_results: Dict[str, List[Dict]], 
                             domain_info: Dict[str, Any],
                             max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Merge results with weighted scoring based on domain confidence.
        
        Args:
            perspective_results: Dict mapping perspective names to result lists
            domain_info: Domain detection information
            max_results: Maximum number of results to return
            
        Returns:
            List of merged and ranked results
        """
        logger.info(f"Merging results from {len(perspective_results)} perspectives")
        
        # Adjust weights based on domain confidence
        weights = self._calculate_dynamic_weights(domain_info)
        
        # Track results by chunk ID
        result_map: Dict[str, Dict[str, Any]] = {}
        
        # Process each perspective's results
        for perspective_name, results in perspective_results.items():
            if not results:
                continue
                
            perspective_weight = weights.get(perspective_name, 0.5)
            logger.debug(f"Processing {len(results)} results from {perspective_name} perspective (weight: {perspective_weight})")
            
            for idx, result in enumerate(results):
                chunk_id = self._extract_chunk_id(result)
                
                # If no chunk_id, create a unique identifier using perspective + index
                if not chunk_id:
                    chunk_id = f"{perspective_name}_{idx}"
                
                # Calculate position-based score (higher rank = lower score)
                position_score = 1.0 / (idx + 1)  # 1.0, 0.5, 0.33, 0.25, etc.
                
                # Get similarity score if available
                similarity_score = self._extract_similarity_score(result)
                
                # Combine scores with perspective weight
                combined_score = (position_score * 0.7 + similarity_score * 0.3) * perspective_weight
                
                # Initialize result entry if not exists
                if chunk_id not in result_map:
                    result_map[chunk_id] = {
                        'chunk_id': chunk_id,
                        'score': 0.0,
                        'rank_boost': 0.0,
                        'perspectives_found': [],
                        'best_score': 0.0,
                        'metadata': self._extract_metadata(result),
                        'source_methods': [],
                        'content': result.get('content', '')  # Add content field
                    }
                
                # Accumulate weighted scores
                result_map[chunk_id]['score'] += combined_score
                result_map[chunk_id]['perspectives_found'].append(perspective_name)
                result_map[chunk_id]['source_methods'].append(perspective_name)
                result_map[chunk_id]['best_score'] = max(
                    result_map[chunk_id]['best_score'], 
                    similarity_score
                )
        
        # Apply rank boosts
        self._apply_rank_boosts(result_map, domain_info)
        
        # Convert to final results and sort
        final_results = []
        for chunk_id, data in result_map.items():
            final_score = data['score'] + data['rank_boost']
            final_results.append({
                'chunk_id': chunk_id,
                'score': final_score,
                'original_score': data['score'],
                'rank_boost': data['rank_boost'],
                'perspectives_found': data['perspectives_found'],
                'num_perspectives': len(set(data['perspectives_found'])),
                'source_methods': list(set(data['source_methods'])),
                'best_similarity_score': data['best_score'],
                'metadata': data['metadata'],
                'content': data.get('content', '')  # Include content in final results
            })
        
        # Sort by final score
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Merged results: {len(final_results)} unique chunks from multi-perspective search")
        return final_results[:max_results]
    
    def _calculate_dynamic_weights(self, domain_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate dynamic weights based on domain confidence.
        
        Args:
            domain_info: Domain detection information
            
        Returns:
            Dict of perspective weights
        """
        weights = self.default_weights.copy()
        
        domain_confidence = domain_info.get('confidence', 5) / 10.0  # Normalize to 0-1
        domain = domain_info.get('domain', 'general')
        
        # Boost domain-specific weight for high-confidence domains
        if domain != 'general' and domain_confidence >= 0.6:
            weights['domain'] = 0.7 + (domain_confidence * 0.3)  # 0.7 to 1.0
            logger.debug(f"Boosted domain weight to {weights['domain']:.2f} for {domain} (confidence: {domain_confidence:.2f})")
        
        # Adjust other weights based on domain confidence
        if domain_confidence >= 0.7:
            # High confidence: boost professional perspective
            weights['professor'] *= (1.0 + domain_confidence * 0.2)
            # Reduce child perspective for technical domains
            if domain in ['technical', 'medical', 'legal']:
                weights['child'] *= 0.8
        
        return weights
    
    def _apply_rank_boosts(self, result_map: Dict[str, Dict[str, Any]], domain_info: Dict[str, Any]):
        """
        Apply rank boosts based on various factors.
        
        Args:
            result_map: Map of chunk results
            domain_info: Domain detection information
        """
        domain = domain_info.get('domain', 'general')
        
        for chunk_id, data in result_map.items():
            perspectives = set(data['perspectives_found'])
            
            # Boost for multiple perspective matches
            if len(perspectives) >= 2:
                boost = len(perspectives) * self.rank_boost_multipliers['multiple_methods']
                data['rank_boost'] += boost
                logger.debug(f"Multiple perspective boost: +{boost:.3f} for chunk {chunk_id}")
            
            # Boost for domain matches (if domain-specific perspective found results)
            if 'domain' in perspectives and domain != 'general':
                boost = self.rank_boost_multipliers['domain_match']
                data['rank_boost'] += boost
                logger.debug(f"Domain match boost: +{boost:.3f} for chunk {chunk_id}")
            
            # Boost for synonym matches
            if 'synonym' in perspectives:
                boost = self.rank_boost_multipliers['synonym_match']
                data['rank_boost'] += boost
                logger.debug(f"Synonym match boost: +{boost:.3f} for chunk {chunk_id}")
    
    def _extract_chunk_id(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Extract chunk ID from result object.
        
        Args:
            result: Result dict from our standardized search format
            
        Returns:
            Chunk ID or None
        """
        # Our search results are always dicts with these possible chunk ID fields
        return (result.get('chunk_id') or 
               result.get('document_chunk_id') or 
               result.get('email_chunk_id') or
               result.get('id'))
    
    def _extract_similarity_score(self, result: Dict[str, Any]) -> float:
        """
        Extract similarity score from result object.
        
        Args:
            result: Result dict from our standardized search format
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Our search results always have similarity_score field
        return float(result.get('similarity_score', 0.0))
    
    def _extract_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from result object.
        
        Args:
            result: Result dict from our standardized search format
            
        Returns:
            Metadata dictionary
        """
        # Our search results have both direct fields and a metadata dict
        return {
            'title': result.get('title'),
            'filename': result.get('filename'),
            'page': result.get('page'),
            'category': result.get('category'),
            'category_type': result.get('category_type'),
            'topics': result.get('topics'),
            'document_id': result.get('document_id'),
            'chunk_id': result.get('chunk_id') or result.get('document_chunk_id'),
            'content_preview': result.get('content', '')[:150] if result.get('content') else '',
            # Include the nested metadata as well
            **result.get('metadata', {})
        }
    
    def merge_email_results(self, original_results: List, professor_results: List,
                           child_results: List, email_domain_results: List,
                           synonym_results: List[List]) -> List[Dict[str, Any]]:
        """
        Merge email search results with email-specific weights.
        
        Args:
            original_results: Results from original query
            professor_results: Results from professor reframing
            child_results: Results from child reframing
            email_domain_results: Results from email domain reframing
            synonym_results: List of result lists from synonym variants
            
        Returns:
            List of merged email results
        """
        perspective_results = {
            'original': original_results,
            'professor': professor_results,
            'child': child_results,
            'email_domain': email_domain_results
        }
        
        # Add synonym results
        for i, synonym_result_set in enumerate(synonym_results):
            perspective_results[f'synonym_{i}'] = synonym_result_set
        
        # Email-specific domain info (assume business context for emails)
        email_domain_info = {
            'domain': 'business',
            'confidence': 7,
            'reasoning': 'Email context defaults to business domain'
        }
        
        # Use email-specific weights
        email_weights = {
            'original': 1.0,
            'professor': 0.7,  # Less relevant for emails
            'child': 0.5,     # Even less relevant for emails
            'email_domain': 0.9,  # High relevance for email-specific reframing
            'synonym_0': 0.6,
            'synonym_1': 0.5,
            'synonym_2': 0.4
        }
        
        # Temporarily override weights
        original_weights = self.default_weights.copy()
        self.default_weights.update(email_weights)
        
        try:
            results = self.merge_weighted_results(perspective_results, email_domain_info)
            return results
        finally:
            # Restore original weights
            self.default_weights = original_weights
    
    def get_fusion_statistics(self, merged_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the fusion process.
        
        Args:
            merged_results: Results from merge operation
            
        Returns:
            Dict containing fusion statistics
        """
        if not merged_results:
            return {}
        
        # Count perspectives used
        all_perspectives = set()
        perspective_counts = defaultdict(int)
        multi_perspective_count = 0
        
        for result in merged_results:
            perspectives = set(result.get('source_methods', []))
            all_perspectives.update(perspectives)
            
            for perspective in perspectives:
                perspective_counts[perspective] += 1
            
            if len(perspectives) > 1:
                multi_perspective_count += 1
        
        return {
            'total_results': len(merged_results),
            'perspectives_used': list(all_perspectives),
            'perspective_counts': dict(perspective_counts),
            'multi_perspective_results': multi_perspective_count,
            'overlap_percentage': (multi_perspective_count / len(merged_results)) * 100 if merged_results else 0,
            'average_perspectives_per_result': sum(len(r.get('source_methods', [])) for r in merged_results) / len(merged_results),
            'top_score': merged_results[0]['score'] if merged_results else 0,
            'score_range': merged_results[0]['score'] - merged_results[-1]['score'] if len(merged_results) > 1 else 0
        }