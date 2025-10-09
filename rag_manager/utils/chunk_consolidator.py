#!/usr/bin/env python3
"""
Chunk consolidation utility for RAG search results.

This module provides functionality to consolidate sequential chunks from the same
documents to provide better context while maintaining proper source attribution.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedChunk:
    """Represents a consolidated chunk with combined content and metadata."""
    chunk_ids: List[str]
    document_id: str
    content: str
    start_page: int
    end_page: int
    highest_score: float
    average_score: float
    source_methods: List[str]
    chunk_count: int
    metadata: Dict[str, Any]


class ChunkConsolidator:
    """
    Consolidates sequential chunks from the same documents to improve context.
    
    This class groups chunks by document, sorts them by position, and merges
    sequential chunks while preserving relevance scores and source attribution.
    """
    
    def __init__(self, 
                 max_chunk_length: int = 8000,
                 overlap_threshold: float = 0.8,
                 min_consolidation_size: int = 2):
        """
        Initialize chunk consolidator.
        
        Args:
            max_chunk_length: Maximum length for consolidated chunks
            overlap_threshold: Threshold for considering chunks as overlapping
            min_consolidation_size: Minimum number of chunks to trigger consolidation
        """
        self.max_chunk_length = max_chunk_length
        self.overlap_threshold = overlap_threshold
        self.min_consolidation_size = min_consolidation_size
    
    def consolidate_chunks(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Consolidate sequential chunks from the same documents.
        
        Args:
            search_results: List of search results from multi-perspective search
            
        Returns:
            List of consolidated search results with improved context
        """
        logger.info(f"Starting chunk consolidation for {len(search_results)} results")
        
        if not search_results:
            return search_results
        
        # Group chunks by document
        document_groups = self._group_by_document(search_results)
        logger.info(f"Grouped chunks into {len(document_groups)} documents")
        
        consolidated_results = []
        consolidation_stats = {
            'original_chunks': len(search_results),
            'consolidated_groups': 0,
            'standalone_chunks': 0,
            'total_consolidated': 0
        }
        
        for document_id, chunks in document_groups.items():
            if len(chunks) >= self.min_consolidation_size:
                # Sort chunks by position within document
                sorted_chunks = self._sort_chunks_by_position(chunks)
                
                # Find sequential groups
                sequential_groups = self._find_sequential_groups(sorted_chunks)
                
                # Consolidate each group
                for group in sequential_groups:
                    if len(group) >= self.min_consolidation_size:
                        consolidated = self._consolidate_group(group, document_id)
                        consolidated_results.append(consolidated)
                        consolidation_stats['consolidated_groups'] += 1
                        consolidation_stats['total_consolidated'] += len(group)
                        logger.debug(f"Consolidated {len(group)} chunks from document {document_id}")
                    else:
                        # Add standalone chunks
                        consolidated_results.extend(self._convert_to_standard_format(group))
                        consolidation_stats['standalone_chunks'] += len(group)
            else:
                # Document has too few chunks, keep as standalone
                consolidated_results.extend(self._convert_to_standard_format(chunks))
                consolidation_stats['standalone_chunks'] += len(chunks)
        
        # Sort final results by highest relevance score
        consolidated_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        logger.info(f"Consolidation complete: {consolidation_stats['original_chunks']} → "
                   f"{len(consolidated_results)} results "
                   f"({consolidation_stats['consolidated_groups']} groups, "
                   f"{consolidation_stats['standalone_chunks']} standalone)")
        
        return consolidated_results
    
    def _group_by_document(self, search_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group search results by document source."""
        groups = defaultdict(list)
        
        for result in search_results:
            metadata = result.get('metadata', {})
            
            # Use filename/title as the primary grouping key for document chunks
            filename = metadata.get('filename', '')
            title = metadata.get('title', '')
            
            # Create a consistent grouping key
            if filename:
                group_key = f"file_{filename}"
            elif title:
                group_key = f"title_{title}"
            else:
                # For chunks without clear document identity, use chunk_id
                chunk_id = result.get('chunk_id', result.get('document_chunk_id', 'unknown'))
                group_key = f"chunk_{chunk_id}"
            
            groups[group_key].append(result)
        
        return dict(groups)
    
    def _sort_chunks_by_position(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort chunks by their position within the document."""
        def get_sort_key(chunk):
            metadata = chunk.get('metadata', {})
            
            # Primary sort: chunk_ordinal (if available)
            chunk_ordinal = metadata.get('chunk_ordinal')
            if chunk_ordinal is not None:
                return (0, chunk_ordinal)
            
            # Secondary sort: page number
            page = metadata.get('page', metadata.get('page_start', 0))
            if page is not None:
                return (1, page)
            
            # Fallback: use relevance score (higher scores first)
            score = chunk.get('score', chunk.get('similarity_score', 0.0))
            return (2, -score)  # Negative for descending order
        
        return sorted(chunks, key=get_sort_key)
    
    def _find_sequential_groups(self, sorted_chunks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Find groups of sequential chunks that can be consolidated."""
        if not sorted_chunks:
            return []
        
        groups = []
        current_group = [sorted_chunks[0]]
        
        for i in range(1, len(sorted_chunks)):
            curr_chunk = sorted_chunks[i]
            prev_chunk = sorted_chunks[i-1]
            
            if self._are_sequential(prev_chunk, curr_chunk):
                current_group.append(curr_chunk)
            else:
                # Start a new group
                groups.append(current_group)
                current_group = [curr_chunk]
        
        # Add the last group
        groups.append(current_group)
        
        return groups
    
    def _are_sequential(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> bool:
        """Check if two chunks are sequential and should be consolidated."""
        meta1 = chunk1.get('metadata', {})
        meta2 = chunk2.get('metadata', {})
        
        # Check chunk ordinal sequence
        ord1 = meta1.get('chunk_ordinal')
        ord2 = meta2.get('chunk_ordinal')
        if ord1 is not None and ord2 is not None:
            return abs(ord2 - ord1) <= 2  # Allow small gaps
        
        # Check page sequence
        page1 = meta1.get('page', meta1.get('page_start', meta1.get('page_end')))
        page2 = meta2.get('page', meta2.get('page_start', meta2.get('page_end')))
        if page1 is not None and page2 is not None:
            return abs(page2 - page1) <= 1  # Adjacent or same page
        
        # If no positional info, check content overlap
        content1 = chunk1.get('content', '')
        content2 = chunk2.get('content', '')
        if content1 and content2:
            overlap = self._calculate_content_overlap(content1, content2)
            return overlap >= self.overlap_threshold
        
        return False
    
    def _calculate_content_overlap(self, content1: str, content2: str) -> float:
        """Calculate content overlap between two chunks."""
        if not content1 or not content2:
            return 0.0
        
        # Simple word-based overlap calculation
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _consolidate_group(self, group: List[Dict[str, Any]], document_id: str) -> Dict[str, Any]:
        """Consolidate a group of sequential chunks into a single result."""
        if len(group) == 1:
            return self._convert_to_standard_format(group)[0]
        
        # Extract basic info
        chunk_ids = []
        contents = []
        scores = []
        source_methods = set()
        pages = []
        
        # Combine metadata from all chunks
        combined_metadata = {}
        first_chunk = group[0]
        
        for chunk in group:
            # Collect chunk IDs
            chunk_id = chunk.get('chunk_id', chunk.get('document_chunk_id', ''))
            if chunk_id:
                chunk_ids.append(chunk_id)
            
            # Collect content
            content = chunk.get('content', '')
            if content and content not in contents:  # Avoid exact duplicates
                contents.append(content)
            
            # Collect scores
            score = chunk.get('score', chunk.get('similarity_score', 0.0))
            scores.append(float(score))
            
            # Collect source methods
            methods = chunk.get('source_methods', [])
            if isinstance(methods, list):
                source_methods.update(methods)
            else:
                source_methods.add(str(methods))
            
            # Collect page info
            metadata = chunk.get('metadata', {})
            page = metadata.get('page', metadata.get('page_start'))
            if page is not None:
                pages.append(page)
            
            # Merge metadata (first chunk takes precedence)
            if not combined_metadata:
                combined_metadata = metadata.copy()
        
        # Create consolidated content
        consolidated_content = self._merge_contents(contents)
        
        # Calculate consolidated scores
        highest_score = max(scores) if scores else 0.0
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        # Update metadata for consolidation
        combined_metadata.update({
            'consolidated': True,
            'original_chunk_count': len(group),
            'chunk_ids': chunk_ids,
            'highest_score': highest_score,
            'average_score': average_score
        })
        
        # Use the consolidated chunk_id as the document_id
        consolidated_chunk_id = f"consolidated_{len(chunk_ids)}_{chunk_ids[0]}" if chunk_ids else f"consolidated_{len(group)}"
        combined_metadata['document_id'] = consolidated_chunk_id
        
        # Determine page range
        if pages:
            start_page = min(pages)
            end_page = max(pages)
            combined_metadata['page_start'] = start_page
            combined_metadata['page_end'] = end_page
            combined_metadata['page'] = start_page  # Use start page as primary
        
        # Create consolidated result
        consolidated = {
            'chunk_id': f"consolidated_{len(chunk_ids)}_{chunk_ids[0]}" if chunk_ids else f"consolidated_{len(group)}",
            'document_chunk_id': f"consolidated_{len(chunk_ids)}_{chunk_ids[0]}" if chunk_ids else f"consolidated_{len(group)}",
            'content': consolidated_content,
            'score': highest_score,  # Use highest score for ranking
            'similarity_score': highest_score,
            'source_methods': list(source_methods),
            'num_perspectives': first_chunk.get('num_perspectives', 1),
            'rank_boost': max(chunk.get('rank_boost', 0.0) for chunk in group),
            'metadata': combined_metadata,
            'consolidation_info': {
                'chunk_count': len(group),
                'chunk_ids': chunk_ids,
                'highest_score': highest_score,
                'average_score': average_score,
                'source_methods': list(source_methods)
            }
        }
        
        return consolidated
    
    def _merge_contents(self, contents: List[str]) -> str:
        """Merge multiple content strings intelligently."""
        if not contents:
            return ""
        
        if len(contents) == 1:
            return contents[0]
        
        # Simple merge: join with paragraph breaks
        merged = "\n\n".join(contents)
        
        # Truncate if too long
        if len(merged) > self.max_chunk_length:
            # Try to truncate at sentence boundary
            sentences = merged.split('. ')
            truncated = ""
            for sentence in sentences:
                test_content = truncated + sentence + ". "
                if len(test_content) <= self.max_chunk_length:
                    truncated = test_content
                else:
                    break
            
            if truncated:
                merged = truncated.rstrip() + "..."
            else:
                # Hard truncate if no good sentence boundary
                merged = merged[:self.max_chunk_length - 3] + "..."
        
        return merged
    
    def _convert_to_standard_format(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert chunks to standard format without consolidation."""
        results = []
        for chunk in chunks:
            # Ensure consistent format
            result = {
                'chunk_id': chunk.get('chunk_id', chunk.get('document_chunk_id', '')),
                'document_chunk_id': chunk.get('chunk_id', chunk.get('document_chunk_id', '')),
                'content': chunk.get('content', ''),
                'score': chunk.get('score', chunk.get('similarity_score', 0.0)),
                'similarity_score': chunk.get('score', chunk.get('similarity_score', 0.0)),
                'source_methods': chunk.get('source_methods', []),
                'num_perspectives': chunk.get('num_perspectives', 1),
                'rank_boost': chunk.get('rank_boost', 0.0),
                'metadata': chunk.get('metadata', {}),
                'consolidation_info': {
                    'chunk_count': 1,
                    'chunk_ids': [chunk.get('chunk_id', chunk.get('document_chunk_id', ''))],
                    'highest_score': chunk.get('score', chunk.get('similarity_score', 0.0)),
                    'average_score': chunk.get('score', chunk.get('similarity_score', 0.0)),
                    'source_methods': chunk.get('source_methods', [])
                }
            }
            results.append(result)
        
        return results
    
    def get_consolidation_stats(self, original_results: List[Dict[str, Any]], 
                               consolidated_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the consolidation process."""
        original_count = len(original_results)
        consolidated_count = len(consolidated_results)
        
        # Count consolidated groups
        consolidated_groups = sum(1 for r in consolidated_results 
                                if r.get('consolidation_info', {}).get('chunk_count', 1) > 1)
        
        # Count total chunks that were consolidated
        total_consolidated_chunks = sum(r.get('consolidation_info', {}).get('chunk_count', 1) 
                                      for r in consolidated_results)
        
        return {
            'original_chunks': original_count,
            'final_results': consolidated_count,
            'consolidated_groups': consolidated_groups,
            'standalone_chunks': consolidated_count - consolidated_groups,
            'total_chunks_processed': total_consolidated_chunks,
            'reduction_ratio': (original_count - consolidated_count) / original_count if original_count > 0 else 0.0,
            'consolidation_efficiency': consolidated_groups / consolidated_count if consolidated_count > 0 else 0.0
        }


def create_chunk_consolidator(config: Any = None) -> ChunkConsolidator:
    """
    Factory function to create a ChunkConsolidator with configuration.
    
    Args:
        config: Configuration object with consolidation settings
        
    Returns:
        Configured ChunkConsolidator instance
    """
    max_chunk_length = getattr(config, 'CONSOLIDATION_MAX_CHUNK_LENGTH', 8000) if config else 8000
    overlap_threshold = getattr(config, 'CONSOLIDATION_OVERLAP_THRESHOLD', 0.8) if config else 0.8
    min_consolidation_size = getattr(config, 'CONSOLIDATION_MIN_SIZE', 2) if config else 2
    
    return ChunkConsolidator(
        max_chunk_length=max_chunk_length,
        overlap_threshold=overlap_threshold,
        min_consolidation_size=min_consolidation_size
    )