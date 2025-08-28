#!/usr/bin/env python3
"""
Utility function to check if a file should be ignored
"""

import os
from pathlib import Path

def should_ignore_file(filename: str) -> bool:
    """
    Check if a file should be ignored during processing.
    
    Args:
        filename: The filename to check
        
    Returns:
        True if the file should be ignored, False otherwise
    """
    # System files to ignore
    system_files = {
        '.DS_Store',
        '.DS_Store?',
        '._*',
        '.Spotlight-V100',
        '.Trashes',
        'ehthumbs.db',
        'Thumbs.db',
        'desktop.ini',
        '*.tmp',
        '*.temp'
    }
    
    # Get just the filename without path
    basename = os.path.basename(filename)
    
    # Check exact matches
    if basename in system_files:
        return True
    
    # Check pattern matches
    if basename.startswith('._'):
        return True
    
    if basename.startswith('.'):
        # Allow some hidden files but block most
        allowed_hidden = {'.env', '.gitignore', '.gitkeep'}
        if basename not in allowed_hidden:
            return True
    
    # Check temporary file patterns
    if basename.endswith(('.tmp', '.temp', '.swp', '~')):
        return True
    
    return False

if __name__ == "__main__":
    # Test the function
    test_files = [
        '.DS_Store',
        'document.pdf',
        '._hidden_file',
        '.env',
        'temp.tmp',
        'normal_file.txt'
    ]
    
    for file in test_files:
        result = should_ignore_file(file)
        print(f"{file:<20}: {'IGNORE' if result else 'PROCESS'}")
