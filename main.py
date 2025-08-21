#!/usr/bin/env python3
"""
Main entry point for RAG Document Handler.

A comprehensive knowledgebase store for storing and retrieving information for RAG implementations.
Refactored to follow Python development best practices with proper module organization.
"""

from rag_manager.app import RAGKnowledgebaseManager


def main() -> None:
    """
    Main entry point for the RAG Document Handler application.
    
    Initializes and runs the Flask web application with all components.
    """
    # Initialize and run the application
    app_manager = RAGKnowledgebaseManager()
    app_manager.run()


if __name__ == "__main__":
    main()
