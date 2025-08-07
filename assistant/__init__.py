"""
Assistant Package - PASSE Bot RAG System

This package provides intelligent assistance capabilities using RAG
(Retrieval-Augmented Generation) technology.
"""

from .rag import (
    Assistant,
    AgentState,
    # create_passe_rag,
    # create_default_rag,
    # create_custom_rag,
)

from .functions import (
    connect_vectorstore,
    make_documents,
    add_documents_to_vectorstore,
    # get_vectorstore_stats,
)

# Version information
__version__ = "1.0.0"
__author__ = "PASSE Team"
__description__ = "Intelligent RAG-based assistance system"

# Default exports for easy access
__all__ = [
    "Assistant",
    "AgentState",
    # "create_passe_rag",
    # "create_default_rag",
    # "create_custom_rag",
    "connect_vectorstore",
    "make_documents",
    "add_documents_to_vectorstore",
    # "get_vectorstore_stats",
]


# Convenience function for quick setup
# def quick_setup(**kwargs):
#     """
#     Quick setup function to get started with PASSE RAG.

#     Returns:
#         Configured RAG instance ready to use
#     """
#     return create_default_rag() if not kwargs else create_passe_rag(**kwargs)
