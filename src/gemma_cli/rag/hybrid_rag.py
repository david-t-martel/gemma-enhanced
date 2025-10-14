
"""Hybrid RAG Manager for Gemma CLI."""

from typing import Any, Optional

from gemma_cli.rag.memory import MemoryEntry
from gemma_cli.rag.python_backend import PythonRAGBackend


class HybridRAGManager:
    """Manages RAG operations, potentially combining multiple backends."""

    def __init__(self) -> None:
        self.python_backend = PythonRAGBackend()  # Initialize the Python backend

    async def initialize(self) -> bool:
        """Initialize the RAG manager and its backends."""
        return await self.python_backend.initialize()

    async def recall_memories(
        self, query: str, memory_type: Optional[str] = None, limit: int = 5
    ) -> list[MemoryEntry]:
        """Recall memories from the RAG system."""
        return await self.python_backend.recall_memories(query, memory_type, limit)

    async def store_memory(
        self,
        content: str,
        memory_type: str,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Store memory in the RAG system."""
        return await self.python_backend.store_memory(content, memory_type, importance, tags)

    async def get_memory_stats(self) -> dict[str, Any]:
        """Get statistics about memory usage across tiers."""
        return await self.python_backend.get_memory_stats()

    async def ingest_document(
        self, file_path: str, memory_type: str = "long_term", chunk_size: int = 500
    ) -> int:
        """
        Ingest a document into the memory system by chunking.
        """
        return await self.python_backend.ingest_document(file_path, memory_type, chunk_size)

    async def close(self) -> None:
        """Close RAG manager and its backends."""
        await self.python_backend.close()
