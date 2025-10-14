"""Python-based RAG backend (fallback implementation)."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

import aiofiles
import numpy as np
import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

from gemma_cli.rag.memory import MemoryEntry, MemoryTier

# Check for optional dependencies
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from tiktoken import get_encoding

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class PythonRAGBackend:
    """Python-based RAG system with 5-tier memory architecture (fallback implementation)."""

    # Memory tier configurations (TTL in seconds)
    TIER_CONFIG = {
        MemoryTier.WORKING: {"ttl": 900, "max_size": 15},  # 15 min
        MemoryTier.SHORT_TERM: {"ttl": 3600, "max_size": 100},  # 1 hour
        MemoryTier.LONG_TERM: {"ttl": 2592000, "max_size": 10000},  # 30 days
        MemoryTier.EPISODIC: {"ttl": 604800, "max_size": 5000},  # 7 days
        MemoryTier.SEMANTIC: {"ttl": None, "max_size": 50000},  # Permanent
    }

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6380,
        redis_db: int = 0,
        pool_size: int = 50,
    ) -> None:
        """
        Initialize Python RAG backend.

        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            pool_size: Connection pool size (default: 50 for production workloads)
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.pool_size = pool_size
        self.redis_pool: Optional[ConnectionPool] = None
        self.async_redis_client: Optional[aioredis.Redis] = None
        self.embedding_model: Optional[Any] = None
        self.encoder: Optional[Any] = None

    async def initialize(self) -> bool:
        """
        Initialize Redis connections and embedding model.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create connection pool for better performance
            self.redis_pool = ConnectionPool(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                max_connections=self.pool_size,
                decode_responses=False,
            )

            self.async_redis_client = aioredis.Redis(connection_pool=self.redis_pool)

            # Test connection
            await self.async_redis_client.ping()

            # Load embedding model if available
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                print("Loading embedding model...")
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Initialize tokenizer if available
            if TIKTOKEN_AVAILABLE:
                self.encoder = get_encoding("cl100k_base")

            print("RAG-Redis system initialized successfully")
            return True

        except (ConnectionError, TimeoutError, OSError) as e:
            print(f"Failed to initialize RAG-Redis system: {e}")
            return False

    def get_redis_key(self, memory_type: str, entry_id: Optional[str] = None) -> str:
        """
        Generate Redis key for memory tier and entry.

        Args:
            memory_type: Memory tier type
            entry_id: Optional entry ID

        Returns:
            Redis key string (sanitized to prevent key injection)
        """
        # Sanitize memory_type to prevent Redis key injection attacks
        # Only allow alphanumeric characters and underscores
        memory_type = re.sub(r'[^a-zA-Z0-9_]', '_', memory_type)

        if entry_id:
            # Sanitize entry_id (allow hyphens for UUID compatibility)
            entry_id = re.sub(r'[^a-zA-Z0-9_-]', '_', entry_id)
            return f"gemma:mem:{memory_type}:{entry_id}"
        return f"gemma:mem:{memory_type}:*"

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        if self.embedding_model:
            return self.embedding_model.encode([text])[0]
        else:
            # Fallback: simple hash-based embedding (for compatibility only)
            hash_val = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
            return np.array([ord(c) for c in hash_val[:384]], dtype=np.float32) / 256.0

    async def _scan_keys(self, pattern: str) -> list[bytes]:
        """
        Scan keys using cursor-based iteration (replaces KEYS for production).

        Args:
            pattern: Redis key pattern

        Returns:
            List of matching keys
        """
        if not self.async_redis_client:
            return []

        keys: list[bytes] = []
        cursor = 0

        while True:
            cursor, partial_keys = await self.async_redis_client.scan(
                cursor, match=pattern, count=100
            )
            keys.extend(partial_keys)
            if cursor == 0:
                break

        return keys

    async def store_memory(
        self,
        content: str,
        memory_type: str,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Store content in specified memory tier.

        Args:
            content: Content to store
            memory_type: Memory tier type
            importance: Importance score (0.0 to 1.0)
            tags: Optional list of tags

        Returns:
            Entry ID if successful, None otherwise
        """
        if not self.async_redis_client:
            return None

        try:
            entry = MemoryEntry(content, memory_type, importance)
            if tags:
                entry.add_tags(*tags)

            # Generate embedding
            entry.embedding = self.get_embedding(content)

            # Store in Redis
            key = self.get_redis_key(memory_type, entry.id)
            data = json.dumps(entry.to_dict())

            config = self.TIER_CONFIG.get(memory_type, self.TIER_CONFIG[MemoryTier.SHORT_TERM])
            if config["ttl"]:
                await self.async_redis_client.setex(key, config["ttl"], data)
            else:
                await self.async_redis_client.set(key, data)

            # Enforce tier size limits
            await self._enforce_tier_limits(memory_type)

            print(f"Stored memory in {memory_type} tier: {entry.id[:8]}...")
            return entry.id

        except (aioredis.RedisError, json.JSONEncodeError, ValueError) as e:
            print(f"Error storing memory: {e}")
            return None

    async def recall_memories(
        self, query: str, memory_type: Optional[str] = None, limit: int = 5
    ) -> list[MemoryEntry]:
        """
        Retrieve memories based on semantic similarity to query.

        Args:
            query: Query text
            memory_type: Optional memory tier to search (None = all tiers)
            limit: Maximum number of results

        Returns:
            List of memory entries sorted by relevance
        """
        if not self.async_redis_client:
            return []

        try:
            query_embedding = self.get_embedding(query)
            results: list[MemoryEntry] = []

            memory_types = [memory_type] if memory_type else list(self.TIER_CONFIG.keys())

            for tier in memory_types:
                pattern = self.get_redis_key(tier)
                keys = await self._scan_keys(pattern)

                # Use pipeline for batch retrieval (10x faster than individual gets)
                if keys:
                    async with self.async_redis_client.pipeline() as pipe:
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()

                    for data_bytes in values:
                        if data_bytes:
                            try:
                                entry_dict = json.loads(data_bytes.decode())
                                entry = MemoryEntry.from_dict(entry_dict)

                                # Calculate similarity score
                                if entry.embedding is not None:
                                    similarity = self._cosine_similarity(
                                        query_embedding, entry.embedding
                                    )
                                    entry.similarity_score = similarity  # type: ignore
                                    results.append(entry)
                            except (json.JSONDecodeError, KeyError, ValueError):
                                continue

            # Sort by combined similarity and importance
            results.sort(key=lambda x: (x.similarity_score * x.importance), reverse=True)  # type: ignore
            return results[:limit]

        except (aioredis.RedisError, ValueError) as e:
            print(f"Error recalling memories: {e}")
            return []

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def search_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> list[MemoryEntry]:
        """
        Search memories by content and importance.

        Args:
            query: Search query
            memory_type: Optional memory tier
            min_importance: Minimum importance threshold

        Returns:
            List of matching memory entries
        """
        if not self.async_redis_client:
            return []

        try:
            results: list[MemoryEntry] = []
            memory_types = [memory_type] if memory_type else list(self.TIER_CONFIG.keys())

            for tier in memory_types:
                pattern = self.get_redis_key(tier)
                keys = await self._scan_keys(pattern)

                if keys:
                    async with self.async_redis_client.pipeline() as pipe:
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()

                    for data_bytes in values:
                        if data_bytes:
                            try:
                                entry_dict = json.loads(data_bytes.decode())
                                entry = MemoryEntry.from_dict(entry_dict)

                                # Filter by importance and content match
                                if (
                                    entry.importance >= min_importance
                                    and query.lower() in entry.content.lower()
                                ):
                                    results.append(entry)
                            except (json.JSONDecodeError, KeyError, ValueError):
                                continue

            # Sort by importance and recency
            results.sort(key=lambda x: (x.importance, x.last_accessed), reverse=True)
            return results

        except (aioredis.RedisError, ValueError) as e:
            print(f"Error searching memories: {e}")
            return []

    async def ingest_document(
        self, file_path: str, memory_type: str = MemoryTier.LONG_TERM, chunk_size: int = 500
    ) -> int:
        """
        Ingest a document into the memory system by chunking.

        Args:
            file_path: Path to document file
            memory_type: Memory tier to store chunks
            chunk_size: Chunk size in tokens (if tiktoken available) or characters

        Returns:
            Number of chunks stored
        """
        try:
            path = Path(file_path)
            if not path.exists():
                print(f"File not found: {file_path}")
                return 0

            # Read file content asynchronously
            async with aiofiles.open(path, encoding="utf-8") as f:
                content = await f.read()

            # Chunk the document
            chunks = self._chunk_text(content, chunk_size)
            stored_count = 0

            # Store chunks with batch operations
            for i, chunk in enumerate(chunks):
                importance = 0.7 if memory_type == MemoryTier.LONG_TERM else 0.5
                tags = [f"document:{path.name}", f"chunk:{i}"]

                entry_id = await self.store_memory(chunk, memory_type, importance, tags)
                if entry_id:
                    stored_count += 1

            print(f"Ingested document: {stored_count} chunks from {path.name}")
            return stored_count

        except (OSError, UnicodeDecodeError, ValueError) as e:
            print(f"Error ingesting document: {e}")
            return 0

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into semantic chunks.

        Args:
            text: Input text
            chunk_size: Target chunk size

        Returns:
            List of text chunks
        """
        if self.encoder:
            # Use tiktoken for intelligent chunking
            tokens = self.encoder.encode(text)
            chunks: list[str] = []

            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i : i + chunk_size]
                chunk_text = self.encoder.decode(chunk_tokens)
                chunks.append(chunk_text)

            return chunks
        else:
            # Simple sentence-based chunking
            sentences = re.split(r"[.!?]+", text)
            chunks_list: list[str] = []
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size * 4:  # Rough word estimate
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks_list.append(current_chunk.strip())
                    current_chunk = sentence + ". "

            if current_chunk:
                chunks_list.append(current_chunk.strip())

            return chunks_list

    async def _enforce_tier_limits(self, memory_type: str) -> None:
        """
        Enforce size limits for memory tiers.

        Args:
            memory_type: Memory tier to enforce limits on
        """
        try:
            config = self.TIER_CONFIG.get(memory_type)
            if not config or not config.get("max_size") or not self.async_redis_client:
                return

            pattern = self.get_redis_key(memory_type)
            keys = await self._scan_keys(pattern)

            if len(keys) > config["max_size"]:
                # Get all entries with timestamps
                entries_to_check: list[tuple[bytes, str]] = []

                async with self.async_redis_client.pipeline() as pipe:
                    for key in keys:
                        pipe.get(key)
                    values = await pipe.execute()

                for key, data_bytes in zip(keys, values):
                    if data_bytes:
                        try:
                            entry_dict = json.loads(data_bytes.decode())
                            entries_to_check.append((key, entry_dict["last_accessed"]))
                        except (json.JSONDecodeError, KeyError):
                            continue

                # Sort by last accessed time and remove oldest
                entries_to_check.sort(key=lambda x: x[1])
                excess_count = len(keys) - config["max_size"]

                # Delete excess entries
                async with self.async_redis_client.pipeline() as pipe:
                    for key, _ in entries_to_check[:excess_count]:
                        pipe.delete(key)
                    await pipe.execute()

        except (aioredis.RedisError, ValueError) as e:
            print(f"Warning: Error enforcing tier limits: {e}")

    async def get_memory_stats(self) -> dict[str, Any]:
        """
        Get statistics about memory usage across tiers.

        Returns:
            Dictionary with memory statistics
        """
        stats: dict[str, Any] = {}
        total_entries = 0

        try:
            if not self.async_redis_client:
                return stats

            for tier in self.TIER_CONFIG.keys():
                pattern = self.get_redis_key(tier)
                keys = await self._scan_keys(pattern)
                count = len(keys)
                stats[tier] = count
                total_entries += count

            stats["total"] = total_entries

            # Try to get Redis memory usage
            try:
                info = await self.async_redis_client.info("memory")
                stats["redis_memory"] = info.get("used_memory", 0)
            except aioredis.RedisError:
                stats["redis_memory"] = 0

        except aioredis.RedisError as e:
            print(f"Error getting memory stats: {e}")

        return stats

    async def cleanup_expired(self) -> int:
        """
        Clean up expired memory entries.

        Returns:
            Number of entries cleaned
        """
        cleaned = 0

        try:
            if not self.async_redis_client:
                return 0

            for tier, config in self.TIER_CONFIG.items():
                if config["ttl"]:  # Skip permanent memories
                    pattern = self.get_redis_key(tier)
                    keys = await self._scan_keys(pattern)

                    # Use pipeline to check TTL for all keys
                    if keys:
                        async with self.async_redis_client.pipeline() as pipe:
                            for key in keys:
                                pipe.ttl(key)
                            ttls = await pipe.execute()

                        # Delete expired keys
                        keys_to_delete = [
                            key for key, ttl in zip(keys, ttls) if ttl == -2  # -2 means expired
                        ]

                        if keys_to_delete:
                            async with self.async_redis_client.pipeline() as pipe:
                                for key in keys_to_delete:
                                    pipe.delete(key)
                                await pipe.execute()
                            cleaned += len(keys_to_delete)

            if cleaned > 0:
                print(f"Cleaned up {cleaned} expired memory entries")

        except aioredis.RedisError as e:
            print(f"Error during cleanup: {e}")

        return cleaned

    async def close(self) -> None:
        """Close Redis connections."""
        if self.async_redis_client:
            await self.async_redis_client.close()
        if self.redis_pool:
            await self.redis_pool.disconnect()
