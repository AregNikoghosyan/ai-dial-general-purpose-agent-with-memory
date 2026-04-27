import os
os.environ['OMP_NUM_THREADS'] = '1'

import json
from datetime import datetime, UTC, timedelta
import numpy as np
import faiss
from aidial_client import AsyncDial
from sentence_transformers import SentenceTransformer

from task.tools.memory._models import Memory, MemoryData, MemoryCollection


class LongTermMemoryStore:
    """
    Manages long-term memory storage for users.

    Storage format: Single JSON file per user in DIAL bucket
    - File: {user_id}/long-memories.json
    - Caching: In-memory cache with conversation_id as key
    - Deduplication: O(n log n) using FAISS batch search
    """

    DEDUP_INTERVAL_HOURS = 24

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache: dict[str, MemoryCollection] = {}
        faiss.omp_set_num_threads(1)

    async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
        """Get the path to the memory file in DIAL bucket."""
        appdata = await dial_client.bucket.get_appdata()
        return f"files/{appdata.raw}/__long-memories/data.json"

    async def _load_memories(self, api_key: str) -> MemoryCollection:
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )
        file_path = await self._get_memory_file_path(dial_client)

        if file_path in self.cache:
            return self.cache[file_path]

        try:
            response = await dial_client.files.download(file_path)
            content = await response.aget_content()
            data = json.loads(content.decode('utf-8'))
            collection = MemoryCollection.model_validate(data)
        except Exception:
            collection = MemoryCollection(updated_at=datetime.now(UTC))

        return collection

    async def _save_memories(self, api_key: str, memories: MemoryCollection):
        """Save memories to DIAL bucket and update cache."""
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )
        file_path = await self._get_memory_file_path(dial_client)
        memories.updated_at = datetime.now(UTC)
        content = memories.model_dump_json()
        await dial_client.files.upload(file_path, file=content.encode('utf-8'))
        self.cache[file_path] = memories

    async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
        """Add a new memory to storage."""
        memories = await self._load_memories(api_key)
        embedding = self.model.encode([content])[0].tolist()
        memory = Memory(
            data=MemoryData(
                id=int(datetime.now(UTC).timestamp()),
                content=content,
                importance=importance,
                category=category,
                topics=topics
            ),
            embedding=embedding
        )
        memories.memories.append(memory)
        await self._save_memories(api_key, memories)
        return f"Memory successfully stored: {content}"

    async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
        """
        Search memories using semantic similarity.

        Returns:
            List of MemoryData objects (without embeddings)
        """
        memories = await self._load_memories(api_key)

        if not memories.memories:
            return []

        if self._needs_deduplication(memories):
            memories = await self._deduplicate_and_save(api_key, memories)

        embeddings = np.array([m.embedding for m in memories.memories], dtype='float32')
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)

        k = min(top_k, len(memories.memories))
        distances, indices = index.search(query_embedding, k)

        return [memories.memories[idx].data for idx in indices[0]]

    def _needs_deduplication(self, collection: MemoryCollection) -> bool:
        """Check if deduplication is needed (>24 hours since last deduplication)."""
        if len(collection.memories) <= 10:
            return False
        if collection.last_deduplicated_at is None:
            return True
        hours_since = (datetime.now(UTC) - collection.last_deduplicated_at).total_seconds() / 3600
        return hours_since > self.DEDUP_INTERVAL_HOURS

    async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
        """
        Deduplicate memories synchronously and save the result.
        Returns the updated collection.
        """
        collection.memories = self._deduplicate_fast(collection.memories)
        collection.last_deduplicated_at = datetime.now(UTC)
        await self._save_memories(api_key, collection)
        return collection

    def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
        """
        Fast deduplication using FAISS batch search with cosine similarity.

        Strategy:
        - Find k nearest neighbors for each memory using cosine similarity
        - Mark duplicates based on similarity threshold (cosine similarity > 0.75)
        - Keep memory with higher importance
        """
        if len(memories) <= 1:
            return memories

        embeddings = np.array([m.embedding for m in memories], dtype='float32')
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        k = min(10, len(memories))
        distances, indices = index.search(embeddings, k)

        removed = set()

        # Process from highest to lowest importance so we keep the most important memories
        order = sorted(range(len(memories)), key=lambda i: -memories[i].data.importance)

        for orig_idx in order:
            if orig_idx in removed:
                continue
            for neighbor_idx, sim in zip(indices[orig_idx], distances[orig_idx]):
                if int(neighbor_idx) == orig_idx:
                    continue
                if sim > 0.75 and int(neighbor_idx) not in removed:
                    removed.add(int(neighbor_idx))

        return [m for i, m in enumerate(memories) if i not in removed]

    async def delete_all_memories(self, api_key: str) -> str:
        """
        Delete all memories for the user.

        Removes the memory file from DIAL bucket and clears the cache
        for the current conversation.
        """
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )
        file_path = await self._get_memory_file_path(dial_client)
        await dial_client.files.delete(file_path)
        self.cache.pop(file_path, None)
        return "All long-term memories have been successfully deleted."
