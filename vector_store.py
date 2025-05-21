import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from config import settings
from logger import log

class QdrantVectorStore:
    """Vector store for chatbot memories using Qdrant."""

    def __init__(
        self,
        embedding_model=None,
        host: str = None,
        port: int = None,
        collection_name: str = None
    ):
        """Initialize the vector store."""
        self.embedding_model = embedding_model
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self.vector_dim = settings.vector_dim

        # Initialize Qdrant client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client and create collection if it doesn't exist."""
        try:
            log.info(f"Connecting to Qdrant at {self.host}:{self.port}")
            self.client = QdrantClient(host=self.host, port=self.port)

            # Check if collection exists, create if not
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                log.info(f"Creating new collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.COSINE
                    )
                )
                log.info(f"Collection {self.collection_name} created successfully")
            else:
                log.info(f"Using existing collection: {self.collection_name}")

        except Exception as e:
            log.error(f"Error initializing Qdrant client: {str(e)}")
            log.info("Falling back to local FAISS index")
            self._initialize_faiss_fallback()

    def _initialize_faiss_fallback(self):
        """Initialize a local FAISS index as fallback."""
        import faiss
        self.use_fallback = True
        self.index_path = f"data/faiss_index"
        self.metadata_path = f"{self.index_path}_metadata.json"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Load existing index and metadata if available
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                log.info(f"Loaded existing FAISS index with {len(self.metadata)} entries")
            except Exception as e:
                log.error(f"Error loading FAISS index: {str(e)}. Creating new index.")
                self._create_new_faiss_index()
        else:
            self._create_new_faiss_index()

    def _create_new_faiss_index(self):
        """Create a new FAISS index."""
        import faiss
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.metadata = []
        log.info("Created new FAISS index")

    def _save_faiss_index(self):
        """Save the FAISS index and metadata to disk."""
        import faiss
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            log.info(f"Saved FAISS index with {len(self.metadata)} entries")
        except Exception as e:
            log.error(f"Error saving FAISS index: {str(e)}")

    def add_memory(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
        embedding: List[float] = None
    ) -> int:
        """Add a memory to the vector store."""
        # Generate embedding if not provided
        if embedding is None:
            embedding = self.embedding_model.embed_query(text)

        timestamp = datetime.now().isoformat()
        memory_metadata = {
            "text": text,
            "timestamp": timestamp,
            **(metadata or {})
        }

        if hasattr(self, 'use_fallback') and self.use_fallback:
            return self._add_memory_faiss(embedding, memory_metadata)
        else:
            return self._add_memory_qdrant(embedding, memory_metadata)

    def _add_memory_qdrant(self, embedding, metadata):
        """Add memory to Qdrant."""
        try:
            # Get the current count to use as ID
            count_result = self.client.count(collection_name=self.collection_name)
            memory_id = count_result.count

            # Add the memory ID to metadata
            metadata["id"] = memory_id

            # Add to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload=metadata
                    )
                ]
            )

            log.info(f"Added memory to Qdrant with ID: {memory_id}")
            return memory_id

        except Exception as e:
            log.error(f"Error adding memory to Qdrant: {str(e)}")
            if not hasattr(self, 'use_fallback'):
                log.info("Initializing FAISS fallback")
                self._initialize_faiss_fallback()
                return self._add_memory_faiss(embedding, metadata)
            return -1

    def _add_memory_faiss(self, embedding, metadata):
        """Add memory to FAISS fallback."""
        # Convert to numpy array of correct shape and type
        embedding_np = np.array([embedding]).astype(np.float32)

        # Add to FAISS index
        self.index.add(embedding_np)

        # Create metadata entry
        memory_id = len(self.metadata)
        metadata["id"] = memory_id

        # Add to metadata store
        self.metadata.append(metadata)

        # Save the updated index
        self._save_faiss_index()

        log.info(f"Added memory to FAISS with ID: {memory_id}")
        return memory_id

    def search(
        self,
        query: str = None,
        k: int = None,
        query_embedding: List[float] = None,
        filter_criteria: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar memories."""
        k = k or settings.memory_k

        # Generate query embedding if not provided
        if query_embedding is None and query is not None:
            query_embedding = self.embedding_model.embed_query(query)
        elif query_embedding is None and query is None:
            raise ValueError("Either query or query_embedding must be provided")

        if hasattr(self, 'use_fallback') and self.use_fallback:
            return self._search_faiss(query_embedding, k, filter_criteria)
        else:
            return self._search_qdrant(query_embedding, k, filter_criteria)

    def _search_qdrant(self, query_embedding, k, filter_criteria=None):
        """Search for memories in Qdrant."""
        try:
            # Convert filter criteria to Qdrant filter format if provided
            filter_obj = None
            if filter_criteria:
                filter_obj = self._create_qdrant_filter(filter_criteria)

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                filter=filter_obj
            )

            results = []
            for scored_point in search_result:
                memory = scored_point.payload.copy()
                memory["similarity"] = float(scored_point.score)
                results.append(memory)

            log.info(f"Found {len(results)} relevant memories in Qdrant")
            return results

        except Exception as e:
            log.error(f"Error searching Qdrant: {str(e)}")
            if not hasattr(self, 'use_fallback'):
                log.info("Initializing FAISS fallback")
                self._initialize_faiss_fallback()
                return self._search_faiss(query_embedding, k, filter_criteria)
            return []

    def _create_qdrant_filter(self, filter_criteria):
        """Convert filter criteria to Qdrant filter format."""
        filter_conditions = []

        for key, value in filter_criteria.items():
            if isinstance(value, list):
                # Handle list values (OR condition)
                or_conditions = [models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=v)
                ) for v in value]

                filter_conditions.append(models.Filter(
                    should=or_conditions,
                    min_should=1
                ))
            else:
                # Handle single value
                filter_conditions.append(models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                ))

        # Combine all conditions with AND
        return models.Filter(
            must=filter_conditions
        )

    def _search_faiss(self, query_embedding, k, filter_criteria=None):
        """Search for memories in FAISS fallback."""
        if len(self.metadata) == 0:
            return []

        k = min(k, len(self.metadata))

        # Convert to numpy array
        query_embedding_np = np.array([query_embedding]).astype(np.float32)

        # Search the index
        distances, indices = self.index.search(query_embedding_np, k * 10)  # Get more results for filtering

        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                memory = self.metadata[idx].copy()
                memory["similarity"] = 1.0 / (1.0 + float(distances[0][i]))

                # Apply filter if provided
                if filter_criteria and not self._matches_filter(memory, filter_criteria):
                    continue

                results.append(memory)

                # Stop once we have enough results
                if len(results) >= k:
                    break

        log.info(f"Found {len(results)} relevant memories in FAISS")
        return results

    def _matches_filter(self, memory, filter_criteria):
        """Check if memory matches filter criteria."""
        for key, value in filter_criteria.items():
            if key not in memory:
                return False

            if isinstance(value, list):
                # Handle list values (OR condition)
                if memory[key] not in value:
                    return False
            else:
                # Handle single value
                if memory[key] != value:
                    return False

        return True

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all memories in the store."""
        if hasattr(self, 'use_fallback') and self.use_fallback:
            return self.metadata.copy()

        try:
            # Get the count of points
            count_result = self.client.count(collection_name=self.collection_name)
            total_points = count_result.count

            if total_points == 0:
                return []

            # Scroll through all points
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=total_points
            )

            memories = []
            for point in scroll_result[0]:
                memories.append(point.payload)

            log.info(f"Retrieved {len(memories)} memories from Qdrant")
            return memories

        except Exception as e:
            log.error(f"Error getting all memories from Qdrant: {str(e)}")
            if not hasattr(self, 'use_fallback'):
                self._initialize_faiss_fallback()
            return self.metadata.copy() if hasattr(self, 'metadata') else []

    def clear(self):
        """Clear all memories from the store."""
        if hasattr(self, 'use_fallback') and self.use_fallback:
            self._create_new_faiss_index()
            self._save_faiss_index()
            log.info("Cleared all memories from FAISS index")
        else:
            try:
                # Recreate the collection
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.COSINE
                    )
                )
                log.info(f"Cleared all memories from Qdrant collection {self.collection_name}")
            except Exception as e:
                log.error(f"Error clearing memories from Qdrant: {str(e)}")
                if not hasattr(self, 'use_fallback'):
                    self._initialize_faiss_fallback()
                self._create_new_faiss_index()
                self._save_faiss_index()
