import httpx
import asyncio
from typing import List, Dict, Any, Optional
import requests
import numpy as np

from config import settings
from logger import log

class AsyncEuriaiEmbeddings:
    """Asynchronous wrapper for Euriai embeddings API."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
    ):
        """Initialize the Euriai embeddings model."""
        self.api_key = api_key or settings.euriai_api_key.get_secret_value()
        self.model = model or settings.embedding_model
        self.embed_url = settings.euriai_embed_url

        if not self.api_key:
            log.error("Euriai API key not provided")
            raise ValueError("Euriai API key not provided")

        log.info(f"Initialized AsyncEuriaiEmbeddings with model: {self.model}")

    def _get_headers(self):
        """Get API request headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents asynchronously."""
        if not texts:
            return []

        # Process in batches if needed for large text collections
        all_embeddings = []
        batch_size = 20  # Adjust based on API limits

        # Split into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        log.info(f"Processing {len(texts)} texts in {len(batches)} batches")

        async with httpx.AsyncClient(timeout=60.0) as client:
            tasks = []
            for batch in batches:
                payload = {
                    "input": batch,
                    "model": self.model
                }

                task = client.post(
                    self.embed_url,
                    headers=self._get_headers(),
                    json=payload
                )
                tasks.append(task)

            # Wait for all requests to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Process responses
            for response in responses:
                if isinstance(response, Exception):
                    log.error(f"Error in async embedding request: {str(response)}")
                    continue

                try:
                    response.raise_for_status()
                    data = response.json()
                    batch_embeddings = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    log.error(f"Error processing embedding response: {str(e)}")

        log.info(f"Generated {len(all_embeddings)} embeddings asynchronously")
        return all_embeddings

    async def embed_query_async(self, text: str) -> List[float]:
        """Generate an embedding for a single query text asynchronously."""
        embeddings = await self.embed_documents_async([text])
        return embeddings[0] if embeddings else []

    # Synchronous methods for backward compatibility
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents (synchronous)."""
        if not texts:
            return []

        # Process in batches if needed for large text collections
        all_embeddings = []

        # For now, process all at once
        payload = {
            "input": texts,
            "model": self.model
        }

        try:
            response = requests.post(
                self.embed_url,
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings from the response
            embeddings = [item["embedding"] for item in data["data"]]
            log.info(f"Generated {len(embeddings)} embeddings synchronously")
            return embeddings

        except Exception as e:
            log.error(f"Error generating embeddings: {str(e)}")
            raise Exception(f"Error generating embeddings: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single query text (synchronous)."""
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []
