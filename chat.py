import httpx
import asyncio
from typing import List, Dict, Any, Optional
import requests

from config import settings
from logger import log

class AsyncEuriaiChat:
    """Asynchronous wrapper for Euriai chat completion API."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        """Initialize the Euriai chat model."""
        self.api_key = api_key or settings.euriai_api_key.get_secret_value()
        self.model = model or settings.chat_model
        self.temperature = temperature or settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens
        self.completion_url = settings.euriai_completion_url
        self.chat_url = settings.euriai_chat_url

        # Import and initialize the Euriai client if available
        try:
            from euriai import EuriaiClient
            self.client = EuriaiClient(
                api_key=self.api_key,
                model=self.model
            )
            self.client_available = True
            log.info(f"Initialized EuriaiClient with model: {self.model}")
        except ImportError:
            log.warning("Euriai client not installed. Using direct API calls instead.")
            self.client_available = False
        except Exception as e:
            log.warning(f"Error initializing Euriai client: {str(e)}. Using direct API calls instead.")
            self.client_available = False

    def _get_headers(self):
        """Get API request headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    async def generate_completion_async(self, prompt: str, **kwargs) -> str:
        """Generate a completion from the model asynchronously."""
        # Extract parameters
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        if self.client_available:
            try:
                # Use the Euriai client to generate a completion
                # Note: If the client doesn't support async, we'll run it in a thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.generate_completion(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                )
                log.info("Generated completion using Euriai client")
                return response
            except Exception as e:
                log.warning(f"Error with Euriai client: {str(e)}. Falling back to direct API.")
                return await self._generate_completion_direct_async(prompt, temperature, max_tokens)
        else:
            return await self._generate_completion_direct_async(prompt, temperature, max_tokens)

    async def _generate_completion_direct_async(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate a completion using direct API call asynchronously."""
        headers = self._get_headers()

        # Try the completions endpoint first
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.completion_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                log.info("Generated completion using direct API (completions endpoint)")
                return data["choices"][0]["text"]

        except Exception as e:
            log.warning(f"Error with completions endpoint: {str(e)}. Trying chat endpoint.")

            # If that fails, try the chat endpoint
            try:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(self.chat_url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()

                    log.info("Generated completion using direct API (chat endpoint)")
                    return data["choices"][0]["message"]["content"]

            except Exception as e2:
                log.error(f"Error generating completion: {str(e)}. Additional error: {str(e2)}")
                raise Exception(f"Error generating completion: {str(e)}. Additional error: {str(e2)}")

    # Synchronous methods for backward compatibility
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate a completion from the model (synchronous)."""
        # Extract parameters
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        if self.client_available:
            try:
                # Use the Euriai client to generate a completion
                response = self.client.generate_completion(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                log.info("Generated completion using Euriai client (sync)")
                return response
            except Exception as e:
                log.warning(f"Error with Euriai client: {str(e)}. Falling back to direct API.")
                return self._generate_completion_direct(prompt, temperature, max_tokens)
        else:
            return self._generate_completion_direct(prompt, temperature, max_tokens)

    def _generate_completion_direct(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate a completion using direct API call (synchronous)."""
        try:
            # Prepare the API request
            headers = self._get_headers()
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Make the API request
            response = requests.post(self.completion_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract the completion text
            log.info("Generated completion using direct API (completions endpoint, sync)")
            return data["choices"][0]["text"]

        except Exception as e:
            # If that fails, try the chat endpoint
            try:
                headers = self._get_headers()
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                response = requests.post(self.chat_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()

                log.info("Generated completion using direct API (chat endpoint, sync)")
                return data["choices"][0]["message"]["content"]

            except Exception as e2:
                log.error(f"Error generating completion: {str(e)}. Additional error: {str(e2)}")
                raise Exception(f"Error generating completion: {str(e)}. Additional error: {str(e2)}")
