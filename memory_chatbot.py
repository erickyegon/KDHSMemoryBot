import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import settings
from logger import log
from embedding import AsyncEuriaiEmbeddings
from chat import AsyncEuriaiChat
from vector_store import QdrantVectorStore
from conversation_store import ConversationStore
from utils import UserInput

class MemoryChatbot:
    """Main chatbot class that ties everything together."""

    def __init__(
        self,
        conversation_store: Optional[ConversationStore] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        chat_model: Optional[AsyncEuriaiChat] = None,
        embedding_model: Optional[AsyncEuriaiEmbeddings] = None,
        user_identity: str = None
    ):
        """Initialize the memory chatbot."""
        self.embedding_model = embedding_model or AsyncEuriaiEmbeddings()
        self.conversation_store = conversation_store or ConversationStore()
        self.vector_store = vector_store or QdrantVectorStore(embedding_model=self.embedding_model)
        self.chat_model = chat_model or AsyncEuriaiChat()
        self.user_identity = user_identity or settings.user_identity
        
        log.info(f"Initialized MemoryChatbot with user identity: {self.user_identity}")

    async def process_input_async(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate a response asynchronously."""
        # Validate input
        validated_input = UserInput(content=user_input)
        user_input = validated_input.content
        
        # Add user message to conversation history
        self.conversation_store.add_user_message(user_input)
        log.info(f"Added user message to conversation history: {user_input[:50]}...")

        # Get conversation history for context
        formatted_history = self.conversation_store.get_formatted_history()

        # Get relevant memories
        relevant_memories = self.vector_store.search(user_input, settings.memory_k)
        log.info(f"Retrieved {len(relevant_memories)} relevant memories")

        # Format relevant memories
        formatted_memories = self._format_memories(relevant_memories)

        # Create the prompt
        prompt = self._create_prompt(user_input, formatted_history, formatted_memories)

        # Generate response
        try:
            # Generate response asynchronously
            response = await self.chat_model.generate_completion_async(prompt)
            log.info(f"Generated response: {response[:50]}...")

        except Exception as e:
            log.error(f"Error generating response: {str(e)}")
            response = f"I apologize, but I encountered an error while processing your request. Please try again or check your API configuration."

        # Add assistant message to conversation history
        message = self.conversation_store.add_assistant_message(response)

        # Add the interaction to memory
        await self._add_interaction_to_memory_async(user_input, response)

        # Prepare response object
        response_obj = {
            "response": response,
            "conversation_history": self.conversation_store.get_history(),
            "metadata": {
                "timestamp": message["timestamp"]
            }
        }

        return response_obj

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate a response (synchronous wrapper)."""
        # For synchronous code paths, we'll use asyncio.run
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an event loop, use run_until_complete
                return loop.run_until_complete(self.process_input_async(user_input))
            else:
                # No event loop running, use asyncio.run
                return asyncio.run(self.process_input_async(user_input))
        except RuntimeError:
            # If we can't get an event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.process_input_async(user_input))
            loop.close()
            return result

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories for the prompt."""
        if not memories:
            return "No relevant memories found."

        formatted = ""
        for i, memory in enumerate(memories):
            formatted += f"{i+1}. {memory['text']} (Relevance: {memory['similarity']:.2f})\n"

        return formatted

    def _create_prompt(self, query: str, conversation_history: str, relevant_memories: str) -> str:
        """Create the prompt for the language model."""
        return f"""
You are a helpful, intelligent memory-based assistant that responds based on the conversation history and relevant memories.

USER IDENTITY:
{self.user_identity}

RELEVANT MEMORIES:
{relevant_memories}

CONVERSATION HISTORY:
{conversation_history}

CURRENT QUERY:
Human: {query}

Your response should be helpful, personalized based on the user's identity and conversation history, and incorporate relevant memories when appropriate.
Assistant: """

    async def _add_interaction_to_memory_async(self, query: str, response: str, metadata: Dict[str, Any] = None) -> int:
        """Add an interaction to the memory store asynchronously."""
        # Create a memory entry from the interaction
        memory_text = f"Human: {query}\nAssistant: {response}"

        # Add to vector store
        memory_id = self.vector_store.add_memory(
            text=memory_text,
            metadata={
                "type": "interaction",
                "query": query,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
        )
        
        log.info(f"Added interaction to memory store with ID: {memory_id}")
        return memory_id

    def _add_interaction_to_memory(self, query: str, response: str, metadata: Dict[str, Any] = None) -> int:
        """Add an interaction to the memory store (synchronous wrapper)."""
        return self.vector_store.add_memory(
            text=f"Human: {query}\nAssistant: {response}",
            metadata={
                "type": "interaction",
                "query": query,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
        )

    def set_user_identity(self, identity: str):
        """Set or update the user identity."""
        self.user_identity = identity
        log.info(f"Updated user identity: {identity[:50]}...")

    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_store.get_history(limit)

    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_store.clear()
        log.info("Cleared conversation history")

    def add_to_memory(self, text: str, metadata: Dict[str, Any] = None) -> int:
        """Add a custom memory entry."""
        memory_id = self.vector_store.add_memory(text, metadata)
        log.info(f"Added custom memory: {text[:50]}... with ID: {memory_id}")
        return memory_id

    def get_suggestions(self, query: str = "", k: int = 3) -> List[Dict[str, Any]]:
        """Get suggestions based on conversation history and memories."""
        # If query is empty, use the last user message
        if not query and self.conversation_store.history:
            for message in reversed(self.conversation_store.history):
                if message["role"] == "user":
                    query = message["content"]
                    break

        if not query:
            return []

        # Get relevant memories
        memories = self.vector_store.search(query, k)
        log.info(f"Retrieved {len(memories)} suggestions for query: {query[:50]}...")

        # Format as suggestions
        suggestions = []
        for memory in memories:
            suggestion = {
                "text": memory["text"],
                "relevance": memory["similarity"],
                "id": memory["id"]
            }
            suggestions.append(suggestion)

        return suggestions
