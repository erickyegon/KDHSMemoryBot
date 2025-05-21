import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import settings
from logger import log

class ConversationStore:
    """Store for conversation history with file-based persistence and future database support."""

    def __init__(self, history_limit: int = None, storage_path: str = "data/conversation_history.json"):
        """Initialize the conversation store."""
        self.history_limit = history_limit or settings.history_limit
        self.storage_path = storage_path
        self.history = []

        # Load existing history if available
        self._load_history()
        
        log.info(f"Initialized ConversationStore with history limit: {self.history_limit}")

    def _load_history(self):
        """Load conversation history from disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self.history = json.load(f)
                log.info(f"Loaded conversation history with {len(self.history)} entries")
            except Exception as e:
                log.error(f"Error loading conversation history: {str(e)}")
                self.history = []
        else:
            log.info("No existing conversation history found")
            self.history = []

    def save(self):
        """Save conversation history to disk."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.history, f)
            log.info(f"Saved conversation history with {len(self.history)} entries")
        except Exception as e:
            log.error(f"Error saving conversation history: {str(e)}")

    def add_user_message(self, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a user message to the conversation history."""
        return self._add_message("user", message, metadata)

    def add_assistant_message(self, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add an assistant message to the conversation history."""
        return self._add_message("assistant", message, metadata)

    def _add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a message to the conversation history."""
        timestamp = datetime.now().isoformat()

        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            **(metadata or {})
        }

        self.history.append(message)

        # Trim history if it exceeds the limit
        if self.history_limit > 0 and len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]
            log.info(f"Trimmed conversation history to {self.history_limit} entries")

        # Save the updated history
        self.save()

        return message

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        if limit is None or limit <= 0:
            return self.history.copy()

        return self.history[-limit:].copy()

    def get_formatted_history(self, limit: Optional[int] = None) -> str:
        """Get formatted conversation history as a string."""
        history = self.get_history(limit)
        formatted = ""

        for entry in history:
            role = "Human" if entry["role"] == "user" else "Assistant"
            formatted += f"{role}: {entry['content']}\n"

        return formatted

    def clear(self):
        """Clear the conversation history."""
        self.history = []
        self.save()
        log.info("Cleared conversation history")
