import os
import re
import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
import secrets
import string
from pydantic import BaseModel, Field, validator

from config import settings
from logger import log

# Input validation models
class UserInput(BaseModel):
    """Model for validating user input."""
    content: str = Field(..., min_length=1, max_length=4000)

    @validator('content')
    def sanitize_content(cls, v):
        """Sanitize user input to prevent injection attacks."""
        # Remove any potentially harmful HTML/script tags
        v = re.sub(r'<script.*?>.*?</script>', '', v, flags=re.DOTALL)
        v = re.sub(r'<.*?>', '', v)
        return v.strip()

class MemoryInput(BaseModel):
    """Model for validating memory input."""
    content: str = Field(..., min_length=1, max_length=10000)

    @validator('content')
    def sanitize_content(cls, v):
        """Sanitize memory input."""
        # Remove any potentially harmful HTML/script tags
        v = re.sub(r'<script.*?>.*?</script>', '', v, flags=re.DOTALL)
        v = re.sub(r'<.*?>', '', v)
        return v.strip()

class IdentityInput(BaseModel):
    """Model for validating user identity input."""
    content: str = Field(..., min_length=1, max_length=1000)

    @validator('content')
    def sanitize_content(cls, v):
        """Sanitize identity input."""
        # Remove any potentially harmful HTML/script tags
        v = re.sub(r'<script.*?>.*?</script>', '', v, flags=re.DOTALL)
        v = re.sub(r'<.*?>', '', v)
        return v.strip()

# Security utilities
def generate_api_key() -> str:
    """Generate a secure API key."""
    # Generate a random string
    alphabet = string.ascii_letters + string.digits
    random_string = ''.join(secrets.choice(alphabet) for _ in range(32))

    # Add a timestamp hash
    timestamp = str(datetime.now().timestamp())
    hash_obj = hashlib.sha256((random_string + timestamp).encode())

    return hash_obj.hexdigest()

def mask_api_key(api_key: str) -> str:
    """Mask an API key for display."""
    if not api_key or len(api_key) < 8:
        return "****"
    return api_key[:4] + "..." + api_key[-4:]

def save_user_identity(identity: str) -> bool:
    """Save user identity to environment variables."""
    try:
        # Validate input
        validated = IdentityInput(content=identity)
        identity = validated.content

        # Update environment variable
        os.environ["USER_IDENTITY"] = identity

        # Update .env file if available
        env_file = ".env"
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                lines = f.readlines()

            # Look for existing USER_IDENTITY line
            updated = False
            for i, line in enumerate(lines):
                if line.startswith("USER_IDENTITY="):
                    lines[i] = f'USER_IDENTITY="{identity}"\n'
                    updated = True
                    break

            # Add new line if not found
            if not updated:
                lines.append(f'USER_IDENTITY="{identity}"\n')

            # Write back to file
            with open(env_file, "w") as f:
                f.writelines(lines)

        log.info("User identity saved successfully")
        return True
    except Exception as e:
        log.error(f"Error saving user identity: {str(e)}")
        return False

# Date formatting
def format_date(timestamp: str) -> str:
    """Format ISO timestamp as a human-readable date string."""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%b %d, %Y %I:%M %p")
    except Exception as e:
        log.error(f"Error formatting date: {str(e)}")
        return timestamp

# API testing
def test_embeddings(api_key=None):
    """Test function for Euriai embeddings API."""
    import requests
    import numpy as np

    url = settings.euriai_embed_url
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or settings.euriai_api_key.get_secret_value()}"
    }
    payload = {
        "input": "The food was delicious and the service was excellent.",
        "model": settings.embedding_model
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Convert to numpy array for vector operations
        embedding = np.array(data['data'][0]['embedding'])

        result = {
            "success": True,
            "shape": embedding.shape,
            "first_5_values": embedding[:5].tolist(),
            "norm": float(np.linalg.norm(embedding))
        }

        log.info("API test successful")
        return result
    except Exception as e:
        log.error(f"API test failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
