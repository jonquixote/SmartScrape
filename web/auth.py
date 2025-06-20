"""
Authentication module for SmartScrape API.
Provides API key validation, JWT token handling, and other authentication utilities.
"""
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os
from pydantic import BaseModel
import secrets
import hashlib
import time

import config

# API Key security scheme
api_key_header = APIKeyHeader(name=config.API_KEY_HEADER, auto_error=False)

# OAuth2 scheme for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Models
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: int

class User(BaseModel):
    username: str
    disabled: Optional[bool] = None
    permissions: List[str] = []

# In-memory storage for API keys (replace with database in production)
# Format: {"api_key_hash": {"user_id": "user1", "permissions": ["read", "write"], "rate_limit": 100}}
API_KEYS = {}

# Load API keys from environment variables
def load_api_keys():
    """Load API keys from environment variables."""
    # Get the master API key from environment
    master_api_key = os.getenv("MASTER_API_KEY")
    if master_api_key:
        key_hash = hash_api_key(master_api_key)
        API_KEYS[key_hash] = {
            "user_id": "admin",
            "permissions": ["admin", "read", "write"],
            "rate_limit": config.RATE_LIMIT_REQUESTS_PER_MINUTE * 2  # Higher limit for admin
        }
    
    # Load additional API keys (format: "API_KEY_user1=key1,API_KEY_user2=key2")
    for env_var, value in os.environ.items():
        if env_var.startswith("API_KEY_") and env_var != "API_KEY_HEADER":
            user_id = env_var[8:]  # Remove "API_KEY_" prefix
            key_hash = hash_api_key(value)
            API_KEYS[key_hash] = {
                "user_id": user_id,
                "permissions": ["read", "write"],
                "rate_limit": config.RATE_LIMIT_REQUESTS_PER_MINUTE
            }

# Hash API key for secure storage
def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()

# Rate limiting storage - keeps track of API usage
# Format: {"api_key_hash": {"counter": 10, "reset_time": 1621234567}}
rate_limits = {}

# Validate API key and enforce rate limits
async def validate_api_key(api_key: str = Security(api_key_header)):
    """
    Validate API key and enforce rate limits.
    Returns user info if valid, raises HTTPException if invalid or rate limited.
    """
    if config.ENVIRONMENT == "development" and not API_KEYS:
        # In development, return a default user if no API keys are configured
        return {"user_id": "dev", "permissions": ["read", "write"]}
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing",
            headers={"WWW-Authenticate": f"API key required in {config.API_KEY_HEADER} header"},
        )
    
    # Hash the provided API key
    key_hash = hash_api_key(api_key)
    
    if key_hash not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": f"Invalid API key provided in {config.API_KEY_HEADER} header"},
        )
    
    # Get user info for this API key
    user_info = API_KEYS[key_hash]
    
    # Check rate limit
    current_time = int(time.time())
    minute_window = current_time // 60
    
    if key_hash not in rate_limits:
        rate_limits[key_hash] = {"counter": 0, "window": minute_window}
    
    # Reset counter if we're in a new minute window
    if rate_limits[key_hash]["window"] < minute_window:
        rate_limits[key_hash] = {"counter": 0, "window": minute_window}
    
    # Increment counter
    rate_limits[key_hash]["counter"] += 1
    
    # Check if over limit
    if rate_limits[key_hash]["counter"] > user_info["rate_limit"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Limit is {user_info['rate_limit']} requests per minute.",
            headers={"Retry-After": str(60 - (current_time % 60))},  # Seconds until next minute
        )
    
    return user_info

# Create JWT token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm="HS256")
    
    return encoded_token, int(expire.timestamp())

# Initialize API keys on module load
load_api_keys()
