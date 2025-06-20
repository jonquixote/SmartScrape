"""
Model Discovery Service

Automated model discovery using public documentation and metadata endpoints.
Uses only no-cost sources - no paid API calls for model usage.
"""

import asyncio
import json
import aiohttp
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelDiscoveryService:
    """
    Automated model discovery using public documentation and metadata endpoints.
    Uses only no-cost sources - no paid API calls for model usage.
    """
    
    def __init__(self, update_interval_hours: int = 24):
        self.update_interval = timedelta(hours=update_interval_hours)
        self.cache_dir = Path("data/model_cache")
        self.models_cache_file = self.cache_dir / "models_cache.json"
        self.last_update_file = self.cache_dir / "last_model_update.json"
        self.provider_cache_files = {
            "google": self.cache_dir / "google_models.json",
            "openai": self.cache_dir / "openai_models.json", 
            "anthropic": self.cache_dir / "anthropic_models.json"
        }
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.running = False
        self._session = None
        
    async def start_background_discovery(self):
        """Start the automated model discovery background task."""
        if self.running:
            return
            
        self.running = True
        logger.info("Starting automated model discovery service")
        
        # Run initial discovery
        await self.discover_all_models()
        
        while self.running:
            try:
                await asyncio.sleep(self.update_interval.total_seconds())
                if self.running:  # Check again after sleep
                    await self.discover_all_models()
            except Exception as e:
                logger.error(f"Error in model discovery: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    def stop_discovery_service(self):
        """Stop the automated model discovery."""
        self.running = False
        logger.info("Stopped automated model discovery")
    
    async def get_models_for_provider(self, provider: str) -> List[Dict[str, Any]]:
        """Get models for a specific provider, using cache if available."""
        provider = provider.lower()
        
        # Try cache first
        cached_models = self._get_cached_provider_models(provider)
        if cached_models and not self._is_cache_stale(provider):
            logger.debug(f"Using cached models for {provider}")
            return cached_models
        
        # Cache is stale or missing, try to discover
        try:
            if provider == "google":
                models = await self._discover_google_models()
            elif provider == "openai":
                models = await self._discover_openai_models()
            elif provider == "anthropic":
                models = await self._discover_anthropic_models()
            else:
                logger.warning(f"Unknown provider: {provider}")
                return []
            
            # Cache the results
            await self._cache_provider_models(provider, models)
            return models
            
        except Exception as e:
            logger.error(f"Failed to discover models for {provider}: {e}")
            # Return cached models if discovery fails
            return cached_models or []
    
    async def discover_all_models(self):
        """Discover models for all providers using documentation and metadata sources."""
        logger.info("Starting model discovery cycle")
        
        all_models = {}
        
        # Discover Google models
        try:
            google_models = await self._discover_google_models()
            all_models["google"] = google_models
            await self._cache_provider_models("google", google_models)
            logger.info(f"Discovered {len(google_models)} Google models")
        except Exception as e:
            logger.error(f"Failed to discover Google models: {e}")
            all_models["google"] = self._get_cached_provider_models("google") or []
        
        # Discover OpenAI models  
        try:
            openai_models = await self._discover_openai_models()
            all_models["openai"] = openai_models
            await self._cache_provider_models("openai", openai_models)
            logger.info(f"Discovered {len(openai_models)} OpenAI models")
        except Exception as e:
            logger.error(f"Failed to discover OpenAI models: {e}")
            all_models["openai"] = self._get_cached_provider_models("openai") or []
        
        # Discover Anthropic models
        try:
            anthropic_models = await self._discover_anthropic_models()
            all_models["anthropic"] = anthropic_models
            await self._cache_provider_models("anthropic", anthropic_models)
            logger.info(f"Discovered {len(anthropic_models)} Anthropic models")
        except Exception as e:
            logger.error(f"Failed to discover Anthropic models: {e}")
            all_models["anthropic"] = self._get_cached_provider_models("anthropic") or []
        
        # Save consolidated cache
        await self._save_models_cache(all_models)
        await self._save_last_update()
        
        logger.info("Model discovery cycle completed")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _discover_google_models(self) -> List[Dict[str, Any]]:
        """Discover Google models using public discovery API."""
        models = []
        
        # Method 1: Use Google's public model discovery (metadata only, no usage)
        google_key = self._get_google_key()
        if google_key:
            try:
                session = await self._get_session()
                url = "https://generativelanguage.googleapis.com/v1beta/models"
                params = {"key": google_key}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for model in data.get("models", []):
                            model_id = model["name"].split("/")[-1]
                            models.append({
                                "model_id": model_id,
                                "name": model.get("displayName", model_id),
                                "description": model.get("description", ""),
                                "capabilities": self._parse_google_capabilities(model),
                                "provider": "google",
                                "context_window": self._get_google_context_window(model_id),
                                "last_updated": datetime.utcnow().isoformat()
                            })
                        logger.info(f"Discovered {len(models)} Google models via API")
                    else:
                        logger.warning(f"Google API returned status {response.status}")
            except Exception as e:
                logger.debug(f"Could not use Google discovery API: {e}")
        
        # Method 2: Fallback to known model patterns
        if not models:
            known_google_models = [
                {
                    "model_id": "gemini-2.0-flash-exp",
                    "name": "Gemini 2.0 Flash (Experimental)",
                    "description": "Latest experimental Gemini model with multimodal capabilities",
                    "capabilities": ["text", "vision", "audio"],
                    "context_window": 1000000
                },
                {
                    "model_id": "gemini-2.0-flash-lite",
                    "name": "Gemini 2.0 Flash Lite", 
                    "description": "Lightweight version of Gemini 2.0",
                    "capabilities": ["text", "vision"],
                    "context_window": 1000000
                },
                {
                    "model_id": "gemini-1.5-pro",
                    "name": "Gemini 1.5 Pro",
                    "description": "Advanced reasoning and multimodal capabilities",
                    "capabilities": ["text", "vision", "audio"],
                    "context_window": 2000000
                },
                {
                    "model_id": "gemini-1.5-flash",
                    "name": "Gemini 1.5 Flash",
                    "description": "Fast and efficient model",
                    "capabilities": ["text", "vision"],
                    "context_window": 1000000
                },
                {
                    "model_id": "gemini-1.5-flash-8b",
                    "name": "Gemini 1.5 Flash 8B",
                    "description": "Smaller, faster model",
                    "capabilities": ["text"],
                    "context_window": 1000000
                }
            ]
            
            for model in known_google_models:
                model.update({
                    "provider": "google",
                    "last_updated": datetime.utcnow().isoformat()
                })
                models.append(model)
            
            logger.info(f"Using {len(models)} known Google models as fallback")
        
        return models
    
    async def _discover_openai_models(self) -> List[Dict[str, Any]]:
        """Discover OpenAI models using their models endpoint (listing only, no usage)."""
        models = []
        
        # Method 1: Use OpenAI's models listing endpoint (no usage charges)
        openai_key = self._get_openai_key()
        if openai_key:
            try:
                session = await self._get_session()
                headers = {"Authorization": f"Bearer {openai_key}"}
                url = "https://api.openai.com/v1/models"
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        for model in data.get("data", []):
                            model_id = model["id"]
                            # Filter to relevant models
                            if model_id.startswith(("gpt-", "o1-", "chatgpt")):
                                models.append({
                                    "model_id": model_id,
                                    "name": self._format_openai_name(model_id),
                                    "description": f"OpenAI {model_id} model",
                                    "capabilities": self._infer_openai_capabilities(model_id),
                                    "provider": "openai",
                                    "context_window": self._get_openai_context_window(model_id),
                                    "last_updated": datetime.utcnow().isoformat()
                                })
                        logger.info(f"Discovered {len(models)} OpenAI models via API")
                    else:
                        logger.warning(f"OpenAI API returned status {response.status}")
            except Exception as e:
                logger.debug(f"Could not use OpenAI models API: {e}")
        
        # Method 2: Fallback to known models
        if not models:
            known_openai_models = [
                {
                    "model_id": "gpt-4o",
                    "name": "GPT-4 Omni",
                    "description": "Most advanced multimodal model",
                    "capabilities": ["text", "vision", "audio"],
                    "context_window": 128000
                },
                {
                    "model_id": "gpt-4o-mini",
                    "name": "GPT-4 Omni Mini", 
                    "description": "Efficient and cost-effective",
                    "capabilities": ["text", "vision"],
                    "context_window": 128000
                },
                {
                    "model_id": "o1-preview",
                    "name": "OpenAI o1 Preview",
                    "description": "Advanced reasoning model",
                    "capabilities": ["text", "reasoning"],
                    "context_window": 128000
                },
                {
                    "model_id": "o1-mini",
                    "name": "OpenAI o1 Mini",
                    "description": "Efficient reasoning model", 
                    "capabilities": ["text", "reasoning"],
                    "context_window": 128000
                },
                {
                    "model_id": "gpt-4-turbo",
                    "name": "GPT-4 Turbo",
                    "description": "Latest GPT-4 with improved speed",
                    "capabilities": ["text", "vision"],
                    "context_window": 128000
                }
            ]
            
            for model in known_openai_models:
                model.update({
                    "provider": "openai",
                    "last_updated": datetime.utcnow().isoformat()
                })
                models.append(model)
            
            logger.info(f"Using {len(models)} known OpenAI models as fallback")
        
        return models
    
    async def _discover_anthropic_models(self) -> List[Dict[str, Any]]:
        """Discover Anthropic models using documentation parsing."""
        models = []
        
        # Method 1: Parse Anthropic documentation
        try:
            session = await self._get_session()
            url = "https://docs.anthropic.com/en/docs/models-overview"
            async with session.get(url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    models.extend(self._parse_anthropic_docs(html_content))
                    if models:
                        logger.info(f"Discovered {len(models)} Anthropic models from docs")
        except Exception as e:
            logger.debug(f"Could not parse Anthropic docs: {e}")
        
        # Method 2: Fallback to known models
        if not models:
            known_anthropic_models = [
                {
                    "model_id": "claude-3-5-sonnet-20241022",
                    "name": "Claude 3.5 Sonnet",
                    "description": "Anthropic's most intelligent model",
                    "capabilities": ["text", "vision"],
                    "context_window": 200000
                },
                {
                    "model_id": "claude-3-5-haiku-20241022", 
                    "name": "Claude 3.5 Haiku",
                    "description": "Fast and lightweight model",
                    "capabilities": ["text"],
                    "context_window": 200000
                },
                {
                    "model_id": "claude-3-opus-20240229",
                    "name": "Claude 3 Opus",
                    "description": "Powerful model for complex tasks",
                    "capabilities": ["text", "vision"],
                    "context_window": 200000
                },
                {
                    "model_id": "claude-3-sonnet-20240229",
                    "name": "Claude 3 Sonnet",
                    "description": "Balanced performance and speed",
                    "capabilities": ["text", "vision"],
                    "context_window": 200000
                },
                {
                    "model_id": "claude-3-haiku-20240307",
                    "name": "Claude 3 Haiku",
                    "description": "Fast model for simple tasks",
                    "capabilities": ["text"],
                    "context_window": 200000
                }
            ]
            
            for model in known_anthropic_models:
                model.update({
                    "provider": "anthropic",
                    "last_updated": datetime.utcnow().isoformat()
                })
                models.append(model)
            
            logger.info(f"Using {len(models)} known Anthropic models as fallback")
        
        return models
    
    def _get_google_key(self) -> Optional[str]:
        """Get Google API key from environment."""
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    def _get_openai_key(self) -> Optional[str]:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")
    
    def _parse_google_capabilities(self, model_info: dict) -> List[str]:
        """Parse model capabilities from Google API response."""
        capabilities = ["text"]  # All models support text
        
        # Add vision capability if mentioned
        if any(term in str(model_info).lower() for term in ["vision", "image", "multimodal"]):
            capabilities.append("vision")
        
        # Add audio capability if mentioned  
        if any(term in str(model_info).lower() for term in ["audio", "speech"]):
            capabilities.append("audio")
            
        return capabilities
    
    def _get_google_context_window(self, model_id: str) -> int:
        """Get context window for Google models."""
        if "2.0" in model_id:
            return 1000000
        elif "1.5-pro" in model_id:
            return 2000000
        else:
            return 1000000
    
    def _format_openai_name(self, model_id: str) -> str:
        """Format OpenAI model ID into a display name."""
        return model_id.replace("-", " ").replace("gpt", "GPT").replace("o1", "O1").title()
    
    def _infer_openai_capabilities(self, model_id: str) -> List[str]:
        """Infer capabilities from OpenAI model ID."""
        capabilities = ["text"]
        
        if "gpt-4" in model_id or "gpt-4o" in model_id:
            capabilities.append("vision")
            
        if "o1" in model_id:
            capabilities.append("reasoning")
            
        return capabilities
    
    def _get_openai_context_window(self, model_id: str) -> int:
        """Get context window for OpenAI models."""
        if "gpt-4" in model_id or "o1" in model_id:
            return 128000
        elif "gpt-3.5" in model_id:
            return 16000
        else:
            return 128000
    
    def _parse_anthropic_docs(self, html_content: str) -> List[Dict[str, Any]]:
        """Parse Anthropic documentation for model information."""
        models = []
        
        # Simple regex parsing for model names
        model_patterns = re.findall(r'claude-[0-9]+-[a-z]+-[0-9]+', html_content, re.IGNORECASE)
        
        for pattern in set(model_patterns):  # Remove duplicates
            name = pattern.replace("-", " ").replace("claude", "Claude").title()
            models.append({
                "model_id": pattern,
                "name": name,
                "description": f"Anthropic {name} model",
                "capabilities": ["text", "vision"] if "opus" in pattern or "sonnet" in pattern else ["text"],
                "provider": "anthropic",
                "context_window": 200000,
                "last_updated": datetime.utcnow().isoformat()
            })
        
        return models
    
    def _get_cached_provider_models(self, provider: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached models for a provider."""
        cache_file = self.provider_cache_files.get(provider)
        if not cache_file or not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to load cached models for {provider}: {e}")
            return None
    
    async def _cache_provider_models(self, provider: str, models: List[Dict[str, Any]]):
        """Cache models for a specific provider."""
        cache_file = self.provider_cache_files.get(provider)
        if not cache_file:
            return
        
        try:
            cache_data = {
                "provider": provider,
                "models": models,
                "last_updated": datetime.utcnow().isoformat(),
                "model_count": len(models)
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to cache models for {provider}: {e}")
    
    def _is_cache_stale(self, provider: str, max_age_hours: int = 12) -> bool:
        """Check if cache is stale for a provider."""
        cache_file = self.provider_cache_files.get(provider)
        if not cache_file or not cache_file.exists():
            return True
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                last_updated = datetime.fromisoformat(data.get("last_updated", ""))
                age = datetime.utcnow() - last_updated
                return age > timedelta(hours=max_age_hours)
        except Exception as e:
            logger.debug(f"Error checking cache staleness for {provider}: {e}")
            return True
    
    async def _save_models_cache(self, models: Dict[str, List[Dict[str, Any]]]):
        """Save consolidated models to cache file."""
        try:
            cache_data = {
                "models": models,
                "last_updated": datetime.utcnow().isoformat(),
                "total_models": sum(len(provider_models) for provider_models in models.values())
            }
            
            with open(self.models_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save models cache: {e}")
    
    async def _save_last_update(self):
        """Save last update timestamp."""
        try:
            update_info = {
                "last_update": datetime.utcnow().isoformat(),
                "next_update": (datetime.utcnow() + self.update_interval).isoformat()
            }
            with open(self.last_update_file, 'w') as f:
                json.dump(update_info, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save last update info: {e}")
    
    def get_cached_models(self, provider: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get models from consolidated cache."""
        try:
            if not self.models_cache_file.exists():
                return {}
                
            with open(self.models_cache_file, 'r') as f:
                data = json.load(f)
                models = data.get("models", {})
                
            if provider:
                return {provider: models.get(provider, [])}
            return models
        except Exception as e:
            logger.error(f"Failed to load models cache: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the service gracefully."""
        self.stop_discovery_service()
        
        if self._session and not self._session.closed:
            await self._session.close()

# Global instance
_model_discovery_service: Optional[ModelDiscoveryService] = None

def get_model_discovery_service() -> ModelDiscoveryService:
    """Get the global model discovery service instance."""
    global _model_discovery_service
    if _model_discovery_service is None:
        _model_discovery_service = ModelDiscoveryService()
    return _model_discovery_service
