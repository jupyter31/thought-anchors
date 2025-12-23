"""Azure Foundry LLM client using Azure AI Inference SDK."""
from typing import Any, Dict, Optional
import os
import logging

from .base_llm_client import BaseLLMClient, ChatResult, retry_on_error

logger = logging.getLogger(__name__)


class AzureFoundryClient(BaseLLMClient):
    """Client for Azure Foundry models using Azure AI Inference SDK with API key authentication."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: int = 300,
    ):
        """
        Initialize Azure Foundry client.
        
        Args:
            endpoint: Azure inference endpoint URL
                     Defaults to AZURE_INFERENCE_SDK_ENDPOINT env var
            model_name: Model deployment name
                       Defaults to DEPLOYMENT_NAME env var
            api_key: Azure API key for authentication
                    Defaults to AZURE_API_KEY env var
            timeout_s: Request timeout in seconds (default: 300)
        """
        try:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError(
                "Azure Foundry client requires 'azure-ai-inference' package. "
                "Install with: pip install azure-ai-inference"
            )
        
        self.endpoint = endpoint or os.getenv(
            "AZURE_INFERENCE_SDK_ENDPOINT",
            "https://model-ft-test.services.ai.azure.com/models"
        )
        self.default_model_name = model_name or os.getenv("DEPLOYMENT_NAME", "DeepSeek-R1-0528")
        self.timeout_s = timeout_s
        
        # Get API key from parameter or environment
        key = api_key or os.getenv("AZURE_API_KEY")
        if not key:
            raise ValueError(
                "API key must be provided either as 'api_key' parameter or "
                "via AZURE_API_KEY environment variable"
            )
        
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(key)
        )
        
        logger.info(f"Initialized AzureFoundryClient with timeout={timeout_s}s, endpoint={self.endpoint}, model={self.default_model_name}")

    @retry_on_error(max_retries=3, initial_wait=30.0)
    def send_chat_request(
        self, model_name: str, request: Dict[str, Any]
    ) -> ChatResult:
        """Send a chat completion request with retry logic for 429/503."""
        from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
        
        # Convert messages to Azure AI Inference format
        messages = []
        for msg in request.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "user":
                messages.append(UserMessage(content=content))
            elif role == "assistant":
                messages.append(AssistantMessage(content=content))
            else:
                # Default to user message for unknown roles
                messages.append(UserMessage(content=content))
        
        # Build kwargs for the API call
        kwargs = {
            "messages": messages,
            "model": model_name or self.default_model_name,
        }
        
        # Add temperature if provided
        if "temperature" in request:
            kwargs["temperature"] = request["temperature"]
        
        # Handle token limits - support both parameter names
        if "max_completion_tokens" in request:
            kwargs["max_tokens"] = request["max_completion_tokens"]
        elif "max_tokens" in request:
            kwargs["max_tokens"] = request["max_tokens"]
        
        # Add seed if provided (may not be supported by all models)
        if "seed" in request:
            kwargs["seed"] = request["seed"]
        
        # Send request with timeout (pass as float seconds)
        logger.debug(f"Sending Azure Foundry request with timeout={self.timeout_s}s")
        response = self.client.complete(**kwargs, timeout=float(self.timeout_s))
        
        # Extract response content
        choices = response.choices if hasattr(response, 'choices') else []
        if choices:
            message = choices[0].message
            content = message.content if hasattr(message, 'content') else ""
        else:
            content = ""
        
        # Extract usage information
        usage = {}
        if hasattr(response, 'usage'):
            usage_obj = response.usage
            if hasattr(usage_obj, 'prompt_tokens'):
                usage["prompt_tokens"] = usage_obj.prompt_tokens
            if hasattr(usage_obj, 'completion_tokens'):
                usage["completion_tokens"] = usage_obj.completion_tokens
            if hasattr(usage_obj, 'total_tokens'):
                usage["total_tokens"] = usage_obj.total_tokens
        
        # Return raw response - <think> tag cleaning is handled centrally in pipeline
        # This ensures consistent behavior across ALL clients
        result: ChatResult = {
            "text": (content or "").strip(),
            "usage": usage,
            "reasoning_text": None,  # Will be extracted centrally if <think> tags present
            "process_tokens": None,
            "flags": {},  # Flags will be set by centralized cleaning
        }
        
        return result


__all__ = ["AzureFoundryClient"]
