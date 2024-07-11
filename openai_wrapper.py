from typing import List, Dict
import os
import openai
from openai import AsyncOpenAI, OpenAI

# client = AsyncOpenAI()
class AsyncOpenAIGPT4V:
    """
    Simple wrapper for an GPT4-V.
    """
    def __init__(
        self, 
        max_tokens=512,
        ):
        """
        Initializes AsyncAzureOpenAI client.
        """
        openai.api_key = os.environ.get("GPT4_API_KEY")
        self.client = AsyncOpenAI(api_key=openai.api_key)
        self.max_tokens = max_tokens

    @property
    def llm_type(self):
        return "GPT-4 V"

    async def __call__(self, 
        messages: List[Dict[str, str]], 
        **kwargs,
    ):
        """
        Make an async API call.
        """
        print(kwargs)
        return await self.client.chat.completions.create(
        # model="gpt-4-vision-preview",
        # model="gpt-4-turbo-2024-04-09",
        model="gpt-4o",
        messages=messages,
        max_tokens=self.max_tokens,
        **kwargs,
    )

class OpenAIGPT4:
    """
    Simple wrapper for an GPT4.
    """
    def __init__(
        self, 
        max_tokens=768,
        ):
        """
        Initializes OpenAI client.
        """
        openai.api_key = os.environ.get("GPT4_API_KEY")
        self.client = OpenAI(api_key=openai.api_key)
        self.max_tokens = max_tokens

    @property
    def llm_type(self):
        return "GPT-4"

    def __call__(self, 
        messages: List[Dict[str, str]], 
        **kwargs,
    ):
        """
        Make an async API call.
        """
        return self.client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=self.max_tokens,
        **kwargs,
    )

class AsyncOpenAIGPT3_5_Turbo:
    """
    Simple wrapper for an GPT3.
    """
    def __init__(
        self, 
        max_tokens=768,
        ):
        """
        Initializes OpenAI client.
        """
        openai.api_key = os.environ.get("GPT4_API_KEY")
        self.client = OpenAI(api_key=openai.api_key)
        self.max_tokens = max_tokens

    @property
    def llm_type(self):
        return "GPT-3.5"

    

class AsyncOpenAIGPT3_Turbo:
    """
    Simple wrapper for an GPT4-V.
    """
    def __init__(
        self, 
        max_tokens=512,
        ):
        """
        Initializes AsyncAzureOpenAI client.
        """
        openai.api_key = os.environ.get("GPT4_API_KEY")
        self.client = AsyncOpenAI(api_key=openai.api_key)
        self.max_tokens = max_tokens

    @property
    def llm_type(self):
        return "GPT-4 V"

    async def __call__(self, 
        messages: List[Dict[str, str]], 
        **kwargs,
    ):
        """
        Make an async API call.
        """
        print(kwargs)
        return await self.client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        max_tokens=self.max_tokens,
        **kwargs,
    )