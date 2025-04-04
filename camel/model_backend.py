import os
import openai
import tiktoken
# from openai.types.chat import ChatCompletion  # for type hints
from typing import Any, Dict
from abc import ABC, abstractmethod

from camel.typing import ModelType
from chatdev.statistics import prompt_cost
from chatdev.utils import log_visualize

# Grab the OpenAI API key and optional base URL
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.environ["BASE_URL"] if "BASE_URL" in os.environ else None

try:
    # If the new openai API version is installed,
    # from openai.types.chat import ChatCompletion
    openai_new_api = True
except ImportError:
    openai_new_api = False


class ModelBackend(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class OpenAIModel(ModelBackend):
    def __init__(self, model_type: ModelType, model_config_dict: Dict) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        # Build a single string from messages to count tokens
        string = "\n".join([message["content"] for message in kwargs["messages"]])
        encoding = tiktoken.encoding_for_model(self.model_type.value)
        num_prompt_tokens = len(encoding.encode(string))
        gap_between_send_receive = 15 * len(kwargs["messages"])
        num_prompt_tokens += gap_between_send_receive

        # -- Ensure openai.api_key is set from environment variable --
        openai.api_key = OPENAI_API_KEY
        if BASE_URL:
            openai.api_base = BASE_URL

        # Map max token counts to model names
        num_max_token_map = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-3.5-turbo-16k": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 100000,
            "gpt-4o": 4096,
            "gpt-4o-mini": 16384,
        }
        num_max_token = num_max_token_map[self.model_type.value]
        num_max_completion_tokens = num_max_token - num_prompt_tokens
        self.model_config_dict["max_tokens"] = num_max_completion_tokens

        # Actually run the query
        response = openai.ChatCompletion.create(
            *args,
            **kwargs,
            model=self.model_type.value,
            **self.model_config_dict,
        )

        # Cost info
        usage = response["usage"]
        cost = prompt_cost(
            self.model_type.value,
            num_prompt_tokens=usage["prompt_tokens"],
            num_completion_tokens=usage["completion_tokens"],
        )
        log_visualize(
            "**[OpenAI_Usage_Info Receive]**\n"
            f"prompt_tokens: {usage['prompt_tokens']}\n"
            f"completion_tokens: {usage['completion_tokens']}\n"
            f"total_tokens: {usage['total_tokens']}\n"
            f"cost: ${cost:.6f}\n"
        )

        if not isinstance(response, dict):
            raise RuntimeError("Unexpected return from OpenAI API")

        return response


class StubModel(ModelBackend):
    """A dummy model used for unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        ARBITRARY_STRING = "Lorem Ipsum"
        return {
            "id": "stub_model_id",
            "usage": {},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": ARBITRARY_STRING, "role": "assistant"},
                }
            ],
        }


class ModelFactory:
    """Factory of backend models."""

    @staticmethod
    def create(model_type: ModelType, model_config_dict: Dict) -> ModelBackend:
        default_model_type = ModelType.GPT_3_5_TURBO

        if model_type in {
            ModelType.GPT_3_5_TURBO,
            ModelType.GPT_3_5_TURBO_NEW,
            ModelType.GPT_4,
            ModelType.GPT_4_32k,
            ModelType.GPT_4_TURBO,
            ModelType.GPT_4_TURBO_V,
            ModelType.GPT_4O,
            ModelType.GPT_4O_MINI,
            None,
        }:
            model_class = OpenAIModel
        elif model_type == ModelType.STUB:
            model_class = StubModel
        else:
            raise ValueError("Unknown model")

        if model_type is None:
            model_type = default_model_type

        return model_class(model_type, model_config_dict)
