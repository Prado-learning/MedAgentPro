import os
from typing import Any

import openai

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "chatgpt-4o-latest")


class LegacyOpenAIClient:
    def __init__(self, module, api_key: str, base_url: str | None = None):
        self.module = module
        self.api_key = api_key
        self.base_url = base_url


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def get_default_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def create_client(api_key: str, base_url: str | None = None) -> Any:
    if OpenAI is not None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    openai.api_key = api_key
    if base_url:
        openai.api_base = base_url
    return LegacyOpenAIClient(openai, api_key=api_key, base_url=base_url)


def create_chat_completion(
    client: Any,
    messages: list[dict[str, Any]],
    model: str | None = None,
    **kwargs: Any,
):
    selected_model = model or get_default_model()

    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return client.chat.completions.create(
            model=selected_model,
            messages=messages,
            **kwargs,
        )

    request_kwargs = {
        "model": selected_model,
        "messages": messages,
    }
    request_kwargs.update(kwargs)
    return client.module.ChatCompletion.create(**request_kwargs)


def extract_text_content(completion) -> str:
    choices = _get_attr_or_key(completion, "choices", [])
    if not choices:
        return ""

    first_choice = choices[0]
    message = _get_attr_or_key(first_choice, "message", {})
    content = _get_attr_or_key(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            text_value = _get_attr_or_key(item, "text", "")
            if text_value:
                text_parts.append(str(text_value))
        return "".join(text_parts)

    if content is None:
        return ""

    return str(content)
