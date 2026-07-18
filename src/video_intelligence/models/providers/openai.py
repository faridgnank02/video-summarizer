from __future__ import annotations

import os

from .base import Provider, ProviderError, Usage, parse_json_response


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI()
        return self._client

    async def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    async def complete(self, model, prompt, schema):
        try:
            resp = await self._get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            raise ProviderError(f"openai request failed: {e}") from e
        try:
            usage = Usage(tokens_in=resp.usage.prompt_tokens, tokens_out=resp.usage.completion_tokens)
            content = resp.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as e:
            raise ProviderError(f"unexpected openai response shape: {e}") from e
        return parse_json_response(content, schema), usage
