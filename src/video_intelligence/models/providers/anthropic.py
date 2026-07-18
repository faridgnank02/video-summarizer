from __future__ import annotations

import os

from .base import Provider, ProviderError, Usage, parse_json_response


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic()
        return self._client

    async def is_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    async def complete(self, model, prompt, schema):
        try:
            resp = await self._get_client().messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            raise ProviderError(f"anthropic request failed: {e}") from e
        usage = Usage(tokens_in=resp.usage.input_tokens, tokens_out=resp.usage.output_tokens)
        return parse_json_response(resp.content[0].text, schema), usage
