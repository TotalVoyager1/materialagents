import os
import json
import logging
import time
from collections.abc import Callable

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it"
INVALID_ANSWER_MARKERS = (
    "user safety:",
    "here's a thinking process",
    "here is a thinking process",
    "let me think through this",
)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "Не найден OPENROUTER_API_KEY. Добавьте ключ в .env или задайте "
                "переменную окружения перед запуском."
            )
        self.api_key = key
        self.model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.last_model = self.model

    def generate(
        self,
        system_instruction: str,
        prompt: str,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        for attempt in range(3):
            parts: list[str] = []
            pending_deltas: list[str] = []
            streamed_chars = 0
            invalid_marker = False
            routed_model = self.model
            try:
                with httpx.stream(
                    "POST",
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "Student Multi-Agent Orchestrator",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 4096,
                        "reasoning": {"enabled": True, "exclude": True},
                        "stream": True,
                    },
                    timeout=90.0,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError as error:
                            raise RuntimeError(
                                "OpenRouter прислал некорректное потоковое событие."
                            ) from error
                        if "error" in event:
                            message = event["error"].get("message", "Ошибка генерации.")
                            raise RuntimeError(f"Ошибка OpenRouter API: {message}")
                        choices = event.get("choices", [])
                        routed_model = event.get("model", routed_model)
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {}).get("content")
                        if isinstance(delta, str) and delta:
                            parts.append(delta)
                            current_text = "".join(parts)
                            if any(
                                marker in current_text.casefold()
                                for marker in INVALID_ANSWER_MARKERS
                            ):
                                invalid_marker = True
                                pending_deltas.clear()
                            elif on_delta is not None and not invalid_marker:
                                if streamed_chars:
                                    on_delta(delta)
                                    streamed_chars += len(delta)
                                else:
                                    pending_deltas.append(delta)
                                    if len(current_text) >= 80:
                                        for pending in pending_deltas:
                                            on_delta(pending)
                                            streamed_chars += len(pending)
                                        pending_deltas.clear()
            except httpx.TransportError as error:
                if (
                    "UNEXPECTED_EOF_WHILE_READING" not in str(error)
                    or attempt == 2
                    or parts
                ):
                    raise
                delay = attempt + 1
                logger.warning(
                    "OpenRouter connection closed unexpectedly; retrying in %s second(s) "
                    "(attempt %s of 3).",
                    delay,
                    attempt + 2,
                )
                time.sleep(delay)
            except httpx.HTTPStatusError as error:
                try:
                    error.response.read()
                    payload = error.response.json()
                    details = payload.get("error", {}).get("message", "")
                except (ValueError, AttributeError):
                    details = ""
                message = details or error.response.reason_phrase
                raise RuntimeError(
                    f"Ошибка OpenRouter API (HTTP {error.response.status_code}): {message}"
                ) from error

            text = "".join(parts).strip()
            self.last_model = routed_model
            invalid_answer = (
                not text
                or len(text) < 40
                or invalid_marker
                or any(marker in text.casefold() for marker in INVALID_ANSWER_MARKERS)
            )
            if invalid_answer:
                if attempt < 2:
                    logger.warning(
                        "OpenRouter returned an unusable answer from %s; retrying "
                        "(attempt %s of 3).",
                        routed_model,
                        attempt + 2,
                    )
                    continue
                raise RuntimeError(
                    "OpenRouter не смог выдать содержательный ответ после трёх попыток "
                    f"(последняя модель: {routed_model}). Попробуйте запустить этап ещё раз."
                )
            if on_delta is not None and not streamed_chars:
                for offset in range(0, len(text), 48):
                    on_delta(text[offset : offset + 48])
            break
        if not text:
            raise RuntimeError("OpenRouter вернул пустой ответ.")
        return text.strip()
