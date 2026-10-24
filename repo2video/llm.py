"""Client interface for interacting with local LLM providers (Ollama or OpenAI-compatible)."""

from __future__ import annotations

import logging
import sys
import time
import urllib.parse
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import requests

from repo2video.config import get_settings
from repo2video.errors import ConfigError, LLMError, Repo2VideoError

if TYPE_CHECKING:
    from repo2video.config import Settings

logger = logging.getLogger(__name__)

RETRY_DELAYS = (1.0, 3.0)
RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        """Send chat messages to the LLM and return the assistant response.

        Args:
            system: System instructions guiding the LLM's role and output format.
            user: User prompt containing the task and input data.
            temperature: Optional per-call sampling temperature override.

        Returns:
            The raw text response content from the model.

        Raises:
            LLMError: If the request fails, times out, or returns invalid content.
        """
        ...

    def check_ready(self) -> None:
        """Verify that the LLM service is reachable and properly configured.

        Raises:
            LLMError: If the service cannot be reached, the model is missing,
                or authentication fails.
        """
        ...


def _matches_model(configured: str, candidate: str) -> bool:
    """Check if a candidate model name matches the configured model name.

    Normalizes model names and compares with and without the ':latest' suffix,
    ignoring registry/library path prefixes if the configured name does not specify one.

    Args:
        configured: The model name from settings (e.g. 'qwen2.5:7b' or 'llama3').
        candidate: An installed model name (e.g. 'qwen2.5:7b' or 'llama3:latest').

    Returns:
        True if the candidate matches the configured model name, False otherwise.
    """

    def normalize(name: str) -> str:
        clean = name.strip()
        if "/" in clean and "/" not in configured:
            clean = clean.split("/")[-1]
        return clean

    def variants(name: str) -> set[str]:
        clean = normalize(name)
        result = {clean}
        if clean.endswith(":latest"):
            result.add(clean[:-7])
        elif ":" not in clean:
            result.add(f"{clean}:latest")
        return result

    return bool(variants(configured) & variants(candidate))


def _is_local_address(url: str) -> bool:
    """Check if the given URL points to a local address (localhost or 127.0.0.1).

    Args:
        url: The URL to inspect.

    Returns:
        True if the hostname is localhost or 127.0.0.1, False otherwise.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = (parsed.hostname or "").lower()
        return hostname in ("localhost", "127.0.0.1")
    except (ValueError, AttributeError):
        return False


def _build_endpoint_url(base_url: str, path: str) -> str:
    """Construct an endpoint URL ensuring /v1 is not duplicated.

    Args:
        base_url: The configured LLM base URL.
        path: The relative API endpoint path, e.g. 'chat/completions' or 'models'.

    Returns:
        The normalized URL string.
    """
    clean_base = base_url.rstrip("/")
    clean_path = path.lstrip("/")
    if clean_base.endswith("/v1"):
        clean_path = clean_path.removeprefix("v1/")
        return f"{clean_base}/{clean_path}"
    if clean_path.startswith("v1/"):
        return f"{clean_base}/{clean_path}"
    return f"{clean_base}/v1/{clean_path}"


def _sanitize_text(text: str, secret: str | None) -> str:
    """Mask sensitive tokens or API keys in error output and logs.

    Args:
        text: Text to sanitize.
        secret: Optional sensitive string to mask.

    Returns:
        The text with secret replaced by '***'.
    """
    if secret:
        return text.replace(secret, "***")
    return text


def _parse_retry_after(header_val: str | None) -> float | None:
    """Parse HTTP Retry-After header into a float duration in seconds.

    Args:
        header_val: The header value string, if present.

    Returns:
        Delay in seconds, or None if invalid or absent.
    """
    if not header_val:
        return None
    try:
        val = float(header_val.strip())
        return val if val >= 0 else None
    except ValueError:
        pass

    try:
        import datetime
        from email.utils import parsedate_to_datetime

        target_dt = parsedate_to_datetime(header_val)
        now_dt = datetime.datetime.now(datetime.UTC)
        diff = (target_dt - now_dt).total_seconds()
        return max(0.0, diff)
    except (ValueError, TypeError, OverflowError):
        return None


def _execute_request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    timeout: int,
    provider: str,
    base_url: str,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    json_payload: dict[str, Any] | None = None,
    allow_404: bool = False,
) -> requests.Response:
    """Execute an HTTP request with retry logic and standardized error mapping.

    Retries at most 2 times on connection errors, timeouts, and HTTP 502, 503, 504
    (waiting 1 second, then 3 seconds). Honours the Retry-After header on HTTP 429
    for delays up to 30 seconds once.

    Args:
        session: Active requests.Session.
        method: HTTP method ('GET' or 'POST').
        url: Destination URL.
        timeout: Request timeout in seconds.
        provider: Provider identifier ('ollama' or 'openai').
        base_url: Configured base URL for reporting connection errors.
        api_key: Optional API key to sanitize from error reports.
        headers: Optional HTTP headers dict.
        json_payload: Optional JSON body.
        allow_404: Whether 404 responses should be returned instead of raised.

    Returns:
        The requests.Response received.

    Raises:
        LLMError: If the request fails, times out, is rate-limited, or fails auth.
    """
    retried_429 = False
    attempts_total = len(RETRY_DELAYS) + 1

    for attempt in range(attempts_total):
        try:
            resp = session.request(
                method=method,
                url=url,
                headers=headers,
                json=json_payload,
                timeout=timeout,
            )

            # HTTP 429 Rate Limit Handling
            if resp.status_code == 429:
                if not retried_429:
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                    if retry_after is not None and 0.0 <= retry_after <= 30.0:
                        retried_429 = True
                        time.sleep(retry_after)
                        continue
                raise LLMError(
                    "Rate limit exceeded (HTTP 429).",
                    hint="Wait a few moments before trying again, or check your rate limits.",
                )

            # Retry on 502, 503, 504
            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
                continue

            return resp

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
                continue

            if isinstance(exc, requests.exceptions.Timeout):
                raise LLMError(
                    f"Request to LLM timed out after {timeout} seconds.",
                    hint="Use a smaller model or increase LLM_TIMEOUT_SECONDS.",
                ) from exc

            # Connection error
            if provider == "ollama":
                raise LLMError(
                    f"Cannot reach Ollama at {base_url}.",
                    hint="Start it with: ollama serve (or open the Ollama app).",
                ) from exc

            raise LLMError(
                f"Cannot reach LLM server at {base_url}.",
                hint="Check that the server is running and the URL is correct.",
            ) from exc

    raise LLMError(f"Request failed after {attempts_total} attempts.")


def extract_json(raw: str) -> str:
    """Extract the first balanced JSON object from raw model output.

    Removes leading byte order marks and whitespace. If a markdown code fence
    is present, extracts the content of the first fenced block. Scans forward
    using a brace-counting state machine to return the substring of the first
    balanced top-level object without parsing the JSON.

    Args:
        raw: The raw text string produced by the LLM.

    Returns:
        The extracted JSON object string.

    Raises:
        LLMError: If no complete balanced JSON object is found or output was truncated.
    """
    text = raw.lstrip("\ufeff").strip()

    # If the text contains a markdown code fence, take the content of the first fenced block
    fence_start = text.find("```")
    if fence_start != -1:
        rest = text[fence_start + 3 :]
        newline_idx = rest.find("\n")
        if newline_idx != -1:
            after_header = rest[newline_idx + 1 :]
            fence_end = after_header.find("```")
            if fence_end != -1:
                text = after_header[:fence_end]
            else:
                text = after_header
        else:
            fence_end = rest.find("```")
            if fence_end != -1:
                inside = rest[:fence_end]
                if inside.lower().startswith("json"):
                    inside = inside[4:]
                text = inside
            else:
                text = rest

    start_idx_brace = text.find("{")
    start_idx_bracket = text.find("[")

    if start_idx_brace == -1 and start_idx_bracket == -1:
        tail = raw[-200:]
        raise LLMError(
            f"The model output was cut off or is not JSON. Output tail: {tail}",
            hint="Increase LLM_NUM_CTX, or use a larger model.",
        )

    if start_idx_bracket != -1 and (start_idx_brace == -1 or start_idx_bracket < start_idx_brace):
        start_idx = start_idx_bracket
        open_char, close_char = "[", "]"
    else:
        start_idx = start_idx_brace
        open_char, close_char = "{", "}"

    depth = 0
    in_string = False
    escape_next = False
    end_idx = -1

    for idx in range(start_idx, len(text)):
        char = text[idx]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            if in_string:
                escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    end_idx = idx + 1
                    break

    if depth != 0 or end_idx == -1:
        tail = raw[-200:]
        raise LLMError(
            f"The model output was cut off or is not JSON. Output tail: {tail}",
            hint="Increase LLM_NUM_CTX, or use a larger model.",
        )

    return text[start_idx:end_idx]


class OllamaClient:
    """LLM client for locally hosted Ollama instances."""

    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            settings: Application settings. If None, loaded via get_settings().
            session: Optional requests.Session instance for dependency injection.
        """
        self.settings = settings if settings is not None else get_settings()
        self.base_url = self.settings.llm_base_url.rstrip("/")
        self.model = self.settings.llm_model
        self.timeout_seconds = self.settings.llm_timeout_seconds
        self.temperature = self.settings.llm_temperature
        self.num_ctx = self.settings.llm_num_ctx
        self.session = session if session is not None else requests.Session()
        self.provider = "ollama"

    def _fetch_installed_models(self) -> list[str]:
        """Query Ollama /api/tags to retrieve installed model names.

        Returns:
            List of model names installed in Ollama.

        Raises:
            LLMError: If Ollama cannot be reached or returns an error.
        """
        url = f"{self.base_url}/api/tags"
        resp = _execute_request_with_retries(
            session=self.session,
            method="GET",
            url=url,
            timeout=self.timeout_seconds,
            provider="ollama",
            base_url=self.base_url,
        )

        if resp.status_code != 200:
            raise LLMError(
                f"LLM request failed with status {resp.status_code}: {resp.text[:300]}",
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise LLMError(
                "Invalid JSON received from Ollama /api/tags.",
            ) from exc

        models_data = data.get("models", [])
        return [str(m["name"]) for m in models_data if isinstance(m, dict) and "name" in m]

    def _model_not_installed_error(self, installed: list[str] | None = None) -> LLMError:
        """Construct an LLMError for a missing Ollama model including installed models.

        Args:
            installed: Optional pre-fetched list of installed model names.

        Returns:
            An LLMError with pull hint and listing of installed models.
        """
        if installed is None:
            try:
                installed = self._fetch_installed_models()
            except (LLMError, requests.RequestException):
                installed = []

        installed_str = ", ".join(installed) if installed else "(none)"
        return LLMError(
            f"Model {self.model} is not installed.",
            hint=f"Run: ollama pull {self.model}. Installed models: {installed_str}.",
        )

    def check_ready(self) -> None:
        """Verify that Ollama is reachable and the configured model is installed.

        Raises:
            LLMError: If Ollama is unreachable or the model is not installed.
        """
        installed = self._fetch_installed_models()
        if not any(_matches_model(self.model, m) for m in installed):
            raise self._model_not_installed_error(installed=installed)

    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        """Send a chat request to Ollama and return the message content.

        Args:
            system: System prompt instructions.
            user: User prompt text.
            temperature: Optional per-call temperature override.

        Returns:
            The raw message content string from the model.

        Raises:
            LLMError: If the request fails, times out, or returns an empty response.
        """
        # Ollama's default context window is small and silently truncates long prompts, which would cut off the README.
        effective_temp = temperature if temperature is not None else self.temperature
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": effective_temp,
                "num_ctx": self.num_ctx,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        url = f"{self.base_url}/api/chat"
        start_time = time.monotonic()

        resp = _execute_request_with_retries(
            session=self.session,
            method="POST",
            url=url,
            timeout=self.timeout_seconds,
            provider="ollama",
            base_url=self.base_url,
            json_payload=payload,
        )

        if resp.status_code == 404:
            raise self._model_not_installed_error()

        if resp.status_code != 200:
            if "model" in resp.text.lower() and "not found" in resp.text.lower():
                raise self._model_not_installed_error()
            raise LLMError(f"LLM request failed with status {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
        except Exception as exc:
            raise LLMError("Invalid JSON received from Ollama response.") from exc

        if "error" in data:
            err_msg = str(data["error"])
            if "not found" in err_msg.lower() and "model" in err_msg.lower():
                raise self._model_not_installed_error()
            raise LLMError(f"Ollama returned error: {err_msg}")

        message_obj = data.get("message")
        content = message_obj.get("content", "") if isinstance(message_obj, dict) else ""

        if not content or not content.strip():
            raise LLMError(
                "The model returned an empty response.",
                hint="Try again or use a different model.",
            )

        elapsed = time.monotonic() - start_time
        prompt_chars = len(system) + len(user)
        resp_chars = len(content)

        logger.info(
            "LLM %s (%s) completed: %d prompt chars, %d response chars in %.2fs",
            self.provider,
            self.model,
            prompt_chars,
            resp_chars,
            elapsed,
        )
        logger.debug("LLM prompt [system]: %s", system)
        logger.debug("LLM prompt [user]: %s", user)
        logger.debug("LLM response: %s", content)

        return content


class OpenAICompatibleClient:
    """LLM client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the OpenAI-compatible client.

        Args:
            settings: Application settings. If None, loaded via get_settings().
            session: Optional requests.Session instance for dependency injection.
        """
        self.settings = settings if settings is not None else get_settings()
        self.base_url = self.settings.llm_base_url.rstrip("/")
        self.model = self.settings.llm_model
        self.api_key = self.settings.llm_api_key
        self.timeout_seconds = self.settings.llm_timeout_seconds
        self.temperature = self.settings.llm_temperature
        self.session = session if session is not None else requests.Session()
        self.provider = "openai"

    def check_ready(self) -> None:
        """Verify that the OpenAI-compatible endpoint is reachable and properly configured.

        Raises:
            LLMError: If no API key is provided for a remote host or the server fails.
        """
        if not self.api_key and not _is_local_address(self.base_url):
            raise LLMError(
                "API key is required for remote OpenAI-compatible endpoint.",
                hint="Set LLM_API_KEY.",
            )

        models_url = _build_endpoint_url(self.base_url, "models")
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = _execute_request_with_retries(
            session=self.session,
            method="GET",
            url=models_url,
            timeout=self.timeout_seconds,
            provider="openai",
            base_url=self.base_url,
            api_key=self.api_key,
            headers=headers,
            allow_404=True,
        )

        # 404 is tolerated since some servers lack the /v1/models endpoint
        if resp.status_code == 404 or 200 <= resp.status_code < 300:
            return

        if resp.status_code in (401, 403):
            raise LLMError(
                f"Authentication failed (HTTP {resp.status_code}).",
                hint="Check that LLM_API_KEY is valid.",
            )

        sanitized_tail = _sanitize_text(resp.text[:300], self.api_key)
        raise LLMError(
            f"LLM request failed with status {resp.status_code}: {sanitized_tail}",
        )

    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        """Send a chat completion request and return the assistant response content.

        Args:
            system: System prompt instructions.
            user: User prompt text.
            temperature: Optional per-call temperature override.

        Returns:
            The raw message content string from the model.

        Raises:
            LLMError: If the request fails, times out, or returns an empty response.
        """
        effective_temp = temperature if temperature is not None else self.temperature
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": effective_temp,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }

        url = _build_endpoint_url(self.base_url, "chat/completions")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start_time = time.monotonic()

        resp = _execute_request_with_retries(
            session=self.session,
            method="POST",
            url=url,
            timeout=self.timeout_seconds,
            provider="openai",
            base_url=self.base_url,
            api_key=self.api_key,
            headers=headers,
            json_payload=payload,
        )

        # Fallback if server rejects response_format
        if resp.status_code == 400 and (
            "response_format" in resp.text.lower() or "json_validate_failed" in resp.text.lower()
        ):
            logger.info("Server does not support response_format; retrying without it.")
            payload_no_rf = dict(payload)
            payload_no_rf.pop("response_format", None)
            resp = _execute_request_with_retries(
                session=self.session,
                method="POST",
                url=url,
                timeout=self.timeout_seconds,
                provider="openai",
                base_url=self.base_url,
                api_key=self.api_key,
                headers=headers,
                json_payload=payload_no_rf,
            )

        if resp.status_code in (401, 403):
            raise LLMError(
                f"Authentication failed (HTTP {resp.status_code}).",
                hint="Check that LLM_API_KEY is valid.",
            )

        if resp.status_code != 200:
            sanitized_tail = _sanitize_text(resp.text[:300], self.api_key)
            raise LLMError(
                f"LLM request failed with status {resp.status_code}: {sanitized_tail}",
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise LLMError("Invalid JSON received from OpenAI-compatible response.") from exc

        choices = data.get("choices", [])
        if not choices or not isinstance(choices, list):
            raise LLMError(
                "The model returned an empty response.",
                hint="Try again or use a different model.",
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LLMError(
                "The model returned an empty response.",
                hint="Try again or use a different model.",
            )

        message_obj = first_choice.get("message")
        content = message_obj.get("content", "") if isinstance(message_obj, dict) else ""

        if not content or not content.strip():
            raise LLMError(
                "The model returned an empty response.",
                hint="Try again or use a different model.",
            )

        elapsed = time.monotonic() - start_time
        prompt_chars = len(system) + len(user)
        resp_chars = len(content)

        logger.info(
            "LLM %s (%s) completed: %d prompt chars, %d response chars in %.2fs",
            self.provider,
            self.model,
            prompt_chars,
            resp_chars,
            elapsed,
        )
        logger.debug("LLM prompt [system]: %s", system)
        logger.debug("LLM prompt [user]: %s", user)
        logger.debug("LLM response: %s", content)

        return content


def get_client(settings: Settings | None = None) -> LLMClient:
    """Instantiate and return the configured LLM client.

    Args:
        settings: Application settings. If None, loaded via get_settings().

    Returns:
        An instance of OllamaClient or OpenAICompatibleClient.

    Raises:
        ConfigError: If the configured llm_provider is not 'ollama' or 'openai'.
    """
    if settings is None:
        settings = get_settings()

    provider = settings.llm_provider.lower().strip()
    if provider == "ollama":
        return OllamaClient(settings=settings)
    if provider == "openai":
        return OpenAICompatibleClient(settings=settings)

    raise ConfigError(
        f"Unknown LLM provider: {settings.llm_provider!r}.",
        hint="Set LLM_PROVIDER to one of: ollama, openai.",
    )


def main() -> None:
    """CLI entrypoint for testing LLM readiness and generation."""
    try:
        client = get_client()
        client.check_ready()
        response_text = client.chat(
            system="You are a helpful assistant that outputs only valid JSON.",
            user="Reply with a JSON object with one key named ok set to true",
        )
        extracted = extract_json(response_text)
        print(extracted)
    except Repo2VideoError as err:
        sys.stderr.write(f"{err}\n")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Unexpected error: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
