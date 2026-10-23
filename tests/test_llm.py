"""Tests for the LLM client layer (Ollama, OpenAI-compatible, extract_json, and HTTP handling)."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
import requests

from repo2video.config import Settings
from repo2video.errors import ConfigError, LLMError
from repo2video.llm import (
    OllamaClient,
    OpenAICompatibleClient,
    _build_endpoint_url,
    _is_local_address,
    _matches_model,
    _parse_retry_after,
    _sanitize_text,
    extract_json,
    get_client,
)


class FakeResponse:
    """Fake requests.Response for offline unit testing."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text if text else (json.dumps(json_data) if json_data is not None else "")
        self.headers = headers or {}

    def json(self) -> Any:
        if self._json_data is not None:
            return self._json_data
        if self.text:
            return json.loads(self.text)
        raise ValueError("No JSON data in FakeResponse")


class FakeSession:
    """Fake requests.Session to inject prepared responses and record calls."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses: list[Any] = list(responses) if responses is not None else []
        self.calls: list[dict[str, Any]] = []

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> FakeResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
                "kwargs": kwargs,
            }
        )
        if not self.responses:
            return FakeResponse(status_code=200, json_data={"message": "ok"})
        if len(self.responses) > 1:
            item = self.responses.pop(0)
        else:
            item = self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request("POST", url, **kwargs)


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Intercept time.sleep to avoid waiting during retries while tracking delays."""
    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("time.sleep", fake_sleep)
    return slept


# ==============================================================================
# 1. extract_json tests
# ==============================================================================


class TestExtractJson:
    """Tests for the extract_json function."""

    def test_plain_json(self) -> None:
        raw = '{"ok": true, "count": 42}'
        assert extract_json(raw) == '{"ok": true, "count": 42}'

    def test_fenced_json_with_lang(self) -> None:
        raw = '```json\n{"ok": true}\n```'
        assert extract_json(raw) == '{"ok": true}'

    def test_fenced_json_without_lang(self) -> None:
        raw = '```\n{"ok": true}\n```'
        assert extract_json(raw) == '{"ok": true}'

    def test_json_with_text_before_and_after(self) -> None:
        raw = 'Sure! Here is the JSON configuration:\n{"key": "value", "num": 1}\nHope this helps!'
        assert extract_json(raw) == '{"key": "value", "num": 1}'

    def test_fenced_json_with_text_before_and_after(self) -> None:
        raw = 'Introductory commentary.\n```json\n{"hello": "world"}\n```\nConcluding remarks.'
        assert extract_json(raw) == '{"hello": "world"}'

    def test_braces_inside_strings(self) -> None:
        raw = '{"template": "Hello {name}, your code is {code}", "active": true}'
        assert extract_json(raw) == raw

    def test_escaped_quotes_inside_strings(self) -> None:
        raw = r'{"dialogue": "He shouted: \"Stop!\", and paused.", "valid": true}'
        assert extract_json(raw) == raw

    def test_escaped_backslashes(self) -> None:
        raw = r'{"filepath": "C:\\Program Files\\App", "ok": true}'
        assert extract_json(raw) == raw

    def test_nested_objects_and_arrays(self) -> None:
        raw = '{"scenes": [{"title": "Scene 1", "data": {"duration": 10}}], "total": 1}'
        assert extract_json(raw) == raw

    def test_leading_bom_and_surrounding_whitespace(self) -> None:
        raw = '\ufeff  \n  {"ok": true}  \t\n'
        assert extract_json(raw) == '{"ok": true}'

    def test_truncated_unbalanced_braces_raises_llmerror(self) -> None:
        raw = '{"scenes": [{"title": "Intro"'
        with pytest.raises(LLMError) as exc_info:
            extract_json(raw)
        err = exc_info.value
        assert "The model output was cut off or is not JSON." in err.message
        assert err.hint == "Increase LLM_NUM_CTX, or use a larger model."
        assert '{"scenes": [{"title": "Intro"' in err.message

    def test_no_json_braces_raises_llmerror(self) -> None:
        raw = "I am sorry, but as an AI language model I cannot fulfill this request."
        with pytest.raises(LLMError) as exc_info:
            extract_json(raw)
        err = exc_info.value
        assert "The model output was cut off or is not JSON." in err.message
        assert err.hint == "Increase LLM_NUM_CTX, or use a larger model."
        assert raw in err.message

    def test_long_truncated_output_includes_last_200_chars(self) -> None:
        tail_marker = "XYZ_END_OF_OUTPUT_MARKER_12345"
        raw = "{" + ("a" * 300) + tail_marker
        with pytest.raises(LLMError) as exc_info:
            extract_json(raw)
        err = exc_info.value
        assert tail_marker in err.message
        # The tail extracted should be at most 200 characters
        assert len(err.message) >= len(tail_marker)


# ==============================================================================
# 2. OllamaClient tests
# ==============================================================================


class TestOllamaClient:
    """Tests for OllamaClient chat and readiness verification."""

    def test_ollama_chat_payload_and_url(self) -> None:
        session = FakeSession(
            [FakeResponse(status_code=200, json_data={"message": {"content": '{"ok": true}'}})]
        )
        settings = Settings(
            llm_provider="ollama",
            llm_base_url="http://localhost:11434",
            llm_model="qwen2.5:7b",
            llm_temperature=0.4,
            llm_num_ctx=8192,
        )
        client = OllamaClient(settings=settings, session=session)  # type: ignore[arg-type]

        result = client.chat(system="Be concise.", user="Generate video script.")

        assert result == '{"ok": true}'
        assert len(session.calls) == 1
        call = session.calls[0]
        assert call["method"] == "POST"
        assert call["url"] == "http://localhost:11434/api/chat"

        body = call["json"]
        assert body["model"] == "qwen2.5:7b"
        assert body["stream"] is False
        assert body["format"] == "json"
        assert body["options"]["temperature"] == 0.4
        assert body["options"]["num_ctx"] == 8192
        assert body["messages"] == [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Generate video script."},
        ]

    def test_ollama_chat_temperature_override(self) -> None:
        session = FakeSession(
            [FakeResponse(status_code=200, json_data={"message": {"content": "ok"}})]
        )
        settings = Settings(llm_temperature=0.4)
        client = OllamaClient(settings=settings, session=session)  # type: ignore[arg-type]

        client.chat(system="sys", user="usr", temperature=0.9)

        assert session.calls[0]["json"]["options"]["temperature"] == 0.9

    def test_ollama_check_ready_success(self) -> None:
        session = FakeSession(
            [
                FakeResponse(
                    status_code=200,
                    json_data={"models": [{"name": "qwen2.5:7b"}, {"name": "llama3:latest"}]},
                )
            ]
        )
        settings = Settings(llm_model="qwen2.5:7b")
        client = OllamaClient(settings=settings, session=session)  # type: ignore[arg-type]

        # Should complete without error
        client.check_ready()
        assert session.calls[0]["url"] == "http://localhost:11434/api/tags"

    def test_ollama_check_ready_model_variant_matching(self) -> None:
        # Configured model without :latest matches installed model with :latest
        session1 = FakeSession(
            [FakeResponse(status_code=200, json_data={"models": [{"name": "llama3:latest"}]})]
        )
        client1 = OllamaClient(
            settings=Settings(llm_model="llama3"),
            session=session1,  # type: ignore[arg-type]
        )
        client1.check_ready()

        # Configured model with registry prefix matches installed model
        session2 = FakeSession(
            [FakeResponse(status_code=200, json_data={"models": [{"name": "library/qwen2.5:7b"}]})]
        )
        client2 = OllamaClient(
            settings=Settings(llm_model="qwen2.5:7b"),
            session=session2,  # type: ignore[arg-type]
        )
        client2.check_ready()

    def test_ollama_check_ready_missing_model_raises_llmerror(self) -> None:
        session = FakeSession(
            [
                FakeResponse(
                    status_code=200,
                    json_data={"models": [{"name": "llama3:8b"}, {"name": "mistral:latest"}]},
                )
            ]
        )
        settings = Settings(llm_model="qwen2.5:7b")
        client = OllamaClient(settings=settings, session=session)  # type: ignore[arg-type]

        with pytest.raises(LLMError) as exc_info:
            client.check_ready()

        err = exc_info.value
        assert "Model qwen2.5:7b is not installed." in err.message
        assert "Run: ollama pull qwen2.5:7b." in str(err.hint)
        assert "Installed models: llama3:8b, mistral:latest." in str(err.hint)

    def test_ollama_check_ready_connection_refused(self) -> None:
        session = FakeSession([requests.exceptions.ConnectionError("Connection refused")])
        client = OllamaClient(
            settings=Settings(llm_base_url="http://localhost:11434"),
            session=session,  # type: ignore[arg-type]
        )

        with pytest.raises(LLMError) as exc_info:
            client.check_ready()

        err = exc_info.value
        assert err.message == "Cannot reach Ollama at http://localhost:11434."
        assert err.hint == "Start it with: ollama serve (or open the Ollama app)."

    def test_ollama_chat_404_model_not_found(self) -> None:
        # First call is POST /api/chat returning 404
        # Followup call to /api/tags retrieves installed models
        session = FakeSession(
            [
                FakeResponse(status_code=404, text="model not found"),
                FakeResponse(status_code=200, json_data={"models": [{"name": "llama3:8b"}]}),
            ]
        )
        client = OllamaClient(
            settings=Settings(llm_model="qwen2.5:7b"),
            session=session,  # type: ignore[arg-type]
        )

        with pytest.raises(LLMError) as exc_info:
            client.chat(system="sys", user="usr")

        err = exc_info.value
        assert "Model qwen2.5:7b is not installed." in err.message
        assert "Run: ollama pull qwen2.5:7b. Installed models: llama3:8b." in str(err.hint)

    def test_ollama_chat_empty_response(self) -> None:
        session = FakeSession(
            [FakeResponse(status_code=200, json_data={"message": {"content": "   "}})]
        )
        client = OllamaClient(session=session)  # type: ignore[arg-type]

        with pytest.raises(LLMError) as exc_info:
            client.chat(system="sys", user="usr")

        err = exc_info.value
        assert err.message == "The model returned an empty response."
        assert err.hint == "Try again or use a different model."


# ==============================================================================
# 3. OpenAICompatibleClient tests
# ==============================================================================


class TestOpenAICompatibleClient:
    """Tests for OpenAICompatibleClient chat and readiness verification."""

    def test_no_v1_duplication(self) -> None:
        urls = [
            ("https://api.openai.com", "https://api.openai.com/v1/chat/completions"),
            ("https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"),
            ("https://api.openai.com/v1/", "https://api.openai.com/v1/chat/completions"),
            ("http://localhost:11434", "http://localhost:11434/v1/chat/completions"),
            ("http://localhost:11434/v1", "http://localhost:11434/v1/chat/completions"),
        ]

        for base_url, expected_chat_url in urls:
            session = FakeSession(
                [
                    FakeResponse(
                        status_code=200,
                        json_data={"choices": [{"message": {"content": '{"ok": true}'}}]},
                    )
                ]
            )
            settings = Settings(
                llm_provider="openai",
                llm_base_url=base_url,
                llm_api_key="test-key",
            )
            client = OpenAICompatibleClient(settings=settings, session=session)  # type: ignore[arg-type]
            client.chat(system="sys", user="usr")

            assert session.calls[0]["url"] == expected_chat_url

    def test_auth_header_sent_only_when_api_key_set(self) -> None:
        # Case 1: Key set
        session1 = FakeSession(
            [
                FakeResponse(
                    status_code=200,
                    json_data={"choices": [{"message": {"content": "ok"}}]},
                )
            ]
        )
        client1 = OpenAICompatibleClient(
            settings=Settings(llm_api_key="secret-123"),
            session=session1,  # type: ignore[arg-type]
        )
        client1.chat(system="sys", user="usr")
        assert session1.calls[0]["headers"]["Authorization"] == "Bearer secret-123"

        # Case 2: Key None
        session2 = FakeSession(
            [
                FakeResponse(
                    status_code=200,
                    json_data={"choices": [{"message": {"content": "ok"}}]},
                )
            ]
        )
        client2 = OpenAICompatibleClient(
            settings=Settings(llm_api_key=None),
            session=session2,  # type: ignore[arg-type]
        )
        client2.chat(system="sys", user="usr")
        assert "Authorization" not in session2.calls[0]["headers"]

    def test_response_format_fallback_after_400(self, caplog: pytest.LogCaptureFixture) -> None:
        session = FakeSession(
            [
                FakeResponse(
                    status_code=400,
                    text='{"error": {"message": "Unsupported parameter: response_format"}}',
                ),
                FakeResponse(
                    status_code=200,
                    json_data={"choices": [{"message": {"content": '{"ok": true}'}}]},
                ),
            ]
        )
        client = OpenAICompatibleClient(session=session)  # type: ignore[arg-type]

        with caplog.at_level(logging.INFO):
            result = client.chat(system="sys", user="usr")

        assert result == '{"ok": true}'
        assert len(session.calls) == 2
        # First call included response_format
        assert "response_format" in session.calls[0]["json"]
        # Second call omitted response_format
        assert "response_format" not in session.calls[1]["json"]
        assert "Server does not support response_format; retrying without it." in caplog.text

    def test_check_ready_remote_missing_api_key_raises_llmerror(self) -> None:
        client = OpenAICompatibleClient(
            settings=Settings(
                llm_provider="openai",
                llm_base_url="https://api.openai.com/v1",
                llm_api_key=None,
            )
        )

        with pytest.raises(LLMError) as exc_info:
            client.check_ready()

        err = exc_info.value
        assert err.hint == "Set LLM_API_KEY."

    def test_check_ready_local_missing_api_key_allowed(self) -> None:
        session = FakeSession([FakeResponse(status_code=200, json_data={"data": []})])
        client = OpenAICompatibleClient(
            settings=Settings(
                llm_provider="openai",
                llm_base_url="http://localhost:8000/v1",
                llm_api_key=None,
            ),
            session=session,  # type: ignore[arg-type]
        )

        # Local endpoint without API key should succeed
        client.check_ready()
        assert session.calls[0]["url"] == "http://localhost:8000/v1/models"

    def test_check_ready_tolerates_404_on_models(self) -> None:
        session = FakeSession([FakeResponse(status_code=404, text="Endpoint Not Found")])
        client = OpenAICompatibleClient(
            settings=Settings(
                llm_provider="openai",
                llm_base_url="https://api.custom-server.com/v1",
                llm_api_key="custom-key",
            ),
            session=session,  # type: ignore[arg-type]
        )

        # 404 on /v1/models must be tolerated
        client.check_ready()

    def test_openai_chat_empty_response(self) -> None:
        session = FakeSession(
            [
                FakeResponse(
                    status_code=200,
                    json_data={"choices": [{"message": {"content": ""}}]},
                )
            ]
        )
        client = OpenAICompatibleClient(session=session)  # type: ignore[arg-type]

        with pytest.raises(LLMError) as exc_info:
            client.chat(system="sys", user="usr")

        err = exc_info.value
        assert err.message == "The model returned an empty response."
        assert err.hint == "Try again or use a different model."


# ==============================================================================
# 4. Shared HTTP behavior tests (Retries, 429, 401/403, Timeouts, Logging, Secrets)
# ==============================================================================


class TestSharedHttpBehavior:
    """Tests for retries, status code mapping, rate limits, and secret masking."""

    def test_retries_on_503_success(self, fast_sleep: list[float]) -> None:
        session = FakeSession(
            [
                FakeResponse(status_code=503, text="Service Unavailable"),
                FakeResponse(status_code=503, text="Service Unavailable"),
                FakeResponse(
                    status_code=200,
                    json_data={"message": {"content": '{"ok": true}'}},
                ),
            ]
        )
        client = OllamaClient(session=session)  # type: ignore[arg-type]

        result = client.chat(system="sys", user="usr")

        assert result == '{"ok": true}'
        assert len(session.calls) == 3
        # Verified delays: 1 second then 3 seconds
        assert fast_sleep == [1.0, 3.0]

    def test_retries_on_503_exhausted(self, fast_sleep: list[float]) -> None:
        session = FakeSession(
            [
                FakeResponse(status_code=503, text="Service Unavailable"),
                FakeResponse(status_code=503, text="Service Unavailable"),
                FakeResponse(status_code=503, text="Service Unavailable"),
            ]
        )
        client = OllamaClient(session=session)  # type: ignore[arg-type]

        with pytest.raises(LLMError) as exc_info:
            client.chat(system="sys", user="usr")

        assert "503" in exc_info.value.message
        assert len(session.calls) == 3
        assert fast_sleep == [1.0, 3.0]

    def test_retry_on_429_within_30s(self, fast_sleep: list[float]) -> None:
        session = FakeSession(
            [
                FakeResponse(
                    status_code=429,
                    headers={"Retry-After": "4"},
                    text="Too Many Requests",
                ),
                FakeResponse(
                    status_code=200,
                    json_data={"choices": [{"message": {"content": "ok"}}]},
                ),
            ]
        )
        client = OpenAICompatibleClient(session=session)  # type: ignore[arg-type]

        result = client.chat(system="sys", user="usr")

        assert result == "ok"
        assert len(session.calls) == 2
        assert 4.0 in fast_sleep

    def test_retry_on_429_exceeding_30s_raises_llmerror(self, fast_sleep: list[float]) -> None:
        session = FakeSession(
            [
                FakeResponse(
                    status_code=429,
                    headers={"Retry-After": "60"},
                    text="Rate limited",
                )
            ]
        )
        client = OpenAICompatibleClient(session=session)  # type: ignore[arg-type]

        with pytest.raises(LLMError) as exc_info:
            client.chat(system="sys", user="usr")

        err = exc_info.value
        assert err.message == "Rate limit exceeded (HTTP 429)."
        assert "Wait a few moments" in str(err.hint)
        # Should not sleep if Retry-After > 30
        assert 60.0 not in fast_sleep

    def test_retry_on_429_still_fails_raises_llmerror(self, fast_sleep: list[float]) -> None:
        session = FakeSession(
            [
                FakeResponse(
                    status_code=429,
                    headers={"Retry-After": "5"},
                    text="Rate limited",
                ),
                FakeResponse(
                    status_code=429,
                    headers={"Retry-After": "5"},
                    text="Still rate limited",
                ),
            ]
        )
        client = OpenAICompatibleClient(session=session)  # type: ignore[arg-type]

        with pytest.raises(LLMError) as exc_info:
            client.chat(system="sys", user="usr")

        assert exc_info.value.message == "Rate limit exceeded (HTTP 429)."

    def test_auth_errors_401_and_403(self) -> None:
        for code in (401, 403):
            session = FakeSession([FakeResponse(status_code=code, text="Unauthorized")])
            client = OpenAICompatibleClient(
                settings=Settings(llm_api_key="bad-key"),
                session=session,  # type: ignore[arg-type]
            )

            with pytest.raises(LLMError) as exc_info:
                client.chat(system="sys", user="usr")

            err = exc_info.value
            assert str(code) in err.message
            assert "Check that LLM_API_KEY is valid." in str(err.hint)

    def test_timeout_raises_llmerror_with_hint(self) -> None:
        session = FakeSession(
            [
                requests.exceptions.Timeout("Read timed out"),
                requests.exceptions.Timeout("Read timed out"),
                requests.exceptions.Timeout("Read timed out"),
            ]
        )
        client = OllamaClient(
            settings=Settings(llm_timeout_seconds=45),
            session=session,  # type: ignore[arg-type]
        )

        with pytest.raises(LLMError) as exc_info:
            client.chat(system="sys", user="usr")

        err = exc_info.value
        assert "Request to LLM timed out after 45 seconds." in err.message
        assert err.hint == "Use a smaller model or increase LLM_TIMEOUT_SECONDS."

    def test_api_keys_never_appear_in_error_text_or_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        sensitive_key = "super-secret-production-token-998877"
        server_error_text = f"Internal crash report containing key={sensitive_key} in stack trace"
        session = FakeSession([FakeResponse(status_code=500, text=server_error_text)])
        settings = Settings(
            llm_provider="openai",
            llm_api_key=sensitive_key,
        )
        client = OpenAICompatibleClient(settings=settings, session=session)  # type: ignore[arg-type]

        with caplog.at_level(logging.DEBUG), pytest.raises(LLMError) as exc_info:
            client.chat(system="sys", user="usr")

        err = exc_info.value
        assert sensitive_key not in str(err)
        assert sensitive_key not in err.message
        assert sensitive_key not in (err.hint or "")
        assert sensitive_key not in caplog.text
        assert "***" in err.message

    def test_logging_info_and_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        session = FakeSession(
            [
                FakeResponse(
                    status_code=200,
                    json_data={"message": {"content": '{"ok": true}'}},
                )
            ]
        )
        client = OllamaClient(session=session)  # type: ignore[arg-type]

        with caplog.at_level(logging.DEBUG):
            client.chat(system="System instructions", user="User message")

        # INFO level checks
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) == 1
        info_msg = info_records[0].getMessage()
        assert "LLM ollama (qwen2.5:7b) completed:" in info_msg
        assert "prompt chars" in info_msg
        assert "response chars" in info_msg

        # DEBUG level checks
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        debug_msgs = [r.getMessage() for r in debug_records]
        assert any("System instructions" in m for m in debug_msgs)
        assert any("User message" in m for m in debug_msgs)
        assert any('{"ok": true}' in m for m in debug_msgs)


# ==============================================================================
# 5. get_client tests
# ==============================================================================


class TestGetClient:
    """Tests for get_client factory function."""

    def test_get_client_ollama(self) -> None:
        settings = Settings(llm_provider="ollama")
        client = get_client(settings=settings)
        assert isinstance(client, OllamaClient)

    def test_get_client_openai(self) -> None:
        settings = Settings(llm_provider="openai")
        client = get_client(settings=settings)
        assert isinstance(client, OpenAICompatibleClient)

    def test_get_client_unknown_raises_config_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Construct via object.__setattr__ to bypass Settings frozen post_init validation
        bad_settings = Settings()
        object.__setattr__(bad_settings, "llm_provider", "anthropic")

        with pytest.raises(ConfigError) as exc_info:
            get_client(settings=bad_settings)

        err = exc_info.value
        assert "Unknown LLM provider: 'anthropic'." in err.message
        assert err.hint == "Set LLM_PROVIDER to one of: ollama, openai."


# ==============================================================================
# 6. Helper functions unit tests
# ==============================================================================


class TestLLMHelpers:
    """Tests for module helper utilities."""

    def test_build_endpoint_url_various(self) -> None:
        assert (
            _build_endpoint_url("http://localhost:11434", "chat/completions")
            == "http://localhost:11434/v1/chat/completions"
        )
        assert (
            _build_endpoint_url("http://localhost:11434/v1", "chat/completions")
            == "http://localhost:11434/v1/chat/completions"
        )
        assert (
            _build_endpoint_url("http://localhost:11434/v1/", "/chat/completions")
            == "http://localhost:11434/v1/chat/completions"
        )
        assert (
            _build_endpoint_url("https://api.openai.com/v1", "models")
            == "https://api.openai.com/v1/models"
        )

    def test_is_local_address(self) -> None:
        assert _is_local_address("http://localhost:11434") is True
        assert _is_local_address("http://127.0.0.1:8000/v1") is True
        assert _is_local_address("https://api.openai.com/v1") is False
        assert _is_local_address("https://remote-llm.example.com") is False

    def test_matches_model(self) -> None:
        assert _matches_model("qwen2.5:7b", "qwen2.5:7b") is True
        assert _matches_model("llama3", "llama3:latest") is True
        assert _matches_model("llama3:latest", "llama3") is True
        assert _matches_model("qwen2.5:7b", "library/qwen2.5:7b") is True
        assert _matches_model("qwen2.5:7b", "qwen2.5:14b") is False

    def test_sanitize_text(self) -> None:
        assert _sanitize_text("safe text", None) == "safe text"
        assert _sanitize_text("key: secret123", "secret123") == "key: ***"

    def test_parse_retry_after(self) -> None:
        assert _parse_retry_after("5") == 5.0
        assert _parse_retry_after("12.5") == 12.5
        assert _parse_retry_after("-1") is None
        assert _parse_retry_after("invalid") is None
        assert _parse_retry_after(None) is None


# ==============================================================================
# 7. CLI __main__ integration tests
# ==============================================================================


class TestMainExecution:
    """Tests for the python -m repo2video.llm entrypoint."""

    def test_main_success(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = FakeSession(
            [
                # check_ready -> /api/tags
                FakeResponse(status_code=200, json_data={"models": [{"name": "qwen2.5:7b"}]}),
                # chat -> /api/chat
                FakeResponse(
                    status_code=200,
                    json_data={"message": {"content": '{"ok": true}'}},
                ),
            ]
        )
        fake_client = OllamaClient(
            settings=Settings(llm_model="qwen2.5:7b"),
            session=session,  # type: ignore[arg-type]
        )
        monkeypatch.setattr("repo2video.llm.get_client", lambda: fake_client)

        from repo2video.llm import main

        main()

        captured = capsys.readouterr()
        assert '{"ok": true}' in captured.out

    def test_main_error_exits_with_code_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = FakeSession([requests.exceptions.ConnectionError("Connection refused")])
        fake_client = OllamaClient(
            settings=Settings(llm_base_url="http://localhost:11434"),
            session=session,  # type: ignore[arg-type]
        )
        monkeypatch.setattr("repo2video.llm.get_client", lambda: fake_client)

        from repo2video.llm import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Cannot reach Ollama at http://localhost:11434." in captured.err
        assert "Start it with: ollama serve (or open the Ollama app)." in captured.err
