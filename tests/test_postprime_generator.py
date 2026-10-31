import sys
import types

import httpx
import pytest

import postprime_generator


def test_request_structured_post_reports_gemini_timeout(monkeypatch):
    observed_timeout = {}

    class FakeAPIError(Exception):
        pass

    class FakeClient:
        def __init__(self, *, api_key, http_options):
            observed_timeout["milliseconds"] = http_options.timeout
            self.models = self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def generate_content(self, **_kwargs):
            raise httpx.ReadTimeout("request timed out")

    fake_types = types.SimpleNamespace(
        HttpOptions=lambda *, timeout: types.SimpleNamespace(timeout=timeout),
        GenerateContentConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    fake_errors = types.SimpleNamespace(APIError=FakeAPIError)
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.errors = fake_errors
    fake_genai.types = fake_types
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    with pytest.raises(RuntimeError, match="タイムアウト"):
        postprime_generator._request_structured_post(
            api_key="test-key",
            model_type=postprime_generator.MacroSectorAnalysisPost,
            user_data={},
            memo="",
        )

    assert observed_timeout["milliseconds"] == 60_000
