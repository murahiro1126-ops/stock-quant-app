import sys
import types

import httpx
import pytest

import postprime_generator


def test_request_structured_post_reports_gemini_timeout(monkeypatch):
    observed_timeout = {}
    calls = {"count": 0}

    class FakeAPIError(Exception):
        pass

    class FakeClient:
        def __init__(self, *, api_key, http_options):
            observed_timeout["milliseconds"] = http_options.timeout
            observed_timeout["sdk_retry_attempts"] = (
                http_options.retry_options.attempts
            )
            self.models = self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def generate_content(self, **_kwargs):
            calls["count"] += 1
            raise httpx.ReadTimeout("request timed out")

    fake_types = types.SimpleNamespace(
        HttpOptions=lambda **kwargs: types.SimpleNamespace(**kwargs),
        HttpRetryOptions=lambda *, attempts: types.SimpleNamespace(
            attempts=attempts
        ),
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
    monkeypatch.setattr("gemini_retry.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="タイムアウト"):
        postprime_generator._request_structured_post(
            api_key="test-key",
            model_type=postprime_generator.MacroSectorAnalysisPost,
            user_data={},
            memo="",
        )

    assert observed_timeout["milliseconds"] == 60_000
    assert observed_timeout["sdk_retry_attempts"] == 1
    assert calls["count"] == 3


def test_request_structured_post_retries_503_then_returns_post(monkeypatch):
    calls = {"count": 0}

    class FakeAPIError(Exception):
        def __init__(self, code):
            super().__init__(f"HTTP {code}")
            self.code = code

    class FakeClient:
        def __init__(self, **_kwargs):
            self.models = self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def generate_content(self, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise FakeAPIError(503)
            return types.SimpleNamespace(
                text=(
                    '{"market_condition":"うねうね・中立",'
                    '"topix_change":"+0.1%",'
                    '"title_headline":"市場を確認",'
                    '"title_target_word":"注目",'
                    '"greeting_summary":"みんなおつかれ！",'
                    '"one_month_leader":"電気機器",'
                    '"reversal_sectors":[],"acceleration_sectors":[],"dark_horses":[],'
                    '"gal_commentary":"データを確認しよう。"}'
                )
            )

    fake_types = types.SimpleNamespace(
        HttpOptions=lambda **kwargs: types.SimpleNamespace(**kwargs),
        HttpRetryOptions=lambda **kwargs: types.SimpleNamespace(**kwargs),
        GenerateContentConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.errors = types.SimpleNamespace(APIError=FakeAPIError)
    fake_genai.types = fake_types
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setattr("gemini_retry.time.sleep", lambda _seconds: None)

    result = postprime_generator._request_structured_post(
        api_key="test-key",
        model_type=postprime_generator.MacroSectorAnalysisPost,
        user_data={},
        memo="",
    )

    assert result.title_headline == "市場を確認"
    assert calls["count"] == 2
