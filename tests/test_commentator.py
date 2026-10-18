from types import SimpleNamespace

import httpx
from google import genai

import commentator
from commentator import generate_ai_comment, generate_postprime_text, get_gemini_api_key


def _arguments():
    return {
        "ticker": "7203.T",
        "company": "トヨタ自動車",
        "buy_confidence": 80,
        "sell_confidence": 20,
        "rank": "A",
        "win_rate": 68.5,
        "range_status": "中間",
        "is_ranging": False,
        "technical_signals": ["MACDがシグナルを上抜け", "RSIが中立より強い範囲"],
        "range_position": 53.2,
        "adx": 25.0,
        "atr_percent": 2.1,
        "fear_greed_score": 64,
        "fear_greed_label": "🙂 貪欲",
        "gemini_api_key": None,
    }


def test_get_gemini_api_key_prefers_streamlit_secrets(monkeypatch):
    class FakeSecrets:
        def get(self, key):
            assert key == "GEMINI_API_KEY"
            return " streamlit-secret "

    import streamlit as st

    monkeypatch.setattr(st, "secrets", FakeSecrets())
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")

    assert get_gemini_api_key() == "streamlit-secret"


def test_get_gemini_api_key_falls_back_to_environment(monkeypatch):
    class FakeSecrets:
        def get(self, key):
            return None

    import streamlit as st

    monkeypatch.setattr(st, "secrets", FakeSecrets())
    monkeypatch.setenv("GEMINI_API_KEY", " environment-key ")

    assert get_gemini_api_key() == "environment-key"


def test_generate_ai_comment_uses_fallback_without_api_key():
    result = generate_ai_comment(**_arguments())

    assert "トヨタ自動車（7203.T）" in result
    assert "ランクA" in result
    assert "MACDがシグナルを上抜け" in result
    assert "Fear & Greedは「🙂 貪欲」（64点）" in result


def test_generate_ai_comment_uses_gemini(monkeypatch):
    captured = {}

    def fake_gemini(api_key, system_prompt, user_prompt, data):
        captured.update(
            api_key=api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            data=data,
        )
        return "買い条件を慎重に見たいね。"

    monkeypatch.setattr(commentator, "_get_gemini_text", fake_gemini)
    result = generate_ai_comment(
        **{**_arguments(), "gemini_api_key": "test-gemini-key"}
    )

    assert result == "買い条件を慎重に見たいね。"
    assert captured["api_key"] == "test-gemini-key"
    assert "ギャル系クオンツアナリスト" in captured["system_prompt"]
    assert captured["data"]["ticker"] == "7203.T"
    assert captured["data"]["fear_greed_score"] == 64
    assert "Fear & Greed" in captured["user_prompt"]


def test_generate_postprime_text_uses_gemini(monkeypatch):
    captured = {}

    def fake_gemini(api_key, system_prompt, user_prompt, data):
        captured.update(api_key=api_key, data=data)
        return "【トヨタ自動車】相場を確認しつつ慎重にいこ。"

    monkeypatch.setattr(commentator, "_get_gemini_text", fake_gemini)
    result = generate_postprime_text(
        **{
            **_arguments(),
            "buy_reasons": ["MACD上昇"],
            "sell_reasons": ["RSI低下"],
            "gemini_api_key": "test-gemini-key",
        }
    )

    assert result.startswith("【トヨタ自動車】")
    assert captured["data"]["fear_greed_label"] == "🙂 貪欲"
    assert captured["data"]["buy_reasons"] == ["MACD上昇"]


def test_get_gemini_text_uses_sdk_config_and_returns_text(monkeypatch):
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(text="Geminiの解説")

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setattr(
        commentator,
        "_read_streamlit_gemini_secret",
        lambda: "streamlit-secret",
    )

    result = commentator._get_gemini_text(
        "explicit-key",
        "system instruction",
        "user instruction",
        {"ticker": "7203.T"},
    )

    assert result == "Geminiの解説"
    assert captured["client"]["api_key"] == "streamlit-secret"
    assert captured["client"]["http_options"].timeout == 30
    assert captured["request"]["model"] == commentator.MODEL
    assert captured["request"]["config"].system_instruction == "system instruction"


def test_generate_ai_comment_falls_back_on_api_error(monkeypatch):
    monkeypatch.setattr(
        commentator,
        "_get_gemini_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("test quota error")),
    )

    result = generate_ai_comment(
        **{**_arguments(), "gemini_api_key": "test-gemini-key"}
    )

    assert "トヨタ自動車（7203.T）" in result
    assert "ランクA" in result


def test_generate_ai_comment_logs_original_api_error_and_status(monkeypatch, capsys):
    class FakeModels:
        def generate_content(self, **kwargs):
            request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
            response = httpx.Response(
                429,
                request=request,
                text='{"error":"quota exceeded"}',
            )
            raise httpx.HTTPStatusError(
                "test quota error",
                request=request,
                response=response,
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)
    result = generate_ai_comment(
        **{**_arguments(), "gemini_api_key": "test-gemini-key"}
    )

    assert "トヨタ自動車（7203.T）" in result
    assert "ランクA" in result
    output = capsys.readouterr().out
    assert "DEBUG GEMINI ERROR:" in output
    assert "test quota error" in output
    assert "status_code=429" in output
    assert "quota exceeded" in output


def test_generate_postprime_text_falls_back_without_api_key():
    result = generate_postprime_text(
        **{**_arguments(), "buy_reasons": ["買い条件"], "sell_reasons": []}
    )

    assert "トヨタ自動車（7203.T）" in result
    assert "Fear & Greed Index: 🙂 貪欲（64点）" in result
    assert "判定理由: 買い条件" in result
