import json

import requests

from commentator import generate_ai_comment


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
    }


def test_generate_ai_comment_uses_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = generate_ai_comment(**_arguments())

    assert "トヨタ自動車（7203.T）" in result
    assert "ランクA" in result
    assert "MACDがシグナルを上抜け" in result


def test_generate_ai_comment_uses_openai_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "買い条件を慎重に見たいね。"}}]}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    result = generate_ai_comment(**_arguments())

    assert result == "買い条件を慎重に見たいね。"
    assert captured["url"].endswith("/chat/completions")
    assert "ギャル系クオンツアナリスト" in captured["json"]["messages"][0]["content"]
    assert json.loads(captured["json"]["messages"][1]["content"].split("\n", 1)[1])[
        "ticker"
    ] == "7203.T"


def test_generate_ai_comment_uses_anthropic_api(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"content": [{"type": "text", "text": "値位置を見て慎重にいこ。"}]}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    result = generate_ai_comment(**_arguments())

    assert result == "値位置を見て慎重にいこ。"
    assert captured["url"].endswith("/messages")
    assert captured["headers"]["x-api-key"] == "test-anthropic-key"


def test_generate_ai_comment_falls_back_on_api_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            requests.Timeout("test timeout")
        ),
    )

    result = generate_ai_comment(**_arguments())

    assert "トヨタ自動車（7203.T）" in result
    assert "ランクA" in result
