import io
import urllib.error

import pytest

from youtube_sentiment import (
    YouTubeSentimentError,
    calculate_sentiment_metrics,
    classify_comments_batch,
    fetch_json,
    get_latest_video,
    get_video_stats_and_comments,
)


def test_calculate_sentiment_metrics():
    result = calculate_sentiment_metrics(
        {
            "Euphoria": 3,
            "Fear": 1,
            "Mockery": 2,
            "Anger": 1,
            "Calm": 3,
        }
    )

    assert result == {
        "net_sentiment": 20.0,
        "friction_index": 30.0,
        "emotional_density": 70.0,
        "total": 10,
    }


def test_calculate_sentiment_metrics_with_zero_total():
    result = calculate_sentiment_metrics(
        {
            "Euphoria": 0,
            "Fear": 0,
            "Mockery": 0,
            "Anger": 0,
            "Calm": 0,
        }
    )

    assert result == {
        "net_sentiment": 0.0,
        "friction_index": 0.0,
        "emotional_density": 0.0,
        "total": 0,
    }


def test_fetch_json_logs_youtube_error_details_without_api_key(monkeypatch, capsys):
    api_key = "private-api-key"
    response_body = (
        b'{"error":{"message":"Daily Limit Exceeded",'
        b'"errors":[{"reason":"quotaExceeded","location":"key",'
        b'"message":"Request quota exceeded"}]}}'
    )

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            {},
            io.BytesIO(response_body),
        )

    monkeypatch.setattr("youtube_sentiment.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(YouTubeSentimentError):
        fetch_json(
            f"https://www.googleapis.com/youtube/v3/videos?key={api_key}"
        )

    output = capsys.readouterr().out
    assert "Daily Limit Exceeded" in output
    assert "quotaExceeded" in output
    assert "Request quota exceeded" in output
    assert api_key not in output
    assert "key=" not in output


def test_fetch_json_logs_first_500_characters_for_non_json_youtube_error(
    monkeypatch,
    capsys,
):
    response_body = b"x" * 600

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            502,
            "Bad Gateway",
            {},
            io.BytesIO(response_body),
        )

    monkeypatch.setattr("youtube_sentiment.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(YouTubeSentimentError):
        fetch_json("https://www.googleapis.com/youtube/v3/videos?key=test-key")

    output = capsys.readouterr().out
    assert "non-JSON response: " + "x" * 500 in output
    assert "x" * 501 not in output


def test_get_latest_video(monkeypatch):
    captured = {}

    def fake_fetch_json(url):
        captured["url"] = url
        return {
            "items": [
                {
                    "snippet": {
                        "resourceId": {"videoId": "video-1"},
                        "title": "Latest title",
                        "publishedAt": "2026-11-28T00:00:00Z",
                    }
                }
            ]
        }

    monkeypatch.setattr(
        "youtube_sentiment.fetch_json",
        fake_fetch_json,
    )

    result = get_latest_video("UCtest-channel-id", "youtube-key")

    assert result["video_id"] == "video-1"
    assert "playlistId=UUtest-channel-id" in captured["url"]


def test_get_video_stats_and_comments(monkeypatch):
    responses = [
        {"items": [{"statistics": {"viewCount": "123", "commentCount": "4"}}]},
        {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {
                            "snippet": {"textDisplay": "  strong market  "}
                        }
                    }
                },
                {
                    "snippet": {
                        "topLevelComment": {
                            "snippet": {"textDisplay": None}
                        }
                    }
                },
            ]
        },
    ]

    monkeypatch.setattr(
        "youtube_sentiment.fetch_json",
        lambda url: responses.pop(0),
    )

    stats, comments = get_video_stats_and_comments(
        "video-1",
        "youtube-key",
        max_comments=30,
    )

    assert stats["viewCount"] == "123"
    assert comments == ["strong market"]


def test_classify_comments_batch(monkeypatch):
    def fake_fetch_json(url, payload):
        assert "gemini-key" in url
        assert "/gemini-3.1-flash-lite:" in url
        assert payload["generationConfig"]["response_mime_type"] == "application/json"
        system_text = payload["system_instruction"]["parts"][0]["text"]
        assert "Euphoria: 相場上昇・利益・強気予想" in system_text
        assert "Fear: 暴落懸念" in system_text
        assert "ちゃお" in system_text
        assert "顔文字（^^、😂など）だけでMockeryと判断しない" in system_text
        assert payload["generationConfig"]["response_schema"]["properties"]["classifications"][
            "items"
        ]["properties"]["label"]["enum"] == [
            "Euphoria",
            "Fear",
            "Mockery",
            "Anger",
            "Calm",
        ]
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{"Euphoria": 1, "Fear": 1, "Mockery": 1, '
                                    '"Anger": 1, "Calm": 1, '
                                    '"classifications": ['
                                    '{"index": 1, "label": "Euphoria", "reason": "利益期待"}, '
                                    '{"index": 2, "label": "Fear", "reason": "暴落不安"}, '
                                    '{"index": 3, "label": "Mockery", "reason": "皮肉"}, '
                                    '{"index": 4, "label": "Anger", "reason": "強い批判"}, '
                                    '{"index": 5, "label": "Calm", "reason": "感謝と挨拶"}'
                                    ']}'
                                )
                            }
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(
        "youtube_sentiment.fetch_json",
        fake_fetch_json,
    )

    result = classify_comments_batch(
        "チャンネル",
        "動画",
        [
            "株価上昇で利益が期待できる",
            "暴落が怖くて不安",
            "また予想を外したね",
            "配信者の発言に強く不満",
            "いつもありがとうございます",
        ],
        "gemini-key",
    )

    assert {key: result[key] for key in ("Euphoria", "Fear", "Mockery", "Anger", "Calm")} == {
        "Euphoria": 1,
        "Fear": 1,
        "Mockery": 1,
        "Anger": 1,
        "Calm": 1,
    }
    assert result["comment_details"] == [
        {
            "index": 1,
            "text": "株価上昇で利益が期待できる",
            "label": "Euphoria",
            "reason": "利益期待",
        },
        {"index": 2, "text": "暴落が怖くて不安", "label": "Fear", "reason": "暴落不安"},
        {"index": 3, "text": "また予想を外したね", "label": "Mockery", "reason": "皮肉"},
        {
            "index": 4,
            "text": "配信者の発言に強く不満",
            "label": "Anger",
            "reason": "強い批判",
        },
        {
            "index": 5,
            "text": "いつもありがとうございます",
            "label": "Calm",
            "reason": "感謝と挨拶",
        },
    ]
