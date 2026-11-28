import csv
import io
import urllib.error
from typing import Any

import pytest

from youtube_sentiment import (
    YouTubeSentimentError,
    calculate_sentiment_metrics,
    classify_comments_batch,
    fetch_json,
    get_latest_video,
    get_video_stats_and_comments,
    save_sentiment_history,
    scan_channels,
    to_jst_published_date,
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


def test_calculate_sentiment_metrics_rounds_to_one_decimal_place():
    result = calculate_sentiment_metrics(
        {
            "Euphoria": 1,
            "Fear": 0,
            "Mockery": 0,
            "Anger": 1,
            "Calm": 1,
        }
    )

    assert result == {
        "net_sentiment": 33.3,
        "friction_index": 33.3,
        "emotional_density": 66.7,
        "total": 3,
    }


def test_to_jst_published_date_converts_utc_to_jst_date():
    assert to_jst_published_date("2026-01-01T18:00:00Z") == "2026-01-02"


def test_scan_channels_returns_timeseries_fields(monkeypatch):
    monkeypatch.setattr(
        "youtube_sentiment.CHANNELS",
        {
            "bull_asakura": {
                "name": "テストチャンネル",
                "channel_id": "UCtest-channel-id",
            }
        },
    )

    def fake_get_latest_video(channel_id, api_key):
        assert channel_id == "UCtest-channel-id"
        assert api_key == "youtube-key"
        return {
            "video_id": "video-1",
            "title": "テスト動画",
            "published_at": "2026-01-01T18:00:00Z",
        }

    def fake_get_video_stats_and_comments(video_id, api_key, max_comments):
        assert video_id == "video-1"
        assert api_key == "youtube-key"
        assert max_comments == 30
        return (
            {"viewCount": "123", "commentCount": "9"},
            ["テストコメント"],
        )

    def fake_classify_comments_batch(channel_name, title, comments, api_key):
        assert channel_name == "テストチャンネル"
        assert title == "テスト動画"
        assert comments == ["テストコメント"]
        assert api_key == "gemini-key"
        return {
            "Euphoria": 1,
            "Fear": 0,
            "Mockery": 0,
            "Anger": 0,
            "Calm": 0,
            "comment_details": [],
        }

    monkeypatch.setattr(
        "youtube_sentiment.get_latest_video",
        fake_get_latest_video,
    )
    monkeypatch.setattr(
        "youtube_sentiment.get_video_stats_and_comments",
        fake_get_video_stats_and_comments,
    )
    monkeypatch.setattr(
        "youtube_sentiment.classify_comments_batch",
        fake_classify_comments_batch,
    )

    result = scan_channels("youtube-key", "gemini-key")[0]

    required_fields = {
        "published_at",
        "published_date",
        "scanned_at",
        "channel_key",
        "channel_name",
        "video_id",
        "video_title",
        "view_count",
        "comment_total_youtube",
        "sample_count",
        "Euphoria",
        "Fear",
        "Mockery",
        "Anger",
        "Calm",
        "net_sentiment",
        "friction_index",
        "emotional_density",
    }
    assert required_fields <= result.keys()
    assert result["published_date"] == "2026-01-02"
    assert result["scanned_at"].endswith("+09:00")
    assert result["channel_key"] == "bull_asakura"
    assert result["channel_name"] == "テストチャンネル"
    assert result["video_title"] == "テスト動画"
    assert result["view_count"] == 123
    assert result["comment_total_youtube"] == 9
    assert result["sample_count"] == 1
    assert isinstance(result["net_sentiment"], float)


def _history_record(
    video_id: str,
    published_date: str,
    scanned_at: str,
) -> dict[str, Any]:
    return {
        "published_at": f"{published_date}T00:00:00Z",
        "published_date": published_date,
        "scanned_at": scanned_at,
        "channel_key": "bull_asakura",
        "channel_name": "テストチャンネル",
        "video_id": video_id,
        "video_title": f"動画 {video_id}",
        "view_count": 100,
        "comment_total_youtube": 10,
        "sample_count": 3,
        "Euphoria": 1,
        "Fear": 0,
        "Mockery": 1,
        "Anger": 0,
        "Calm": 1,
        "net_sentiment": 33.3,
        "friction_index": 33.3,
        "emotional_density": 66.7,
    }


def test_save_sentiment_history_upserts_and_sorts_records(tmp_path):
    history_path = tmp_path / "data" / "sentiment_history.csv"
    first = _history_record(
        "video-later",
        "2026-01-03",
        "2026-01-04T10:00:00+09:00",
    )
    older = _history_record(
        "video-earlier",
        "2026-01-01",
        "2026-01-02T10:00:00+09:00",
    )

    first_save = save_sentiment_history([first, older], history_path)
    updated = {
        **first,
        "scanned_at": "2026-01-05T10:00:00+09:00",
        "sample_count": 4,
        "Euphoria": 2,
        "net_sentiment": 50.0,
    }
    second_save = save_sentiment_history([updated], history_path)
    stale = {
        **first,
        "scanned_at": "2026-01-03T10:00:00+09:00",
        "sample_count": 99,
    }
    stale_save = save_sentiment_history([stale], history_path)

    with history_path.open("r", newline="", encoding="utf-8") as history_file:
        rows = list(csv.DictReader(history_file))

    assert first_save == {"added": 2, "updated": 0, "total": 2}
    assert second_save == {"added": 0, "updated": 1, "total": 2}
    assert stale_save == {"added": 0, "updated": 0, "total": 2}
    assert [row["video_id"] for row in rows] == ["video-earlier", "video-later"]
    assert rows[1]["scanned_at"] == "2026-01-05T10:00:00+09:00"
    assert rows[1]["sample_count"] == "4"
    assert rows[1]["net_sentiment"] == "50.0"


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
