from youtube_sentiment import (
    classify_comments_batch,
    get_latest_video,
    get_video_stats_and_comments,
)


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
