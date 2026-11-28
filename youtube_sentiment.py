"""YouTube market-sentiment collection and Gemini comment classification."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


CHANNELS = {
    "bull_asakura": {
        "name": "朝倉慶のASK1（強気派）",
        "channel_id": "UCax09PmcRoY1R8mfJFBfv0g",
    },
    "bear_buffett": {
        "name": "バフェット太郎（弱気派）",
        "channel_id": "UCQPPXy9LCznUQHHG_kh6Bpg",
    },
    "entertainer_kamioka": {
        "name": "上岡正明（風見鶏・エンタメ派）",
        "channel_id": "UC0gvn7xAe2wHtFZYPqsf6bA",
    },
}

EMOTION_KEYS = ("Euphoria", "Fear", "Mockery", "Anger", "Calm")


class YouTubeSentimentError(RuntimeError):
    """An expected YouTube, Gemini, or configuration error."""


def fetch_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    if payload is None:
        request = urllib.request.Request(url)
    else:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise YouTubeSentimentError(
            f"APIリクエストが失敗しました（HTTP {exc.code}）"
        ) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise YouTubeSentimentError("APIへの接続またはレスポンス解析に失敗しました") from exc


def get_latest_video(
    channel_id: str,
    youtube_api_key: str,
) -> dict[str, str] | None:
    """Get the newest upload using the channel uploads playlist."""
    uploads_playlist_id = "UU" + channel_id[2:]
    params = urllib.parse.urlencode(
        {
            "part": "snippet",
            "playlistId": uploads_playlist_id,
            "maxResults": 1,
            "key": youtube_api_key,
        }
    )
    data = fetch_json(
        f"https://www.googleapis.com/youtube/v3/playlistItems?{params}"
    )
    items = data.get("items", [])
    if not items:
        return None

    snippet = items[0]["snippet"]
    return {
        "video_id": snippet["resourceId"]["videoId"],
        "title": snippet["title"],
        "published_at": snippet["publishedAt"],
    }


def get_video_stats_and_comments(
    video_id: str,
    youtube_api_key: str,
    max_comments: int = 30,
) -> tuple[dict[str, str], list[str]]:
    v_params = urllib.parse.urlencode(
        {
            "part": "statistics",
            "id": video_id,
            "key": youtube_api_key,
        }
    )
    video_data = fetch_json(
        f"https://www.googleapis.com/youtube/v3/videos?{v_params}"
    )
    stats = (
        video_data["items"][0]["statistics"]
        if video_data.get("items")
        else {}
    )

    c_params = urllib.parse.urlencode(
        {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": max_comments,
            "order": "relevance",
            "textFormat": "plainText",
            "key": youtube_api_key,
        }
    )
    try:
        comment_data = fetch_json(
            f"https://www.googleapis.com/youtube/v3/commentThreads?{c_params}"
        )
    except YouTubeSentimentError as exc:
        if "HTTP 403" in str(exc):
            return stats, []
        raise

    comments = []
    for item in comment_data.get("items", []):
        text = (
            item.get("snippet", {})
            .get("topLevelComment", {})
            .get("snippet", {})
            .get("textDisplay")
        )
        if isinstance(text, str) and text.strip():
            comments.append(text.strip())
    return stats, comments


def classify_comments_batch(
    channel_name: str,
    video_title: str,
    comments: list[str],
    gemini_api_key: str,
) -> dict[str, Any]:
    if not comments:
        return {key: 0 for key in EMOTION_KEYS} | {"comment_details": []}
    if not gemini_api_key:
        raise YouTubeSentimentError("GEMINI_API_KEYが設定されていません")

    model = "gemini-3.1-flash-lite"
    params = urllib.parse.urlencode({"key": gemini_api_key})
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?{params}"
    )
    numbered_comments = "\n".join(
        f"{index + 1}. {comment[:200]}"
        for index, comment in enumerate(comments)
    )

    system_prompt = (
        "あなたは冷徹で正確な金融センチメント分析官です。"
        "投資系YouTube動画の各コメントを以下の5カテゴリのいずれか1つに厳密に分類してください。\n"
        "- Euphoria: 相場上昇・利益・強気予想への過度な熱狂、盲信、歓喜のみ。"
        "相場と無関係な応援や番組への感想には使わない。\n"
        "- Fear: 暴落懸念、相場への恐怖、損失の痛み・後悔、投げ売り、悲観、先行き不安。\n"
        "- Mockery: 嘲笑、煽り、皮肉、逆神扱い、他者や予想への見下し。\n"
        "- Anger: 配信者や市場への強い怒り、批判、不満、掌返しへの非難。\n"
        "- Calm: 冷静な市場分析、客観的意見、質問、挨拶、視聴感謝、"
        "出演者や番組への応援・雑談。\n"
        "ノイズ除外を厳守: チャンネル固有の挨拶・定型句（「ちゃお」「結界」「グッドラック」など）、"
        "歌やトークへの感想、社員・配信者への単なる応援や感謝は、熱量が高くても必ずCalm。"
        "これらをEuphoriaにしない。\n"
        "顔文字（^^、😂など）だけでMockeryと判断しない。文脈が冷静な分析ならCalm。\n"
        "各コメントにindex（1始まり）、label、reason（判定理由15文字以内）を付け、"
        "5カテゴリの件数と分類明細を同じJSONに含めてください。件数は明細の分類と一致させてください。"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"チャンネル: {channel_name}\n"
                            f"動画タイトル: {video_title}\n\n"
                            f"コメント一覧（全{len(comments)}件）:\n"
                            f"{numbered_comments}"
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "OBJECT",
                "properties": {
                    "Euphoria": {"type": "INTEGER"},
                    "Fear": {"type": "INTEGER"},
                    "Mockery": {"type": "INTEGER"},
                    "Anger": {"type": "INTEGER"},
                    "Calm": {"type": "INTEGER"},
                    "classifications": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "index": {"type": "INTEGER"},
                                "label": {
                                    "type": "STRING",
                                    "enum": list(EMOTION_KEYS),
                                },
                                "reason": {"type": "STRING"},
                            },
                            "required": ["index", "label", "reason"],
                        },
                    },
                },
                "required": [*EMOTION_KEYS, "classifications"],
            },
        },
    }

    response = fetch_json(url, payload)
    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
        raise YouTubeSentimentError("Geminiから分類結果を解析できませんでした") from exc

    normalized = {}
    for key in EMOTION_KEYS:
        value = result.get(key, 0)
        normalized[key] = int(value) if isinstance(value, int) else 0

    detail_by_index: dict[int, dict[str, str]] = {}
    raw_details = result.get("classifications", [])
    if isinstance(raw_details, list):
        for detail in raw_details:
            if not isinstance(detail, dict):
                continue
            try:
                index = int(detail["index"])
            except (KeyError, TypeError, ValueError):
                continue
            label = detail.get("label")
            if label not in EMOTION_KEYS:
                continue
            reason = detail.get("reason")
            if not isinstance(reason, str):
                reason = ""
            detail_by_index[index] = {
                "label": label,
                "reason": reason.strip()[:15],
            }

    comment_details = []
    counts = {key: 0 for key in EMOTION_KEYS}
    for index, comment in enumerate(comments, start=1):
        detail = detail_by_index.get(index, {})
        label = detail.get("label", "Calm")
        if label not in EMOTION_KEYS:
            label = "Calm"
        reason = detail.get("reason", "分類可能な感情なし")
        counts[label] += 1
        comment_details.append(
            {
                "index": index,
                "text": comment,
                "label": label,
                "reason": reason[:15],
            }
        )

    normalized["comment_details"] = comment_details
    for key in EMOTION_KEYS:
        normalized[key] = counts[key]
    return normalized


def scan_channels(
    youtube_api_key: str,
    gemini_api_key: str,
    max_comments: int = 30,
) -> list[dict[str, Any]]:
    if not youtube_api_key:
        raise YouTubeSentimentError("YOUTUBE_API_KEYが設定されていません")

    results = []
    for info in CHANNELS.values():
        try:
            video = get_latest_video(info["channel_id"], youtube_api_key)
            if not video:
                results.append(
                    {
                        **info,
                        "error": "動画が取得できませんでした",
                    }
                )
                continue

            stats, comments = get_video_stats_and_comments(
                video["video_id"],
                youtube_api_key,
                max_comments=max_comments,
            )
            emotions = classify_comments_batch(
                info["name"],
                video["title"],
                comments,
                gemini_api_key,
            )
            results.append(
                {
                    **info,
                    **video,
                    "stats": stats,
                    "comment_count": len(comments),
                    "emotions": emotions,
                    "comment_details": emotions.get("comment_details", []),
                }
            )
        except YouTubeSentimentError as exc:
            results.append(
                {
                    **info,
                    "error": str(exc),
                }
            )
    return results
