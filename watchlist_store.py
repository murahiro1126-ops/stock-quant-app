"""Load and persist the application's editable stock watchlist."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

from config import WATCHLIST

WATCHLIST_FILE = Path(__file__).resolve().with_name("watchlist.json")
_JAPANESE_TICKER_PATTERN = re.compile(r"^[0-9A-Z]{4}\.T$")


def load_watchlist(path: str | Path | None = None) -> dict[str, str]:
    """Load a saved watchlist, or use the built-in defaults if none is saved."""
    file_path = Path(path) if path is not None else WATCHLIST_FILE
    if not file_path.exists():
        return dict(WATCHLIST)

    try:
        saved = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"ウォッチリストを読み込めません: {file_path}: {exc}") from exc

    if not isinstance(saved, dict):
        raise ValueError(f"ウォッチリストの形式が正しくありません: {file_path}")
    return validate_watchlist(saved)


def validate_watchlist(entries: Mapping[object, object]) -> dict[str, str]:
    """Normalize and validate ticker/name pairs before saving."""
    normalized: dict[str, str] = {}
    seen_tickers: set[str] = set()

    for raw_ticker, raw_name in entries.items():
        if not isinstance(raw_ticker, str):
            raise ValueError("銘柄コードを文字列で入力してください。")
        if not isinstance(raw_name, str):
            raise ValueError(f"銘柄コード「{raw_ticker}」の銘柄名を入力してください。")
        ticker = str(raw_ticker).strip().upper()
        if re.fullmatch(r"[0-9A-Z]{4}", ticker):
            ticker = f"{ticker}.T"
        name = str(raw_name).strip()
        if not _JAPANESE_TICKER_PATTERN.fullmatch(ticker):
            raise ValueError(
                f"銘柄コード「{raw_ticker}」が不正です。4文字の日本株コード "
                "（例: 7203 または 7203.T）を入力してください。"
            )
        if not name:
            raise ValueError(f"銘柄コード「{ticker}」の銘柄名を入力してください。")
        if ticker in seen_tickers:
            raise ValueError(f"銘柄コード「{ticker}」が重複しています。")
        normalized[ticker] = name
        seen_tickers.add(ticker)

    return normalized


def save_watchlist(
    entries: Mapping[object, object], path: str | Path | None = None
) -> dict[str, str]:
    """Validate and atomically persist ticker/name pairs as UTF-8 JSON."""
    file_path = Path(path) if path is not None else WATCHLIST_FILE
    watchlist = validate_watchlist(entries)
    temporary_path = file_path.with_name(f"{file_path.name}.tmp")

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(watchlist, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(file_path)
    except OSError as exc:
        raise OSError(f"ウォッチリストを保存できません: {file_path}: {exc}") from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return watchlist
