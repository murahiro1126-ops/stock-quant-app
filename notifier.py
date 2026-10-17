"""Scheduled watchlist screening and Discord webhook notifications."""

from __future__ import annotations

import os
import sys

import requests

from analyzer import (
    add_signal_flags,
    backtest_strategy,
    calculate_indicators,
    fetch_stock_data,
    get_google_trends_sentiment,
    score_signal,
)
from config import NOTIFICATION_THRESHOLD
from watchlist_store import load_watchlist


def screen_watchlist() -> list[dict[str, object]]:
    alerts = []
    watchlist = load_watchlist()
    for ticker, company in watchlist.items():
        try:
            indicators = calculate_indicators(fetch_stock_data(ticker))
            latest = indicators.iloc[-1]
            latest_signals = add_signal_flags(
                indicators, threshold=NOTIFICATION_THRESHOLD
            ).iloc[-1]
        except Exception as exc:
            print(f"ERROR: {ticker} ({company}) の分析に失敗: {exc}", file=sys.stderr)
            continue

        sentiment_rising = None
        try:
            sentiment_rising = get_google_trends_sentiment(company)["rising"]
        except Exception as exc:
            print(
                f"WARNING: {ticker} ({company}) のGoogleトレンドを取得できません。"
                f"テクニカルスコアのみで判定します: {exc}",
                file=sys.stderr,
            )
        scores = score_signal(indicators, sentiment_rising=sentiment_rising)
        backtest = backtest_strategy(
            indicators, threshold=NOTIFICATION_THRESHOLD
        )

        for side, key in (("買い", "buy_confidence"), ("売り", "sell_confidence")):
            confidence = int(scores[key])
            transition = (
                bool(latest_signals["Buy_Signal"])
                if side == "買い"
                else bool(latest_signals["Sell_Signal"])
            )
            if (
                confidence >= NOTIFICATION_THRESHOLD
                and transition
                and not scores["is_ranging"]
            ):
                reason_key = "buy_reasons" if side == "買い" else "sell_reasons"
                alerts.append(
                    {
                        "ticker": ticker,
                        "company": company,
                        "side": side,
                        "confidence": confidence,
                        "close": float(latest["Close"]),
                        "reasons": scores[reason_key],
                        "range_position": scores["range_position"],
                        "range_status": scores["range_status"],
                        "rank": backtest["rank"],
                        "backtest_win_rate": backtest["win_rate"],
                        "sentiment": (
                            "上昇" if sentiment_rising else "非上昇"
                        ) if sentiment_rising is not None else "未取得（テクニカルのみ）",
                    }
                )
    return alerts


def send_discord_notifications(alerts: list[dict[str, object]], webhook_url: str) -> None:
    for alert in alerts:
        reasons = "、".join(alert["reasons"])
        range_position = alert["range_position"]
        range_text = (
            f"{range_position:.1f}%" if range_position is not None else "判定不可"
        )
        message = (
            f"**{alert['side']}シグナル｜{alert['company']}（{alert['ticker']}）**\n"
            f"確度: **{alert['confidence']}%** / 終値: {alert['close']:,.2f}円\n"
            f"5年レンジ位置: {alert['range_status']}（{range_text}）\n"
            f"バックテストAI信頼度: ランク{alert['rank']} "
            f"（勝率{alert['backtest_win_rate']:.1f}%）\n"
            f"Googleトレンド7日移動平均: {alert['sentiment']}\n"
            f"判定理由: {reasons or '条件なし'}"
        )
        response = requests.post(webhook_url, json={"content": message}, timeout=15)
        response.raise_for_status()


def main() -> int:
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("ERROR: DISCORD_WEBHOOK_URL が設定されていません。", file=sys.stderr)
        return 1

    alerts = screen_watchlist()
    if not alerts:
        print("通知対象のシグナルはありませんでした。")
        return 0

    try:
        send_discord_notifications(alerts, webhook_url)
    except requests.RequestException as exc:
        print(f"ERROR: Discordへの通知に失敗しました: {exc}", file=sys.stderr)
        return 1
    print(f"{len(alerts)} 件のシグナルをDiscordへ通知しました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
