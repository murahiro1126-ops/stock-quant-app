"""Formatting helpers for chart titles and static chart rendering."""

from __future__ import annotations

from math import floor

CHART_FONT_FAMILY = "Noto Sans CJK JP, sans-serif"

_FEAR_GREED_LABELS = {
    "😱 極度の恐怖": "極度の恐怖",
    "😨 恐怖": "恐怖",
    "😐 中立": "中立",
    "🙂 貪欲": "貪欲",
    "🔥 極度の熱狂": "極度の熱狂",
}


def build_chart_title(
    ticker: str,
    company: str,
    buy_confidence: int,
    sell_confidence: int,
    range_status: str,
    fear_greed_label: str,
    fear_greed_score: float | None,
    rank: str,
    win_rate: float,
) -> str:
    """Build a chart title matching the on-screen metric order."""
    fear_greed_text = _FEAR_GREED_LABELS.get(
        fear_greed_label, fear_greed_label
    )
    if fear_greed_score is not None:
        fear_greed_text += f" ({int(floor(fear_greed_score + 0.5))})"

    return (
        f"{ticker} {company} | 買い確度: {buy_confidence}% "
        f"/ 売り確度: {sell_confidence}% "
        f"/ 5年レンジ位置: {range_status} "
        f"/ 恐怖指数: {fear_greed_text} "
        f"/ AI信頼度: ランク{rank} (勝率{win_rate:.1f}%)"
    )
