"""Reusable PostPrime text and chart-image export helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
import re
import unicodedata

import plotly.graph_objects as go

PNG_EXPORT_SCALE = 2

_EMOJI_PATTERN = re.compile(
    r"[\U0001F000-\U0001FAFF\u2600-\u27BF]"
    r"(?:[\uFE0E\uFE0F\u200D\u20E3\U0001F3FB-\U0001F3FF"
    r"\U0001F000-\U0001FAFF\u2600-\u27BF])*"
)
_EMOJI_LABELS = {
    "🔥": "[HOT]",
    "🚀": "[UP]",
    "⚠": "[WARN]",
    "💪": "[STRONG]",
    "❄": "[COOL]",
    "📈": "[CHART]",
    "📉": "[DOWN]",
    "✅": "[OK]",
    "❌": "[NG]",
    "🟢": "[POSITIVE]",
    "🔴": "[NEGATIVE]",
}


def normalize_emoji_text(text: str) -> str:
    """Replace emoji with readable ASCII labels for deterministic static exports."""
    def replace(match: re.Match[str]) -> str:
        emoji = match.group().replace("\ufe0e", "").replace("\ufe0f", "")
        if emoji in _EMOJI_LABELS:
            return _EMOJI_LABELS[emoji]

        names = [
            unicodedata.name(character, "ICON")
            for character in emoji
            if character not in {"\u200d", "\u20e3"}
            and not "\U0001F3FB" <= character <= "\U0001F3FF"
        ]
        return "".join(f"[{name}]" for name in names) or "[ICON]"

    return _EMOJI_PATTERN.sub(replace, text)


def _normalize_plotly_value(value: object) -> object:
    if isinstance(value, str):
        return normalize_emoji_text(value)
    if isinstance(value, dict):
        return {
            key: _normalize_plotly_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_plotly_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _normalize_plotly_value(value.tolist())
    if hasattr(value, "item"):
        return _normalize_plotly_value(value.item())
    return value


def prepare_figure_for_png(figure: go.Figure) -> go.Figure:
    """Copy a Plotly figure and replace emoji unsupported by server-side fonts."""
    normalized = _normalize_plotly_value(figure.to_plotly_json())
    return go.Figure(normalized)


def build_postprime_export_text(
    mode_name: str,
    comments: Mapping[str, str],
) -> str:
    """Combine a reusable article prompt with named analysis-comment sections."""
    prompt = (
        f"ギャルたん！{mode_name}の最新クオンツ分析データと"
        "添付画像(A)を共有するね！\n"
        "以下の分析コメントと画像を参考に、数値を誇張せず、"
        "リスクにも触れたPostPrime投稿用の記事を作成してください。"
    )
    sections = [
        f"【{title}】\n{comment.strip()}"
        for title, comment in comments.items()
        if comment.strip()
    ]
    if not sections:
        sections.append("【分析コメント】第0層の分析データは取得できませんでした。")
    return "\n\n".join((prompt, *sections))


def render_plotly_png(
    figure: go.Figure,
    *,
    width: int = 1600,
    height: int | None = None,
    scale: int = PNG_EXPORT_SCALE,
) -> bytes:
    """Render a font-safe, high-resolution Plotly PNG for any app chart."""
    if width < 1 or (height is not None and height < 1) or scale < 1:
        raise ValueError("画像幅・高さ・解像度スケールは1以上にしてください。")
    render_options: dict[str, int | str] = {
        "format": "png",
        "width": width,
        "scale": scale,
    }
    if height is not None:
        render_options["height"] = height
    return prepare_figure_for_png(figure).to_image(**render_options)


def compose_plotly_png(
    figures: Sequence[tuple[str, go.Figure]],
    *,
    width: int = 1600,
    gap: int = 24,
    scale: int = PNG_EXPORT_SCALE,
) -> bytes:
    """Render named Plotly figures and stack them into one reusable PNG image."""
    if not figures:
        raise ValueError("画像にまとめるチャートがありません。")
    if width < 1 or gap < 0 or scale < 1:
        raise ValueError("画像幅・解像度スケールは1以上、チャート間隔は0以上にしてください。")

    from PIL import Image, ImageDraw, ImageFont

    rendered: list[tuple[str, Image.Image]] = []
    for title, figure in figures:
        png_data = render_plotly_png(figure, width=width, scale=scale)
        with Image.open(BytesIO(png_data)) as image:
            rgb_image = image.convert("RGB")
        output_width = width * scale
        if rgb_image.width != output_width:
            resized_height = round(rgb_image.height * output_width / rgb_image.width)
            rgb_image = rgb_image.resize(
                (output_width, resized_height),
                Image.Resampling.LANCZOS,
            )
        rendered.append((title, rgb_image))

    output_width = width * scale
    heading_height = 32 * scale
    output_gap = gap * scale
    canvas_height = sum(
        heading_height + image.height for _, image in rendered
    ) + output_gap * (len(rendered) - 1)
    canvas = Image.new("RGB", (output_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14 * scale)
    top = 0
    for title, image in rendered:
        safe_title = normalize_emoji_text(title)
        if safe_title.isascii():
            draw.text(
                (16 * scale, top + 8 * scale),
                safe_title,
                fill="#26374a",
                font=font,
            )
        top += heading_height
        canvas.paste(image, (0, top))
        top += image.height + output_gap

    output = BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()
