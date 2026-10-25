"""Reusable PostPrime text and chart-image export helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO

import plotly.graph_objects as go


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


def render_plotly_png(figure: go.Figure, *, width: int = 1600) -> bytes:
    """Render one Plotly figure as a PNG for a standalone download."""
    if width < 1:
        raise ValueError("画像幅は1以上にしてください。")
    return figure.to_image(format="png", width=width)


def compose_plotly_png(
    figures: Sequence[tuple[str, go.Figure]],
    *,
    width: int = 1600,
    gap: int = 24,
) -> bytes:
    """Render named Plotly figures and stack them into one reusable PNG image."""
    if not figures:
        raise ValueError("画像にまとめるチャートがありません。")
    if width < 1 or gap < 0:
        raise ValueError("画像幅は1以上、チャート間隔は0以上にしてください。")

    from PIL import Image, ImageDraw

    rendered: list[tuple[str, Image.Image]] = []
    for title, figure in figures:
        png_data = render_plotly_png(figure, width=width)
        with Image.open(BytesIO(png_data)) as image:
            rgb_image = image.convert("RGB")
        if rgb_image.width != width:
            resized_height = round(rgb_image.height * width / rgb_image.width)
            rgb_image = rgb_image.resize(
                (width, resized_height),
                Image.Resampling.LANCZOS,
            )
        rendered.append((title, rgb_image))

    heading_height = 32
    canvas_height = sum(
        heading_height + image.height for _, image in rendered
    ) + gap * (len(rendered) - 1)
    canvas = Image.new("RGB", (width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    top = 0
    for title, image in rendered:
        draw.text((16, top + 8), title, fill="#26374a")
        top += heading_height
        canvas.paste(image, (0, top))
        top += image.height + gap

    output = BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()
