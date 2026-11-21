"""Reusable PostPrime text and chart-image export helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from chart_format import CHART_FONT_FAMILY
from sector_flow import build_matrix_dataframe

PNG_EXPORT_SCALE = 2
POSTPRIME_CARD_WIDTH = 1600
POSTPRIME_CARD_HEIGHT = 900
PNG_EMOJI_FONT_FAMILIES = (
    "Noto Color Emoji",
    "Segoe UI Emoji",
    "Apple Color Emoji",
)
_EMOJI_FONT_FAMILY_NAMES = {
    family.lower() for family in PNG_EMOJI_FONT_FAMILIES
}


def _with_emoji_font_fallbacks(family: object) -> str:
    existing_families = [
        name.strip()
        for name in str(family or "Noto Sans CJK JP").split(",")
        if name.strip()
        and name.strip().lower()
        not in {"sans-serif", *_EMOJI_FONT_FAMILY_NAMES}
    ]
    families = [*PNG_EMOJI_FONT_FAMILIES, *existing_families]
    families.append("sans-serif")
    return ", ".join(families)


def _prepare_plotly_value(value: object) -> object:
    if isinstance(value, dict):
        prepared = {
            key: _prepare_plotly_value(item)
            for key, item in value.items()
        }
        for key, font in prepared.items():
            if key.lower().endswith("font") and isinstance(font, dict):
                font["family"] = _with_emoji_font_fallbacks(
                    font.get("family")
                )
        return prepared
    if isinstance(value, (list, tuple)):
        return [_prepare_plotly_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _prepare_plotly_value(value.tolist())
    if hasattr(value, "item"):
        return _prepare_plotly_value(value.item())
    return value


def prepare_figure_for_png(figure: go.Figure) -> go.Figure:
    """Copy a Plotly figure and add emoji-font fallbacks without altering text."""
    prepared = _prepare_plotly_value(figure.to_plotly_json())
    layout = prepared.setdefault("layout", {})
    layout_font = layout.setdefault("font", {})
    if isinstance(layout_font, dict):
        layout_font["family"] = _with_emoji_font_fallbacks(
            layout_font.get("family")
        )
    return go.Figure(prepared)


def build_postprime_export_text(
    mode_name: str,
    comments: Mapping[str, str],
) -> str:
    """Combine a reusable article prompt with named analysis-comment sections."""
    prompt = (
        f"ギャルたん！{mode_name}の最新クオンツ分析データと"
        "第0層 東証33業種 資金循環の分析画像(RRG、モメンタム、多期間比較マトリックス)を共有するね！\n"
        "以下の分析コメントと分析画像を参考に、PostPrime投稿用の記事を作成してくれるかな？"
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
    """Render a high-resolution Plotly PNG with emoji-font fallbacks."""
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


def select_postprime_card_rows(
    matrix: pd.DataFrame,
    *,
    max_rows: int = 7,
) -> pd.DataFrame:
    """Prioritize early signals, then fill a compact card with notable returns."""
    if max_rows < 1:
        raise ValueError("表示行数は1以上にしてください。")
    required = {"直近5日リターン", "初動判定"}
    missing = required.difference(matrix.columns)
    if missing:
        raise ValueError(
            "カード表示用のマトリックス列が不足しています: "
            + ", ".join(sorted(missing))
        )
    if matrix.empty:
        raise ValueError("カードに表示する業種データがありません。")

    ordered = matrix.assign(
        _card_return=pd.to_numeric(
            matrix["直近5日リターン"], errors="coerce"
        )
    ).sort_values("_card_return", ascending=False, na_position="last")
    early_signals = {"🔥 転換初動", "🚀 加速初動"}
    early = ordered.loc[ordered["初動判定"].isin(early_signals)]
    peak_out = ordered.loc[ordered["初動判定"].eq("⚠️ ピークアウト")]

    selected: list[object] = []

    def append_sectors(rows: pd.DataFrame) -> None:
        for sector in rows.index:
            if sector not in selected and len(selected) < max_rows:
                selected.append(sector)

    append_sectors(early)
    append_sectors(peak_out)
    if early.empty:
        append_sectors(ordered.head(4))
        append_sectors(ordered.tail(min(3, len(ordered))))
    append_sectors(ordered)
    return matrix.loc[selected].copy()


def build_postprime_card_figure(
    metrics: pd.DataFrame,
    mode_name: str,
) -> go.Figure:
    """Build the compact 16:9 ranking-and-sector-signal post card."""
    required = {"Date", "Sector", "Return_20D", "Market_Return_20D"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            "カード画像用の指標が不足しています: " + ", ".join(sorted(missing))
        )
    if metrics.empty:
        raise ValueError("カード画像を描画する分析データがありません。")

    latest = (
        metrics.assign(Date=pd.to_datetime(metrics["Date"], errors="coerce"))
        .dropna(subset=["Date"])
        .sort_values("Date")
        .groupby("Sector", sort=False, as_index=False)
        .tail(1)
        .set_index("Sector")
    )
    ranking = latest.copy()
    ranking["Return_20D"] = pd.to_numeric(
        ranking["Return_20D"], errors="coerce"
    )
    ranking = ranking.dropna(subset=["Return_20D"])
    if ranking.empty:
        raise ValueError("20営業日リターンのランキングデータがありません。")

    top = ranking.nlargest(5, "Return_20D").sort_values("Return_20D")
    bottom = ranking.nsmallest(5, "Return_20D").sort_values("Return_20D")
    top_sectors = set(top.index)
    extremes = pd.concat([top, bottom]).loc[lambda rows: ~rows.index.duplicated()]
    sector_order = list(
        dict.fromkeys(
            list(bottom.sort_values("Return_20D", ascending=False).index)
            + list(top.sort_values("Return_20D").index)
        )
    )
    extremes = extremes.reindex(sector_order)

    market_values = pd.to_numeric(
        latest["Market_Return_20D"], errors="coerce"
    ).dropna()
    market_return = float(market_values.iloc[-1]) if not market_values.empty else 0.0
    value_min = min(float(extremes["Return_20D"].min()), market_return, 0)
    value_max = max(float(extremes["Return_20D"].max()), market_return, 0)
    value_span = float(extremes["Return_20D"].max() - extremes["Return_20D"].min())
    latest_date = pd.to_datetime(metrics["Date"], errors="coerce").max()
    date_label = latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "—"

    matrix = build_matrix_dataframe(metrics)
    card_rows = select_postprime_card_rows(matrix)
    table_values = [
        [str(sector) for sector in card_rows.index],
        [
            f"{value:+.2f}%"
            if pd.notna(value) and np.isfinite(value)
            else "-"
            for value in card_rows["直近5日リターン"]
        ],
        [
            f"{value:+.2f}%"
            if pd.notna(value) and np.isfinite(value)
            else "-"
            for value in card_rows["前月リターン"]
        ],
        [
            "-" if signal == "—" else str(signal)
            for signal in card_rows["初動判定"]
        ],
    ]
    row_colors = [
        (
            "#FCE6B8"
            if signal in {"🔥 転換初動", "🚀 加速初動"}
            else "#F9D7D2"
            if signal == "⚠️ ピークアウト"
            else "#FFFFFF"
        )
        for signal in card_rows["初動判定"]
    ]
    table_height = 62 + len(card_rows) * 78
    table_center = (0.065 + 0.825) / 2
    table_half_domain = table_height / (2 * POSTPRIME_CARD_HEIGHT)

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "table"}]],
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.035,
    )
    figure.add_trace(
        go.Bar(
            x=extremes["Return_20D"],
            y=extremes.index.astype(str),
            orientation="h",
            marker={
                "color": [
                    "#D84343" if sector in top_sectors else "#347DB1"
                    for sector in extremes.index
                ],
                "line": {"color": "#FFFFFF", "width": 1},
            },
            text=[f"{value:+.1f}%" for value in extremes["Return_20D"]],
            textposition="outside",
            textfont={
                "size": 21,
                "color": "#172033",
                "family": CHART_FONT_FAMILY,
                "weight": 700,
            },
            cliponaxis=False,
            hovertemplate="%{y}<br>20営業日リターン: %{x:+.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Table(
            columnwidth=[1.55, 0.92, 0.85, 1.65],
            header={
                "values": ["業種", "直近5日", "前月", "判定"],
                "fill_color": "#17243A",
                "font": {
                    "color": "#FFFFFF",
                    "size": 23,
                    "family": CHART_FONT_FAMILY,
                    "weight": 700,
                },
                "align": ["left", "center", "center", "center"],
                "height": 62,
                "line": {"color": "#D6DCE5", "width": 1},
            },
            cells={
                "values": table_values,
                "fill_color": [row_colors] * 4,
                "font": {
                    "color": "#101828",
                    "size": 20,
                    "family": CHART_FONT_FAMILY,
                    "weight": 700,
                },
                "align": ["left", "center", "center", "center"],
                "height": 78,
                "line": {"color": "#D6DCE5", "width": 1},
            },
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(
        range=[
            value_min - max(6.0, value_span * 0.15),
            value_max + max(3.0, value_span * 0.08),
        ],
        tickfont={"size": 17, "color": "#253247", "weight": 600},
        ticksuffix="%",
        showgrid=True,
        gridcolor="#E1E6ED",
        zeroline=False,
        domain=[0.065, 0.56],
        row=1,
        col=1,
    )
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=sector_order,
        tickfont={
            "size": 21,
            "color": "#172033",
            "family": CHART_FONT_FAMILY,
            "weight": 700,
        },
        ticklen=0,
        showgrid=True,
        gridcolor="#E1E6ED",
        domain=[0.105, 0.785],
        row=1,
        col=1,
    )
    figure.data[1].domain = {
        "x": [0.605, 0.985],
        "y": [
            table_center - table_half_domain,
            table_center + table_half_domain,
        ],
    }
    figure.update_layout(
        width=POSTPRIME_CARD_WIDTH,
        height=POSTPRIME_CARD_HEIGHT,
        autosize=False,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FBFCFE",
        font={
            "family": CHART_FONT_FAMILY,
            "color": "#101828",
            "weight": 600,
        },
        margin={"l": 22, "r": 22, "t": 24, "b": 18},
        showlegend=False,
        bargap=0.25,
        shapes=[
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "x0": 0.015,
                "x1": 0.58,
                "y0": 0.065,
                "y1": 0.825,
                "line": {"color": "#D7DEE8", "width": 1},
                "fillcolor": "rgba(248,250,253,0.45)",
                "layer": "below",
            },
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "x0": 0.59,
                "x1": 0.99,
                "y0": 0.065,
                "y1": 0.825,
                "line": {"color": "#D7DEE8", "width": 1},
                "fillcolor": "rgba(248,250,253,0.45)",
                "layer": "below",
            },
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": 0.015,
                "x1": 0.99,
                "y0": 0.875,
                "y1": 0.875,
                "line": {"color": "#BFC9D6", "width": 1},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y domain",
                "x0": 0,
                "x1": 0,
                "y0": 0,
                "y1": 1,
                "line": {"color": "#667085", "dash": "dash", "width": 2},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y domain",
                "x0": market_return,
                "x1": market_return,
                "y0": 0,
                "y1": 1,
                "line": {"color": "#8E63B0", "dash": "dot", "width": 2},
            },
        ],
        annotations=[
            {
                "text": (
                    f"{mode_name} クオンツ分析 | "
                    f"TOPIX {market_return:+.2f}% ({date_label})"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.94,
                "showarrow": False,
                "font": {
                    "size": 34,
                    "color": "#101828",
                    "family": CHART_FONT_FAMILY,
                    "weight": 700,
                },
                "xanchor": "center",
            },
            {
                "text": "業種 20営業日リターン TOP5 / ワースト5",
                "xref": "paper",
                "yref": "paper",
                "x": 0.295,
                "y": 0.845,
                "showarrow": False,
                "font": {
                    "size": 26,
                    "color": "#172033",
                    "family": CHART_FONT_FAMILY,
                    "weight": 700,
                },
                "xanchor": "center",
            },
            {
                "text": "🔥 注目・初動セクターピックアップ",
                "xref": "paper",
                "yref": "paper",
                "x": 0.79,
                "y": 0.845,
                "showarrow": False,
                "font": {
                    "size": 26,
                    "color": "#172033",
                    "family": CHART_FONT_FAMILY,
                    "weight": 700,
                },
                "xanchor": "center",
            },
            {
                "text": f"TOPIX {market_return:+.2f}%",
                "xref": "x",
                "yref": "paper",
                "x": market_return,
                "y": 0.79,
                "showarrow": False,
                "font": {
                    "size": 16,
                    "color": "#7445A5",
                    "weight": 700,
                },
                "bgcolor": "#FFFFFF",
                "bordercolor": "#D8DDE5",
                "borderwidth": 1,
                "borderpad": 4,
                "xanchor": "center",
            },
        ],
    )
    return figure


def render_postprime_card_png(metrics: pd.DataFrame, mode_name: str) -> bytes:
    """Render the card at exactly 1600 by 900 pixels."""
    return render_plotly_png(
        build_postprime_card_figure(metrics, mode_name),
        width=POSTPRIME_CARD_WIDTH,
        height=POSTPRIME_CARD_HEIGHT,
        scale=1,
    )


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
        if title.isascii():
            draw.text(
                (16 * scale, top + 8 * scale),
                title,
                fill="#26374a",
                font=font,
            )
        top += heading_height
        canvas.paste(image, (0, top))
        top += image.height + output_gap

    output = BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()
