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
POSTPRIME_STOCK_CARD_WIDTH = 1600
POSTPRIME_STOCK_CARD_HEIGHT = 900
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


def _with_card_font_fallbacks(family: object) -> str:
    primary_families = [
        name.strip()
        for name in str(family or CHART_FONT_FAMILY).split(",")
        if name.strip()
        and name.strip().lower()
        not in {"sans-serif", *_EMOJI_FONT_FAMILY_NAMES}
    ]
    return ", ".join(
        [*primary_families, *PNG_EMOJI_FONT_FAMILIES, "sans-serif"]
    )


def _prepare_postprime_card_figure(figure: go.Figure) -> go.Figure:
    """Keep normal text in the CJK font and use emoji fonts only as fallbacks."""
    prepared = figure.to_plotly_json()

    def set_card_font_fallbacks(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower().endswith("font") and isinstance(item, dict):
                    item["family"] = _with_card_font_fallbacks(
                        item.get("family")
                    )
                set_card_font_fallbacks(item)
        elif isinstance(value, list):
            for item in value:
                set_card_font_fallbacks(item)

    set_card_font_fallbacks(prepared)
    for trace in prepared.get("data", []):
        if trace.get("type") != "table":
            continue
        cells_font = trace.get("cells", {}).get("font")
        if not isinstance(cells_font, dict):
            continue
        text_family = cells_font.get("family", CHART_FONT_FAMILY)
        cells_font["family"] = [
            _with_card_font_fallbacks(text_family),
            _with_card_font_fallbacks(text_family),
            _with_card_font_fallbacks(text_family),
            _with_emoji_font_fallbacks(text_family),
        ]
    return go.Figure(prepared)


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
            columnwidth=[0.31, 0.21, 0.21, 0.27],
            header={
                "values": ["業種", "直近5日", "前月", "判定"],
                "fill_color": "#17243A",
                "font": {
                    "color": "#FFFFFF",
                    "size": 22,
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
                    "size": 18,
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
            value_min - max(10.0, value_span * 0.18),
            value_max + max(10.0, value_span * 0.18),
        ],
        tickfont={"size": 17, "color": "#253247", "weight": 600},
        ticksuffix="%",
        showgrid=True,
        gridcolor="#E1E6ED",
        zeroline=False,
        domain=[0.15, 0.56],
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
        "x": [0.595, 0.985],
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
                "y": 0.802,
                "showarrow": False,
                "font": {
                    "size": 20,
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
    figure = _prepare_postprime_card_figure(
        build_postprime_card_figure(metrics, mode_name)
    )
    return figure.to_image(
        format="png",
        width=POSTPRIME_CARD_WIDTH,
        height=POSTPRIME_CARD_HEIGHT,
        scale=1,
    )


def extract_latest_stock_card_values(
    indicators: pd.DataFrame,
) -> dict[str, float | None]:
    """Return the latest calculated MACD, RSI, ADX, and Fear & Greed values."""
    columns = {
        "MACD": "MACD",
        "RSI": "RSI",
        "ADX": "ADX",
        "Fear & Greed": "Fear_Greed_Score",
    }
    if not isinstance(indicators, pd.DataFrame) or indicators.empty:
        raise ValueError("テクニカル指標データが空です。")
    missing = set(columns.values()).difference(indicators.columns)
    if missing:
        raise ValueError(
            "カード表示用の指標が不足しています: " + ", ".join(sorted(missing))
        )

    latest = indicators.iloc[-1]
    values: dict[str, float | None] = {}
    for label, column in columns.items():
        value = pd.to_numeric(pd.Series([latest[column]]), errors="coerce").iloc[0]
        values[label] = (
            float(value)
            if pd.notna(value) and np.isfinite(float(value))
            else None
        )
    return values


def _stock_card_gauge_style(
    name: str,
    value: float | None,
) -> dict[str, object]:
    """Keep gauge thresholds and visual state separate from trace rendering."""
    if name == "MACD":
        radius = max(1.0, abs(value) * 1.5) if value is not None else 1.0
        color = (
            "#16A085"
            if value is not None and value > 0
            else "#E74C3C"
            if value is not None and value < 0
            else "#98A2B3"
        )
        state = (
            "プラス圏"
            if value is not None and value > 0
            else "マイナス圏"
            if value is not None and value < 0
            else "ゼロ近辺"
            if value is not None
            else "取得不可"
        )
        value_range = [-radius, radius]
        value_format = ".2f"
        suffix = ""
    elif name == "RSI":
        color = (
            "#E74C3C"
            if value is not None and value >= 70
            else "#3478C5"
            if value is not None and value <= 30
            else "#16A085"
            if value is not None
            else "#98A2B3"
        )
        state = (
            "買われ過ぎ"
            if value is not None and value >= 70
            else "売られ過ぎ"
            if value is not None and value <= 30
            else "中立域"
            if value is not None
            else "取得不可"
        )
        value_range = [0, 100]
        value_format = ".1f"
        suffix = ""
    elif name == "ADX":
        color = (
            "#16A085"
            if value is not None and value >= 25
            else "#F39C12"
            if value is not None and value >= 20
            else "#98A2B3"
        )
        state = (
            "強いトレンド"
            if value is not None and value >= 25
            else "トレンド発生"
            if value is not None and value >= 20
            else "弱いトレンド"
            if value is not None
            else "取得不可"
        )
        value_range = [0, 100]
        value_format = ".1f"
        suffix = ""
    else:
        color = (
            "#E74C3C"
            if value is not None and value <= 25
            else "#F39C12"
            if value is not None and value <= 40
            else "#98A2B3"
            if value is not None and value <= 60
            else "#16A085"
            if value is not None
            else "#98A2B3"
        )
        state = (
            "恐怖"
            if value is not None and value <= 40
            else "中立"
            if value is not None and value <= 60
            else "貪欲"
            if value is not None
            else "取得不可"
        )
        value_range = [0, 100]
        value_format = ".0f"
        suffix = "/100"
    return {
        "range": value_range,
        "color": color,
        "state": state,
        "valueformat": value_format,
        "suffix": suffix,
    }


def build_postprime_stock_card_figure(
    indicators: pd.DataFrame,
    ticker: str,
    company: str,
) -> go.Figure:
    """Build a 16:9 stock chart with today's four technical indicator gauges."""
    chart_columns = {"Close", "Volume", "SMA_25", "SMA_75", "SMA_200"}
    if not isinstance(indicators, pd.DataFrame) or indicators.empty:
        raise ValueError("カード画像を描画する株価データがありません。")
    missing_chart_columns = chart_columns.difference(indicators.columns)
    if missing_chart_columns:
        raise ValueError(
            "カード画像用の株価列が不足しています: "
            + ", ".join(sorted(missing_chart_columns))
        )
    latest_values = extract_latest_stock_card_values(indicators)
    chart_data = indicators.tail(120)
    date_value = pd.to_datetime(indicators.index[-1], errors="coerce")
    date_label = (
        date_value.strftime("%Y-%m-%d")
        if pd.notna(date_value)
        else str(indicators.index[-1])
    )

    figure = make_subplots(
        rows=4,
        cols=3,
        specs=[
            [
                {"type": "xy", "rowspan": 3},
                {"type": "domain", "rowspan": 2},
                {"type": "domain", "rowspan": 2},
            ],
            [None, None, None],
            [
                None,
                {"type": "domain", "rowspan": 2},
                {"type": "domain", "rowspan": 2},
            ],
            [{"type": "xy"}, None, None],
        ],
        column_widths=[0.64, 0.18, 0.18],
        row_heights=[0.25, 0.25, 0.25, 0.25],
        horizontal_spacing=0.035,
        vertical_spacing=0.025,
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["Close"],
            mode="lines",
            name="終値",
            line={"color": "#17243A", "width": 2.5},
        ),
        row=1,
        col=1,
    )
    for column, label, color in (
        ("SMA_25", "25日線", "#E67E22"),
        ("SMA_75", "75日線", "#2980B9"),
        ("SMA_200", "200日線", "#8E44AD"),
    ):
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data[column],
                mode="lines",
                name=label,
                line={"color": color, "width": 1.7},
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Bar(
            x=chart_data.index,
            y=chart_data["Volume"],
            name="出来高",
            marker_color="#AAB6C4",
        ),
        row=4,
        col=1,
    )
    figure.update_yaxes(
        title_text="株価",
        tickfont={"size": 11},
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="出来高",
        tickfont={"size": 10},
        row=4,
        col=1,
    )
    figure.update_xaxes(
        showticklabels=False,
        row=1,
        col=1,
    )
    figure.update_xaxes(
        tickfont={"size": 10},
        row=4,
        col=1,
    )

    gauge_positions = (
        ("MACD", 1, 2),
        ("RSI", 1, 3),
        ("ADX", 3, 2),
        ("Fear & Greed", 3, 3),
    )
    for name, row, column in gauge_positions:
        value = latest_values[name]
        style = _stock_card_gauge_style(name, value)
        figure.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={
                    "text": name,
                    "font": {
                        "family": CHART_FONT_FAMILY,
                        "size": 17,
                        "color": "#17243A",
                    },
                },
                number={
                    "valueformat": str(style["valueformat"]),
                    "suffix": str(style["suffix"]),
                    "font": {
                        "family": CHART_FONT_FAMILY,
                        "size": 30,
                        "color": "#17243A",
                    },
                },
                gauge={
                    "axis": {
                        "range": style["range"],
                        "tickwidth": 1,
                        "tickcolor": "#98A2B3",
                    },
                    "bar": {
                        "color": style["color"],
                        "thickness": 0.28,
                    },
                    "bgcolor": "#E9EEF5",
                    "borderwidth": 0,
                },
                meta={"status": style["state"]},
            ),
            row=row,
            col=column,
        )

    figure.update_layout(
        width=POSTPRIME_STOCK_CARD_WIDTH,
        height=POSTPRIME_STOCK_CARD_HEIGHT,
        autosize=False,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FBFCFE",
        font={
            "family": CHART_FONT_FAMILY,
            "color": "#17243A",
        },
        title={
            "text": f"{ticker} {company} | 本日のテクニカル分析 ({date_label})",
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "family": CHART_FONT_FAMILY,
                "size": 27,
                "color": "#17243A",
            },
        },
        margin={"l": 48, "r": 38, "t": 84, "b": 35},
        showlegend=True,
        legend={
            "orientation": "h",
            "x": 0.02,
            "y": 0.985,
            "font": {"size": 12},
        },
        bargap=0.18,
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor="#E1E6ED",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#E1E6ED",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        showgrid=False,
        row=4,
        col=1,
    )
    return figure


def render_postprime_stock_card_png(
    indicators: pd.DataFrame,
    ticker: str,
    company: str,
) -> bytes:
    """Render the stock card at exactly 1600 by 900 pixels."""
    figure = _prepare_postprime_card_figure(
        build_postprime_stock_card_figure(indicators, ticker, company)
    )
    return figure.to_image(
        format="png",
        width=POSTPRIME_STOCK_CARD_WIDTH,
        height=POSTPRIME_STOCK_CARD_HEIGHT,
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
