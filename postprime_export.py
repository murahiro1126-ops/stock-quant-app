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


def _stock_card_gauge_flag(
    name: str,
    indicators: pd.DataFrame,
) -> str | None:
    """Return the highest-priority recent event label for one gauge."""
    if not isinstance(indicators, pd.DataFrame) or indicators.empty:
        return None

    def numeric_column(column: str) -> pd.Series | None:
        if column not in indicators:
            return None
        values = pd.to_numeric(indicators[column].tail(30), errors="coerce")
        return values.replace([np.inf, -np.inf], np.nan)

    def latest_value(values: pd.Series | None) -> float | None:
        if values is None or values.empty or pd.isna(values.iloc[-1]):
            return None
        return float(values.iloc[-1])

    def crossed(
        values: pd.Series,
        threshold: float,
        upward: bool,
        sessions: int = 3,
    ) -> bool:
        recent = values.tail(sessions + 1)
        previous = recent.shift(1)
        event = (
            (previous <= threshold) & (recent > threshold)
            if upward
            else (previous >= threshold) & (recent < threshold)
        )
        return bool(event.iloc[1:].fillna(False).any())

    def crossed_between(
        values: pd.Series,
        previous_limit: float,
        current_limit: float,
        upward: bool,
        sessions: int = 3,
    ) -> bool:
        recent = values.tail(sessions + 1)
        previous = recent.shift(1)
        event = (
            (previous <= previous_limit) & (recent >= current_limit)
            if upward
            else (previous >= previous_limit) & (recent <= current_limit)
        )
        return bool(event.iloc[1:].fillna(False).any())

    if name == "MACD":
        macd = numeric_column("MACD")
        signal = numeric_column("MACD_Signal")
        if macd is None or signal is None:
            return None
        relative = macd - signal
        if crossed(relative, 0, upward=True):
            return "GC点灯"
        if crossed(relative, 0, upward=False):
            return "DC点灯"
        if crossed(macd, 0, upward=True):
            return "ゼロ上抜"
        if crossed(macd, 0, upward=False):
            return "ゼロ下抜"
        return None

    if name == "RSI":
        rsi = numeric_column("RSI")
        if rsi is None:
            return None
        current_rsi = latest_value(rsi)
        close = numeric_column("Close")
        if current_rsi is not None and close is not None:
            history = pd.concat(
                {"close": close, "rsi": rsi},
                axis=1,
            ).dropna().tail(30)
            if len(history) == 30:
                previous_window = history.iloc[:15]
                recent_window = history.iloc[15:]
                previous_low = int(previous_window["close"].to_numpy().argmin())
                recent_low = int(recent_window["close"].to_numpy().argmin())
                if (
                    current_rsi < 40
                    and recent_window["close"].iloc[recent_low]
                    < previous_window["close"].iloc[previous_low]
                    and recent_window["rsi"].iloc[recent_low]
                    > previous_window["rsi"].iloc[previous_low]
                ):
                    return "強気逆行"
                previous_high = int(previous_window["close"].to_numpy().argmax())
                recent_high = int(recent_window["close"].to_numpy().argmax())
                if (
                    current_rsi > 60
                    and recent_window["close"].iloc[recent_high]
                    > previous_window["close"].iloc[previous_high]
                    and recent_window["rsi"].iloc[recent_high]
                    < previous_window["rsi"].iloc[previous_high]
                ):
                    return "弱気逆行"
        if crossed(rsi, 50, upward=True):
            return "50上抜け"
        if crossed(rsi, 50, upward=False):
            return "50割れ"
        return None

    if name == "ADX":
        adx = numeric_column("ADX")
        if adx is None:
            return None
        current_adx = latest_value(adx)
        atr_percent = latest_value(numeric_column("ATR_Percent"))
        ranging = (
            indicators["Is_Ranging"].iloc[-1]
            if "Is_Ranging" in indicators
            else False
        )
        if (
            pd.notna(ranging)
            and bool(ranging)
            and current_adx is not None
            and current_adx < 20
            and atr_percent is not None
            and atr_percent < 1.5
        ):
            return "レンジ警戒"
        if crossed(adx, 20, upward=True):
            return "発生初動"
        recent = adx.tail(4)
        changes = recent.diff().iloc[1:]
        if (
            len(changes) == 3
            and not changes.isna().any()
            and current_adx is not None
            and current_adx >= 25
            and (changes > 0).all()
        ):
            return "勢い加速"
        if (
            len(changes) == 3
            and not changes.isna().any()
            and current_adx is not None
            and current_adx >= 30
            and (changes < 0).all()
        ):
            return "勢い一服"
        return None

    if name == "Fear & Greed":
        score = numeric_column("Fear_Greed_Score")
        if score is None:
            return None
        current_score = latest_value(score)
        if current_score is None:
            return None
        if len(score) >= 6 and pd.notna(score.iloc[-6]):
            change = current_score - float(score.iloc[-6])
            if change <= -15:
                return "急冷え込み"
            if change >= 15:
                return "急過熱"
        if crossed_between(score, 24, 25, upward=True):
            return "底打ち反転"
        if crossed_between(score, 75, 74, upward=False):
            return "過熱一服"
    return None


def _stock_card_gauge_style(
    name: str,
    value: float | None,
) -> dict[str, object]:
    """Map each indicator value to its gauge range, zones, and state label."""
    if name == "MACD":
        radius = max(1.0, abs(value) * 1.5) if value is not None else 1.0
        neutral_limit = max(radius * 0.1, 0.05)
        color = "#16A085" if value is not None and value > neutral_limit else (
            "#E74C3C"
            if value is not None and value < -neutral_limit
            else "#F39C12"
            if value is not None
            else "#98A2B3"
        )
        state = (
            "プラス圏"
            if value is not None and value > neutral_limit
            else "マイナス圏"
            if value is not None and value < -neutral_limit
            else "転換付近"
            if value is not None
            else "取得不可"
        )
        value_range = [-radius, radius]
        steps = [
            {"range": [-radius, -neutral_limit], "color": "#F8D8D5"},
            {"range": [-neutral_limit, neutral_limit], "color": "#FCE8C7"},
            {"range": [neutral_limit, radius], "color": "#D6F1E8"},
        ]
        arc_zones = [
            (-radius, -neutral_limit, "#2563EB", "#DBEAFE"),
            (-neutral_limit, neutral_limit, "#475569", "#E2E8F0"),
            (neutral_limit, radius, "#16A34A", "#DCFCE7"),
        ]
        tick_values = [-radius, 0, radius]
        value_format = ".2f"
        suffix = ""
        state_color = (
            "#16A34A"
            if value is not None and value > neutral_limit
            else "#2563EB"
            if value is not None and value < -neutral_limit
            else "#475569"
        )
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
        steps = [
            {"range": [0, 30], "color": "#F8D8D5"},
            {"range": [30, 70], "color": "#E5EBF2"},
            {"range": [70, 100], "color": "#F8D8D5"},
        ]
        arc_zones = [
            (0, 30, "#2563EB", "#DBEAFE"),
            (30, 70, "#64748B", "#E2E8F0"),
            (70, 100, "#DC2626", "#FEE2E2"),
        ]
        tick_values = [0, 30, 70, 100]
        value_format = ".1f"
        suffix = ""
        state_color = (
            "#DC2626"
            if value is not None and value >= 70
            else "#2563EB"
            if value is not None and value <= 30
            else "#64748B"
        )
    elif name == "ADX":
        color = (
            "#16A085"
            if value is not None and value >= 40
            else "#F39C12"
            if value is not None and value >= 20
            else "#98A2B3"
        )
        state = (
            "非常に強い"
            if value is not None and value >= 40
            else "トレンド発生"
            if value is not None and value >= 20
            else "レンジ気味"
            if value is not None
            else "取得不可"
        )
        value_range = [0, 100]
        steps = [
            {"range": [0, 20], "color": "#E5EBF2"},
            {"range": [20, 40], "color": "#FCE8C7"},
            {"range": [40, 100], "color": "#D6F1E8"},
        ]
        arc_zones = [
            (0, 20, "#64748B", "#E2E8F0"),
            (20, 40, "#D97706", "#FEF3C7"),
            (40, 100, "#059669", "#D1FAE5"),
        ]
        tick_values = [0, 20, 40, 100]
        value_format = ".1f"
        suffix = ""
        state_color = (
            "#059669"
            if value is not None and value >= 40
            else "#D97706"
            if value is not None and value >= 20
            else "#64748B"
        )
    else:
        rounded_score = (
            int(np.clip(np.floor(value + 0.5), 0, 100))
            if value is not None
            else None
        )
        if rounded_score is None:
            color, state = "#98A2B3", "取得不可"
        elif rounded_score <= 20:
            color, state = "#DC2626", "極度の恐怖"
        elif rounded_score <= 40:
            color, state = "#D97706", "恐怖"
        elif rounded_score <= 60:
            color, state = "#64748B", "中立"
        elif rounded_score <= 80:
            color, state = "#16A34A", "貪欲"
        else:
            color, state = "#C2410C", "極度の熱狂"
        value_range = [0, 100]
        steps = [
            {"range": [0, 20], "color": "#FEE2E2"},
            {"range": [20, 40], "color": "#FEF3C7"},
            {"range": [40, 60], "color": "#E2E8F0"},
            {"range": [60, 80], "color": "#DCFCE7"},
            {"range": [80, 100], "color": "#FFEDD5"},
        ]
        arc_zones = [
            (0, 20, "#DC2626", "#FEE2E2"),
            (20, 40, "#D97706", "#FEF3C7"),
            (40, 60, "#64748B", "#E2E8F0"),
            (60, 80, "#16A34A", "#DCFCE7"),
            (80, 100, "#C2410C", "#FFEDD5"),
        ]
        tick_values = [0, 20, 40, 60, 80, 100]
        value_format = ".0f"
        suffix = "/100"
        state_color = color
    return {
        "range": value_range,
        "color": color,
        "state": state,
        "steps": steps,
        "arc_zones": arc_zones,
        "state_color": state_color,
        "tickvals": tick_values,
        "valueformat": value_format,
        "suffix": suffix,
    }


def _stock_card_gauge_visuals(
    name: str,
    value: float | None,
    style: Mapping[str, object],
    domain: tuple[list[float], list[float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Create the colored gauge arcs, ticks, and labels for one stock-card meter."""
    x_domain, y_domain = domain
    center_x = sum(x_domain) / 2
    baseline_y = y_domain[0] + 0.105
    radius_x = 0.053
    radius_y = 0.1
    is_macd = name == "MACD"
    zones = style["arc_zones"]
    if not isinstance(zones, list):
        raise TypeError("ゲージの色ゾーン設定が不正です。")

    def angle_for(raw_value: float) -> float:
        if is_macd:
            macd_range = style["range"]
            if not isinstance(macd_range, list):
                raise TypeError("MACDゲージの範囲設定が不正です。")
            radius = float(macd_range[1])
            return np.pi / 2 - raw_value / radius * np.pi / 2
        return np.pi * (1 - raw_value / 100)

    shapes: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []

    def add_arc(start_value: float, end_value: float, color: str, width: int) -> None:
        start_angle, end_angle = sorted(
            (angle_for(start_value), angle_for(end_value))
        )
        angles = np.linspace(start_angle, end_angle, 25)
        points = [
            (
                center_x + radius_x * float(np.cos(angle)),
                baseline_y + radius_y * float(np.sin(angle)),
            )
            for angle in angles
        ]
        path = "M " + " L ".join(f"{x:.6f},{y:.6f}" for x, y in points)
        shapes.append(
            {
                "type": "path",
                "xref": "paper",
                "yref": "paper",
                "path": path,
                "line": {"color": color, "width": width},
                "layer": "above",
            }
        )

    for zone in zones:
        low, high, active_color, track_color = zone
        low_value = float(low)
        high_value = float(high)
        if high_value <= low_value:
            continue
        if is_macd and low_value < 0 < high_value and value != 0:
            continue
        add_arc(low_value, high_value, str(track_color), 28)

    if value is not None:
        if is_macd:
            progress_low, progress_high = sorted((0.0, float(value)))
        else:
            progress_low, progress_high = 0.0, float(np.clip(value, 0, 100))
        for zone in zones:
            low, high, active_color, _ = zone
            low_value = float(low)
            high_value = float(high)
            if high_value <= low_value:
                continue
            segment_low = max(low_value, progress_low)
            segment_high = min(high_value, progress_high)
            if segment_high <= segment_low:
                continue
            progress_color = str(active_color)
            if is_macd and low_value < 0 < high_value and value != 0:
                progress_color = "#16A34A" if value > 0 else "#2563EB"
            add_arc(segment_low, segment_high, progress_color, 9)

    tick_values = style["tickvals"]
    if not isinstance(tick_values, list):
        raise TypeError("ゲージの目盛り設定が不正です。")
    for tick in tick_values:
        tick_value = float(tick)
        angle = angle_for(tick_value)
        cos_angle = float(np.cos(angle))
        sin_angle = float(np.sin(angle))
        tick_inner_x = center_x + (radius_x + 0.006) * cos_angle
        tick_inner_y = baseline_y + (radius_y + 0.006) * sin_angle
        tick_outer_x = center_x + (radius_x + 0.012) * cos_angle
        tick_outer_y = baseline_y + (radius_y + 0.012) * sin_angle
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": tick_inner_x,
                "y0": tick_inner_y,
                "x1": tick_outer_x,
                "y1": tick_outer_y,
                "line": {"color": "#64748B", "width": 1.5},
                "layer": "above",
            }
        )
        tick_label = f"{tick_value:.3g}" if is_macd else f"{tick_value:g}"
        label_radius_offset = 0.013 if abs(cos_angle) > 0.85 else 0.025
        label_x = center_x + (radius_x + label_radius_offset) * cos_angle
        label_y = baseline_y + (radius_y + label_radius_offset) * sin_angle
        if abs(cos_angle) > 0.85:
            label_xanchor = "right" if cos_angle < 0 else "left"
            label_yanchor = "middle"
        else:
            label_xanchor = "center"
            label_yanchor = "bottom"
        annotations.append(
            {
                "text": tick_label,
                "xref": "paper",
                "yref": "paper",
                "x": label_x,
                "y": label_y,
                "showarrow": False,
                "xanchor": label_xanchor,
                "yanchor": label_yanchor,
                "font": {
                    "family": CHART_FONT_FAMILY,
                    "size": 13,
                    "color": "#475467",
                    "weight": 600,
                },
            }
        )

    if value is None:
        display_value = "N/A"
    elif is_macd:
        display_value = f"{value:.3g}"
        if len(display_value) > 5 or "e" in display_value.lower():
            mantissa, exponent = f"{value:.2e}".split("e")
            display_value = f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    elif name in {"RSI", "ADX"}:
        display_value = f"{value:.1f}"
    else:
        display_value = f"{value:.0f}"
    value_font_size = (
        max(12, int(38 * 3 / len(display_value)))
        if is_macd
        else 38
    )
    annotations.extend(
        [
            {
                "text": name,
                "xref": "paper",
                "yref": "paper",
                "x": center_x,
                "y": y_domain[1] + 0.035,
                "showarrow": False,
                "xanchor": "center",
                "yanchor": "top",
                "font": {
                    "family": CHART_FONT_FAMILY,
                    "size": 23,
                    "color": "#17243A",
                    "weight": 700,
                },
            },
            {
                "text": display_value,
                "xref": "paper",
                "yref": "paper",
                "x": center_x,
                "y": baseline_y,
                "showarrow": False,
                "xanchor": "center",
                "yanchor": "middle",
                "font": {
                    "family": CHART_FONT_FAMILY,
                    "size": value_font_size,
                    "color": "#17243A",
                    "weight": 700,
                },
            },
        ]
    )
    return shapes, annotations


def build_postprime_stock_card_figure(
    indicators: pd.DataFrame,
    ticker: str,
    company: str,
    *,
    range_position: float | None = None,
    buy_confidence: int | None = None,
    sell_confidence: int | None = None,
    rank: str | None = None,
) -> go.Figure:
    """Build a 16:9 stock card with a summary bar and four indicator gauges."""
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
    chart_data = indicators.tail(252)
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
        tickfont={"size": 16, "weight": 700},
        title_font={"size": 17, "weight": 700},
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="出来高",
        tickfont={"size": 15, "weight": 700},
        title_font={"size": 16, "weight": 700},
        row=4,
        col=1,
    )
    figure.update_xaxes(
        showticklabels=False,
        row=1,
        col=1,
    )
    figure.update_xaxes(
        tickfont={"size": 15, "weight": 700},
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
                        "size": 23,
                        "color": "#17243A",
                        "weight": 700,
                    },
                },
                number={
                    "valueformat": str(style["valueformat"]),
                    "suffix": str(style["suffix"]),
                    "font": {
                        "family": CHART_FONT_FAMILY,
                        "size": 38,
                        "color": "#17243A",
                        "weight": 700,
                    },
                },
                gauge={
                    "axis": {
                        "range": style["range"],
                        "tickvals": style["tickvals"],
                        "tickwidth": 1,
                        "tickcolor": "#98A2B3",
                        "tickfont": {
                            "family": CHART_FONT_FAMILY,
                            "size": 11 if name == "MACD" else 13,
                            "color": "#667085",
                        },
                    },
                    "bar": {
                        "color": "rgba(0,0,0,0)",
                        "thickness": 0.12,
                    },
                    "steps": style["steps"],
                    "threshold": (
                        {
                            "value": value,
                            "line": {"color": style["color"], "width": 0},
                            "thickness": 0,
                        }
                        if value is not None
                        else {}
                    ),
                    "bgcolor": "#E9EEF5",
                    "borderwidth": 0,
                },
                meta={"status": style["state"]},
                visible=False,
            ),
            row=row,
            col=column,
        )
    gauge_domains = {
        "MACD": ([0.65, 0.81], [0.52, 0.77]),
        "RSI": ([0.83, 0.99], [0.52, 0.77]),
        "ADX": ([0.65, 0.81], [0.17, 0.42]),
        "Fear & Greed": ([0.83, 0.99], [0.17, 0.42]),
    }
    gauge_flags = {
        name: _stock_card_gauge_flag(name, indicators)
        for name, _, _ in gauge_positions
    }
    gauge_status_labels: dict[str, tuple[str, int]] = {}
    for name, _, _ in gauge_positions:
        state = str(_stock_card_gauge_style(name, latest_values[name])["state"])
        flag = gauge_flags[name]
        text = f"{state} · {flag}" if flag is not None else state
        font_size = max(18, min(21, round(21 * 9 / max(9, len(text)))))
        gauge_status_labels[name] = (text, font_size)
    for trace in figure.data:
        if trace.type != "indicator":
            continue
        x_domain, y_domain = gauge_domains[trace.title.text]
        trace.domain = {"x": x_domain, "y": y_domain}

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
            "y": 0.985,
            "yanchor": "top",
            "font": {
                "family": CHART_FONT_FAMILY,
                "size": 27,
                "color": "#17243A",
            },
        },
        margin={"l": 48, "r": 28, "t": 24, "b": 30},
        showlegend=True,
        legend={
            "orientation": "h",
            "x": 0.02,
            "y": 0.805,
            "font": {
                "size": 16,
                "weight": 700,
                "family": CHART_FONT_FAMILY,
            },
        },
        bargap=0.18,
    )
    figure.layout.xaxis.domain = [0.06, 0.61]
    figure.layout.yaxis.domain = [0.205, 0.785]
    figure.layout.xaxis2.domain = [0.06, 0.61]
    figure.layout.yaxis2.domain = [0.075, 0.175]
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
    range_value = (
        float(range_position)
        if range_position is not None and np.isfinite(range_position)
        else None
    )
    range_label = f"{range_value:.1f}%" if range_value is not None else "N/A"
    range_track_left = 0.055
    range_track_right = 0.345
    range_x = (
        range_track_left
        + (range_track_right - range_track_left)
        * float(np.clip(range_value, 0, 100))
        / 100
        if range_value is not None
        else range_track_left
    )
    buy_label = f"{buy_confidence}%" if buy_confidence is not None else "N/A"
    sell_label = f"{sell_confidence}%" if sell_confidence is not None else "N/A"
    buy_value = (
        float(np.clip(buy_confidence, 0, 100))
        if buy_confidence is not None
        else None
    )
    buy_track_left = 0.41
    buy_track_right = 0.69
    buy_x = (
        buy_track_left
        + (buy_track_right - buy_track_left) * buy_value / 100
        if buy_value is not None
        else buy_track_left
    )
    rank_label = f"Rank {rank}" if rank else "Rank N/A"
    figure.update_layout(
        shapes=[
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "x0": 0.04,
                "x1": 0.96,
                "y0": 0.825,
                "y1": 0.955,
                "line": {"color": "#D8E0EA", "width": 1},
                "fillcolor": "#F7F9FC",
                "layer": "below",
            },
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": 0.055,
                "x1": range_track_right,
                "y0": 0.875,
                "y1": 0.875,
                "line": {"color": "#D8E0EA", "width": 8},
            },
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": 0.055,
                "x1": range_x,
                "y0": 0.875,
                "y1": 0.875,
                "line": {
                    "color": "#1E293B",
                    "width": 8,
                },
            },
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": buy_track_left,
                "x1": buy_track_right,
                "y0": 0.875,
                "y1": 0.875,
                "line": {"color": "#D8E0EA", "width": 8},
            },
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": buy_track_left,
                "x1": buy_x,
                "y0": 0.875,
                "y1": 0.875,
                "line": {
                    "color": "#1E293B",
                    "width": 8,
                },
            },
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": 0.39,
                "x1": 0.39,
                "y0": 0.855,
                "y1": 0.94,
                "line": {"color": "#D8E0EA", "width": 1},
            },
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": 0.72,
                "x1": 0.72,
                "y0": 0.855,
                "y1": 0.94,
                "line": {"color": "#D8E0EA", "width": 1},
            },
        ],
        annotations=[
            {
                "text": "5年レンジ位置",
                "xref": "paper",
                "yref": "paper",
                "x": 0.055,
                "y": 0.93,
                "xanchor": "left",
                "showarrow": False,
                "font": {
                    "size": 22,
                    "color": "#475467",
                    "weight": 700,
                    "family": CHART_FONT_FAMILY,
                },
            },
            {
                "text": range_label,
                "xref": "paper",
                "yref": "paper",
                "x": 0.345,
                "y": 0.93,
                "xanchor": "right",
                "showarrow": False,
                "font": {
                    "size": 28,
                    "color": "#17243A",
                    "weight": 700,
                    "family": CHART_FONT_FAMILY,
                },
            },
            {
                "text": "買い確度",
                "xref": "paper",
                "yref": "paper",
                "x": buy_track_left,
                "y": 0.93,
                "xanchor": "left",
                "showarrow": False,
                "font": {
                    "size": 20,
                    "color": "#475467",
                    "family": CHART_FONT_FAMILY,
                    "weight": 700,
                },
            },
            {
                "text": buy_label,
                "xref": "paper",
                "yref": "paper",
                "x": buy_track_right,
                "y": 0.93,
                "xanchor": "right",
                "showarrow": False,
                "font": {
                    "size": 25,
                    "color": "#17243A",
                    "family": CHART_FONT_FAMILY,
                    "weight": 700,
                },
            },
            {
                "text": f"売り確度 {sell_label}（独立スコア）",
                "xref": "paper",
                "yref": "paper",
                "x": 0.55,
                "y": 0.855,
                "xanchor": "center",
                "showarrow": False,
                "font": {
                    "size": 17,
                    "color": "#475467",
                    "family": CHART_FONT_FAMILY,
                    "weight": 600,
                },
            },
            {
                "text": "AI信頼度",
                "xref": "paper",
                "yref": "paper",
                "x": 0.755,
                "y": 0.93,
                "xanchor": "left",
                "showarrow": False,
                "font": {
                    "size": 22,
                    "color": "#475467",
                    "family": CHART_FONT_FAMILY,
                    "weight": 700,
                },
            },
            {
                "text": rank_label,
                "xref": "paper",
                "yref": "paper",
                "x": 0.94,
                "y": 0.905,
                "xanchor": "right",
                "showarrow": False,
                "font": {
                    "size": 28,
                    "color": "#17243A",
                    "family": CHART_FONT_FAMILY,
                    "weight": 700,
                },
            },
            *[
                {
                    "text": gauge_status_labels[name][0],
                    "xref": "paper",
                    "yref": "paper",
                    "x": sum(gauge_domains[name][0]) / 2,
                    "y": gauge_domains[name][1][0] + 0.005,
                    "xanchor": "center",
                    "showarrow": False,
                    "font": {
                        "family": CHART_FONT_FAMILY,
                        "size": gauge_status_labels[name][1],
                        "color": "#1E293B",
                        "weight": 700,
                    },
                }
                for name, _, _ in gauge_positions
            ],
        ],
    )
    visual_shapes: list[dict[str, object]] = []
    visual_annotations: list[dict[str, object]] = []
    for name, _, _ in gauge_positions:
        shapes, annotations = _stock_card_gauge_visuals(
            name,
            latest_values[name],
            _stock_card_gauge_style(name, latest_values[name]),
            gauge_domains[name],
        )
        visual_shapes.extend(shapes)
        visual_annotations.extend(annotations)
    figure.update_layout(
        shapes=[*figure.layout.shapes, *visual_shapes],
        annotations=[*figure.layout.annotations, *visual_annotations],
    )
    return figure


def render_postprime_stock_card_png(
    indicators: pd.DataFrame,
    ticker: str,
    company: str,
    *,
    range_position: float | None = None,
    buy_confidence: int | None = None,
    sell_confidence: int | None = None,
    rank: str | None = None,
) -> bytes:
    """Render the stock card at exactly 1600 by 900 pixels."""
    figure = _prepare_postprime_card_figure(
        build_postprime_stock_card_figure(
            indicators,
            ticker,
            company,
            range_position=range_position,
            buy_confidence=buy_confidence,
            sell_confidence=sell_confidence,
            rank=rank,
        )
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
