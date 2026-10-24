"""TOPIX-17 sector flow data, metrics, and chart builders."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from chart_format import CHART_FONT_FAMILY

SECTOR_ETFS: dict[str, str] = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信・サービスその他",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融（除く銀行）",
    "1633.T": "不動産",
}
MARKET_TICKER = "1306.T"
MARKET_COLUMN = "TOPIX"
RS_LOOKBACK = 20
RS_MOMENTUM_LOOKBACK = 10

_QUADRANT_COLORS = {
    "Leading": "#2e8b57",
    "Weakening": "#e0a326",
    "Lagging": "#d9534f",
    "Improving": "#4386c6",
}


def _extract_close_prices(
    downloaded: pd.DataFrame, ticker_labels: Mapping[str, str]
) -> pd.DataFrame:
    """Extract adjusted closes from either yfinance MultiIndex column order."""
    if downloaded.empty:
        raise ValueError("Yahoo Financeから株価データを取得できませんでした。")

    close_prices: dict[str, pd.Series] = {}
    for ticker, label in ticker_labels.items():
        matching_columns = []
        if isinstance(downloaded.columns, pd.MultiIndex):
            for column in downloaded.columns:
                parts = [str(part) for part in column]
                if ticker in parts and any(part.lower() == "close" for part in parts):
                    matching_columns.append(column)
        elif len(ticker_labels) == 1:
            matching_columns = [
                column
                for column in downloaded.columns
                if str(column).lower() == "close"
            ]

        close_prices[label] = (
            pd.to_numeric(downloaded[matching_columns[0]], errors="coerce")
            if matching_columns
            else pd.Series(np.nan, index=downloaded.index, dtype=float)
        )

    prices = pd.DataFrame(close_prices, index=downloaded.index)
    prices = prices.sort_index().dropna(how="all")
    if MARKET_COLUMN not in prices or prices[MARKET_COLUMN].notna().sum() == 0:
        raise ValueError(f"市場ベンチマーク（{MARKET_TICKER}）の終値がありません。")
    if prices.drop(columns=MARKET_COLUMN).notna().any().sum() == 0:
        raise ValueError("利用可能なセクターETFの終値がありません。")
    return prices


def fetch_sector_close_prices() -> pd.DataFrame:
    """Download one year of adjusted daily closes for TOPIX-17 ETFs and TOPIX."""
    import yfinance as yf
    from yfinance.exceptions import YFException

    ticker_labels = {**SECTOR_ETFS, MARKET_TICKER: MARKET_COLUMN}
    try:
        downloaded = yf.download(
            tickers=list(ticker_labels),
            period="1y",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except (OSError, RuntimeError, ValueError, YFException) as exc:
        raise RuntimeError(
            f"Yahoo Financeからセクターデータを取得できません: {exc}"
        ) from exc
    return _extract_close_prices(downloaded, ticker_labels)


def calculate_sector_metrics(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Build the shared long-form return and relative-momentum dataset."""
    if not isinstance(close_prices, pd.DataFrame) or close_prices.empty:
        raise ValueError("セクター株価データが空です。")
    if MARKET_COLUMN not in close_prices:
        raise ValueError(f"市場ベンチマーク列 {MARKET_COLUMN} がありません。")

    market = pd.to_numeric(close_prices[MARKET_COLUMN], errors="coerce")
    metrics_by_sector: list[pd.DataFrame] = []
    for sector in close_prices.columns:
        if sector == MARKET_COLUMN:
            continue

        pair = pd.DataFrame(
            {
                "Close": pd.to_numeric(close_prices[sector], errors="coerce"),
                "Market_Close": market,
            }
        ).dropna()
        if len(pair) < 43:
            continue

        relative_strength = pair["Close"] / pair["Market_Close"].where(
            pair["Market_Close"] != 0
        )
        rs_ratio = (
            relative_strength
            / relative_strength.rolling(RS_LOOKBACK, min_periods=10).mean()
            * 100
        )
        pair["RS_Ratio"] = rs_ratio
        pair["RS_Momentum"] = (
            rs_ratio / rs_ratio.shift(RS_MOMENTUM_LOOKBACK) * 100
        )
        pair["Return_1D"] = pair["Close"].pct_change() * 100
        pair["Return_5D"] = pair["Close"].pct_change(5) * 100
        pair["Return_PreviousWeek"] = pair["Close"].shift(5).pct_change(5) * 100
        pair["Return_PreviousMonth"] = pair["Close"].shift(21).pct_change(21) * 100
        pair["Momentum_20D"] = pair["Close"].pct_change(20) * 100
        relative_return_5d = relative_strength.pct_change(5)
        pair["EarlySignal"] = (
            pair["Return_5D"].gt(0)
            & pair["Return_PreviousWeek"].le(0)
            & relative_return_5d.gt(0)
        )
        pair["Sector"] = str(sector)
        metrics_by_sector.append(
            pair.drop(columns="Market_Close").reset_index(names="Date")
        )

    if not metrics_by_sector:
        raise ValueError("指標計算に必要な日次データが43営業日分ありません。")
    return pd.concat(metrics_by_sector, ignore_index=True).sort_values(
        ["Date", "Sector"], ignore_index=True
    )


def _latest_by_sector(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "Sector"}
    if not required.issubset(metrics.columns):
        raise ValueError("セクター指標データに Date / Sector 列が必要です。")
    return (
        metrics.sort_values("Date")
        .groupby("Sector", sort=False, as_index=False)
        .tail(1)
        .set_index("Sector")
    )


def build_rrg_figure(metrics: pd.DataFrame) -> go.Figure:
    latest = _latest_by_sector(metrics).dropna(subset=["RS_Ratio", "RS_Momentum"])
    if latest.empty:
        raise ValueError("RRGを描画できる相対指標データがありません。")

    x_values = latest["RS_Ratio"].astype(float)
    y_values = latest["RS_Momentum"].astype(float)
    x_min, x_max = min(float(x_values.min()), 100.0), max(float(x_values.max()), 100.0)
    y_min, y_max = min(float(y_values.min()), 100.0), max(float(y_values.max()), 100.0)
    x_padding = max((x_max - x_min) * 0.15, 1.0)
    y_padding = max((y_max - y_min) * 0.15, 1.0)
    x_range = [x_min - x_padding, x_max + x_padding]
    y_range = [y_min - y_padding, y_max + y_padding]

    figure = go.Figure()
    quadrants = (
        ("Leading", x_values.ge(100) & y_values.ge(100)),
        ("Weakening", x_values.ge(100) & y_values.lt(100)),
        ("Lagging", x_values.lt(100) & y_values.lt(100)),
        ("Improving", x_values.lt(100) & y_values.ge(100)),
    )
    for label, condition in quadrants:
        selected = latest.loc[condition]
        if selected.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=selected["RS_Ratio"],
                y=selected["RS_Momentum"],
                text=selected.index,
                mode="markers+text",
                textposition="top center",
                name=label,
                marker={"size": 12, "color": _QUADRANT_COLORS[label]},
                hovertemplate=(
                    "%{text}<br>RS-Ratio: %{x:.2f}"
                    "<br>RS-Momentum: %{y:.2f}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    for x0, x1, y0, y1, label in (
        (x_range[0], 100, 100, y_range[1], "Leading"),
        (100, x_range[1], 100, y_range[1], "Weakening"),
        (x_range[0], 100, y_range[0], 100, "Improving"),
        (100, x_range[1], y_range[0], 100, "Lagging"),
    ):
        figure.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor=_QUADRANT_COLORS[label],
            opacity=0.06,
            line_width=0,
            layer="below",
        )
    figure.add_hline(y=100, line_dash="dash", line_color="#666")
    figure.add_vline(x=100, line_dash="dash", line_color="#666")
    figure.update_layout(
        title="セクター相対強度ローテーション（RRG）",
        xaxis_title="相対強度（RS-Ratio）",
        yaxis_title="相対モメンタム（RS-Momentum）",
        xaxis={"range": x_range, "zeroline": False},
        yaxis={"range": y_range, "zeroline": False},
        font={"family": CHART_FONT_FAMILY},
        height=560,
        hovermode="closest",
        margin={"t": 70, "b": 50, "l": 70, "r": 30},
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def build_ranking_figure(metrics: pd.DataFrame) -> go.Figure:
    latest = _latest_by_sector(metrics).dropna(subset=["Momentum_20D"])
    if latest.empty:
        raise ValueError("ランキングを描画できる20日モメンタムがありません。")
    extremes = pd.concat(
        [
            latest.nsmallest(5, "Momentum_20D"),
            latest.nlargest(5, "Momentum_20D"),
        ]
    )
    extremes = extremes.loc[~extremes.index.duplicated(keep="first")].sort_values(
        "Momentum_20D"
    )
    values = extremes["Momentum_20D"].astype(float)
    figure = go.Figure(
        go.Bar(
            x=values,
            y=extremes.index,
            orientation="h",
            marker_color=[
                "#d9534f" if value >= 0 else "#4386c6" for value in values
            ],
            text=[f"{value:+.1f}%" for value in values],
            textposition="outside",
            hovertemplate="%{y}<br>20日リターン: %{x:.2f}%<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color="#666", line_dash="dash")
    figure.update_layout(
        title="20営業日モメンタム TOP5 / ワースト5",
        xaxis_title="20営業日リターン（%）",
        yaxis_title=None,
        font={"family": CHART_FONT_FAMILY},
        height=480,
        margin={"t": 70, "b": 45, "l": 180, "r": 70},
        showlegend=False,
    )
    return figure


def build_matrix_figure(metrics: pd.DataFrame) -> go.Figure:
    latest = _latest_by_sector(metrics).sort_values(
        "Momentum_20D", ascending=False, na_position="last"
    )
    if latest.empty:
        raise ValueError("比較マトリックスを描画するデータがありません。")

    values: list[list[str]] = [[str(sector) for sector in latest.index]]
    for metric in (
        "Return_5D",
        "Return_PreviousWeek",
        "Return_PreviousMonth",
    ):
        values.append(
            [
                f"{value:+.2f}%"
                if pd.notna(value) and np.isfinite(value)
                else "—"
                for value in latest[metric]
            ]
        )
    values.append(["初動" if value else "—" for value in latest["EarlySignal"]])

    row_colors = [
        "#f5f7fa" if index % 2 == 0 else "#ffffff"
        for index in range(len(latest))
    ]
    early_colors = [
        "#f8d7da" if value else row_colors[index]
        for index, value in enumerate(latest["EarlySignal"])
    ]
    figure = go.Figure(
        go.Table(
            header={
                "values": [
                    "セクター",
                    "直近5日リターン",
                    "前週5日リターン",
                    "前月リターン",
                    "初動判定",
                ],
                "fill_color": "#26374a",
                "font": {"color": "white", "size": 13, "family": CHART_FONT_FAMILY},
                "align": "center",
                "height": 36,
            },
            cells={
                "values": values,
                "fill_color": [
                    row_colors,
                    row_colors,
                    row_colors,
                    row_colors,
                    early_colors,
                ],
                "font": {"color": "#202833", "size": 12, "family": CHART_FONT_FAMILY},
                "align": ["left", "center", "center", "center", "center"],
                "height": 30,
            },
        )
    )
    figure.update_layout(
        title="多期間比較マトリックス",
        font={"family": CHART_FONT_FAMILY},
        height=650,
        margin={"t": 65, "b": 15, "l": 10, "r": 10},
    )
    return figure


def get_rrg_comment(metrics: pd.DataFrame) -> str:
    latest = _latest_by_sector(metrics).dropna(subset=["RS_Ratio", "RS_Momentum"])
    leading = int(
        (latest["RS_Ratio"].ge(100) & latest["RS_Momentum"].ge(100)).sum()
    )
    improving = int((latest["RS_Ratio"].lt(100) & latest["RS_Momentum"].ge(100)).sum())
    return (
        f"現在は「先導」象限が{leading}業種、「改善」象限が{improving}業種です。"
        "市場に対する相対強度とその勢いを示す参考値です。"
    )


def get_ranking_comment(metrics: pd.DataFrame) -> str:
    latest = _latest_by_sector(metrics).dropna(subset=["Momentum_20D"])
    if latest.empty:
        return "ランキング対象となるデータがありません。"
    strongest = latest["Momentum_20D"].idxmax()
    weakest = latest["Momentum_20D"].idxmin()
    return (
        f"20営業日モメンタムの最上位は「{strongest}」、"
        f"最下位は「{weakest}」です。業種ETFの値動きに基づく集計です。"
    )


def get_matrix_comment(metrics: pd.DataFrame) -> str:
    latest = _latest_by_sector(metrics)
    early_count = int(latest["EarlySignal"].fillna(False).sum())
    return (
        f"初動フラグは{early_count}業種です。"
        "直近5日が上昇、前週5日が非上昇、かつ市場比の直近5日リターンが"
        "プラスの場合に点灯します。"
    )
