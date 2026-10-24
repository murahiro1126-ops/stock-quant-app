"""TSE 33-industry proxy data, metrics, and chart builders."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from chart_format import CHART_FONT_FAMILY

TSE33_SECTOR_TICKERS: dict[str, tuple[str, ...]] = {
    "水産・農林業": ("1332.T", "1377.T"),
    "鉱業": ("1605.T", "1662.T"),
    "建設業": ("1801.T", "1802.T", "1803.T"),
    "食料品": ("2502.T", "2503.T", "2802.T"),
    "繊維製品": ("3401.T", "3402.T", "3101.T"),
    "パルプ・紙": ("3861.T", "3863.T"),
    "化学": ("4005.T", "4063.T", "4188.T"),
    "医薬品": ("4502.T", "4503.T", "4568.T"),
    "石油・石炭製品": ("5019.T", "5020.T"),
    "ゴム製品": ("5101.T", "5108.T", "5110.T"),
    "ガラス・土石製品": ("5201.T", "5233.T", "5333.T"),
    "鉄鋼": ("5401.T", "5411.T"),
    "非鉄金属": ("5711.T", "5713.T", "5802.T"),
    "金属製品": ("5929.T", "5938.T", "5947.T"),
    "機械": ("6301.T", "6367.T", "7011.T"),
    "電気機器": ("6501.T", "6752.T", "6758.T"),
    "輸送用機器": ("7203.T", "7267.T", "6902.T"),
    "精密機器": ("7731.T", "7733.T", "7741.T"),
    "その他製品": ("7911.T", "7832.T", "7974.T"),
    "電気・ガス業": ("9501.T", "9503.T", "9531.T"),
    "陸運業": ("9020.T", "9022.T", "9064.T"),
    "海運業": ("9101.T", "9104.T", "9107.T"),
    "空運業": ("9201.T", "9202.T"),
    "倉庫・運輸関連業": ("9069.T", "9301.T", "9364.T"),
    "情報・通信業": ("9432.T", "9433.T", "9434.T"),
    "卸売業": ("8001.T", "8002.T", "8058.T"),
    "小売業": ("3382.T", "8267.T", "9843.T"),
    "銀行業": ("8306.T", "8316.T", "8411.T"),
    "証券・商品先物取引業": ("8601.T", "8604.T", "8697.T"),
    "保険業": ("8766.T", "8725.T", "8630.T"),
    "その他金融業": ("8253.T", "8591.T", "8593.T"),
    "不動産業": ("8801.T", "8802.T", "8830.T"),
    "サービス業": ("4661.T", "6098.T", "9735.T"),
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
_QUADRANT_LABELS = {
    "Leading": "先導（右上）",
    "Weakening": "減速（右下）",
    "Lagging": "停滞（左下）",
    "Improving": "改善（左上）",
}


def _extract_close_prices(
    downloaded: pd.DataFrame,
    sector_tickers: Mapping[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Aggregate representative-stock daily returns into normalized sector indices."""
    if downloaded.empty:
        raise ValueError("Yahoo Financeから株価データを取得できませんでした。")

    tickers = {
        ticker
        for sector_symbols in sector_tickers.values()
        for ticker in sector_symbols
    }
    tickers.add(MARKET_TICKER)
    close_prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        matching_columns = []
        if isinstance(downloaded.columns, pd.MultiIndex):
            for column in downloaded.columns:
                parts = [str(part) for part in column]
                if ticker in parts and any(part.lower() == "close" for part in parts):
                    matching_columns.append(column)
        elif len(tickers) == 1:
            matching_columns = [
                column
                for column in downloaded.columns
                if str(column).lower() == "close"
            ]

        close_prices[ticker] = (
            pd.to_numeric(downloaded[matching_columns[0]], errors="coerce")
            if matching_columns
            else pd.Series(np.nan, index=downloaded.index, dtype=float)
        )

    raw_prices = pd.DataFrame(close_prices, index=downloaded.index)
    raw_prices = raw_prices.sort_index().dropna(how="all")
    if MARKET_TICKER not in raw_prices or raw_prices[MARKET_TICKER].notna().sum() == 0:
        raise ValueError(f"市場ベンチマーク（{MARKET_TICKER}）の終値がありません。")

    ticker_returns = raw_prices.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )
    sector_returns: dict[str, pd.Series] = {}
    for sector, sector_symbols in sector_tickers.items():
        available_symbols = [
            ticker
            for ticker in sector_symbols
            if ticker_returns[ticker].notna().any()
        ]
        if not available_symbols:
            continue
        sector_return = ticker_returns[available_symbols].mean(axis=1)
        if sector_return.notna().any():
            sector_returns[sector] = sector_return

    if not sector_returns:
        raise ValueError("利用可能な業種代表銘柄の終値がありません。")

    returns = pd.DataFrame(sector_returns, index=raw_prices.index)
    returns[MARKET_COLUMN] = ticker_returns[MARKET_TICKER]
    return (returns.fillna(0).add(1).cumprod() * 100).dropna(how="all")


def fetch_sector_close_prices() -> pd.DataFrame:
    """Download one year of adjusted daily closes for TSE 33 proxies and TOPIX."""
    import yfinance as yf
    from yfinance.exceptions import YFException

    tickers = sorted(
        {
            ticker
            for symbols in TSE33_SECTOR_TICKERS.values()
            for ticker in symbols
        }
        | {MARKET_TICKER}
    )
    try:
        downloaded = yf.download(
            tickers=tickers,
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
    return _extract_close_prices(downloaded, TSE33_SECTOR_TICKERS)


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


def _rrg_quadrant(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "Leading"
    if rs_ratio >= 100:
        return "Weakening"
    if rs_momentum >= 100:
        return "Improving"
    return "Lagging"


def build_rrg_figure(metrics: pd.DataFrame) -> go.Figure:
    latest = _latest_by_sector(metrics).dropna(subset=["RS_Ratio", "RS_Momentum"])
    if latest.empty:
        raise ValueError("RRGを描画できる相対指標データがありません。")

    recent_points = (
        metrics.sort_values("Date")
        .groupby("Sector", sort=False)
        .tail(5)
        .dropna(subset=["RS_Ratio", "RS_Momentum"])
    )
    x_values = pd.concat([latest["RS_Ratio"], recent_points["RS_Ratio"]]).astype(float)
    y_values = pd.concat(
        [latest["RS_Momentum"], recent_points["RS_Momentum"]]
    ).astype(float)
    x_min, x_max = min(float(x_values.min()), 100.0), max(float(x_values.max()), 100.0)
    y_min, y_max = min(float(y_values.min()), 100.0), max(float(y_values.max()), 100.0)
    x_padding = max((x_max - x_min) * 0.12, 1.0)
    y_padding = max((y_max - y_min) * 0.12, 1.0)
    x_range = [x_min - x_padding, x_max + x_padding]
    y_range = [y_min - y_padding, y_max + y_padding]

    figure = go.Figure()
    latest_x = latest["RS_Ratio"].astype(float)
    latest_y = latest["RS_Momentum"].astype(float)
    improving = latest.loc[latest_x.lt(100) & latest_y.ge(100)]
    momentum_leaders = latest.nlargest(3, "Momentum_20D")
    labeled_sectors = set(improving.index) | set(momentum_leaders.index)

    for sector, sector_history in metrics.groupby("Sector", sort=False):
        tail = (
            sector_history.sort_values("Date")
            .dropna(subset=["RS_Ratio", "RS_Momentum"])
            .tail(5)
        )
        if len(tail) < 2:
            continue
        quadrant = _rrg_quadrant(
            float(tail["RS_Ratio"].iloc[-1]),
            float(tail["RS_Momentum"].iloc[-1]),
        )
        figure.add_trace(
            go.Scatter(
                x=tail["RS_Ratio"],
                y=tail["RS_Momentum"],
                customdata=np.column_stack(
                    [
                        tail["Date"].astype(str),
                        np.full(len(tail), sector, dtype=object),
                    ]
                ),
                mode="lines+markers",
                name=f"{sector} 軌跡",
                showlegend=False,
                line={"color": _QUADRANT_COLORS[quadrant], "width": 1.5},
                marker={
                    "color": _QUADRANT_COLORS[quadrant],
                    "size": [4, 5, 6, 7, 8][-len(tail) :],
                },
                hovertemplate=(
                    "%{customdata[1]}<br>日付: %{customdata[0]}"
                    "<br>RS-Ratio: %{x:.2f}"
                    "<br>RS-Momentum: %{y:.2f}<extra>移動軌跡</extra>"
                ),
            )
        )
        figure.add_annotation(
            x=float(tail["RS_Ratio"].iloc[-1]),
            y=float(tail["RS_Momentum"].iloc[-1]),
            ax=float(tail["RS_Ratio"].iloc[-2]),
            ay=float(tail["RS_Momentum"].iloc[-2]),
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowwidth=1.2,
            arrowcolor=_QUADRANT_COLORS[quadrant],
        )

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
        customdata = np.column_stack(
            [
                selected.index,
                [_QUADRANT_LABELS[label]] * len(selected),
                selected["Momentum_20D"].map(
                    lambda value: f"{value:+.2f}%"
                    if pd.notna(value) and np.isfinite(value)
                    else "—"
                ),
            ]
        )
        figure.add_trace(
            go.Scatter(
                x=selected["RS_Ratio"],
                y=selected["RS_Momentum"],
                customdata=customdata,
                text=[
                    sector if sector in labeled_sectors else ""
                    for sector in selected.index
                ],
                textposition="top center",
                mode="markers+text",
                name=label,
                marker={
                    "size": 12 if label == "Improving" else 10,
                    "color": _QUADRANT_COLORS[label],
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate=(
                    "業種: %{customdata[0]}"
                    "<br>象限: %{customdata[1]}"
                    "<br>RS-Ratio: %{x:.2f}"
                    "<br>RS-Momentum: %{y:.2f}"
                    "<br>20日モメンタム: %{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    for x0, x1, y0, y1, label in (
        (x_range[0], 100, 100, y_range[1], "Improving"),
        (100, x_range[1], 100, y_range[1], "Leading"),
        (x_range[0], 100, y_range[0], 100, "Lagging"),
        (100, x_range[1], y_range[0], 100, "Weakening"),
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
        xaxis={"range": x_range, "zeroline": False, "autorange": False},
        yaxis={"range": y_range, "zeroline": False, "autorange": False},
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
        height=max(650, 105 + len(latest) * 31),
        margin={"t": 65, "b": 15, "l": 10, "r": 10},
    )
    return figure


def get_rrg_comment(metrics: pd.DataFrame) -> str:
    required = {"Date", "Sector", "RS_Ratio", "RS_Momentum"}
    if not required.issubset(metrics.columns):
        raise ValueError("RRGコメントには Date / Sector / RS_Ratio / RS_Momentum が必要です。")

    valid = metrics.dropna(subset=["RS_Ratio", "RS_Momentum"]).sort_values("Date")
    latest = _latest_by_sector(valid)
    improving_mask = latest["RS_Ratio"].lt(100) & latest["RS_Momentum"].ge(100)
    improving = latest.loc[improving_mask]
    leading = int(
        (latest["RS_Ratio"].ge(100) & latest["RS_Momentum"].ge(100)).sum()
    )
    stagnating = int(
        (latest["RS_Ratio"].lt(100) & latest["RS_Momentum"].lt(100)).sum()
    )

    histories = {
        sector: group.sort_values("Date")
        for sector, group in valid.groupby("Sector", sort=False)
    }
    initial_entries: list[str] = []
    rising_improving: list[str] = []
    for sector in improving.index:
        history = histories[str(sector)].tail(2)
        if len(history) < 2:
            continue
        previous = history.iloc[-2]
        if previous["RS_Ratio"] < 100 and previous["RS_Momentum"] < 100:
            initial_entries.append(str(sector))
        if history["RS_Momentum"].iloc[-1] > history["RS_Momentum"].iloc[-2]:
            rising_improving.append(str(sector))

    if initial_entries:
        initial_signal = (
            "停滞（左下）から改善（左上）へ直近で進入した初動セクター: "
            + "、".join(initial_entries)
            + "。"
        )
    else:
        initial_signal = (
            "停滞（左下）から改善（左上）へ直近で進入した初動セクターはありません。"
        )

    vector_summary = (
        "軌跡でもモメンタムが上向いている改善業種: "
        + "、".join(rising_improving)
        + "。"
        if rising_improving
        else "改善ゾーン内でモメンタムが上向く業種はありません。"
    )
    return (
        f"現在は先導（右上）が{leading}業種、改善（左上）が{len(improving)}業種、"
        f"停滞（左下）が{stagnating}業種です。{initial_signal}{vector_summary}"
        "座標の直近4営業日の軌跡と矢印で移動方向を示しています。"
    )


def get_ranking_comment(metrics: pd.DataFrame) -> str:
    latest = _latest_by_sector(metrics).dropna(subset=["Momentum_20D"])
    if latest.empty:
        return "ランキング対象となるデータがありません。"
    strongest = latest["Momentum_20D"].idxmax()
    weakest = latest["Momentum_20D"].idxmin()
    return (
        f"代表銘柄バスケットの20営業日モメンタム最上位は「{strongest}」、"
        f"最下位は「{weakest}」です。各業種内の代表銘柄を等ウェイトで集計しています。"
    )


def get_matrix_comment(metrics: pd.DataFrame) -> str:
    latest = _latest_by_sector(metrics)
    early_count = int(latest["EarlySignal"].fillna(False).sum())
    return (
        f"初動フラグは{early_count}業種です。"
        "直近5日が上昇、前週5日が非上昇、かつ市場比の直近5日リターンが"
        "プラスの場合に点灯します。"
    )
