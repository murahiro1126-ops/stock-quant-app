"""TSE 33-industry proxy data, metrics, and chart builders."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed

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
TICKER_NAME_MAP: dict[str, str] = {
    "1332": "ニッスイ",
    "1377": "サカタのタネ",
    "1605": "INPEX",
    "1662": "石油資源開発",
    "1801": "大成建設",
    "1802": "大林組",
    "1803": "清水建設",
    "2502": "アサヒグループホールディングス",
    "2503": "キリンホールディングス",
    "2802": "味の素",
    "3401": "帝人",
    "3402": "東レ",
    "3101": "東洋紡",
    "3861": "王子ホールディングス",
    "3863": "日本製紙",
    "4005": "住友化学",
    "4063": "信越化学工業",
    "4188": "三菱ケミカルグループ",
    "4502": "武田薬品工業",
    "4503": "アステラス製薬",
    "4568": "第一三共",
    "5019": "出光興産",
    "5020": "ENEOSホールディングス",
    "5101": "横浜ゴム",
    "5108": "ブリヂストン",
    "5110": "住友ゴム工業",
    "5201": "AGC",
    "5233": "太平洋セメント",
    "5333": "日本ガイシ",
    "5401": "日本製鉄",
    "5411": "JFEホールディングス",
    "5711": "三菱マテリアル",
    "5713": "住友金属鉱山",
    "5802": "住友電気工業",
    "5929": "三和ホールディングス",
    "5938": "LIXIL",
    "5947": "リンナイ",
    "6301": "小松製作所",
    "6367": "ダイキン工業",
    "7011": "三菱重工業",
    "6501": "日立製作所",
    "6752": "パナソニック ホールディングス",
    "6758": "ソニーグループ",
    "7203": "トヨタ自動車",
    "7267": "本田技研工業",
    "6902": "デンソー",
    "7731": "ニコン",
    "7733": "オリンパス",
    "7741": "HOYA",
    "7911": "TOPPANホールディングス",
    "7832": "バンダイナムコホールディングス",
    "7974": "任天堂",
    "9501": "東京電力ホールディングス",
    "9503": "関西電力",
    "9531": "東京ガス",
    "9020": "東日本旅客鉄道",
    "9022": "東海旅客鉄道",
    "9064": "ヤマトホールディングス",
    "9101": "日本郵船",
    "9104": "商船三井",
    "9107": "川崎汽船",
    "9201": "日本航空",
    "9202": "ANAホールディングス",
    "9069": "センコーグループホールディングス",
    "9301": "三菱倉庫",
    "9364": "上組",
    "9432": "NTT",
    "9433": "KDDI",
    "9434": "ソフトバンク",
    "8001": "伊藤忠商事",
    "8002": "丸紅",
    "8058": "三菱商事",
    "3382": "セブン＆アイ・ホールディングス",
    "8267": "イオン",
    "9843": "ニトリホールディングス",
    "8306": "三菱UFJフィナンシャル・グループ",
    "8316": "三井住友フィナンシャルグループ",
    "8411": "みずほフィナンシャルグループ",
    "8601": "大和証券グループ本社",
    "8604": "野村ホールディングス",
    "8697": "日本取引所グループ",
    "8766": "東京海上ホールディングス",
    "8725": "MS＆ADインシュアランスグループホールディングス",
    "8630": "SOMPOホールディングス",
    "8253": "クレディセゾン",
    "8591": "オリックス",
    "8593": "三菱HCキャピタル",
    "8801": "三井不動産",
    "8802": "三菱地所",
    "8830": "住友不動産",
    "4661": "オリエンタルランド",
    "6098": "リクルートホールディングス",
    "9735": "セコム",
}
MARKET_TICKER = "1306.T"
MARKET_COLUMN = "TOPIX"
RS_LOOKBACK = 20
RS_MOMENTUM_LOOKBACK = 10
RANKING_PERIODS: dict[str, int] = {
    "5営業日（1週間）": 5,
    "20営業日（1ヶ月）": 20,
    "60営業日（3ヶ月）": 60,
}
MATRIX_SIGNAL_LABELS = (
    "🔥 転換初動",
    "🚀 加速初動",
    "💪 継続トレンド",
    "⚠️ ピークアウト",
    "❄️ 継続停滞",
    "—",
    "データ不足",
)
MATRIX_ACCELERATION_THRESHOLD = 1.0


def get_ticker_name(ticker: str, fallback: str | None = None) -> str | None:
    """Return the Japanese master name for a ticker, or its supplied fallback."""
    ticker_code = ticker.strip().upper().removesuffix(".T")
    return TICKER_NAME_MAP.get(ticker_code, fallback)

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
        raise ValueError("Yahoo Financeから分析対象の株価データを取得できませんでした。")

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
        raise ValueError("利用可能な分析対象銘柄の終値がありません。")

    returns = pd.DataFrame(sector_returns, index=raw_prices.index)
    returns[MARKET_COLUMN] = ticker_returns[MARKET_TICKER]
    return (returns.fillna(0).add(1).cumprod() * 100).dropna(how="all")


def fetch_basket_close_prices(
    baskets: Mapping[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Download adjusted daily closes and aggregate the supplied stock baskets."""
    import yfinance as yf
    from yfinance.exceptions import YFException

    if not baskets:
        raise ValueError("分析対象の銘柄バスケットがありません。")
    tickers = sorted(
        {
            ticker
            for symbols in baskets.values()
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
            f"Yahoo Financeから対象銘柄データを取得できません: {exc}"
        ) from exc
    return _extract_close_prices(downloaded, baskets)


def fetch_sector_close_prices() -> pd.DataFrame:
    """Download one year of adjusted daily closes for TSE 33 proxies and TOPIX."""
    return fetch_basket_close_prices(TSE33_SECTOR_TICKERS)


def fetch_ticker_names(
    tickers: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve ticker names from Yahoo Finance, returning names and lookup errors."""
    import yfinance as yf

    def resolve(ticker: str) -> tuple[str, str]:
        master_name = get_ticker_name(ticker)
        if master_name is not None:
            return ticker, master_name
        info = yf.Ticker(ticker).get_info()
        name = info.get("shortName") or info.get("longName")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Yahoo Financeの銘柄情報に銘柄名がありません。")
        return ticker, name.strip()

    names: dict[str, str] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(resolve, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                resolved_ticker, name = future.result()
            except Exception as exc:
                errors[ticker] = str(exc)
            else:
                names[resolved_ticker] = name
    return names, errors


def calculate_sector_metrics(
    close_prices: pd.DataFrame,
    constituents: Mapping[str, tuple[str, ...]] | None = None,
    ticker_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build the shared long-form return and relative-momentum dataset."""
    if not isinstance(close_prices, pd.DataFrame) or close_prices.empty:
        raise ValueError("分析対象の株価データが空です。")
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
        pair["Return_20D"] = pair["Close"].pct_change(20) * 100
        pair["Return_60D"] = pair["Close"].pct_change(60) * 100
        pair["Return_PreviousWeek"] = pair["Close"].shift(5).pct_change(5) * 100
        pair["Return_PreviousMonth"] = pair["Close"].shift(21).pct_change(21) * 100
        pair["Momentum_20D"] = pair["Return_20D"]
        for period in RANKING_PERIODS.values():
            pair[f"Market_Return_{period}D"] = (
                pair["Market_Close"].pct_change(period) * 100
            )
        relative_return_5d = relative_strength.pct_change(5)
        pair["EarlySignal"] = (
            pair["Return_5D"].gt(0)
            & pair["Return_PreviousWeek"].le(0)
            & relative_return_5d.gt(0)
        )
        pair["Sector"] = str(sector)
        constituent_tickers = (constituents or TSE33_SECTOR_TICKERS).get(
            str(sector), ()
        )
        pair["Constituents"] = "、".join(constituent_tickers)
        pair["ConstituentNames"] = "、".join(
            (ticker_names or {}).get(ticker, "") for ticker in constituent_tickers
        )
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


def _format_sector_tickers(
    sector: str,
    constituents: object | None = None,
    constituent_names: object | None = None,
) -> str:
    tickers = (
        constituents.split("、")
        if isinstance(constituents, str) and constituents
        else list(TSE33_SECTOR_TICKERS.get(sector, ()))
    )
    if not tickers:
        return "代表銘柄情報なし"
    names = (
        constituent_names.split("、")
        if isinstance(constituent_names, str)
        else []
    )
    names.extend([""] * (len(tickers) - len(names)))
    return "、".join(
        f"{ticker.removesuffix('.T')}({get_ticker_name(ticker, name)})"
        if get_ticker_name(ticker, name)
        else ticker.removesuffix(".T")
        for ticker, name in zip(tickers, names)
    )


def _universe_term(metrics: pd.DataFrame) -> str:
    if "Constituents" in metrics.columns:
        constituents = metrics["Constituents"].dropna().astype(str)
        constituents = constituents.loc[constituents.ne("")]
        if not constituents.empty and constituents.map(
            lambda value: len(value.split("、")) == 1
        ).all():
            return "銘柄"
    return "業種"


def _constituent_label(metrics: pd.DataFrame) -> str:
    return "対象" if _universe_term(metrics) == "銘柄" else "代表"


def get_rrg_selected_sectors(selection_state: object) -> set[str]:
    """Read selected industry names from Streamlit's Plotly selection state."""
    if isinstance(selection_state, Mapping):
        selection = selection_state.get("selection", {})
    else:
        selection = getattr(selection_state, "selection", {})

    if isinstance(selection, Mapping):
        points = selection.get("points", [])
    else:
        points = getattr(selection, "points", [])

    selected: set[str] = set()
    for point in points:
        if not isinstance(point, Mapping):
            continue
        customdata = point.get("customdata")
        if isinstance(customdata, (list, tuple, np.ndarray)) and len(customdata) > 0:
            selected.add(str(customdata[0]))
    return selected


def build_rrg_figure(
    metrics: pd.DataFrame,
    highlighted_sectors: set[str] | frozenset[str] = frozenset(),
) -> go.Figure:
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
        if len(tail) < 2 or sector not in highlighted_sectors:
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
                line={"color": _QUADRANT_COLORS[quadrant], "width": 2.4},
                marker={
                    "color": [
                        _hex_to_rgba(_QUADRANT_COLORS[quadrant], alpha)
                        for alpha in (0.16, 0.32, 0.52, 0.76, 1.0)[-len(tail) :]
                    ],
                    "size": [5, 6, 7, 8, 10][-len(tail) :],
                },
                hoverinfo="skip",
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
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=_QUADRANT_COLORS[quadrant],
        )

    quadrants = (
        ("Leading", latest_x.ge(100) & latest_y.ge(100)),
        ("Weakening", latest_x.ge(100) & latest_y.lt(100)),
        ("Lagging", latest_x.lt(100) & latest_y.lt(100)),
        ("Improving", latest_x.lt(100) & latest_y.ge(100)),
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
                hoveron="points",
                hovertemplate=(
                    "対象: %{customdata[0]}"
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
        title=f"{_universe_term(metrics)}相対強度ローテーション（RRG）",
        xaxis_title="相対強度（RS-Ratio）",
        yaxis_title="相対モメンタム（RS-Momentum）",
        xaxis={"range": x_range, "zeroline": False, "autorange": False},
        yaxis={"range": y_range, "zeroline": False, "autorange": False},
        font={"family": CHART_FONT_FAMILY},
        height=560,
        hovermode="closest",
        hoverdistance=20,
        margin={"t": 70, "b": 50, "l": 70, "r": 30},
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convert a #RRGGBB color to rgba for the RRG tail gradient."""
    red, green, blue = (
        int(color[index : index + 2], 16) for index in (1, 3, 5)
    )
    return f"rgba({red}, {green}, {blue}, {alpha:.2f})"


def build_ranking_figure(metrics: pd.DataFrame, period: int = 20) -> go.Figure:
    if period not in RANKING_PERIODS.values():
        raise ValueError(f"未対応のランキング期間です: {period}")
    return_column = f"Return_{period}D"
    market_column = f"Market_Return_{period}D"
    required = {return_column, market_column}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            "ランキング指標が不足しています: " + ", ".join(sorted(missing))
        )

    latest = _latest_by_sector(metrics).dropna(subset=[return_column])
    if latest.empty:
        raise ValueError(
            f"ランキングを描画できる{period}営業日リターンがありません。"
        )
    extremes = pd.concat(
        [latest.nsmallest(5, return_column), latest.nlargest(5, return_column)]
    )
    extremes = extremes.loc[~extremes.index.duplicated(keep="first")].sort_values(
        return_column
    )
    values = extremes[return_column].astype(float)
    market_values = latest[market_column].dropna()
    market_return = float(market_values.iloc[-1]) if not market_values.empty else 0.0
    top_sectors = set(latest.nlargest(5, return_column).index)
    rank_labels = [
        "TOP5" if sector in top_sectors else "ワースト5"
        for sector in extremes.index
    ]
    ticker_labels = [
        _format_sector_tickers(
            str(sector),
            extremes.loc[sector, "Constituents"]
            if "Constituents" in extremes.columns
            else None,
            extremes.loc[sector, "ConstituentNames"]
            if "ConstituentNames" in extremes.columns
            else None,
        )
        for sector in extremes.index
    ]
    member_label = _constituent_label(metrics)
    figure = go.Figure(
        go.Bar(
            x=values,
            y=extremes.index,
            orientation="h",
            marker_color=[
                "#c0392b" if sector in top_sectors else "#2874a6"
                for sector in extremes.index
            ],
            text=[f"{value:+.1f}%" for value in values],
            textposition="outside",
            customdata=np.column_stack([rank_labels, ticker_labels]),
            hovertemplate=(
                f"%{{y}}（%{{customdata[0]}}）<br>{member_label}: "
                "%{customdata[1]}"
                f"<br>{period}営業日リターン: "
                "%{x:.2f}%<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_color="#666", line_dash="dash")
    figure.add_vline(
        x=market_return,
        line_color="#8e44ad",
        line_dash="dot",
        annotation_text=f"TOPIX {market_return:+.2f}%",
        annotation_position="top",
    )
    figure.update_layout(
        title=f"{_universe_term(metrics)} {period}営業日リターン TOP5 / ワースト5",
        xaxis_title=f"{period}営業日リターン（%）",
        yaxis_title=None,
        font={"family": CHART_FONT_FAMILY},
        height=max(480, 150 + len(extremes) * 34),
        margin={"t": 70, "b": 45, "l": 180, "r": 100},
        showlegend=False,
    )
    return figure


def classify_matrix_signal(
    return_5d: float | None,
    previous_week: float | None,
    previous_month: float | None,
) -> str:
    returns = (return_5d, previous_week, previous_month)
    if any(value is None or not np.isfinite(value) for value in returns):
        return "データ不足"

    current, previous, month = (float(value) for value in returns)
    if month < 0 and current > 0:
        return "🔥 転換初動"
    if current > 0 and current - previous > MATRIX_ACCELERATION_THRESHOLD:
        return "🚀 加速初動"
    if current > 0 and previous > 0 and month > 0:
        return "💪 継続トレンド"
    if month > 0 and previous > 0 and current < 0:
        return "⚠️ ピークアウト"
    if current < 0 and previous < 0 and month < 0:
        return "❄️ 継続停滞"
    return "—"


def build_matrix_dataframe(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Return_5D",
        "Return_PreviousWeek",
        "Return_PreviousMonth",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            "マトリックス指標が不足しています: " + ", ".join(sorted(missing))
        )

    latest = _latest_by_sector(metrics).copy()
    if latest.empty:
        raise ValueError("比較マトリックスを描画するデータがありません。")
    latest["初動判定"] = [
        classify_matrix_signal(current, previous, month)
        for current, previous, month in zip(
            latest["Return_5D"],
            latest["Return_PreviousWeek"],
            latest["Return_PreviousMonth"],
        )
    ]
    matrix = latest.rename(
        columns={
            "Return_5D": "直近5日リターン",
            "Return_PreviousWeek": "前週5日リターン",
            "Return_PreviousMonth": "前月リターン",
        }
    )[
        [
            "直近5日リターン",
            "前週5日リターン",
            "前月リターン",
            "初動判定",
        ]
    ]
    return matrix.sort_values(
        "直近5日リターン", ascending=False, na_position="last"
    )


def _return_cell_style(value: object) -> str:
    if pd.isna(value) or not np.isfinite(value):
        return "background-color: #f1f3f5; color: #59636e;"
    numeric_value = float(value)
    intensity = min(abs(numeric_value) / 8.0, 1.0)
    if numeric_value >= 0:
        pale = np.array([232, 245, 233], dtype=float)
        saturated = np.array([46, 125, 50], dtype=float)
        color = pale + (saturated - pale) * intensity
        text_color = "#ffffff" if intensity > 0.55 else "#1b5e20"
    else:
        pale = np.array([227, 242, 253], dtype=float)
        saturated = np.array([21, 101, 192], dtype=float)
        color = pale + (saturated - pale) * intensity
        text_color = "#ffffff" if intensity > 0.55 else "#0d47a1"
    red, green, blue = color.astype(int)
    return f"background-color: rgb({red}, {green}, {blue}); color: {text_color};"


def style_matrix_dataframe(matrix: pd.DataFrame) -> pd.io.formats.style.Styler:
    return matrix.style.map(
        _return_cell_style,
        subset=["直近5日リターン", "前週5日リターン", "前月リターン"],
    )


def build_matrix_figure(
    metrics: pd.DataFrame,
    sectors: list[str] | None = None,
) -> go.Figure:
    matrix = build_matrix_dataframe(metrics)
    universe_term = _universe_term(metrics)
    if sectors is not None:
        matrix = matrix.loc[matrix.index.isin(sectors)]
    values: list[list[str]] = [[str(sector) for sector in matrix.index]]
    for column in (
        "直近5日リターン",
        "前週5日リターン",
        "前月リターン",
    ):
        values.append(
            [
                f"{value:+.2f}%"
                if pd.notna(value) and np.isfinite(value)
                else "—"
                for value in matrix[column]
            ]
        )
    values.append(matrix["初動判定"].tolist())

    row_colors = [
        "#f5f7fa" if index % 2 == 0 else "#ffffff"
        for index in range(len(matrix))
    ]
    signal_colors = [
        {
            "🔥 転換初動": "#f8c471",
            "🚀 加速初動": "#f5b041",
            "💪 継続トレンド": "#abebc6",
            "⚠️ ピークアウト": "#f5b7b1",
            "❄️ 継続停滞": "#aed6f1",
        }.get(signal, row_colors[index])
        for index, signal in enumerate(matrix["初動判定"])
    ]
    figure = go.Figure(
        go.Table(
            header={
                "values": [
                    universe_term,
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
                    signal_colors,
                ],
                "font": {"color": "#202833", "size": 12, "family": CHART_FONT_FAMILY},
                "align": ["left", "center", "center", "center", "center"],
                "height": 30,
            },
        )
    )
    figure.update_layout(
        title=f"{universe_term} 多期間比較マトリックス",
        font={"family": CHART_FONT_FAMILY},
        height=max(480, 105 + len(matrix) * 31),
        margin={"t": 65, "b": 15, "l": 10, "r": 10},
    )
    return figure


def get_rrg_comment(metrics: pd.DataFrame) -> str:
    required = {"Date", "Sector", "RS_Ratio", "RS_Momentum"}
    if not required.issubset(metrics.columns):
        raise ValueError("RRGコメントには Date / Sector / RS_Ratio / RS_Momentum が必要です。")

    valid = metrics.dropna(subset=["RS_Ratio", "RS_Momentum"]).sort_values("Date")
    histories = {
        sector: group.sort_values("Date")
        for sector, group in valid.groupby("Sector", sort=False)
    }
    universe_term = _universe_term(metrics)
    member_label = _constituent_label(metrics)
    constituents_by_sector = {
        str(sector): group.iloc[-1].get("Constituents")
        for sector, group in histories.items()
    }
    constituent_names_by_sector = {
        str(sector): group.iloc[-1].get("ConstituentNames")
        for sector, group in histories.items()
    }
    quadrant_counts = {quadrant: 0 for quadrant in _QUADRANT_LABELS}
    quadrant_shifts: list[tuple[str, str, str]] = []
    initial_entries: list[str] = []
    improving_candidates: list[tuple[str, float, float, float]] = []
    distances: dict[str, float] = {}
    speeds: dict[str, float] = {}
    accelerations: dict[str, float] = {}

    for sector, history in histories.items():
        recent = history.tail(5)
        current = recent.iloc[-1]
        current_quadrant = _rrg_quadrant(
            float(current["RS_Ratio"]), float(current["RS_Momentum"])
        )
        quadrant_counts[current_quadrant] += 1

        if len(recent) > 1:
            previous = recent.iloc[0]
            previous_quadrant = _rrg_quadrant(
                float(previous["RS_Ratio"]), float(previous["RS_Momentum"])
            )
            if previous_quadrant != current_quadrant:
                quadrant_shifts.append(
                    (str(sector), previous_quadrant, current_quadrant)
                )
            if (
                previous_quadrant == "Lagging"
                and current_quadrant == "Improving"
            ):
                initial_entries.append(str(sector))

        if len(recent) > 1:
            x_delta = recent["RS_Ratio"].astype(float).diff()
            y_delta = recent["RS_Momentum"].astype(float).diff()
            distance = float(np.hypot(x_delta, y_delta).iloc[1:].sum())
            steps = len(recent) - 1
            distances[str(sector)] = distance
            speeds[str(sector)] = distance / steps
            if steps > 1:
                acceleration = np.hypot(x_delta.diff(), y_delta.diff()).iloc[2:]
                accelerations[str(sector)] = float(acceleration.mean())
            else:
                accelerations[str(sector)] = 0.0

        if current_quadrant == "Improving":
            improving_candidates.append(
                (
                    str(sector),
                    distances.get(str(sector), 0.0),
                    speeds.get(str(sector), 0.0),
                    accelerations.get(str(sector), 0.0),
                )
            )

    quadrant_shifts.sort(
        key=lambda item: distances.get(item[0], 0.0),
        reverse=True,
    )
    if quadrant_shifts:
        shift_summary = "直近最大4営業日の象限シフト: " + "、".join(
            f"{sector}（{_QUADRANT_LABELS[old]}→{_QUADRANT_LABELS[new]}）"
            for sector, old, new in quadrant_shifts[:3]
        ) + "。"
    else:
        shift_summary = f"直近最大4営業日で象限を移動した{universe_term}はありません。"

    if initial_entries:
        initial_details = [
            (
                sector,
                _format_sector_tickers(
                    sector,
                    constituents_by_sector.get(sector),
                    constituent_names_by_sector.get(sector),
                ),
            )
            for sector in initial_entries[:3]
        ]
        initial_signal = (
            "🔥 【資金流入の初動候補】停滞（左下）から改善（左上）へ"
            f"直近最大4営業日でシフトした{universe_term}: "
            + "、".join(
                f"{sector}（{member_label}: {tickers}）"
                for sector, tickers in initial_details
            )
            + "。"
        )
    else:
        initial_signal = (
            "直近最大4営業日で停滞（左下）から改善（左上）へシフトした"
            "初動候補はありません。"
        )

    ranked_distances = sorted(
        distances.items(), key=lambda item: item[1], reverse=True
    )
    if ranked_distances:
        fastest_sector, fastest_distance = ranked_distances[0]
        fastest_tickers = _format_sector_tickers(
            fastest_sector,
            constituents_by_sector.get(fastest_sector),
            constituent_names_by_sector.get(fastest_sector),
        )
        speed_summary = (
            f"テール移動距離の最大は「{fastest_sector}」の{fastest_distance:.2f}"
            f"ポイント（1営業日平均{speeds[fastest_sector]:.2f}ポイント、"
            f"平均加速度{accelerations[fastest_sector]:.2f}ポイント/営業日²）で、"
            f"{member_label}銘柄は {fastest_tickers} です。"
        )
    else:
        fastest_sector = None
        speed_summary = "移動距離を算出できるテールデータがありません。"

    improving_candidates.sort(key=lambda item: item[1], reverse=True)
    if improving_candidates:
        improving_details = [
            (
                sector,
                _format_sector_tickers(
                    sector,
                    constituents_by_sector.get(sector),
                    constituent_names_by_sector.get(sector),
                ),
            )
            for sector, _, _, _ in improving_candidates[:3]
        ]
        improving_summary = "改善ゾーンの注目候補: " + "、".join(
            f"{sector}（{member_label}: {tickers}）"
            for sector, tickers in improving_details
        ) + "。"
        inflow_sector = improving_candidates[0]
        inflow_signal = (
            f"改善ゾーンでは「{inflow_sector[0]}」の移動距離が最大で、"
            f"相対指標の勢いが強い状態です（平均加速度{inflow_sector[3]:.2f}"
            "ポイント/営業日²）。"
        )
    else:
        improving_summary = f"現在、改善ゾーンにある{universe_term}はありません。"
        inflow_sector = None
        inflow_signal = (
            f"改善ゾーン内で資金流入の初動候補となる{universe_term}は確認できません。"
        )

    if inflow_sector is not None:
        inflow_tickers = _format_sector_tickers(
            inflow_sector[0],
            constituents_by_sector.get(inflow_sector[0]),
            constituent_names_by_sector.get(inflow_sector[0]),
        )
        action = (
            f"アクション案: 「{inflow_sector[0]}」の{member_label}銘柄 "
            f"({inflow_tickers}) を監視候補に加え、"
            "価格・出来高の継続性と個別材料を確認してください。"
        )
    else:
        action = (
            "アクション案: 改善ゾーンへの移行が確認できるまで、"
            f"{member_label}銘柄の価格・出来高を継続観察してください。"
        )

    return (
        f"先導 {quadrant_counts['Leading']}{universe_term} / 改善 {quadrant_counts['Improving']}{universe_term} / "
        f"停滞 {quadrant_counts['Lagging']}{universe_term} / 減速 {quadrant_counts['Weakening']}{universe_term}。"
        f"{shift_summary}{initial_signal}{speed_summary}{improving_summary}"
        f"{inflow_signal}{action}"
        "※移動距離は相対強度・相対モメンタム座標上の変化であり、"
        "実際の資金流出入や売買推奨を示すものではありません。"
    )


def get_ranking_comment(metrics: pd.DataFrame, period: int = 20) -> str:
    if period not in RANKING_PERIODS.values():
        raise ValueError(f"未対応のランキング期間です: {period}")
    return_column = f"Return_{period}D"
    market_column = f"Market_Return_{period}D"
    required = {return_column, market_column}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            "ランキング指標が不足しています: " + ", ".join(sorted(missing))
        )

    latest = _latest_by_sector(metrics).dropna(subset=[return_column])
    if latest.empty:
        return "ランキング対象となるデータがありません。"

    strongest = latest[return_column].idxmax()
    weakest = latest[return_column].idxmin()
    strongest_return = float(latest.loc[strongest, return_column])
    weakest_return = float(latest.loc[weakest, return_column])
    spread = strongest_return - weakest_return
    market_values = latest[market_column].dropna()
    market_return = float(market_values.iloc[-1]) if not market_values.empty else 0.0
    universe_term = _universe_term(metrics)
    member_label = _constituent_label(metrics)
    excess = latest[return_column] - latest[market_column]
    positive_excess = excess.clip(lower=0)
    total_positive_excess = float(positive_excess.sum())

    if total_positive_excess > 0:
        top_three = positive_excess.nlargest(min(3, len(positive_excess)))
        concentration = float(top_three.sum() / total_positive_excess * 100)
        concentration_comment = (
            f"TOPIX超過分の{concentration:.0f}%が上位{len(top_three)}{universe_term}に集中しています。"
        )
    else:
        concentration_comment = (
            f"TOPIXを上回る{universe_term}がなく、上昇テーマの広がりは限定的です。"
        )

    theme_leaders = excess.nlargest(min(3, len(excess)))
    theme_summaries = []
    for sector, value in theme_leaders.items():
        if value > 0:
            tickers = _format_sector_tickers(
                str(sector),
                latest.loc[sector].get("Constituents"),
                latest.loc[sector].get("ConstituentNames"),
            )
            theme_summaries.append(
                f"{sector}（TOPIX比 {value:+.2f}pt、{member_label}: {tickers}）"
            )
    theme_summary = "、".join(theme_summaries)
    if theme_summary:
        theme_comment = f"相対優位テーマは{theme_summary}です。"
    else:
        theme_comment = f"TOPIXを上回る{universe_term}がなく、相対優位テーマは見られません。"

    if spread >= 10:
        dispersion_comment = f"{universe_term}間格差が大きく、選別色の強い相場です。"
    elif spread >= 5:
        dispersion_comment = f"{universe_term}間格差は中程度で、強弱の見極めが重要です。"
    else:
        dispersion_comment = f"{universe_term}間格差は小さく、選別の優位性は限定的です。"

    basis_comment = (
        "各銘柄の個別リターンをTOPIXと比較しています。"
        if universe_term == "銘柄"
        else "代表銘柄は業種内で等ウェイト集計した代理指標です。"
    )
    return (
        f"【{period}営業日分析】最強は「{strongest}」({strongest_return:+.2f}%)、"
        f"最弱は「{weakest}」({weakest_return:+.2f}%)で、格差は{spread:.2f}ptです。"
        f"{dispersion_comment} TOPIXは{market_return:+.2f}%。"
        f"{concentration_comment}{theme_comment}"
        f"{basis_comment}"
        "実際の資金フローや売買推奨を示すものではありません。"
    )


def get_matrix_comment(metrics: pd.DataFrame) -> str:
    required = {
        "Date",
        "Sector",
        "Return_5D",
        "Return_PreviousWeek",
        "Return_PreviousMonth",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            "マトリックスコメントに必要な指標が不足しています: "
            + ", ".join(sorted(missing))
        )

    member_label = _constituent_label(metrics)
    newly_lit: list[
        tuple[str, str, float, float, float, object | None, object | None]
    ] = []
    for sector, history in metrics.sort_values("Date").groupby(
        "Sector", sort=False
    ):
        recent = history.tail(2)
        if len(recent) < 2:
            continue
        previous, current = recent.iloc[0], recent.iloc[1]
        signal = classify_matrix_signal(
            current["Return_5D"],
            current["Return_PreviousWeek"],
            current["Return_PreviousMonth"],
        )
        previous_signal = classify_matrix_signal(
            previous["Return_5D"],
            previous["Return_PreviousWeek"],
            previous["Return_PreviousMonth"],
        )
        if (
            signal in {"🔥 転換初動", "🚀 加速初動"}
            and signal != previous_signal
        ):
            newly_lit.append(
                (
                    str(sector),
                    signal,
                    float(current["Return_5D"]),
                    float(current["Return_PreviousWeek"]),
                    float(current["Return_PreviousMonth"]),
                    current.get("Constituents"),
                    current.get("ConstituentNames"),
                )
            )

    priority = {"🔥 転換初動": 0, "🚀 加速初動": 1}
    newly_lit.sort(key=lambda item: (priority[item[1]], item[0]))
    if not newly_lit:
        return (
            "本日新たに点灯した転換初動・加速初動シグナルはありません。"
            f"初動判定は{member_label}銘柄のリターンから算出しています。"
        )

    bullets = "\n".join(
        f"- **{signal}｜{sector}**: 直近5日 {return_5d:+.2f}%、"
        f"前週 {previous_week:+.2f}%、前月 {previous_month:+.2f}%。"
        f"{member_label}銘柄: "
        f"{_format_sector_tickers(sector, constituents, constituent_names)}"
        for (
            sector,
            signal,
            return_5d,
            previous_week,
            previous_month,
            constituents,
            constituent_names,
        ) in newly_lit
    )
    return (
        "本日新たに点灯した初動シグナルです。個別銘柄の価格・出来高や材料を"
        "確認する候補としてご覧ください。\n\n"
        f"{bullets}\n\n"
        f"※{member_label}銘柄の価格リターンに基づく参考情報であり、売買推奨ではありません。"
    )
