import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from sector_flow import (
    MARKET_COLUMN,
    MARKET_TICKER,
    RANKING_PERIODS,
    MATRIX_SIGNAL_LABELS,
    TSE33_SECTOR_TICKERS,
    _extract_close_prices,
    build_matrix_figure,
    build_matrix_dataframe,
    build_ranking_figure,
    build_rrg_figure,
    calculate_sector_metrics,
    classify_matrix_signal,
    fetch_sector_close_prices,
    get_matrix_comment,
    get_ranking_comment,
    get_rrg_comment,
    get_rrg_selected_sectors,
)


@pytest.fixture
def sample_sector_prices():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    trend = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "食品": 100 + trend * 0.3,
            "機械": 100 + trend * 0.1,
            MARKET_COLUMN: 100 + trend * 0.2,
        },
        index=dates,
    )


def test_extract_close_prices_supports_ticker_first_multiindex():
    dates = pd.date_range("2025-01-01", periods=2, freq="B")
    columns = pd.MultiIndex.from_tuples(
        [
            ("1332.T", "Close"),
            ("1377.T", "Close"),
            ("1306.T", "Close"),
        ]
    )
    downloaded = pd.DataFrame(
        [[100, 100, 200], [110, 90, 220]], index=dates, columns=columns
    )

    result = _extract_close_prices(
        downloaded,
        {"水産・農林業": ("1332.T", "1377.T")},
    )

    assert result.columns.tolist() == ["水産・農林業", MARKET_COLUMN]
    assert result.loc[dates[0], "水産・農林業"] == pytest.approx(100)
    assert result.loc[dates[-1], "水産・農林業"] == pytest.approx(100)
    assert result.loc[dates[-1], MARKET_COLUMN] == pytest.approx(110)


def test_fetch_sector_close_prices_downloads_all_tickers_in_one_request(monkeypatch):
    dates = pd.date_range("2025-01-01", periods=2, freq="B")
    expected_tickers = {
        ticker for symbols in TSE33_SECTOR_TICKERS.values() for ticker in symbols
    } | {MARKET_TICKER}
    tickers = sorted(expected_tickers)
    columns = pd.MultiIndex.from_product([tickers, ["Close"]])
    downloaded = pd.DataFrame(
        np.tile(np.arange(100, 100 + len(tickers), dtype=float), (2, 1)),
        index=dates,
        columns=columns,
    )
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return downloaded

    monkeypatch.setattr("yfinance.download", fake_download)

    result = fetch_sector_close_prices()

    assert len(calls) == 1
    assert calls[0]["period"] == "1y"
    assert calls[0]["interval"] == "1d"
    assert calls[0]["auto_adjust"] is True
    assert set(calls[0]["tickers"]) == expected_tickers
    assert set(result.columns) == {*TSE33_SECTOR_TICKERS, MARKET_COLUMN}
    assert len(result.columns) == 34


def test_calculate_sector_metrics_builds_shared_returns_and_early_flag(
    sample_sector_prices,
):
    prices = sample_sector_prices.copy()
    prices["食品"] = 100.0
    prices.loc[prices.index[-11], "食品"] = 105
    prices.loc[prices.index[-6], "食品"] = 100
    prices.loc[prices.index[-5:], "食品"] = np.linspace(100, 106, 5)

    metrics = calculate_sector_metrics(prices)
    latest_food = metrics.loc[metrics["Sector"] == "食品"].iloc[-1]

    assert {
        "RS_Ratio",
        "RS_Momentum",
        "Return_1D",
        "Return_5D",
        "Return_20D",
        "Return_60D",
        "Market_Return_5D",
        "Market_Return_20D",
        "Market_Return_60D",
        "Return_PreviousWeek",
        "Return_PreviousMonth",
        "Momentum_20D",
        "EarlySignal",
    }.issubset(metrics.columns)
    assert latest_food["Return_5D"] > 0
    assert latest_food["Return_PreviousWeek"] < 0
    assert bool(latest_food["EarlySignal"])


@pytest.mark.parametrize(
    "figure_builder",
    [build_rrg_figure, build_ranking_figure, build_matrix_figure],
)
def test_sector_chart_builders_return_plotly_figures(
    sample_sector_prices, figure_builder
):
    metrics = calculate_sector_metrics(sample_sector_prices)

    figure = figure_builder(metrics)

    assert isinstance(figure, go.Figure)
    assert figure.data


def test_metric_and_chart_functions_validate_missing_market_column():
    with pytest.raises(ValueError, match="ベンチマーク"):
        calculate_sector_metrics(pd.DataFrame({"食品": [1, 2, 3]}))


def test_matrix_comment_reports_early_signal_count(sample_sector_prices):
    metrics = calculate_sector_metrics(sample_sector_prices)

    assert "本日新たに点灯した" in get_matrix_comment(metrics)


@pytest.mark.parametrize(
    ("returns", "expected"),
    [
        ((1.0, -1.0, -3.0), "🔥 転換初動"),
        ((2.0, 0.5, 1.0), "🚀 加速初動"),
        ((2.0, 1.0, 0.5), "💪 継続トレンド"),
        ((-1.0, 1.0, 2.0), "⚠️ ピークアウト"),
        ((-1.0, -2.0, -3.0), "❄️ 継続停滞"),
        ((0.0, 1.0, -1.0), "—"),
        ((np.nan, 1.0, 2.0), "データ不足"),
    ],
)
def test_classify_matrix_signal_patterns(returns, expected):
    assert classify_matrix_signal(*returns) == expected


def test_matrix_dataframe_sorts_latest_return_descending_and_has_flags(
    sample_sector_prices,
):
    matrix = build_matrix_dataframe(calculate_sector_metrics(sample_sector_prices))

    assert matrix.index.is_unique
    assert matrix["直近5日リターン"].is_monotonic_decreasing
    assert matrix["初動判定"].isin(MATRIX_SIGNAL_LABELS).all()


def test_matrix_comment_lists_new_signals_with_representative_tickers():
    dates = pd.date_range("2025-01-01", periods=2, freq="B")
    metrics = pd.DataFrame(
        [
            {
                "Date": date,
                "Sector": "証券・商品先物取引業",
                "Return_5D": current,
                "Return_PreviousWeek": previous,
                "Return_PreviousMonth": month,
            }
            for date, current, previous, month in (
                (dates[0], -1.0, -0.5, -2.0),
                (dates[1], 1.2, -0.5, -2.0),
            )
        ]
    )

    comment = get_matrix_comment(metrics)

    assert "🔥 転換初動｜証券・商品先物取引業" in comment
    assert "代表銘柄: 8601、8604、8697" in comment


def test_matrix_comment_does_not_repeat_existing_signal():
    dates = pd.date_range("2025-01-01", periods=2, freq="B")
    metrics = pd.DataFrame(
        [
            {
                "Date": date,
                "Sector": "保険業",
                "Return_5D": 1.2,
                "Return_PreviousWeek": -0.5,
                "Return_PreviousMonth": -2.0,
            }
            for date in dates
        ]
    )

    assert "新たに点灯した転換初動・加速初動シグナルはありません" in get_matrix_comment(
        metrics
    )


@pytest.mark.parametrize("period", RANKING_PERIODS.values())
def test_ranking_figure_and_comment_follow_selected_period(
    sample_sector_prices, period
):
    metrics = calculate_sector_metrics(sample_sector_prices)

    figure = build_ranking_figure(metrics, period)
    comment = get_ranking_comment(metrics, period)

    assert f"{period}営業日" in figure.layout.title.text
    assert len(figure.data[0].x) == 2
    assert figure.data[0].customdata.shape == (2, 2)
    assert len(figure.layout.shapes) == 2
    assert any(
        annotation.text.startswith("TOPIX ")
        for annotation in figure.layout.annotations
    )
    assert f"【{period}営業日分析】" in comment
    assert "格差は" in comment
    assert "TOPIX超過分" in comment or "TOPIXを上回る業種がなく" in comment
    assert "代表:" in comment or "相対優位テーマは見られません" in comment


def test_ranking_rejects_unsupported_period(sample_sector_prices):
    metrics = calculate_sector_metrics(sample_sector_prices)

    with pytest.raises(ValueError, match="未対応"):
        build_ranking_figure(metrics, 30)


def test_rrg_comment_detects_recent_lagging_to_improving_transition():
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    metrics = pd.DataFrame(
        [
            {
                "Date": date,
                "Sector": "証券・商品先物取引業",
                "RS_Ratio": ratio,
                "RS_Momentum": momentum,
                "Momentum_20D": 1.0,
            }
            for date, ratio, momentum in zip(
                dates, [97.0, 97.5, 99.0], [96.0, 99.0, 101.0]
            )
        ]
    )

    comment = get_rrg_comment(metrics)

    assert "直近最大4営業日の象限シフト" in comment
    assert "停滞（左下）→改善（左上）" in comment
    assert "【資金流入の初動候補】" in comment
    assert "証券・商品先物取引業（代表: 8601、8604、8697）" in comment
    assert "移動距離の最大は「証券・商品先物取引業」" in comment
    assert "平均加速度1.41ポイント/営業日²" in comment
    assert "価格・出来高の継続性" in comment
    assert "実際の資金流出入や売買推奨を示すものではありません" in comment


def test_rrg_figure_hides_tails_by_default_and_highlights_selected_sector(
    sample_sector_prices,
):
    metrics = calculate_sector_metrics(sample_sector_prices)
    default_figure = build_rrg_figure(metrics)
    highlighted_figure = build_rrg_figure(metrics, {"食品"})

    default_tail_traces = [
        trace for trace in default_figure.data if trace.mode == "lines+markers"
    ]
    highlighted_tail_traces = [
        trace for trace in highlighted_figure.data if trace.mode == "lines+markers"
    ]
    point_traces = [
        trace for trace in default_figure.data if trace.mode == "markers+text"
    ]

    assert not default_tail_traces
    assert len(highlighted_tail_traces) == 1
    assert len(highlighted_tail_traces[0].x) == 5
    assert highlighted_tail_traces[0].hoverinfo == "skip"
    marker_colors = highlighted_tail_traces[0].marker.color
    assert "0.16)" in marker_colors[0]
    assert "1.00)" in marker_colors[-1]
    assert len(point_traces) >= 1
    assert len(default_figure.layout.annotations) == 0
    assert len(highlighted_figure.layout.annotations) == 1
    assert default_figure.layout.xaxis.range[0] < 100 < default_figure.layout.xaxis.range[1]
    assert default_figure.layout.yaxis.range[0] < 100 < default_figure.layout.yaxis.range[1]


def test_rrg_selected_sectors_reads_plotly_selection_customdata():
    selection_state = {
        "selection": {
            "points": [
                {"customdata": ["輸送用機器", "改善（左上）", "+3.4%"]},
                {"customdata": ["保険業", "先導（右上）", "+1.2%"]},
            ]
        }
    }

    assert get_rrg_selected_sectors(selection_state) == {"輸送用機器", "保険業"}


def test_ticker_basket_covers_33_unique_industries_and_tickers():
    tickers = [
        ticker
        for symbols in TSE33_SECTOR_TICKERS.values()
        for ticker in symbols
    ]

    assert len(TSE33_SECTOR_TICKERS) == 33
    assert all(len(symbols) >= 2 for symbols in TSE33_SECTOR_TICKERS.values())
    assert len(tickers) == len(set(tickers))


def test_transport_and_insurance_baskets_use_requested_representative_stocks():
    assert TSE33_SECTOR_TICKERS["輸送用機器"] == (
        "7203.T",
        "7267.T",
        "6902.T",
    )
    assert TSE33_SECTOR_TICKERS["保険業"] == (
        "8766.T",
        "8725.T",
        "8630.T",
    )


def test_metrics_and_charts_scale_to_33_industries():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    trend = np.arange(len(dates), dtype=float)
    prices = pd.DataFrame(
        {
            **{
                sector: 100 + trend * (0.1 + index * 0.001)
                for index, sector in enumerate(TSE33_SECTOR_TICKERS)
            },
            MARKET_COLUMN: 100 + trend * 0.2,
        },
        index=dates,
    )

    metrics = calculate_sector_metrics(prices)
    figure = build_rrg_figure(metrics)

    assert metrics["Sector"].nunique() == 33
    assert sum(
        len(trace.x) for trace in figure.data if trace.mode == "markers+text"
    ) == 33
    assert not [trace for trace in figure.data if trace.mode == "lines+markers"]
    selected_figure = build_rrg_figure(metrics, {"水産・農林業", "鉱業"})
    assert sum(
        len(trace.x) for trace in selected_figure.data if trace.mode == "lines+markers"
    ) == 2 * 5
    assert len(selected_figure.layout.annotations) == 2
    assert len(build_ranking_figure(metrics).data[0].y) == 10
    assert len(build_matrix_figure(metrics).data[0].cells.values[0]) == 33
    assert build_matrix_figure(metrics).layout.height >= 33 * 31 + 105
