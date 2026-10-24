import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from sector_flow import (
    MARKET_COLUMN,
    MARKET_TICKER,
    TSE33_SECTOR_TICKERS,
    _extract_close_prices,
    build_matrix_figure,
    build_ranking_figure,
    build_rrg_figure,
    calculate_sector_metrics,
    fetch_sector_close_prices,
    get_matrix_comment,
    get_rrg_comment,
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

    assert get_matrix_comment(metrics).startswith("初動フラグは")


def test_rrg_comment_detects_recent_lagging_to_improving_transition():
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    metrics = pd.DataFrame(
        [
            {
                "Date": date,
                "Sector": "初動業種",
                "RS_Ratio": ratio,
                "RS_Momentum": momentum,
                "Momentum_20D": 1.0,
            }
            for date, ratio, momentum in zip(
                dates, [97.0, 98.0, 99.0], [96.0, 98.0, 101.0]
            )
        ]
    )

    comment = get_rrg_comment(metrics)

    assert "停滞（左下）から改善（左上）へ直近で進入した初動セクター: 初動業種" in comment
    assert "軌跡でもモメンタムが上向いている改善業種: 初動業種" in comment


def test_rrg_figure_draws_four_prior_steps_and_only_dynamic_labels(
    sample_sector_prices,
):
    metrics = calculate_sector_metrics(sample_sector_prices)
    figure = build_rrg_figure(metrics)

    tail_traces = [trace for trace in figure.data if trace.mode == "lines+markers"]
    point_traces = [trace for trace in figure.data if trace.mode == "markers+text"]

    assert len(tail_traces) == 2
    assert all(len(trace.x) == 5 for trace in tail_traces)
    assert len(point_traces) >= 1
    assert len(figure.layout.annotations) >= 2
    assert figure.layout.xaxis.range[0] < 100 < figure.layout.xaxis.range[1]
    assert figure.layout.yaxis.range[0] < 100 < figure.layout.yaxis.range[1]


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
    assert sum(
        len(trace.x) for trace in figure.data if trace.mode == "lines+markers"
    ) == 33 * 5
    assert len(figure.layout.annotations) == 33
    assert len(build_ranking_figure(metrics).data[0].y) == 10
    assert len(build_matrix_figure(metrics).data[0].cells.values[0]) == 33
    assert build_matrix_figure(metrics).layout.height >= 33 * 31 + 105
