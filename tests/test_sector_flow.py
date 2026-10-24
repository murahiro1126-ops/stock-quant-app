import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from sector_flow import (
    MARKET_COLUMN,
    MARKET_TICKER,
    SECTOR_ETFS,
    _extract_close_prices,
    build_matrix_figure,
    build_ranking_figure,
    build_rrg_figure,
    calculate_sector_metrics,
    fetch_sector_close_prices,
    get_matrix_comment,
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
            ("1617.T", "Close"),
            ("1306.T", "Close"),
        ]
    )
    downloaded = pd.DataFrame([[100, 200], [101, 202]], index=dates, columns=columns)

    result = _extract_close_prices(
        downloaded,
        {"1617.T": "食品", "1306.T": MARKET_COLUMN},
    )

    assert result.columns.tolist() == ["食品", MARKET_COLUMN]
    assert result.loc[dates[-1], "食品"] == pytest.approx(101)
    assert result.loc[dates[-1], MARKET_COLUMN] == pytest.approx(202)


def test_fetch_sector_close_prices_downloads_all_tickers_in_one_request(monkeypatch):
    dates = pd.date_range("2025-01-01", periods=2, freq="B")
    tickers = [*SECTOR_ETFS, MARKET_TICKER]
    columns = pd.MultiIndex.from_product([tickers, ["Close"]])
    downloaded = pd.DataFrame(
        np.tile(np.arange(1, len(tickers) + 1, dtype=float), (2, 1)),
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
    assert set(calls[0]["tickers"]) == {*SECTOR_ETFS, MARKET_TICKER}
    assert result.columns.tolist() == [*SECTOR_ETFS.values(), MARKET_COLUMN]


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
    assert len(SECTOR_ETFS) == 17
