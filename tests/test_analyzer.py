import numpy as np
import pandas as pd
import pytest

from analyzer import (
    add_signal_flags,
    backtest_strategy,
    calculate_indicators,
    calculate_fear_greed_index,
    classify_fear_greed_score,
    classify_range_position,
    score_signal,
)


@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range("2024-01-01", periods=240, freq="B")
    close = np.linspace(100, 220, len(dates))
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(len(dates), 1_000_000),
        },
        index=dates,
    )


def test_calculate_indicators_adds_expected_columns(sample_ohlcv):
    result = calculate_indicators(sample_ohlcv)

    for column in (
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "RSI",
        "SMA_25",
        "SMA_75",
        "SMA_200",
        "Volume_SMA_20",
        "ADX",
        "ATR",
        "ATR_Percent",
        "Range_Position",
        "Is_Ranging",
    ):
        assert column in result.columns

    assert result["SMA_25"].iloc[-1] == pytest.approx(sample_ohlcv["Close"].iloc[-25:].mean())
    assert result["SMA_75"].notna().any()
    assert result["SMA_200"].notna().any()
    assert result["RSI"].iloc[-1] == pytest.approx(100)
    assert result["ATR"].notna().any()
    assert result["ADX"].notna().any()
    assert result["Range_Position"].iloc[-1] == pytest.approx(
        (sample_ohlcv["Close"].iloc[-1] - sample_ohlcv["Low"].min())
        / (sample_ohlcv["High"].max() - sample_ohlcv["Low"].min())
        * 100
    )
    assert "SMA_25" not in sample_ohlcv.columns


def test_calculate_indicators_rejects_missing_data():
    with pytest.raises(ValueError, match="必須列"):
        calculate_indicators(pd.DataFrame({"Close": [1, 2, 3]}))


def test_calculate_fear_greed_index_uses_linear_components_and_volume_boost():
    history = pd.DataFrame(
        {
            "Close": [100] * 20,
            "SMA_25": [100] * 20,
            "RSI": [50] * 20,
            "Volume": [100] * 20,
        }
    )
    indicators = pd.DataFrame(
        {
            "Close": [110, 90, 110],
            "SMA_25": [100, 100, 100],
            "RSI": [70, 30, 70],
            "Volume": [200, 210, 100],
        }
    )

    result = calculate_fear_greed_index(
        pd.concat([history, indicators], ignore_index=True)
    )
    result = result.tail(3)

    assert result["Fear_Greed_Score"].tolist() == pytest.approx([100, 0, 90])
    assert result["Volume_Ratio"].tolist() == pytest.approx([2, 2, 100 / 110.5])
    assert result["Fear_Greed_Label"].tolist() == [
        "🔥 極度の熱狂",
        "😱 極度の恐怖",
        "🔥 極度の熱狂",
    ]


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "😱 極度の恐怖"),
        (20, "😱 極度の恐怖"),
        (21, "😨 恐怖"),
        (40, "😨 恐怖"),
        (41, "😐 中立"),
        (60, "😐 中立"),
        (61, "🙂 貪欲"),
        (80, "🙂 貪欲"),
        (81, "🔥 極度の熱狂"),
        (100, "🔥 極度の熱狂"),
    ],
)
def test_classify_fear_greed_score(score, expected):
    assert classify_fear_greed_score(score) == expected


def test_calculate_fear_greed_index_rejects_missing_indicators():
    with pytest.raises(ValueError, match="恐怖指数"):
        calculate_fear_greed_index(pd.DataFrame({"Close": [100]}))


def test_score_signal_scores_bullish_conditions_and_sentiment():
    indicators = pd.DataFrame(
        [
            {
                "Close": 110,
                "MACD": 2,
                "MACD_Signal": 1,
                "RSI": 60,
                "SMA_25": 100,
                "SMA_75": 90,
                "Volume": 200,
                "Volume_SMA_20": 100,
            }
        ]
    )

    result = score_signal(indicators, sentiment_rising=True)

    assert result["buy_confidence"] == 100
    assert result["sell_confidence"] == 0
    assert "Googleトレンドの7日移動平均が上昇" in result["buy_reasons"]


def test_score_signal_scores_bearish_conditions_without_sentiment():
    indicators = pd.DataFrame(
        [
            {
                "Close": 80,
                "MACD": -2,
                "MACD_Signal": -1,
                "RSI": 40,
                "SMA_25": 100,
                "SMA_75": 110,
                "Volume": 200,
                "Volume_SMA_20": 100,
            }
        ]
    )

    result = score_signal(indicators)

    assert result["buy_confidence"] == 0
    assert result["sell_confidence"] == 100


def test_add_signal_flags_only_marks_transition_into_threshold():
    rows = [
        {
            "Close": 110,
            "MACD": 2,
            "MACD_Signal": 1,
            "RSI": rsi,
            "SMA_25": 100,
            "SMA_75": 90,
            "Volume": volume,
            "Volume_SMA_20": 100,
        }
        for rsi, volume in ((60, 200), (60, 250), (60, 300), (40, 50), (60, 200))
    ]
    indicators = pd.DataFrame(rows)

    result = add_signal_flags(indicators, threshold=80)

    assert result["Buy_Score"].tolist() == [100, 100, 100, 60, 100]
    assert result["Buy_Signal"].tolist() == [True, False, False, False, True]
    assert result["Sell_Signal"].tolist() == [False] * len(rows)
    assert result.loc[0, "Buy_Signal_Reason"]
    assert result.loc[4, "Buy_Signal_Reason"]


def test_add_signal_flags_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        add_signal_flags(pd.DataFrame(), threshold=101)


@pytest.mark.parametrize(
    ("position", "expected"),
    [(85, "高値圏"), (20, "底値圏"), (50, "中間"), (None, "判定不可")],
)
def test_classify_range_position(position, expected):
    assert classify_range_position(position) == expected


def test_score_signal_adjusts_for_range_position_and_disables_ranging():
    bullish = {
        "Close": 110,
        "MACD": 2,
        "MACD_Signal": 1,
        "RSI": 60,
        "SMA_25": 100,
        "SMA_75": 90,
        "Volume": 200,
        "Volume_SMA_20": 100,
        "Range_Position": 90,
        "Is_Ranging": False,
        "ADX": 15,
        "ATR_Percent": 1.0,
    }

    high_position_score = score_signal(pd.Series(bullish))
    assert high_position_score["buy_confidence"] == 80
    assert high_position_score["range_status"] == "高値圏"

    bullish["Range_Position"] = 10
    bullish["Is_Ranging"] = True
    ranging_score = score_signal(pd.Series(bullish))
    assert ranging_score["buy_confidence"] == 0
    assert ranging_score["sell_confidence"] == 0
    assert ranging_score["is_ranging"] is True


def test_add_signal_flags_excludes_ranging_market_from_signals():
    data = pd.DataFrame(
        [
            {
                "Close": 110,
                "MACD": 2,
                "MACD_Signal": 1,
                "RSI": 60,
                "SMA_25": 100,
                "SMA_75": 90,
                "Volume": 200,
                "Volume_SMA_20": 100,
                "Is_Ranging": True,
            }
        ]
    )

    result = add_signal_flags(data, threshold=75)

    assert result.loc[0, "Buy_Score"] == 0
    assert not result.loc[0, "Buy_Signal"]
    assert not result.loc[0, "Sell_Signal"]


def test_backtest_returns_trades_equity_and_bounded_confidence():
    bullish = {
        "Open": 100,
        "Close": 100,
        "MACD": 2,
        "MACD_Signal": 1,
        "RSI": 60,
        "SMA_25": 90,
        "SMA_75": 80,
        "Volume": 200,
        "Volume_SMA_20": 100,
        "Range_Position": 50,
        "Is_Ranging": False,
    }
    bearish = {
        "Open": 120,
        "Close": 120,
        "MACD": -2,
        "MACD_Signal": -1,
        "RSI": 40,
        "SMA_25": 130,
        "SMA_75": 140,
        "Volume": 200,
        "Volume_SMA_20": 100,
        "Range_Position": 50,
        "Is_Ranging": False,
    }
    data = pd.DataFrame(
        [
            bullish,
            bullish | {"Open": 100, "Close": 110},
            bearish | {"Open": 110, "Close": 120},
            bearish | {"Open": 120, "Close": 100},
        ]
    )

    result = backtest_strategy(data, threshold=75, initial_capital=10_000)

    assert result["trade_count"] == 2
    assert result["win_rate"] == pytest.approx(100)
    assert result["profit_factor"] > 0
    assert result["max_drawdown_percent"] >= 0
    assert result["equity_curve"].iloc[0] == pytest.approx(10_000)
    assert result["confidence_score"] in range(101)
    assert result["rank"] in {"S", "A", "B", "C"}
    assert len(result["trades"]) == result["trade_count"]


def test_backtest_rejects_nonpositive_initial_capital(sample_ohlcv):
    indicators = calculate_indicators(sample_ohlcv)
    with pytest.raises(ValueError):
        backtest_strategy(indicators, initial_capital=0)
