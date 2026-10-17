"""Market data, technical analysis, scoring, and simple backtesting."""

from __future__ import annotations

from math import inf
from typing import Any

import numpy as np
import pandas as pd

from config import (
    BACKTEST_ADX_THRESHOLD,
    BACKTEST_HIGH_RANGE_PERCENTILE,
    BACKTEST_LOW_ATR_PERCENT,
    BACKTEST_LOW_RANGE_PERCENTILE,
    SIGNAL_THRESHOLD,
)


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance's single- and multi-ticker column layouts."""
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        expected = {"close", "high", "low", "open", "volume"}
        for level in range(frame.columns.nlevels):
            values = {
                str(value).lower() for value in frame.columns.get_level_values(level)
            }
            if expected.issubset(values):
                frame.columns = frame.columns.get_level_values(level)
                break
        else:
            frame.columns = frame.columns.get_level_values(-1)
    frame.columns = [str(column).strip().title() for column in frame.columns]
    return frame


def calculate_indicators(
    data: pd.DataFrame,
    range_lookback: int = 1250,
    adx_threshold: float = BACKTEST_ADX_THRESHOLD,
    low_atr_percent: float = BACKTEST_LOW_ATR_PERCENT,
) -> pd.DataFrame:
    """Calculate price indicators and rolling range/regime measures."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("株価データが空です。")

    frame = _normalize_columns(data)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"株価データに必須列がありません: {', '.join(sorted(missing))}")
    if range_lookback < 1:
        raise ValueError("range_lookback は1以上で指定してください。")

    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce")

    for window in (25, 75, 200):
        frame[f"SMA_{window}"] = close.rolling(window, min_periods=window).mean()

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    frame["MACD"] = ema_12 - ema_26
    frame["MACD_Signal"] = frame["MACD"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    frame["MACD_Hist"] = frame["MACD"] - frame["MACD_Signal"]

    delta = close.diff()
    average_gain = delta.clip(lower=0).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    average_loss = -delta.clip(upper=0).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50)
    frame["RSI"] = rsi
    frame["Volume_SMA_20"] = volume.rolling(20, min_periods=20).mean()

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["ATR"] = true_range.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    frame["ATR_Percent"] = frame["ATR"] / close.abs() * 100

    upward_move = high.diff()
    downward_move = -low.diff()
    positive_dm = upward_move.where(
        (upward_move > downward_move) & (upward_move > 0), 0.0
    )
    negative_dm = downward_move.where(
        (downward_move > upward_move) & (downward_move > 0), 0.0
    )
    smoothed_tr = true_range.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    positive_di = 100 * positive_dm.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean() / smoothed_tr.replace(0, np.nan)
    negative_di = 100 * negative_dm.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean() / smoothed_tr.replace(0, np.nan)
    directional_index = (
        100
        * (positive_di - negative_di).abs()
        / (positive_di + negative_di).replace(0, np.nan)
    )
    frame["ADX"] = directional_index.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    rolling_high = high.rolling(range_lookback, min_periods=1).max()
    rolling_low = low.rolling(range_lookback, min_periods=1).min()
    span = rolling_high - rolling_low
    frame["Range_Position"] = (
        (close - rolling_low) / span.replace(0, np.nan) * 100
    ).clip(0, 100)
    frame["Range_High"] = rolling_high
    frame["Range_Low"] = rolling_low
    frame["Is_Ranging"] = (
        (frame["ADX"] < adx_threshold)
        & (frame["ATR_Percent"] < low_atr_percent)
    ).fillna(False)
    return frame


def fetch_stock_data(
    ticker: str, period: str = "5y", interval: str = "1d"
) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    import yfinance as yf

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        raise ValueError(f"{ticker} の株価データを取得できませんでした。")
    return _normalize_columns(data)


def classify_range_position(position: float | None) -> str:
    """Describe where the latest close sits within its rolling range."""
    if position is None or not np.isfinite(position):
        return "判定不可"
    if position >= BACKTEST_HIGH_RANGE_PERCENTILE:
        return "高値圏"
    if position <= BACKTEST_LOW_RANGE_PERCENTILE:
        return "底値圏"
    return "中間"


def _latest_row(indicators: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(indicators, pd.DataFrame):
        if indicators.empty:
            raise ValueError("指標データが空です。")
        row = indicators.iloc[-1]
    elif isinstance(indicators, pd.Series):
        row = indicators
    else:
        raise TypeError(
            "indicators は pandas.DataFrame または pandas.Series を指定してください。"
        )
    required = {
        "Close",
        "MACD",
        "MACD_Signal",
        "RSI",
        "SMA_25",
        "SMA_75",
        "Volume",
        "Volume_SMA_20",
    }
    missing = required.difference(row.index)
    if missing:
        raise ValueError(
            f"スコア計算に必要な指標がありません: {', '.join(sorted(missing))}"
        )
    return row


def _finite_value(row: pd.Series, name: str) -> float | None:
    try:
        result = float(row[name])
    except (KeyError, TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def score_signal(
    indicators: pd.DataFrame | pd.Series,
    sentiment_rising: bool | None = None,
) -> dict[str, Any]:
    """Score technical and sentiment conditions, then apply risk filters."""
    row = _latest_row(indicators)
    close = _finite_value(row, "Close")
    macd = _finite_value(row, "MACD")
    macd_signal = _finite_value(row, "MACD_Signal")
    rsi = _finite_value(row, "RSI")
    sma_25 = _finite_value(row, "SMA_25")
    sma_75 = _finite_value(row, "SMA_75")
    volume = _finite_value(row, "Volume")
    volume_average = _finite_value(row, "Volume_SMA_20")
    range_position = _finite_value(row, "Range_Position")

    buy_conditions = [
        macd is not None and macd_signal is not None and macd > macd_signal,
        rsi is not None and 50 <= rsi < 70,
        close is not None and sma_25 is not None and close > sma_25,
        sma_25 is not None and sma_75 is not None and sma_25 > sma_75,
        volume is not None
        and volume_average is not None
        and volume > volume_average
        and close is not None
        and sma_25 is not None
        and close > sma_25,
    ]
    sell_conditions = [
        macd is not None and macd_signal is not None and macd < macd_signal,
        rsi is not None and (rsi < 50 or rsi >= 70),
        close is not None and sma_25 is not None and close < sma_25,
        sma_25 is not None and sma_75 is not None and sma_25 < sma_75,
        volume is not None
        and volume_average is not None
        and volume > volume_average
        and close is not None
        and sma_25 is not None
        and close < sma_25,
    ]
    buy_reasons = [
        reason
        for condition, reason in zip(
            buy_conditions,
            (
                "MACDがシグナルを上抜け",
                "RSIが中立より強い範囲",
                "株価が25日移動平均線より上",
                "25日線が75日線より上",
                "上昇トレンド中に出来高が20日平均を上回る",
            ),
        )
        if condition
    ]
    sell_reasons = [
        reason
        for condition, reason in zip(
            sell_conditions,
            (
                "MACDがシグナルを下抜け",
                "RSIが弱気または買われ過ぎ",
                "株価が25日移動平均線より下",
                "25日線が75日線より下",
                "下落トレンド中に出来高が20日平均を上回る",
            ),
        )
        if condition
    ]
    buy_points = sum(buy_conditions) * 20
    sell_points = sum(sell_conditions) * 20

    if range_position is not None:
        if range_position >= BACKTEST_HIGH_RANGE_PERCENTILE:
            buy_points -= 20
            sell_points += 10
            buy_reasons.append("5年レンジ高値圏のため買いを割引")
            sell_reasons.append("5年レンジ高値圏")
        elif range_position <= BACKTEST_LOW_RANGE_PERCENTILE:
            buy_points += 10
            sell_points -= 20
            buy_reasons.append("5年レンジ底値圏の反発を考慮")
            sell_reasons.append("5年レンジ底値圏のため売りを割引")

    denominator = 100
    if sentiment_rising is not None:
        denominator = 110
        if sentiment_rising:
            buy_points += 10
            buy_reasons.append("Googleトレンドの7日移動平均が上昇")
        else:
            sell_points += 10
            sell_reasons.append("Googleトレンドの7日移動平均が上昇していない")

    is_ranging = bool(row.get("Is_Ranging", False))
    adx = _finite_value(row, "ADX")
    atr_percent = _finite_value(row, "ATR_Percent")
    buy_confidence = int(np.clip(round(buy_points / denominator * 100), 0, 100))
    sell_confidence = int(np.clip(round(sell_points / denominator * 100), 0, 100))
    if is_ranging:
        buy_confidence = 0
        sell_confidence = 0
        buy_reasons.append("ADX・ATRが低いレンジ相場のためシグナル無効")
        sell_reasons.append("ADX・ATRが低いレンジ相場のためシグナル無効")

    return {
        "buy_confidence": buy_confidence,
        "sell_confidence": sell_confidence,
        "buy_reasons": buy_reasons,
        "sell_reasons": sell_reasons,
        "range_position": range_position,
        "range_status": classify_range_position(range_position),
        "is_ranging": is_ranging,
        "adx": adx,
        "atr_percent": atr_percent,
    }


def add_signal_flags(
    indicators: pd.DataFrame, threshold: int = SIGNAL_THRESHOLD
) -> pd.DataFrame:
    """Mark only the first non-ranging bar to cross a confidence threshold."""
    if not isinstance(indicators, pd.DataFrame) or indicators.empty:
        raise ValueError("シグナル判定用の指標データが空です。")
    if not 0 <= threshold <= 100:
        raise ValueError("threshold は0から100の範囲で指定してください。")

    required = {
        "Close",
        "MACD",
        "MACD_Signal",
        "RSI",
        "SMA_25",
        "SMA_75",
        "Volume",
        "Volume_SMA_20",
    }
    missing = required.difference(indicators.columns)
    if missing:
        raise ValueError(
            f"シグナル判定に必要な指標がありません: {', '.join(sorted(missing))}"
        )

    values = {
        column: pd.to_numeric(indicators[column], errors="coerce")
        for column in required
    }
    buy_conditions = (
        values["MACD"] > values["MACD_Signal"],
        values["RSI"].between(50, 70, inclusive="left"),
        values["Close"] > values["SMA_25"],
        values["SMA_25"] > values["SMA_75"],
        (values["Volume"] > values["Volume_SMA_20"])
        & (values["Close"] > values["SMA_25"]),
    )
    sell_conditions = (
        values["MACD"] < values["MACD_Signal"],
        (values["RSI"] < 50) | (values["RSI"] >= 70),
        values["Close"] < values["SMA_25"],
        values["SMA_25"] < values["SMA_75"],
        (values["Volume"] > values["Volume_SMA_20"])
        & (values["Close"] < values["SMA_25"]),
    )
    buy_score = sum(
        condition.fillna(False).astype(int) * 20 for condition in buy_conditions
    )
    sell_score = sum(
        condition.fillna(False).astype(int) * 20 for condition in sell_conditions
    )

    range_position = (
        pd.to_numeric(indicators["Range_Position"], errors="coerce")
        if "Range_Position" in indicators
        else pd.Series(np.nan, index=indicators.index)
    )
    high_zone = range_position >= BACKTEST_HIGH_RANGE_PERCENTILE
    low_zone = range_position <= BACKTEST_LOW_RANGE_PERCENTILE
    buy_score = (buy_score - high_zone.astype(int) * 20 + low_zone.astype(int) * 10).clip(0, 100)
    sell_score = (sell_score + high_zone.astype(int) * 10 - low_zone.astype(int) * 20).clip(0, 100)

    is_ranging = (
        indicators["Is_Ranging"].fillna(False).astype(bool)
        if "Is_Ranging" in indicators
        else pd.Series(False, index=indicators.index)
    )
    buy_score = buy_score.mask(is_ranging, 0)
    sell_score = sell_score.mask(is_ranging, 0)
    buy_qualified = buy_score >= threshold
    sell_qualified = sell_score >= threshold

    result = indicators.copy()
    result["Buy_Score"] = buy_score.astype(int)
    result["Sell_Score"] = sell_score.astype(int)
    result["Buy_Signal"] = buy_qualified & ~buy_qualified.shift(
        1, fill_value=False
    )
    result["Sell_Signal"] = sell_qualified & ~sell_qualified.shift(
        1, fill_value=False
    )
    result["Buy_Signal_Reason"] = ""
    result["Sell_Signal_Reason"] = ""
    for index in result.index[result["Buy_Signal"] | result["Sell_Signal"]]:
        scores = score_signal(result.loc[index])
        if result.at[index, "Buy_Signal"]:
            result.at[index, "Buy_Signal_Reason"] = " / ".join(scores["buy_reasons"])
        if result.at[index, "Sell_Signal"]:
            result.at[index, "Sell_Signal_Reason"] = " / ".join(scores["sell_reasons"])
    return result


def backtest_strategy(
    indicators: pd.DataFrame,
    threshold: int = SIGNAL_THRESHOLD,
    initial_capital: float = 1_000_000,
) -> dict[str, Any]:
    """Backtest signals at the following session's open to avoid lookahead."""
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital は正の数で指定してください。")
    signals = add_signal_flags(indicators, threshold=threshold)
    if "Open" not in signals:
        raise ValueError("バックテストにはOpen列（始値）が必要です。")
    position = 0
    trades: list[dict[str, Any]] = []
    entry_date: Any = None
    entry_price = 0.0
    entry_side = ""
    equity_values: list[float] = []
    equity = float(initial_capital)
    previous_close: float | None = None

    for index, (date, row) in enumerate(signals.iterrows()):
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        if (
            not np.isfinite(open_price)
            or open_price <= 0
            or not np.isfinite(close_price)
            or close_price <= 0
        ):
            raise ValueError(f"{date} の始値または終値が不正なためバックテストできません。")

        if index and previous_close and previous_close > 0:
            overnight_return = open_price / previous_close - 1
            equity *= 1 + position * overnight_return

        if index:
            previous_signal = signals.iloc[index - 1]
            action = (
                1
                if previous_signal["Buy_Signal"]
                else -1
                if previous_signal["Sell_Signal"]
                else 0
            )
        else:
            action = 0

        if action and action != position:
            if position:
                trade_return = position * (open_price / entry_price - 1)
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "side": entry_side,
                        "entry_price": entry_price,
                        "exit_price": open_price,
                        "return_percent": trade_return * 100,
                    }
                )
            position = action
            entry_date = date
            entry_price = open_price
            entry_side = "買い" if action == 1 else "売り"
        intraday_return = close_price / open_price - 1
        equity *= 1 + position * intraday_return
        equity_values.append(equity)
        previous_close = close_price

    if position and len(signals):
        last_date = signals.index[-1]
        last_price = float(signals["Close"].iloc[-1])
        trade_return = position * (last_price / entry_price - 1)
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": last_date,
                "side": entry_side,
                "entry_price": entry_price,
                "exit_price": last_price,
                "return_percent": trade_return * 100,
            }
        )

    equity_curve = pd.Series(equity_values, index=signals.index, name="資産推移")
    drawdown = equity_curve / equity_curve.cummax() - 1

    returns = np.array([trade["return_percent"] / 100 for trade in trades])
    winning = returns[returns > 0]
    losing = returns[returns < 0]
    gross_profit = float(winning.sum())
    gross_loss = float(abs(losing.sum()))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else inf if gross_profit > 0 else 0.0
    )
    win_rate = float((returns > 0).mean() * 100) if len(returns) else 0.0
    trade_count = len(returns)
    pf_quality = min(profit_factor / 2, 1) if np.isfinite(profit_factor) else 1.0
    sample_quality = min(trade_count / 30, 1)
    confidence = round(
        min(100, max(0, win_rate * 0.5 + pf_quality * 30 + sample_quality * 20))
    )
    rank = (
        "S"
        if confidence >= 80 and trade_count >= 20
        else "A"
        if confidence >= 65 and trade_count >= 10
        else "B"
        if confidence >= 45 and trade_count >= 5
        else "C"
    )

    return {
        "trade_count": trade_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_percent": (
            float(abs(drawdown.min()) * 100) if not drawdown.empty else 0.0
        ),
        "confidence_score": confidence,
        "rank": rank,
        "equity_curve": equity_curve.rename("資産推移"),
        "trades": pd.DataFrame(trades),
    }


def get_google_trends_sentiment(
    keyword: str, timeframe: str = "today 3-m"
) -> dict[str, Any]:
    """Check whether Google's daily-interest seven-day moving average is rising."""
    if not keyword or not keyword.strip():
        raise ValueError("Googleトレンド検索語を指定してください。")

    from pytrends.request import TrendReq

    trends = TrendReq(hl="ja-JP", tz=540)
    trends.build_payload([keyword.strip()], timeframe=timeframe, geo="JP")
    interest = trends.interest_over_time()
    if interest is None or interest.empty or keyword.strip() not in interest.columns:
        raise ValueError(f"「{keyword}」のGoogleトレンドデータを取得できませんでした。")

    series = pd.to_numeric(interest[keyword.strip()], errors="coerce").dropna()
    moving_average = series.rolling(window=7, min_periods=7).mean().dropna()
    if len(moving_average) < 2:
        raise ValueError("7日移動平均の傾向判定には8日以上のデータが必要です。")
    latest = float(moving_average.iloc[-1])
    previous = float(moving_average.iloc[-2])
    return {
        "rising": latest > previous,
        "latest_7ma": latest,
        "previous_7ma": previous,
        "series": series,
        "moving_average": moving_average,
    }
