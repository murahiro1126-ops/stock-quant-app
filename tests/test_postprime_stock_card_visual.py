import numpy as np
import pandas as pd
import pytest

from postprime_export import (
    _stock_card_gauge_flag,
    _stock_card_gauge_style,
    _stock_card_gauge_visuals,
    build_postprime_stock_card_figure,
)


def _make_gauge_flag_frame(rows=30):
    close = np.linspace(100.0, 115.0, rows)
    return pd.DataFrame(
        {
            "Close": close,
            "Volume": np.full(rows, 1000),
            "SMA_25": close - 1,
            "SMA_75": close - 2,
            "SMA_200": close - 3,
            "MACD": np.full(rows, -0.5),
            "MACD_Signal": np.full(rows, -0.25),
            "RSI": np.full(rows, 45.0),
            "ADX": np.full(rows, 15.0),
            "ATR_Percent": np.full(rows, 1.0),
            "Is_Ranging": np.zeros(rows, dtype=bool),
            "Fear_Greed_Score": np.full(rows, 50.0),
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="B"),
    )


def _assert_gauge_flag(name, indicators, expected):
    flag = _stock_card_gauge_flag(name, indicators)
    assert flag == expected
    if flag is not None:
        assert 2 <= len(flag) <= 6


def test_stock_card_uses_dark_summary_bars_and_flat_zoned_gauge_arcs():
    indicators = pd.DataFrame(
        {
            "Close": [100.0],
            "Volume": [1000],
            "SMA_25": [99.0],
            "SMA_75": [98.0],
            "SMA_200": [95.0],
            "MACD": [0.0001],
            "RSI": [22.0],
            "ADX": [45.0],
            "Fear_Greed_Score": [80.0],
        },
        index=pd.to_datetime(["2026-02-03"]),
    )

    figure = build_postprime_stock_card_figure(
        indicators,
        "1234.T",
        "テスト銘柄",
        range_position=63.4,
        buy_confidence=72,
        sell_confidence=18,
        rank="A",
    )

    summary_rect = next(
        shape
        for shape in figure.layout.shapes
        if shape.type == "rect" and shape.x0 == 0.04
    )
    assert summary_rect.y0 <= 0.85 <= summary_rect.y1
    dark_bar_shapes = [
        shape
        for shape in figure.layout.shapes
        if shape.type == "line"
        and shape.line.color == "#1E293B"
        and shape.y0 == shape.y1 == 0.875
    ]
    assert len(dark_bar_shapes) == 2
    assert all(shape.line.width == 8 for shape in dark_bar_shapes)

    tiny_macd_arc = next(
        shape
        for shape in figure.layout.shapes
        if shape.type == "path"
        and shape.line.color == "#16A34A"
        and shape.line.width == 9
    )
    assert " L " in tiny_macd_arc.path
    arc_start_x = float(tiny_macd_arc.path.split()[1].split(",")[0])
    arc_end_x = float(tiny_macd_arc.path.split()[-1].split(",")[0])
    assert not np.isclose(arc_start_x, arc_end_x, rtol=0, atol=1e-7)
    assert any(
        shape.type == "path" and shape.line.width == 28
        for shape in figure.layout.shapes
    )
    track_index = next(
        index
        for index, shape in enumerate(figure.layout.shapes)
        if shape.type == "path" and shape.line.width == 28
    )
    active_index = next(
        index
        for index, shape in enumerate(figure.layout.shapes)
        if shape.type == "path" and shape.line.width == 9
    )
    assert track_index < active_index
    assert all(
        trace.visible is False for trace in figure.data if trace.type == "indicator"
    )

    annotations = {annotation.text: annotation for annotation in figure.layout.annotations}
    assert annotations["売り確度 18%（独立スコア）"].y == 0.855
    assert annotations["MACD"].font.size >= 23
    assert annotations["MACD"].y >= 0.8
    assert annotations["20"].y > annotations["0"].y
    assert annotations["1e-4"].font.size < 38
    assert np.isclose(annotations["1e-4"].y, 0.625)
    assert {"30", "70", "20", "40", "60", "80", "100"}.issubset(annotations)
    assert {"25", "45", "55", "56", "75"}.isdisjoint(annotations)
    fear_tick_marks = [
        shape
        for shape in figure.layout.shapes
        if shape.type == "line"
        and 0.83 < shape.x0 < 0.99
        and 0.83 < shape.x1 < 0.99
        and 0.17 < shape.y0 < 0.42
        and 0.17 < shape.y1 < 0.42
    ]
    assert len(fear_tick_marks) == 6
    assert annotations["80"].text == "80"
    adx_right_tick = next(
        annotation
        for annotation in figure.layout.annotations
        if annotation.text == "100"
        and annotation.x < 0.83
        and annotation.y < 0.4
    )
    fear_left_tick = next(
        annotation
        for annotation in figure.layout.annotations
        if annotation.text == "0"
        and 0.83 < annotation.x < 0.92
        and annotation.y < 0.4
    )
    assert fear_left_tick.x - adx_right_tick.x >= 0.04
    assert {
        annotations[state].font.color
        for state in {"転換付近", "売られ過ぎ", "非常に強い", "貪欲"}
    } == {"#1E293B"}
    assert all(shape.type != "circle" for shape in figure.layout.shapes)
    assert annotations["買い確度"].font.size == 20
    assert annotations["72%"].font.size == 25


@pytest.mark.parametrize(
    ("score", "expected_state"),
    [
        (0, "極度の恐怖"),
        (20, "極度の恐怖"),
        (20.5, "恐怖"),
        (40, "恐怖"),
        (40.5, "中立"),
        (60, "中立"),
        (60.5, "貪欲"),
        (80, "貪欲"),
        (80.5, "極度の熱狂"),
        (100, "極度の熱狂"),
    ],
)
def test_stock_card_fear_greed_zones_match_analyzer_boundaries(
    score, expected_state
):
    assert _stock_card_gauge_style("Fear & Greed", score)["state"] == expected_state


def test_stock_card_fear_greed_ticks_and_colors_match_its_zone_boundaries():
    style = _stock_card_gauge_style("Fear & Greed", 50)
    zones = style["arc_zones"]

    assert style["tickvals"] == [0, 20, 40, 60, 80, 100]
    assert [(zone[0], zone[1]) for zone in zones] == [
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 80),
        (80, 100),
    ]
    assert [zone[2] for zone in zones] == [
        "#DC2626",
        "#D97706",
        "#64748B",
        "#16A34A",
        "#C2410C",
    ]
    assert [zone[3] for zone in zones] == [
        "#FEE2E2",
        "#FEF3C7",
        "#E2E8F0",
        "#DCFCE7",
        "#FFEDD5",
    ]


@pytest.mark.parametrize("value", [-0.0001, 0.0001])
def test_macd_nonzero_value_omits_neutral_gray_track(value):
    style = _stock_card_gauge_style("MACD", value)
    shapes, _ = _stock_card_gauge_visuals(
        "MACD", value, style, ([0.15, 0.33], [0.65, 0.92])
    )

    assert not any(
        shape["type"] == "path" and shape["line"]["color"] == "#E2E8F0"
        for shape in shapes
    )


def test_macd_zero_value_shows_neutral_gray_track_without_rounded_caps():
    style = _stock_card_gauge_style("MACD", 0)
    shapes, _ = _stock_card_gauge_visuals(
        "MACD", 0, style, ([0.15, 0.33], [0.65, 0.92])
    )

    assert any(
        shape["type"] == "path" and shape["line"]["color"] == "#E2E8F0"
        for shape in shapes
    )
    assert all(shape["type"] != "circle" for shape in shapes)


@pytest.mark.parametrize(
    ("macd", "signal", "expected"),
    [
        ([-1, -1, 1, 1], [-0.5, -0.5, 0.5, 0.5], "GC点灯"),
        ([1, 1, -1, -1], [0.5, 0.5, -0.5, -0.5], "DC点灯"),
        ([-1, -0.5, 0.5, 1], [-2, -2, -2, -2], "ゼロ上抜"),
        ([1, 0.5, -0.5, -1], [2, 2, 2, 2], "ゼロ下抜"),
    ],
)
def test_stock_card_macd_flags_and_priority(macd, signal, expected):
    indicators = _make_gauge_flag_frame(rows=4)
    indicators["MACD"] = macd
    indicators["MACD_Signal"] = signal

    _assert_gauge_flag("MACD", indicators, expected)


def test_stock_card_rsi_divergence_has_priority_over_recent_50_cross():
    indicators = _make_gauge_flag_frame()
    indicators["Close"] = np.concatenate(
        [np.linspace(100, 90, 15), np.linspace(95, 85, 15)]
    )
    indicators["RSI"] = np.full(30, 45.0)
    indicators.iloc[14, indicators.columns.get_loc("RSI")] = 25
    indicators.iloc[-4:, indicators.columns.get_loc("RSI")] = [49, 51, 52, 35]

    _assert_gauge_flag("RSI", indicators, "強気逆行")


def test_stock_card_rsi_bearish_divergence():
    indicators = _make_gauge_flag_frame()
    indicators["Close"] = np.concatenate(
        [np.linspace(80, 100, 15), np.linspace(90, 110, 15)]
    )
    indicators["RSI"] = np.full(30, 65.0)
    indicators.iloc[14, indicators.columns.get_loc("RSI")] = 80
    indicators.iloc[-1, indicators.columns.get_loc("RSI")] = 70

    _assert_gauge_flag("RSI", indicators, "弱気逆行")


@pytest.mark.parametrize(
    ("rsi", "expected"),
    [
        ([49, 50, 51, 52], "50上抜け"),
        ([51, 50, 49, 48], "50割れ"),
    ],
)
def test_stock_card_rsi_50_cross_flags(rsi, expected):
    indicators = _make_gauge_flag_frame(rows=4)
    indicators["RSI"] = rsi

    _assert_gauge_flag("RSI", indicators, expected)


@pytest.mark.parametrize(
    ("adx", "is_ranging", "atr_percent", "expected"),
    [
        ([18, 21, 19], True, [1.0, 1.0, 1.0], "レンジ警戒"),
        ([18, 19, 20, 21], False, [2.0] * 4, "発生初動"),
        ([30, 31, 32, 33], False, [2.0] * 4, "勢い加速"),
        ([38, 36, 33, 30], False, [2.0] * 4, "勢い一服"),
        ([18, 21, 23, 25], False, [2.0] * 4, "発生初動"),
    ],
)
def test_stock_card_adx_flags_and_priority(adx, is_ranging, atr_percent, expected):
    indicators = _make_gauge_flag_frame(rows=len(adx))
    indicators["ADX"] = adx
    indicators["Is_Ranging"] = [is_ranging] * len(adx)
    indicators["ATR_Percent"] = atr_percent

    _assert_gauge_flag("ADX", indicators, expected)


def test_stock_card_adx_range_requires_both_regime_thresholds():
    indicators = _make_gauge_flag_frame(rows=4)
    indicators["Is_Ranging"] = True
    indicators["ATR_Percent"] = 1.5

    _assert_gauge_flag("ADX", indicators, None)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([50, 50, 50, 50, 50, 35], "急冷え込み"),
        ([50, 50, 50, 50, 50, 65], "急過熱"),
        ([20, 24, 25], "底打ち反転"),
        ([80, 75, 74], "過熱一服"),
        ([70, 70, 70, 70, 24, 25], "急冷え込み"),
        ([10, 10, 10, 10, 10, 25], "急過熱"),
    ],
)
def test_stock_card_fear_greed_flags_and_priority(scores, expected):
    indicators = _make_gauge_flag_frame(rows=len(scores))
    indicators["Fear_Greed_Score"] = scores

    _assert_gauge_flag("Fear & Greed", indicators, expected)


@pytest.mark.parametrize(
    "name",
    ["MACD", "RSI", "ADX", "Fear & Greed"],
)
def test_stock_card_gauge_flag_is_none_when_no_condition_matches(name):
    _assert_gauge_flag(name, _make_gauge_flag_frame(), None)


def test_stock_card_fear_greed_flag_requires_history():
    indicators = _make_gauge_flag_frame(rows=1)
    indicators["Fear_Greed_Score"] = [25]

    _assert_gauge_flag("Fear & Greed", indicators, None)


def test_stock_card_combines_centered_values_and_status_labels_on_one_line():
    indicators = _make_gauge_flag_frame()
    indicators.iloc[-2:, indicators.columns.get_loc("MACD")] = [-0.1, 0.1]
    indicators.iloc[-2:, indicators.columns.get_loc("MACD_Signal")] = [0, 0]
    indicators.iloc[-4:, indicators.columns.get_loc("RSI")] = [49, 51, 52, 53]
    indicators.iloc[-1, indicators.columns.get_loc("Is_Ranging")] = True
    indicators.iloc[-6:, indicators.columns.get_loc("Fear_Greed_Score")] = [
        80,
        80,
        80,
        80,
        80,
        20,
    ]

    figure = build_postprime_stock_card_figure(
        indicators,
        "1234.T",
        "テスト銘柄",
        range_position=63.4,
        buy_confidence=72,
        sell_confidence=18,
        rank="A",
    )
    annotations = figure.layout.annotations
    expected = {
        "転換付近 · GC点灯": (0.73, 0.525, 0.625, "0.1"),
        "中立域 · 50上抜け": (0.91, 0.525, 0.625, "53.0"),
        "レンジ気味 · レンジ警戒": (0.73, 0.175, 0.275, "15.0"),
        "極度の恐怖 · 急冷え込み": (0.91, 0.175, 0.275, "20"),
    }
    for label, (center_x, label_y, value_y, value_text) in expected.items():
        label_annotation = next(item for item in annotations if item.text == label)
        value_annotation = next(
            item
            for item in annotations
            if np.isclose(item.x, center_x)
            and np.isclose(item.y, value_y)
            and item.text == value_text
        )
        assert np.isclose(label_annotation.x, center_x)
        assert np.isclose(label_annotation.y, label_y)
        assert label_annotation.xanchor == "center"
        assert label_annotation.font.size < value_annotation.font.size
        assert 18 <= label_annotation.font.size <= 21
        assert len(label) * label_annotation.font.size <= 0.16 * 1600
        assert label_annotation.font.color == "#1E293B"
        assert label_annotation.font.weight == 700
        assert label_annotation.y < value_annotation.y


def test_stock_card_does_not_render_empty_flag_placeholders():
    indicators = _make_gauge_flag_frame(rows=1)
    figure = build_postprime_stock_card_figure(
        indicators,
        "1234.T",
        "テスト銘柄",
        range_position=63.4,
        buy_confidence=72,
        sell_confidence=18,
        rank="A",
    )

    flag_labels = {
        "GC点灯",
        "DC点灯",
        "ゼロ上抜",
        "ゼロ下抜",
        "強気逆行",
        "弱気逆行",
        "50上抜け",
        "50割れ",
        "レンジ警戒",
        "発生初動",
        "勢い加速",
        "勢い一服",
        "急冷え込み",
        "急過熱",
        "底打ち反転",
        "過熱一服",
    }
    texts = {annotation.text for annotation in figure.layout.annotations}
    assert not flag_labels & texts
    assert not any(" · " in text for text in texts)
    assert not any("該当なし" in text for text in texts)
