import numpy as np
import pandas as pd

from postprime_export import build_postprime_stock_card_figure


def test_stock_card_uses_dark_summary_bars_and_round_zoned_gauge_arcs():
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
        and shape.line.color == "#475569"
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
    assert {"30", "70", "25", "45", "55", "75", "100"}.issubset(annotations)
    assert "56" not in annotations
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
        for state in {"転換付近", "売られ過ぎ", "非常に強い", "極度の熱狂"}
    } == {"#1E293B"}
    assert annotations["買い確度"].font.size == 20
    assert annotations["72%"].font.size == 25
