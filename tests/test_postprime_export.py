from io import BytesIO

from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import pytest

import postprime_export
from postprime_export import (
    PNG_EMOJI_FONT_FAMILIES,
    PNG_EXPORT_SCALE,
    POSTPRIME_CARD_HEIGHT,
    POSTPRIME_CARD_WIDTH,
    build_postprime_export_text,
    build_postprime_card_figure,
    compose_plotly_png,
    prepare_figure_for_png,
    render_plotly_png,
    render_postprime_card_png,
    select_postprime_card_rows,
)


def test_build_postprime_export_text_includes_prompt_and_ordered_comments():
    text = build_postprime_export_text(
        "東証33業種",
        {
            "RRG自動コメント": "RRGコメント",
            "モメンタム自動コメント": "ランキングコメント",
            "マトリックス自動コメント": "初動銘柄コメント",
        },
    )

    assert text.index("添付画像(A)") < text.index("【RRG自動コメント】")
    assert text.index("【RRG自動コメント】") < text.index(
        "【モメンタム自動コメント】"
    )
    assert text.index("【モメンタム自動コメント】") < text.index(
        "【マトリックス自動コメント】"
    )
    assert "初動銘柄コメント" in text


def test_build_postprime_export_text_reports_missing_analysis_comments():
    text = build_postprime_export_text("Myポートフォリオ", {})

    assert "Myポートフォリオ" in text
    assert "分析データは取得できませんでした" in text


def test_render_plotly_png_preserves_icons_and_uses_emoji_font_fallbacks(monkeypatch):
    calls = []
    rendered_figures = []

    def fake_to_image(figure, *, format, width, scale):
        calls.append((format, width, scale))
        rendered_figures.append(figure)
        return b"png-data"

    monkeypatch.setattr(go.Figure, "to_image", fake_to_image)

    figure = go.Figure(
        data=[
            go.Scatter(
                name="🔥 Signal",
                text=["🚀", "⚠️"],
                textfont={"family": "Noto Sans CJK JP, sans-serif"},
            )
        ]
    )
    figure.update_layout(title="Momentum 🚀")

    assert render_plotly_png(figure, width=800) == b"png-data"
    assert calls == [("png", 800, PNG_EXPORT_SCALE)]
    rendered_json = rendered_figures[0].to_plotly_json()
    assert "🔥" in str(rendered_json)
    assert "🚀" in str(rendered_json)
    assert "⚠️" in str(rendered_json)
    trace_font_family = rendered_json["data"][0]["textfont"]["family"]
    assert all(
        name in trace_font_family for name in PNG_EMOJI_FONT_FAMILIES
    ), repr(trace_font_family)
    assert trace_font_family.startswith(PNG_EMOJI_FONT_FAMILIES[0])
    assert all(
        name in rendered_json["layout"]["font"]["family"]
        for name in PNG_EMOJI_FONT_FAMILIES
    )
    assert rendered_json["layout"]["font"]["family"].startswith(
        PNG_EMOJI_FONT_FAMILIES[0]
    )
    assert "🔥" in str(figure.to_plotly_json())


def test_prepare_figure_for_png_does_not_mutate_original_figure():
    figure = go.Figure(data=[go.Table(cells={"values": [["🔥 Initial"]]})])
    figure.update_layout(font={"family": "Noto Sans CJK JP, sans-serif"})

    prepared = prepare_figure_for_png(figure)

    assert "🔥 Initial" in prepared.data[0].cells.values[0][0]
    assert "🔥 Initial" in figure.data[0].cells.values[0][0]
    assert "Noto Color Emoji" in prepared.layout.font.family


def test_select_postprime_card_rows_prioritizes_initial_signals_and_peak_out():
    matrix = pd.DataFrame(
        {
            "直近5日リターン": [8, 7, 6, 5, 4, 3, 2, 1, 0, -1],
            "前月リターン": [0] * 10,
            "初動判定": [
                "—",
                "🔥 転換初動",
                "🚀 加速初動",
                "⚠️ ピークアウト",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
            ],
        },
        index=[f"業種{i}" for i in range(10)],
    )

    selected = select_postprime_card_rows(matrix)

    assert len(selected) == 7
    assert list(selected.index[:3]) == ["業種1", "業種2", "業種3"]


def test_select_postprime_card_rows_falls_back_to_high_and_low_returns():
    matrix = pd.DataFrame(
        {
            "直近5日リターン": range(10),
            "前月リターン": [0] * 10,
            "初動判定": ["—"] * 10,
        },
        index=[f"業種{i}" for i in range(10)],
    )

    selected = select_postprime_card_rows(matrix)

    assert len(selected) == 7
    assert {"業種9", "業種8", "業種7", "業種6"}.issubset(selected.index)
    assert {"業種2", "業種1", "業種0"}.issubset(selected.index)


def test_render_postprime_card_uses_exact_16_by_9_dimensions(monkeypatch):
    rows = []
    for index in range(10):
        for date in ("2026-01-01", "2026-02-01"):
            rows.append(
                {
                    "Date": date,
                    "Sector": f"業種{index}",
                    "Return_20D": float(index - 4),
                    "Market_Return_20D": -8.18,
                    "Return_5D": float(10 - index),
                    "Return_PreviousWeek": -1.0,
                    "Return_PreviousMonth": -5.0,
                }
            )
    metrics = pd.DataFrame(rows)
    calls = []

    def fake_to_image(_figure, **options):
        calls.append(options)
        return b"card-png"

    monkeypatch.setattr(go.Figure, "to_image", fake_to_image)

    assert render_postprime_card_png(metrics, "東証33業種") == b"card-png"

    assert calls == [
        {
            "format": "png",
            "width": POSTPRIME_CARD_WIDTH,
            "height": POSTPRIME_CARD_HEIGHT,
            "scale": 1,
        }
    ]
    figure = build_postprime_card_figure(metrics, "東証33業種")
    assert figure.layout.width == 1600
    assert figure.layout.height == 900
    assert "TOPIX -8.18%" in str(figure.layout.annotations[0].text)
    assert "業種 20営業日リターン TOP5 / ワースト5" in str(
        figure.layout.annotations[1].text
    )


def test_compose_plotly_png_stacks_named_chart_images(monkeypatch):
    colors = ["red", "blue"]
    rendered_scales = []

    def fake_render(_figure, *, width, scale):
        rendered_scales.append(scale)
        image = Image.new("RGB", (width * scale, 20 * scale), colors.pop(0))
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    monkeypatch.setattr(postprime_export, "render_plotly_png", fake_render)
    png_data = compose_plotly_png(
        (("RRG", go.Figure()), ("Matrix", go.Figure())),
        width=120,
        gap=5,
    )

    with Image.open(BytesIO(png_data)) as combined:
        assert combined.format == "PNG"
        assert combined.size == (240, 64 + 40 + 10 + 64 + 40)
        assert combined.getpixel((0, 64)) == (255, 0, 0)
        assert combined.getpixel((0, 64 + 40 + 10 + 64)) == (0, 0, 255)
    assert rendered_scales == [PNG_EXPORT_SCALE, PNG_EXPORT_SCALE]


@pytest.mark.parametrize(
    ("width", "gap", "message"),
    [
        (0, 0, "画像幅"),
        (100, -1, "画像幅"),
        (100, 0, "解像度スケール"),
    ],
)
def test_compose_plotly_png_rejects_invalid_dimensions(width, gap, message):
    scale = 0 if message == "解像度スケール" else 2
    with pytest.raises(ValueError, match=message):
        compose_plotly_png(
            (("RRG", go.Figure()),), width=width, gap=gap, scale=scale
        )
