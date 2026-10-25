from io import BytesIO

from PIL import Image
import plotly.graph_objects as go
import pytest

import postprime_export
from postprime_export import (
    PNG_EMOJI_FONT_FAMILIES,
    PNG_EXPORT_SCALE,
    build_postprime_export_text,
    compose_plotly_png,
    prepare_figure_for_png,
    render_plotly_png,
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
