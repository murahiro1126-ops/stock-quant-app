from io import BytesIO

from PIL import Image
import plotly.graph_objects as go
import pytest

import postprime_export
from postprime_export import (
    PNG_EXPORT_SCALE,
    build_postprime_export_text,
    compose_plotly_png,
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


def test_render_plotly_png_delegates_width_and_png_format(monkeypatch):
    calls = []
    rendered_figures = []

    def fake_to_image(figure, *, format, width, scale):
        calls.append((format, width, scale))
        rendered_figures.append(figure)
        return b"png-data"

    monkeypatch.setattr(go.Figure, "to_image", fake_to_image)

    figure = go.Figure(
        data=[go.Scatter(name="🔥 Signal", text=["🚀", "⚠️"])]
    )
    figure.update_layout(title="Momentum 🚀")

    assert render_plotly_png(figure, width=800) == b"png-data"
    assert calls == [("png", 800, PNG_EXPORT_SCALE)]
    assert "🔥" not in str(rendered_figures[0].to_plotly_json())
    assert "🚀" not in str(rendered_figures[0].to_plotly_json())
    assert "⚠️" not in str(rendered_figures[0].to_plotly_json())
    assert "🔥" in str(figure.to_plotly_json())


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
        scale=1,
    )

    with Image.open(BytesIO(png_data)) as combined:
        assert combined.format == "PNG"
        assert combined.size == (120, 32 + 20 + 5 + 32 + 20)
        assert combined.getpixel((0, 32)) == (255, 0, 0)
        assert combined.getpixel((0, 32 + 20 + 5 + 32)) == (0, 0, 255)
    assert rendered_scales == [1, 1]


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
