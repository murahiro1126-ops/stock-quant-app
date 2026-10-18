from chart_format import CHART_FONT_FAMILY, build_chart_title


def test_build_chart_title_matches_header_metric_order_and_uses_text_label():
    title = build_chart_title(
        ticker="285A.T",
        company="キオクシア",
        buy_confidence=9,
        sell_confidence=73,
        range_status="中間",
        fear_greed_label="😱 極度の恐怖",
        fear_greed_score=12,
        rank="B",
        win_rate=35,
    )

    assert title == (
        "285A.T キオクシア | 買い確度: 9% / 売り確度: 73% "
        "/ 5年レンジ位置: 中間 / 恐怖指数: 極度の恐怖 (12) "
        "/ AI信頼度: ランクB (勝率35.0%)"
    )
    assert "😱" not in title


def test_build_chart_title_handles_unknown_fear_greed_label_without_score():
    title = build_chart_title(
        ticker="7203.T",
        company="トヨタ自動車",
        buy_confidence=80,
        sell_confidence=20,
        range_status="上限",
        fear_greed_label="判定不可",
        fear_greed_score=None,
        rank="A",
        win_rate=68.5,
    )

    assert "恐怖指数: 判定不可 / AI信頼度" in title
    assert "(0)" not in title


def test_chart_font_supports_japanese_glyphs():
    assert CHART_FONT_FAMILY.startswith("Noto Sans CJK JP")
