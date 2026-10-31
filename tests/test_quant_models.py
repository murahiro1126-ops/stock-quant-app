from quant_models import (
    MacroSectorAnalysisPost,
    MarketCondition,
    MicroStockAnalysisPost,
    TechnicalReason,
    TradeMode,
)


def test_macro_render_post_has_blank_lines_and_postprime_links():
    post = MacroSectorAnalysisPost(
        market_condition=MarketCondition.NEUTRAL,
        topix_change="+0.5%",
        title_headline="循環に変化",
        title_target_word="転換",
        greeting_summary="みんなおつかれ！",
        one_month_leader="電気機器",
        reversal_sectors=[],
        acceleration_sectors=[],
        dark_horses=[],
        gal_commentary="数値を確認しよう。",
    )

    rendered = post.render_post()

    assert "みんなおつかれ！\n\n📊" in rendered
    assert "個別株のディープな" in rendered
    assert "**" not in rendered
    assert "---" not in rendered


def test_micro_render_post_separates_technical_reasons():
    post = MicroStockAnalysisPost(
        trade_mode=TradeMode.BUY,
        stock_code="7203",
        stock_name="トヨタ自動車",
        sector_name="輸送用機器",
        accuracy_score=70,
        story_hook_phrase="データで見る",
        intro_character_intro="マクロでも注目",
        price_position="中間",
        fear_greed_display="中立",
        signal_summary="買い優勢",
        ai_reliability="参考値",
        technical_reasons=[
            TechnicalReason(point_number=1, description="MACD上向き"),
            TechnicalReason(point_number=2, description="RSI中立"),
        ],
        gal_commentary_body="定量データを確認。",
    )

    rendered = post.render_post()

    assert "1. MACD上向き\n\n2. RSI中立" in rendered
    assert "本日のマクロ分析との連動チェック" in rendered
    assert "**" not in rendered
    assert "---" not in rendered
