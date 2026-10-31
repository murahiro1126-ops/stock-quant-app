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

    title = rendered.splitlines()[0]
    assert title.startswith("【東証33業種クオンツ分析】")
    assert title.endswith("クオンツ直感チェック📈✨")
    assert len(title) <= 80
    assert "みんなおつかれ！\n\n📊" in rendered
    assert "個別銘柄のディープな" in rendered
    assert "ねぇねぇ、みんな" in rendered
    assert "♡" not in rendered
    assert "**" not in rendered
    assert "---" not in rendered
    assert all(
        word.startswith("#")
        for word in rendered.splitlines()[-1].split()
    )


def test_micro_render_post_separates_technical_reasons():
    post = MicroStockAnalysisPost(
        trade_mode=TradeMode.BUY,
        stock_code="7203",
        stock_name="トヨタ自動車",
        sector_name="輸送用機器",
        accuracy_score=70,
        story_hook_phrase="データで見る" * 50,
        intro_character_intro="マクロでも注目",
        price_position="中間",
        fear_greed_display="中立",
        signal_summary="買い優勢",
        ai_reliability="参考値",
        technical_reasons=[
            TechnicalReason(point_number=1, description="MACD上向き"),
            TechnicalReason(point_number=2, description="RSI中立"),
        ],
        gal_commentary_body="ねぇねぇ、みんな定量データを確認。",
        custom_hashtags=["TOPIX", "#ギャルたん"],
    )

    rendered = post.render_post()
    title = rendered.splitlines()[0]

    assert title.startswith("【トヨタ自動車（7203）】")
    assert title.endswith("クオンツ直感チェック📈✨")
    assert len(title) <= 80
    assert "1. MACD上向き\n\n2. RSI中立" in rendered
    assert "本日のマクロ分析との連動チェック" in rendered
    assert "💬 ギャルたんの直感＆定量解説\n\nねぇねぇ、みんな" in rendered
    assert "#TOPIX" in rendered
    assert rendered.count("#ギャルたん") == 1
    assert "**" not in rendered
    assert "---" not in rendered
