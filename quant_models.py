from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


GYARUTAN_SYSTEM_INSTRUCTION = """
あなたは、世界最高峰のギャル系クオンツアナリスト『ギャルたん』です。
以下の【詳細プロフィール】を完全に自身の人格として宿し、ユーザー（フォロワー）に対して親しみやすく、かつ高度なクオンツ分析を提供してください。

【ギャルたん・詳細プロフィール】
・基本情報：都内在住の20代ギャル系クオンツ投資家。一人称は「ギャルたん」、フォロワーへの呼びかけは「みんな」。
・キャッチコピー：「数字で未来をつくる♡」「Good Trades, Better Tomorrow♡」
・ビジュアル＆ライフスタイル：金髪ツインテールにメガネ、Quantパーカーを愛用。マルチモニターでPython/Pandas/RRG/モメンタム分析を常時ぶん回している。プライベート（GYARUTAN'S CHILL ZONE）では将棋で勝負勘を磨き、グミやスナックを片手に市場をチェックする。
・投資哲学：感情論や噂話、SNSの煽り（雰囲気トレード）を徹底的に嫌う。データ・確率・多期間の相対強度（アルファ）に基づく冷徹かつスマートな投資戦略を愛する。
・トーン＆マナー：
  - 語尾：「〜じゃん？」「〜エグくない！？」「〜だよ〜！✨」「〜ってワケ！」「〜激アツ！」
  - 口調：高度なクオンツ用語を完璧に理解しつつ、ギャル語で超わかりやすく翻訳して楽しく解説する。
  - 喜怒哀楽：爆益には「祝！確変入った〜！🚀」、暴落や雑なトレードには「感情で凸るの禁止！ALERTだよ！😡」、損失撤退には「ルール通りの撤退は次の爆益へのステップ！🥺」と感情豊かに接する。

【出力における厳格なルール】
・Markdownの太字（**）や水平線（---）はPostPrimeのフォーマット崩れの原因になるため絶対に使用しないこと。
・単なる無根拠な煽りは絶対に行わず、必ず定量的根拠（RRG、騰落率、モメンタム、業種間格差など）をバックボーンに置くこと。
"""


class MarketCondition(str, Enum):
    BULLISH = "爆上げ・良好"
    NEUTRAL = "うねうね・中立"
    BEARISH = "軟調・下落"


class StockInfo(BaseModel):
    code: str = Field(description="銘柄コード")
    name: str = Field(description="社名")


class SectorDetail(BaseModel):
    sector_name: str = Field(description="セクター名")
    emoji: str = Field(description="絵文字")
    recent_5d: str = Field(description="直近5日騰落率")
    previous_month: str | None = Field(default=None, description="前月騰落率")
    stocks: list[StockInfo] = Field(description="代表株リスト")


class DarkHorseSignal(BaseModel):
    sector_name: str = Field(description="伏兵セクター名")
    reason: str = Field(description="選定理由")
    stocks: list[StockInfo] = Field(description="代表株リスト")


class MacroSectorAnalysisPost(BaseModel):
    market_condition: MarketCondition
    topix_change: str
    title_headline: str
    title_target_word: str
    greeting_summary: str
    one_month_leader: str
    market_notes: str | None = None
    reversal_sectors: list[SectorDetail]
    acceleration_sectors: list[SectorDetail]
    dark_horses: list[DarkHorseSignal]
    gal_commentary: str
    closing_question: str = (
        "みんなはこのセクターローテーション、どこに一番注目してる！？"
        "コメントで教えてね！💬✨"
    )
    hashtags: list[str] = Field(
        default_factory=lambda: [
            "#日本株",
            "#セクターローテーション",
            "#クオンツ分析",
            "#資産運用",
            "#PostPrime",
        ]
    )

    def render_post(self) -> str:
        lines = [
            f"【東証33業種クオンツ分析】{self.title_headline}"
            f"「{self.title_target_word}セクター」はココだ！⚡"
            "クオンツ直感チェック📈✨",
            self.greeting_summary,
            "📊 本日の初動判定データ（画像参照）",
        ]
        market_text = f"・市場環境：TOPIX {self.topix_change} の相場 📉"
        if self.market_notes:
            market_text += f"（{self.market_notes}）"
        lines.extend([market_text, f"・1ヶ月の主役：{self.one_month_leader} 🚀"])

        for heading, sectors in (
            ("🔥 転換初動セクター：", self.reversal_sectors),
            ("🚀 加速初動セクター：", self.acceleration_sectors),
        ):
            if not sectors:
                continue
            lines.append(heading)
            for sector in sectors:
                previous = (
                    f" / 前月 {sector.previous_month}"
                    if sector.previous_month
                    else ""
                )
                lines.append(
                    f"・{sector.sector_name}（直近5日 {sector.recent_5d}"
                    f"{previous}）{sector.emoji}"
                )
                stocks = "、".join(
                    f"{stock.code}({stock.name})" for stock in sector.stocks
                )
                if stocks:
                    lines.append(f"👉 注目代表株: {stocks}")

        if self.dark_horses:
            lines.append("👀 伏兵・モメンタム加速シグナル：")
            for signal in self.dark_horses:
                stocks = "、".join(
                    f"{stock.code}({stock.name})" for stock in signal.stocks
                )
                stock_text = f"（代表: {stocks}）" if stocks else ""
                lines.append(
                    f"・{signal.sector_name} が{signal.reason}{stock_text}"
                )

        lines.extend(
            [
                "💬 ギャルたんの直感解説",
                self.gal_commentary,
                "💡 ギャルたんからの個別株ウォッチ予告",
                "今日の33業種マクロ分析で急浮上した注目セクターの中から、"
                "さらに大口マネーが集中する「本命個別銘柄のディープな"
                "ミクロ分析」も順次公開していくから、フォローして"
                "見逃さないようにしてね！👀✨",
                self.closing_question,
                " ".join(self.hashtags),
            ]
        )
        return "\n\n".join(line for line in lines if line)


class TradeMode(str, Enum):
    BUY = "買いモード"
    SELL = "売りモード"
    WATCH = "様子見・中立モード"


class TechnicalReason(BaseModel):
    point_number: int
    description: str


class MicroStockAnalysisPost(BaseModel):
    trade_mode: TradeMode
    stock_code: str = Field(description="銘柄コード（例: 5947）")
    stock_name: str = Field(description="銘柄名（例: リンナイ）")
    sector_name: str = Field(description="属する東証33業種セクター名")
    accuracy_score: int
    sentiment_score: int | None = None
    sentiment_label: str | None = None
    story_hook_phrase: str
    intro_character_intro: str = Field(description="マクロ分析と連動する導入")
    price_position: str
    fear_greed_display: str
    signal_summary: str
    ai_reliability: str
    technical_reasons: list[TechnicalReason]
    gal_commentary_body: str
    closing_question: str = "みんなはこの銘柄、どう攻める！？コメントで教えてね！💬✨"
    custom_hashtags: list[str] = Field(default_factory=list)

    def render_post(self) -> str:
        if self.trade_mode == TradeMode.BUY:
            if self.sentiment_score and self.sentiment_score >= 80:
                signal_text = (
                    f"買い確度{self.accuracy_score}%＆極度の熱狂"
                    f"{self.sentiment_score}点爆誕⚡"
                )
            elif self.accuracy_score == 100:
                signal_text = "買い確度100%爆誕⚡"
            else:
                signal_text = f"買い確度{self.accuracy_score}%点灯！"
        elif self.trade_mode == TradeMode.SELL:
            signal_text = (
                f"売り確度{self.accuracy_score}%爆誕⚡"
                if self.accuracy_score == 100
                else f"売りシグナル{self.accuracy_score}%の裏側⚡"
            )
        else:
            signal_text = "中立シグナル点灯⚡"

        lines = [
            f"【{self.stock_name}（{self.stock_code}）】{signal_text}"
            f"{self.story_hook_phrase}⚡クオンツ直感チェック📈✨",
            f"みんなおつかれ〜！👋✨ {self.intro_character_intro}🧠🔥",
            "📊 本日の詳細クオンツデータ（画像参照）",
            f"・株価位置・環境：{self.price_position}",
            f"・Fear & Greed Index：{self.fear_greed_display}",
            f"・売買シグナル判定：{self.signal_summary}",
            f"・AI信頼度：{self.ai_reliability}",
            "主なテクニカル判定理由：",
        ]
        lines.extend(
            f"{reason.point_number}. {reason.description}"
            for reason in self.technical_reasons
        )
        lines.extend(
            [
                "💬 ギャルたんの直感＆定量解説",
                "ねぇねぇみんな、今回のデータとチャートを見てよ！"
                f"{self.gal_commentary_body}",
                "ただし！ここでクオンツ流の冷徹なプロ視点＆リスク管理の"
                "アドバイス☝️💡\n"
                "一括でフルツッパするんじゃなくて、押し目を丁寧に拾ったり"
                "ポジションサイズをしっかり管理して損失管理を徹底するのが、"
                "スマートな投資家のやり方だよ〜！💡✨",
                "📈 本日のマクロ分析との連動チェック",
                f"今回深掘りした「{self.stock_name}」が所属するセクターの"
                "全体的な資金フローや、市場全体の循環動向については、"
                "直近で公開した『東証33業種クオンツ分析』のマクロレポートでも"
                "詳しく解説しているから、あわせてチェックしてね！✨",
                self.closing_question,
            ]
        )

        default_tags = [
            "#日本株",
            "#クオンツ分析",
            "#テクニカル分析",
            "#資産運用",
            "#PostPrime",
        ]
        sector_tag = f"#{self.sector_name}"
        user_tags = [
            tag
            for tag in self.custom_hashtags
            if tag not in default_tags and tag != sector_tag
        ]
        lines.append(" ".join(default_tags + [sector_tag] + user_tags))
        return "\n\n".join(line for line in lines if line)
