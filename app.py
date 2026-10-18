from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analyzer import (
    add_signal_flags,
    backtest_strategy,
    calculate_indicators,
    calculate_fear_greed_index,
    fetch_stock_data,
    get_google_trends_sentiment,
    score_signal,
)
from config import (
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_PERIOD,
    SIGNAL_THRESHOLD,
)
from commentator import generate_ai_comment
from watchlist_store import load_watchlist, save_watchlist

st.set_page_config(page_title="日本株シグナル検証", page_icon="📈", layout="wide")
st.title("日本株テクニカル・シグナル検証")
st.caption(
    "テクニカル・センチメント・値位置を使った参考分析です。"
    "バックテスト結果や信頼度ランクは将来の利益を保証するものではありません。"
)


@st.cache_data(ttl=900, show_spinner=False)
def load_analysis(ticker: str) -> pd.DataFrame:
    indicators = calculate_indicators(
        fetch_stock_data(ticker, period=BACKTEST_PERIOD)
    )
    return calculate_fear_greed_index(indicators)


@st.cache_data(ttl=3600, show_spinner=False)
def load_trends(keyword: str) -> dict:
    return get_google_trends_sentiment(keyword)


st.subheader("ウォッチリスト管理")
try:
    active_watchlist = load_watchlist()
except (OSError, ValueError) as exc:
    st.error(f"ウォッチリストを読み込めません: {exc}")
    st.stop()

with st.expander("銘柄の追加・編集・削除", expanded=False):
    st.caption(
        "表のセルを編集し、行を追加・削除してから保存してください。"
        "銘柄コードは4文字（例: 7203）または .T 付きで入力できます。"
    )
    watchlist_frame = pd.DataFrame(
        list(active_watchlist.items()),
        columns=["銘柄コード", "銘柄名"],
    ).astype({"銘柄コード": "string", "銘柄名": "string"})
    revision = hashlib.sha256(
        json.dumps(active_watchlist, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    with st.form("watchlist_form"):
        edited_watchlist = st.data_editor(
            watchlist_frame,
            key=f"watchlist_editor_{revision}",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "銘柄コード": st.column_config.TextColumn(
                    "銘柄コード", help="例: 7203 または 7203.T", required=True
                ),
                "銘柄名": st.column_config.TextColumn("銘柄名", required=True),
            },
            alt="編集可能な銘柄ウォッチリスト",
        )
        save_changes = st.form_submit_button("変更を保存", type="primary")
    if save_changes:
        try:
            active_watchlist = save_watchlist(
                {
                    row["銘柄コード"]: row["銘柄名"]
                    for _, row in edited_watchlist.iterrows()
                }
            )
        except (OSError, ValueError) as exc:
            st.error(f"変更を保存できません: {exc}")
        else:
            st.cache_data.clear()
            st.session_state.pop("chart_png_data", None)
            st.session_state.pop("chart_png_ticker", None)
            st.success("ウォッチリストを保存しました。")
            st.rerun()

st.subheader("第1段階｜高速スクリーニング")
if st.button("ウォッチリストを更新"):
    st.cache_data.clear()
    st.session_state.pop("chart_png_data", None)
    st.session_state.pop("chart_png_ticker", None)

rows: list[dict] = []
data_by_ticker: dict[str, pd.DataFrame] = {}
errors: list[str] = []
with st.spinner("ウォッチリストを分析しています…"):
    for ticker, company in active_watchlist.items():
        try:
            indicators = load_analysis(ticker)
            data_by_ticker[ticker] = indicators
            latest = indicators.iloc[-1]
            score = score_signal(indicators)
            adx_value = score["adx"]
            if adx_value is None:
                adx_display = "判定不可 (ADX取得不可)"
            else:
                adx_level = "強" if adx_value >= 25 else "中" if adx_value >= 20 else "弱"
                adx_display = f"{adx_level} ({adx_value:.1f})"

            atr_value = score["atr_percent"]
            if atr_value is None:
                atr_display = "判定不可 (ATR%取得不可)"
            else:
                atr_level = (
                    "大"
                    if atr_value >= 3.0
                    else "中"
                    if atr_value >= 1.5
                    else "小"
                )
                atr_display = f"{atr_level} ({atr_value:.2f}%)"

            rows.append(
                {
                    "銘柄": f"{ticker}: {company}",
                    "終値": latest["Close"],
                    "買い確度": score["buy_confidence"],
                    "売り確度": score["sell_confidence"],
                    "5年レンジ位置": score["range_position"],
                    "値位置": score["range_status"],
                    "相場判定": "レンジ" if score["is_ranging"] else "通常",
                    "トレンドの強さ": adx_display,
                    "値動きの激しさ": atr_display,
                    "恐怖指数": (
                        f"{latest['Fear_Greed_Label']} "
                        f"({int(np.floor(latest['Fear_Greed_Score'] + 0.5))})"
                        if pd.notna(latest["Fear_Greed_Score"])
                        else "判定不可"
                    ),
                }
            )
        except Exception as exc:
            errors.append(f"{ticker} ({company}): {exc}")

for error in errors:
    st.warning(f"分析に失敗しました — {error}")

if rows:
    summary = pd.DataFrame(rows).set_index("銘柄")
    st.dataframe(
        summary,
        width="stretch",
        alt="ウォッチリスト銘柄の売買確度、テクニカル指標、恐怖指数",
        column_config={
            "終値": st.column_config.NumberColumn(format="%.2f"),
            "買い確度": st.column_config.NumberColumn(format="%d%%"),
            "売り確度": st.column_config.NumberColumn(format="%d%%"),
            "5年レンジ位置": st.column_config.NumberColumn(format="%.1f%%"),
            "相場判定": st.column_config.TextColumn(
                help=(
                    "相場環境の判定状態。"
                    "・通常：上記以外の相場状態"
                    "・レンジ：ADX < 20 かつ ATR% < 1.5%"
                    "※レンジ判定では売買確度は自動で0%になります"
                )
            ),
            "トレンドの強さ": st.column_config.TextColumn(
                help=(
                    "ADXによるトレンドの強さ（方向性は示しません、日足14日基準）。"
                    "強：ADX 25以上、中：ADX 20以上25未満、弱：ADX 20未満。"
                    "括弧内は実際のADX値です。"
                ),
            ),
            "値動きの激しさ": st.column_config.TextColumn(
                help=(
                    "ATR%による値動きの大きさ（終値比、日足14日基準）。"
                    "大（激しい）：ATR% 3.0%以上、中（標準）：ATR% 1.5%以上3.0%未満、"
                    "小（穏やか）：ATR% 1.5%未満。括弧内は実際のATR%値です。"
                ),
            ),
            "恐怖指数": st.column_config.TextColumn(
                help=(
                    "25日移動平均乖離率（最大±25点）とRSI（最大±15点）を"
                    "基準点50に加算し、出来高が20日平均の2倍以上なら補正値を"
                    "1.3倍にします。スコアは0〜100に制限します。"
                    "判定：0〜20 極度の恐怖、21〜40 恐怖、41〜60 中立、"
                    "61〜80 貪欲、81〜100 極度の熱狂。"
                ),
            ),
        },
    )

    if st.session_state.get("selected_ticker") not in data_by_ticker:
        st.session_state["selected_ticker"] = next(iter(data_by_ticker))
    selected = st.selectbox(
        "詳細分析する銘柄",
        options=list(data_by_ticker),
        format_func=lambda ticker: f"{ticker}: {active_watchlist[ticker]}",
        key="selected_ticker",
    )
    indicators = data_by_ticker[selected]
    company = active_watchlist[selected]

    st.subheader("第2段階｜詳細分析・バックテスト")
    trend_data = None
    try:
        trend_data = load_trends(company)
    except Exception as exc:
        st.warning(
            "Googleトレンドを取得できませんでした。センチメント加点なしで分析します。"
            f"詳細: {exc}"
        )

    sentiment_rising = trend_data["rising"] if trend_data else None
    current_score = score_signal(indicators, sentiment_rising=sentiment_rising)
    fear_greed_score = indicators["Fear_Greed_Score"]
    latest_fear_greed = fear_greed_score.iloc[-1]
    previous_fear_greed = (
        fear_greed_score.iloc[-2] if len(fear_greed_score) > 1 else np.nan
    )
    latest_fear_greed_label = indicators["Fear_Greed_Label"].iloc[-1]
    fear_greed_delta = (
        int(np.floor(latest_fear_greed + 0.5))
        - int(np.floor(previous_fear_greed + 0.5))
        if pd.notna(latest_fear_greed) and pd.notna(previous_fear_greed)
        else None
    )
    backtest = backtest_strategy(
        indicators,
        threshold=SIGNAL_THRESHOLD,
        initial_capital=BACKTEST_INITIAL_CAPITAL,
    )
    rank, win_rate, confidence_score = (
        backtest["rank"],
        backtest["win_rate"],
        backtest["confidence_score"],
    )

    latest_reasons = list(
        dict.fromkeys(current_score["buy_reasons"] + current_score["sell_reasons"])
    )
    if current_score["adx"] is not None:
        latest_reasons.append(f"ADX {current_score['adx']:.1f}")
    if current_score["atr_percent"] is not None:
        latest_reasons.append(f"ATR {current_score['atr_percent']:.2f}%")
    comment_cache_key = (
        f"ai-comment-{selected}-{current_score['buy_confidence']}-"
        f"{current_score['sell_confidence']}-{rank}-{win_rate:.4f}-"
        f"{current_score['range_status']}-{current_score['is_ranging']}-"
        f"{sentiment_rising}"
    )
    if st.button("AI解説を再生成", key=f"refresh-{comment_cache_key}"):
        st.session_state.pop(comment_cache_key, None)
    if comment_cache_key not in st.session_state:
        with st.spinner("クオンツ解説を作成しています…"):
            st.session_state[comment_cache_key] = generate_ai_comment(
                ticker=selected,
                company=company,
                buy_confidence=current_score["buy_confidence"],
                sell_confidence=current_score["sell_confidence"],
                rank=rank,
                win_rate=win_rate,
                range_status=current_score["range_status"],
                is_ranging=current_score["is_ranging"],
                technical_signals=latest_reasons,
                range_position=current_score["range_position"],
                adx=current_score["adx"],
                atr_percent=current_score["atr_percent"],
            )
    st.subheader("🤖 ギャルたんクオンツの直感解説")
    st.info(st.session_state[comment_cache_key], icon="🤖", width="stretch")

    metric_columns = st.columns(6)
    metric_columns[0].metric("買い確度", f"{current_score['buy_confidence']}%")
    metric_columns[1].metric("売り確度", f"{current_score['sell_confidence']}%")
    metric_columns[2].metric("5年レンジ位置", current_score["range_status"])
    fear_greed_value = (
        f"{latest_fear_greed_label} "
        f"({int(np.floor(latest_fear_greed + 0.5))})"
        if pd.notna(latest_fear_greed)
        else "判定不可"
    )
    metric_columns[3].metric(
        "恐怖指数",
        fear_greed_value,
        f"{fear_greed_delta:+d}pt" if fear_greed_delta is not None else None,
        delta_color="off",
        help="括弧内は0〜100のスコアです。デルタは前営業日からのスコア差です。",
    )
    metric_columns[4].metric("AI信頼度", f"ランク{rank}", f"{confidence_score}%")
    metric_columns[5].metric("バックテスト勝率", f"{win_rate:.1f}%")

    signal_data = add_signal_flags(indicators, threshold=SIGNAL_THRESHOLD)
    figure = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.38, 0.12, 0.16, 0.11, 0.1, 0.13],
        subplot_titles=(
            "株価・移動平均・売買シグナル",
            "出来高",
            "MACD",
            "RSI",
            "ADX",
            "個別株 Fear & Greed Index",
        ),
    )
    figure.add_trace(
        go.Candlestick(
            x=indicators.index,
            open=indicators["Open"],
            high=indicators["High"],
            low=indicators["Low"],
            close=indicators["Close"],
            name="株価",
        ),
        row=1,
        col=1,
    )
    for column, label, color in (
        ("SMA_25", "25日線", "#f39c12"),
        ("SMA_75", "75日線", "#3498db"),
        ("SMA_200", "200日線", "#9b59b6"),
    ):
        figure.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators[column],
                name=label,
                line={"color": color},
            ),
            row=1,
            col=1,
        )
    for action, flag, price_col, reason_col, symbol, color in (
        ("買い", "Buy_Signal", "Low", "Buy_Signal_Reason", "triangle-up", "#16a085"),
        ("売り", "Sell_Signal", "High", "Sell_Signal_Reason", "triangle-down", "#e74c3c"),
    ):
        points = signal_data.loc[signal_data[flag]]
        if not points.empty:
            figure.add_trace(
                go.Scatter(
                    x=points.index,
                    y=points[price_col],
                    mode="markers",
                    name=f"{action}シグナル",
                    marker={"symbol": symbol, "size": 12, "color": color},
                    text=points[reason_col],
                    hovertemplate="%{x}<br>" + action + "シグナル<br>%{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    figure.add_trace(
        go.Bar(
            x=indicators.index,
            y=indicators["Volume"],
            name="出来高",
            marker_color="#7f8c8d",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=indicators.index,
            y=indicators["MACD_Hist"],
            name="MACDヒストグラム",
            marker_color="#95a5a6",
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=indicators.index, y=indicators["MACD"], name="MACD"),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=indicators.index,
            y=indicators["MACD_Signal"],
            name="MACDシグナル",
        ),
        row=3,
        col=1,
    )
    figure.add_hline(y=70, line_dash="dot", line_color="red", row=4, col=1)
    figure.add_hline(y=30, line_dash="dot", line_color="green", row=4, col=1)
    figure.add_trace(
        go.Scatter(x=indicators.index, y=indicators["RSI"], name="RSI"),
        row=4,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=indicators.index, y=indicators["ADX"], name="ADX"),
        row=5,
        col=1,
    )
    figure.add_hline(y=20, line_dash="dot", line_color="#e74c3c", row=6, col=1)
    figure.add_hline(y=80, line_dash="dot", line_color="#2ecc71", row=6, col=1)
    figure.add_trace(
        go.Scatter(
            x=indicators.index,
            y=fear_greed_score,
            name="Fear & Greed スコア",
            line={"color": "#8e44ad"},
        ),
        row=6,
        col=1,
    )
    figure.update_yaxes(title_text="出来高", row=2, col=1)
    figure.update_yaxes(range=[0, 100], title_text="スコア", row=6, col=1)

    figure.update_layout(
        title={
            "text": (
                f"{selected} {company} | 買い確度: {current_score['buy_confidence']}% "
                f"/ 売り確度: {current_score['sell_confidence']}% | "
                f"AI信頼度: ランク{rank} (勝率{win_rate:.1f}%)"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        height=1300,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02},
        margin={"t": 130},
    )
    figure.update_xaxes(range=[indicators.index.min(), indicators.index.max()])
    st.plotly_chart(
        figure,
        width="stretch",
        alt=(
            f"{selected} {company} の株価・出来高・テクニカル指標と"
            "Fear & Greed スコアの推移。"
            f"買い確度{current_score['buy_confidence']}%、"
            f"売り確度{current_score['sell_confidence']}%"
        ),
        theme="streamlit",
        key=f"detail-chart-{selected}",
    )

    st.markdown("**現在の判定理由**")
    st.write("買い:", " / ".join(current_score["buy_reasons"]) or "条件なし")
    st.write("売り:", " / ".join(current_score["sell_reasons"]) or "条件なし")
    if current_score["is_ranging"]:
        st.warning("ADX・ATRが低いレンジ相場と判定され、現在のシグナルは無効化されています。")
    if trend_data is None:
        st.caption("Googleトレンドのセンチメント加点はありません。")

    with st.container(border=True):
        st.subheader("簡易バックテスト")
        st.caption(
            f"{BACKTEST_PERIOD}の日次データ、シグナル閾値{SIGNAL_THRESHOLD}%、"
            f"初期資金{BACKTEST_INITIAL_CAPITAL:,.0f}円。"
            "終値で確定したシグナルを翌営業日の始値で約定する手数料・税金なしの簡易検証です。"
            "AI信頼度は勝率・PF・試行回数を使ったルールベース指標で、学習モデルではありません。"
        )
        bt_columns = st.columns(5)
        bt_columns[0].metric("信頼度ランク", f"ランク{rank}")
        bt_columns[1].metric("信頼度スコア", f"{confidence_score}%")
        bt_columns[2].metric("勝率 / 試行回数", f"{win_rate:.1f}%", f"{backtest['trade_count']}回")
        pf = backtest["profit_factor"]
        bt_columns[3].metric(
            "プロフィットファクター",
            "∞" if np.isinf(pf) else f"{pf:.2f}",
        )
        bt_columns[4].metric("最大ドローダウン", f"{backtest['max_drawdown_percent']:.1f}%")

        equity_figure = go.Figure(
            go.Scatter(
                x=backtest["equity_curve"].index,
                y=backtest["equity_curve"],
                mode="lines",
                name="資産推移",
                line={"color": "#3498db"},
            )
        )
        equity_figure.update_layout(
            title="バックテスト資産推移",
            xaxis_title="日付",
            yaxis_title="資産（円）",
            height=350,
            margin={"t": 55},
        )
        st.plotly_chart(
            equity_figure,
            width="stretch",
            alt="バックテスト期間の資産推移",
            key=f"equity-chart-{selected}",
        )
        if not backtest["trades"].empty:
            st.dataframe(
                backtest["trades"],
                width="stretch",
                hide_index=True,
                alt="バックテスト各トレードの結果",
                column_config={
                    "entry_date": "エントリー日",
                    "exit_date": "決済日",
                    "side": "売買",
                    "entry_price": st.column_config.NumberColumn("建値", format="%.2f"),
                    "exit_price": st.column_config.NumberColumn("決済値", format="%.2f"),
                    "return_percent": st.column_config.NumberColumn("損益率", format="%.2f%%"),
                },
            )
        else:
            st.info("指定した期間・条件ではバックテスト対象のトレードがありません。")

    if st.button("PNG画像を生成", key=f"generate-png-{selected}"):
        try:
            st.session_state["chart_png_data"] = figure.to_image(format="png")
            st.session_state["chart_png_ticker"] = selected
        except (ValueError, ImportError, RuntimeError) as exc:
            st.session_state.pop("chart_png_data", None)
            st.session_state.pop("chart_png_ticker", None)
            st.error(f"PNG出力に失敗しました。Kaleidoの状態を確認してください: {exc}")
    if (
        st.session_state.get("chart_png_ticker") == selected
        and st.session_state.get("chart_png_data")
    ):
        st.download_button(
            "生成したチャートをPNGでダウンロード",
            data=st.session_state["chart_png_data"],
            file_name=f"{selected}_analysis.png",
            mime="image/png",
            key=f"download-png-{selected}",
        )

    direction = (
        "買い"
        if current_score["buy_confidence"] >= current_score["sell_confidence"]
        else "売り"
    )
    confidence = max(current_score["buy_confidence"], current_score["sell_confidence"])
    reasons = current_score["buy_reasons"] if direction == "買い" else current_score["sell_reasons"]
    post_text = (
        f"【{company}（{selected}）テクニカル分析】\n"
        f"買い確度: {current_score['buy_confidence']}% / "
        f"売り確度: {current_score['sell_confidence']}%\n"
        f"値位置: {current_score['range_status']} "
        f"（{current_score['range_position']:.1f}%）\n"
        f"AI信頼度: ランク{rank}（勝率{win_rate:.1f}%）\n"
        f"シグナル: {direction}（確度 {confidence}%）\n"
        f"判定理由: {'、'.join(reasons) if reasons else '明確な条件なし'}\n"
        "※バックテストは手数料・税金を含まない参考値です。投資判断はご自身の責任でお願いします。"
    )
    st.markdown("**PostPrime投稿用テキスト**")
    st.code(post_text, language=None)
else:
    st.info("分析できる銘柄がありません。銘柄コードと株価データ接続を確認してください。")
