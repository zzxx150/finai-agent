"""
رادار بوش (Boosh Radar) — منصة تحليل الأخبار اللحظية وتوجيه المتداول
================================================================
تشغيل محلي:
    streamlit run app.py

المتطلبات:
    pip install -r requirements.txt

المفاتيح المطلوبة (يمكن إدخالها من الشريط الجانبي مباشرة أو عبر .env):
    - Finnhub API Key (مجاني): https://finnhub.io/register
    - OpenAI API Key: https://platform.openai.com/api-keys
    - (اختياري) Telegram Bot Token + Chat ID للتنبيهات
"""

import os
import datetime as dt

import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

from auth import verify_login, get_display_name, hash_password
from utils import (
    fetch_market_news,
    fetch_company_news,
    fetch_stock_snapshot,
    check_short_squeeze_potential,
    analyze_news_with_ai,
    classify_news_impact,
    filter_high_impact_news,
    calculate_position_size,
    send_telegram_alert,
    format_large_number,
    news_fingerprint,
    save_recommendation,
    get_all_recommendations,
    fetch_top_gainers,
    fetch_stocks_under_price,
    check_liquidity_activity,
    fetch_extended_hours_data,
    get_market_status,
    get_market_status_v2,
    calculate_support_resistance,
    send_ntfy_alert,
    translate_texts_to_arabic,
    translate_texts_free,
    calculate_technical_indicators,
    get_analyst_consensus,
    check_and_update_open_recommendations,
    get_win_rate_stats,
    create_app_user,
    get_app_user,
    add_price_alert,
    get_active_price_alerts,
    delete_price_alert,
    check_price_alerts,
    fetch_earnings_calendar,
    run_technical_backtest,
    has_recent_open_recommendation,
    calculate_confluence_score,
    fetch_market_trend,
    calculate_atr,
    check_sector_concentration,
    get_confluence_accuracy_stats,
    get_ticker_data,
    check_shariah_compliance,
    get_news_ticker_items,
    get_next_market_transition,
    get_market_sentiment_gauge,
    add_portfolio_position,
    get_portfolio_positions,
    delete_portfolio_position,
    calculate_portfolio_pnl,
    cleanup_duplicate_recommendations,
    enforce_atr_stop_floor,
)

load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    """
    يقرأ المفتاح من Streamlit Secrets (عند النشر على Streamlit Community Cloud)
    أو من متغيرات البيئة/.env (عند التشغيل المحلي).
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


def get_arabic_translations(texts, openai_api_key: str = "", model: str = "gpt-4o-mini"):
    """
    يترجم قائمة نصوص للعربية بالذكاء الاصطناعي (أدق وأوثق من الترجمة المجانية،
    وبدون حد يومي محدود يسبب رجوع النص الإنجليزي بصمت)، مع تخزين مؤقت بذاكرة
    الجلسة (session_state) بحيث أي عنوان خبر تُرجم مرة ما يُعاد ترجمته ثانية.
    لو ما توفر مفتاح OpenAI، يرجع تلقائياً للترجمة المجانية (MyMemory).
    """
    if "translation_cache" not in st.session_state:
        st.session_state["translation_cache"] = {}
    cache = st.session_state["translation_cache"]

    if not texts:
        return list(texts)

    if not openai_api_key:
        openai_api_key = get_secret("OPENAI_API_KEY", "")

    to_translate = [t for t in texts if t not in cache]
    if to_translate:
        if openai_api_key:
            translated = translate_texts_to_arabic(openai_api_key, to_translate, model=model)
        else:
            translated = translate_texts_free(to_translate)
        for orig, trans in zip(to_translate, translated):
            cache[orig] = trans

    return [cache.get(t, t) for t in texts]


def suppress_short_if_disabled(analysis: dict, shorts_enabled: bool) -> dict:
    """
    لو توصيات الشورت معطّلة بإعدادات المستخدم، يحوّل أي توصية 'دخول بيع (شورت)'
    إلى 'انتظار' مع توضيح السبب، بدل ما يعرضها أو يحفظها كصفقة قابلة للتنفيذ.
    """
    if shorts_enabled:
        return analysis
    plan = analysis.get("trade_plan", {})
    action = plan.get("action", "")
    if "بيع" in action or "شورت" in action:
        plan["action"] = "انتظار (توصية شورت مُعطّلة بإعداداتك)"
        plan["entry_note"] = "تم تعطيل توصيات البيع على المكشوف من الشريط الجانبي — فعّلها لو تبي تشوف هذا النوع من التوصيات."
        plan["entry_price"] = None
        plan["stop_loss_price"] = None
        plan["target_price"] = None
        analysis["trade_plan"] = plan
    return analysis


def enforce_neutral_wait(analysis: dict) -> dict:
    """
    منع تناقض منطقي: لو معنويات الخبر 'محايد'، ما يُسمح للإجراء يكون
    'دخول شراء' أو 'دخول بيع' — لازم يتحول لـ'انتظار' تلقائياً. هذا يمنع
    حالات كان الذكاء الاصطناعي يقترح فيها صفقة فعلية رغم إشارة ضعيفة/غير
    واضحة (خصوصاً بالتحليل الفني بدون خبر).
    """
    plan = analysis.get("trade_plan", {})
    action = plan.get("action", "")
    if analysis.get("sentiment") == "محايد" and ("دخول" in action):
        plan["action"] = "انتظار (معنويات محايدة — لا توجد إشارة واضحة كفاية للدخول)"
        plan["entry_price"] = None
        plan["stop_loss_price"] = None
        plan["target_price"] = None
        analysis["trade_plan"] = plan
    return analysis


st.set_page_config(
    page_title="رادار بوش (Boosh Radar)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# تنسيق مخصص: خط عربي أنيق، ألوان متناسقة، بطاقات مرتبة، مسافات محسّنة
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Tajawal:wght@400;500;700;900&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, span, div, label {
    font-family: 'Tajawal', 'Inter', 'Segoe UI', sans-serif !important;
}

/* ---------- خلفية عالمية احترافية: تدرج هادئ + شبكة نقطية خفيفة ---------- */
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: radial-gradient(rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size: 26px 26px;
    pointer-events: none;
    z-index: 0;
}
html, body {
    background-color: #06080D !important;
}
#bg-video {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    z-index: -100;
    opacity: 0.35;
}
#bg-video-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(160deg, rgba(6,8,13,0.75) 0%, rgba(10,14,22,0.85) 45%, rgba(13,18,32,0.9) 100%);
    z-index: -90;
}
/* ---------- كرات ضوء متوهجة متحركة (Glow Orbs) — حيوية قوية بالخلفية ---------- */
.glow-orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(70px);
    pointer-events: none;
    z-index: -30;
    opacity: 0.55;
}
.glow-orb-1 {
    width: 340px; height: 340px; top: -60px; right: -60px;
    background: radial-gradient(circle, #22D3A8, transparent 70%);
    animation: float-orb-1 14s ease-in-out infinite;
}
.glow-orb-2 {
    width: 300px; height: 300px; bottom: -80px; left: -60px;
    background: radial-gradient(circle, #4FD1FF, transparent 70%);
    animation: float-orb-2 18s ease-in-out infinite;
}
.glow-orb-3 {
    width: 240px; height: 240px; top: 40%; left: 45%;
    background: radial-gradient(circle, #A78BFA, transparent 70%);
    animation: float-orb-3 22s ease-in-out infinite;
}
@keyframes float-orb-1 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50%      { transform: translate(-40px, 50px) scale(1.15); }
}
@keyframes float-orb-2 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50%      { transform: translate(50px, -40px) scale(1.2); }
}
@keyframes float-orb-3 {
    0%, 100% { transform: translate(0, 0) scale(0.9); opacity: 0.35; }
    50%      { transform: translate(-30px, -30px) scale(1.1); opacity: 0.55; }
}

/* ---------- تأثير رادار حي يدور بالخلفية (يناسب اسم "رادار بوش") ---------- */
.radar-container {
    position: fixed;
    top: 50%; left: 50%;
    width: 900px; height: 900px;
    transform: translate(-50%, -50%);
    pointer-events: none;
    z-index: -20;
    opacity: 0.20;
}
.radar-ring {
    position: absolute;
    top: 50%; left: 50%;
    border: 1px solid rgba(34,211,168,0.35);
    border-radius: 50%;
    transform: translate(-50%, -50%);
}
.radar-ring-1 { width: 220px; height: 220px; }
.radar-ring-2 { width: 440px; height: 440px; }
.radar-ring-3 { width: 660px; height: 660px; }
.radar-ring-4 { width: 880px; height: 880px; }
.radar-sweep {
    position: absolute;
    top: 50%; left: 50%;
    width: 450px; height: 450px;
    transform: translate(-50%, -50%);
    border-radius: 50%;
    background: conic-gradient(
        from 0deg,
        rgba(34,211,168,0.55) 0deg,
        rgba(34,211,168,0.18) 25deg,
        transparent 70deg,
        transparent 360deg
    );
    animation: radar-spin 5s linear infinite;
}
@keyframes radar-spin {
    0%   { transform: translate(-50%, -50%) rotate(0deg); }
    100% { transform: translate(-50%, -50%) rotate(360deg); }
}
.radar-dot {
    position: absolute;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4FD1FF;
    box-shadow: 0 0 10px 2px rgba(79,209,255,0.8);
    animation: radar-blip 3s ease-in-out infinite;
}
@keyframes radar-blip {
    0%, 100% { opacity: 0.2; transform: scale(0.8); }
    50%      { opacity: 1; transform: scale(1.3); }
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090C13 0%, #0B0F18 100%);
    border-left: 1px solid rgba(255,255,255,0.06);
}

/* ---------- عنوان الصفحة الرئيسي ---------- */
h1 {
    font-weight: 900 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(100deg, #22D3A8 0%, #4FD1FF 55%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 4px;
}
h2, h3, h4 {
    font-weight: 700 !important;
    color: #EDF1F7 !important;
    letter-spacing: -0.2px;
}
h4 { border-right: 3px solid #22D3A8; padding-right: 10px; }

/* ---------- بطاقات المؤشرات (st.metric) بمستوى عالمي ---------- */
[data-testid="stMetric"] {
    background: linear-gradient(155deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: all 0.25s cubic-bezier(.2,.8,.2,1);
}
[data-testid="stMetric"]:hover {
    border-color: rgba(34,211,168,0.45);
    box-shadow: 0 8px 28px rgba(34,211,168,0.12), inset 0 1px 0 rgba(255,255,255,0.06);
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] { color: #8B96A8 !important; font-size: 0.82rem !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { color: #F5F8FC !important; font-weight: 800 !important; letter-spacing: -0.3px; }

/* ---------- التبويبات: كبسولات أنيقة بلمسة عالمية ---------- */
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 999px !important;
    padding: 7px 20px !important;
    margin-left: 4px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #A9B2C3 !important;
    font-weight: 500 !important;
    transition: all 0.2s ease;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    background: rgba(255,255,255,0.06);
    color: #EDF1F7 !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(100deg, #22D3A8, #4FD1FF) !important;
    color: #06110D !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(34,211,168,0.25);
}

/* ---------- الأزرار ---------- */
.stButton button, .stFormSubmitButton button, .stDownloadButton button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1px;
    transition: all 0.2s cubic-bezier(.2,.8,.2,1);
}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
    background: linear-gradient(100deg, #22D3A8, #1BB894) !important;
    border: none !important;
    color: #06110D !important;
}
.stButton button:hover, .stFormSubmitButton button:hover, .stDownloadButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(34,211,168,0.22);
}

/* ---------- صناديق التنبيهات ---------- */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    backdrop-filter: blur(6px);
}

/* ---------- الجداول ---------- */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.09);
    box-shadow: 0 4px 18px rgba(0,0,0,0.18);
}

/* ---------- الفاصل ---------- */
hr { border-color: rgba(255,255,255,0.08) !important; margin: 1.5rem 0 !important; }

/* ---------- الـ expander كبطاقة عالمية ---------- */
[data-testid="stExpander"] {
    background: linear-gradient(155deg, rgba(255,255,255,0.03), rgba(255,255,255,0.008));
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15);
}

/* ---------- خانات الإدخال ---------- */
input, textarea, select, .stSelectbox div[data-baseweb="select"] {
    border-radius: 10px !important;
}

/* ---------- شريط التمرير بلمسة عالمية ---------- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0B0E14; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #22D3A8, #1BB894); border-radius: 10px; }

/* ---------- تأثيرات ثلاثية الأبعاد (3D) للبطاقات ---------- */
[data-testid="stMetric"], [data-testid="stExpander"], [data-testid="stAlert"] {
    transform-style: preserve-3d;
    perspective: 800px;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px) rotateX(3deg) rotateY(-2deg) scale(1.01);
    box-shadow: 0 18px 38px rgba(0,0,0,0.35), 0 0 0 1px rgba(34,211,168,0.25), inset 0 1px 0 rgba(255,255,255,0.08);
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(34,211,168,0.3), 0 2px 0 rgba(0,0,0,0.3);
}
.stButton button[kind="primary"]:hover, .stFormSubmitButton button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.015);
    box-shadow: 0 10px 24px rgba(34,211,168,0.35), 0 2px 0 rgba(0,0,0,0.2);
}

/* ---------- الشريط المتحرك (Ticker) ---------- */
.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: linear-gradient(90deg, #0A0E16, #10141F, #0A0E16);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px 0;
    margin-bottom: 14px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
}
.ticker-track {
    display: inline-flex;
    white-space: nowrap;
    animation: ticker-scroll 40s linear infinite;
}
.ticker-wrap:hover .ticker-track { animation-play-state: paused; }
@keyframes ticker-scroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.ticker-item {
    display: inline-flex;
    align-items: center;
    padding: 0 22px;
    font-weight: 600;
    font-size: 0.92rem;
    border-left: 1px solid rgba(255,255,255,0.08);
}
.ticker-symbol { color: #C7CFDD; margin-left: 8px; }
.ticker-price { color: #F5F8FC; margin-left: 8px; }
.ticker-up { color: #34D399; }
.ticker-down { color: #F87171; }

/* ---------- شريط الأخبار العاجلة (الثاني) ---------- */
.news-ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: linear-gradient(90deg, #1A0E12, #240F14, #1A0E12);
    border: 1px solid rgba(248,113,113,0.25);
    border-radius: 12px;
    padding: 9px 0;
    margin-bottom: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03);
}
.news-ticker-track {
    display: inline-flex;
    white-space: nowrap;
    animation: news-ticker-scroll 55s linear infinite;
}
.news-ticker-wrap:hover .news-ticker-track { animation-play-state: paused; }
@keyframes news-ticker-scroll {
    0%   { transform: translateX(-50%); }
    100% { transform: translateX(0); }
}
.news-ticker-item {
    display: inline-flex;
    align-items: center;
    padding: 0 28px;
    font-weight: 500;
    font-size: 0.9rem;
    color: #F1D9DC;
    border-left: 1px solid rgba(248,113,113,0.2);
}
.news-ticker-tag {
    background: #F87171; color: #1A0E12; font-weight: 800;
    font-size: 0.72rem; padding: 2px 8px; border-radius: 999px; margin-left: 10px;
}

/* ---------- حيوية إضافية عامة بكل الموقع ---------- */
[data-testid="stAppViewContainer"] .main .block-container {
    animation: fade-slide-in 0.55s cubic-bezier(.2,.8,.2,1);
}
@keyframes fade-slide-in {
    0%   { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}
.stCheckbox, .stRadio, .stSelectbox, .stSlider {
    transition: all 0.2s ease;
}
.stCheckbox:hover, .stRadio:hover { transform: translateX(-2px); }
[data-testid="stDataFrame"] tbody tr:hover {
    background: rgba(34,211,168,0.06) !important;
    transition: background 0.15s ease;
}
.live-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #34D399; margin-left: 6px;
    box-shadow: 0 0 0 rgba(52,211,153,0.6);
    animation: pulse-live 1.8s infinite;
}
@keyframes pulse-live {
    0%   { box-shadow: 0 0 0 0 rgba(52,211,153,0.55); }
    70%  { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
    100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<video id="bg-video" autoplay loop muted playsinline>'
    '<source src="data:video/mp4;base64,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" type="video/mp4">'
    '</video>'
    '<div id="bg-video-overlay"></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="glow-orb glow-orb-1"></div>'
    '<div class="glow-orb glow-orb-2"></div>'
    '<div class="glow-orb glow-orb-3"></div>'
    '<div class="radar-container">'
    '  <div class="radar-ring radar-ring-1"></div>'
    '  <div class="radar-ring radar-ring-2"></div>'
    '  <div class="radar-ring radar-ring-3"></div>'
    '  <div class="radar-ring radar-ring-4"></div>'
    '  <div class="radar-sweep"></div>'
    '  <div class="radar-dot" style="top:30%; left:65%; animation-delay:0s;"></div>'
    '  <div class="radar-dot" style="top:60%; left:40%; animation-delay:1.2s;"></div>'
    '  <div class="radar-dot" style="top:45%; left:75%; animation-delay:2.1s;"></div>'
    '</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# شريط الأسعار المتحرك (Ticker) — يظهر بأعلى كل صفحات التطبيق
# ----------------------------------------------------------------------
st_autorefresh(interval=60_000, key="header_ticker_autorefresh")

if "ticker_data" not in st.session_state or "ticker_fetched_at" not in st.session_state or \
   (dt.datetime.now() - st.session_state.get("ticker_fetched_at", dt.datetime.min)).total_seconds() > 60:
    st.session_state["ticker_data"] = get_ticker_data()
    st.session_state["ticker_fetched_at"] = dt.datetime.now()

_ticker_rows = st.session_state.get("ticker_data", [])
if _ticker_rows:
    _items_html = ""
    for _row in _ticker_rows * 2:  # تكرار القائمة مرتين لضمان تمرير سلس بلا فراغ
        _cls = "ticker-up" if _row["change_pct"] >= 0 else "ticker-down"
        _arrow = "▲" if _row["change_pct"] >= 0 else "▼"
        _items_html += (
            f'<span class="ticker-item"><span class="ticker-symbol">{_row["symbol"]}</span>'
            f'<span class="ticker-price">${_row["price"]:,}</span>'
            f'<span class="{_cls}">{_arrow} {abs(_row["change_pct"])}%</span></span>'
        )
    st.markdown(
        f'<div class="ticker-wrap"><div class="ticker-track">{_items_html}</div></div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# شريط الأخبار العاجلة المتحرك (الثاني) — أهم الأخبار المؤثرة بالسوق
# ----------------------------------------------------------------------
if "news_ticker_data" not in st.session_state or "news_ticker_fetched_at" not in st.session_state or \
   (dt.datetime.now() - st.session_state.get("news_ticker_fetched_at", dt.datetime.min)).total_seconds() > 60:
    _finnhub_key_for_ticker = get_secret("FINNHUB_API_KEY", "")
    st.session_state["news_ticker_data"] = get_news_ticker_items(_finnhub_key_for_ticker) if _finnhub_key_for_ticker else []
    st.session_state["news_ticker_fetched_at"] = dt.datetime.now()

_news_ticker_rows = st.session_state.get("news_ticker_data", [])
if _news_ticker_rows:
    _translated_headlines = get_arabic_translations([r["headline"] for r in _news_ticker_rows])
    _news_items_html = ""
    for _row, _headline_ar in list(zip(_news_ticker_rows, _translated_headlines)) * 2:
        _news_items_html += (
            f'<span class="news-ticker-item"><span class="news-ticker-tag">عاجل</span>'
            f'{_headline_ar} — {_row.get("source", "")}</span>'
        )
    st.markdown(
        f'<div class="news-ticker-wrap"><div class="news-ticker-track">{_news_items_html}</div></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# بوابة تسجيل الدخول — تظهر قبل أي محتوى آخر في التطبيق
# ----------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None

if not st.session_state["authenticated"]:
    st.title("🔐 تسجيل الدخول")
    st.caption("منصة رادار بوش (Boosh Radar) — الدخول مقتصر على المستخدمين المصرّح لهم")

    login_tab, signup_tab = st.tabs(["دخول", "🆕 حساب جديد"])

    with login_tab:
        with st.form("login_form"):
            login_user = st.text_input("اسم المستخدم")
            login_pass = st.text_input("كلمة المرور", type="password")
            submitted = st.form_submit_button("دخول", type="primary", use_container_width=True)
        if submitted:
            if verify_login(login_user, login_pass):
                st.session_state["authenticated"] = True
                st.session_state["username"] = login_user
                st.rerun()
            else:
                st.error("اسم المستخدم أو كلمة المرور غير صحيحة.")

    with signup_tab:
        signup_code_required = get_secret("SIGNUP_CODE", "")
        if not signup_code_required:
            st.info("إنشاء حساب جديد غير مفعّل حالياً. تواصل مع صاحب المنصة.")
        else:
            st.caption("اطلب رمز الدعوة من صاحب المنصة قبل التسجيل.")
            with st.form("signup_form"):
                new_username = st.text_input("اسم المستخدم الجديد (بالإنجليزي، بدون مسافات)")
                new_name = st.text_input("اسمك (يظهر بالترحيب)")
                new_pass = st.text_input("كلمة المرور", type="password")
                new_pass_confirm = st.text_input("تأكيد كلمة المرور", type="password")
                invite_code = st.text_input("رمز الدعوة", type="password")
                signup_submitted = st.form_submit_button("إنشاء الحساب", type="primary", use_container_width=True)

            if signup_submitted:
                if invite_code != signup_code_required:
                    st.error("رمز الدعوة غير صحيح.")
                elif not new_username or not new_pass:
                    st.error("لازم تعبّي اسم المستخدم وكلمة المرور.")
                elif " " in new_username:
                    st.error("اسم المستخدم ما يقبل مسافات.")
                elif new_pass != new_pass_confirm:
                    st.error("كلمتا المرور غير متطابقتين.")
                elif len(new_pass) < 6:
                    st.error("كلمة المرور لازم تكون 6 أحرف على الأقل.")
                elif get_app_user(new_username) or new_username in ["admin"]:
                    st.error("اسم المستخدم محجوز، اختر اسماً ثانياً.")
                else:
                    created = create_app_user(new_username, new_name or new_username, hash_password(new_pass))
                    if created:
                        st.success("✅ تم إنشاء الحساب بنجاح! روح لتبويب 'دخول' وسجّل دخولك الآن.")
                    else:
                        st.error("اسم المستخدم محجوز، اختر اسماً ثانياً.")

    st.stop()

# ----------------------------------------------------------------------
# الشريط الجانبي: إعدادات المفاتيح
# ----------------------------------------------------------------------
with st.sidebar:
    st.success(f"مرحباً {get_display_name(st.session_state['username'])} 👋")
    if st.button("🚪 تسجيل الخروج", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.rerun()

    st.divider()
    st.title("⚙️ الإعدادات")
    st.caption("أدخل مفاتيحك الخاصة، أو ضعها في ملف .env")

    finnhub_key = st.text_input(
        "Finnhub API Key",
        value=get_secret("FINNHUB_API_KEY", ""),
        type="password",
        help="مجاني من finnhub.io",
    )
    openai_key = st.text_input(
        "OpenAI API Key",
        value=get_secret("OPENAI_API_KEY", ""),
        type="password",
    )

    st.divider()
    st.subheader("🔔 تنبيهات تلغرام (اختياري)")
    telegram_token = st.text_input("Telegram Bot Token", value=get_secret("TELEGRAM_BOT_TOKEN", ""), type="password")
    telegram_chat_id = st.text_input("Telegram Chat ID", value=get_secret("TELEGRAM_CHAT_ID", ""))
    enable_telegram = st.checkbox("تفعيل إرسال تنبيه عند العثور على خبر عالي التأثير", value=False)

    st.divider()
    st.subheader("📱 تنبيهات ntfy (اختياري، بديل مجاني)")
    ntfy_topic = st.text_input(
        "اسم قناة ntfy السرية",
        value=get_secret("NTFY_TOPIC", ""),
        help="مثال: finai_alerts_boosh_2026_x7k9m",
    )
    enable_ntfy = st.checkbox("تفعيل إرسال إشعار ntfy عند العثور على خبر عالي التأثير", value=False)

    st.divider()
    st.markdown("**📨 إرسال رسالة يدوية للجميع**")
    broadcast_text = st.text_area(
        "اكتب أي نص وأرسله فوراً لكل المشتركين (تلغرام و/أو ntfy)",
        key="broadcast_text",
        height=80,
        placeholder="مثال: خبر عاجل، تنبيه شخصي، أو أي رسالة تبي ترسلها...",
    )
    if st.button("📤 إرسال الآن", use_container_width=True, key="send_broadcast"):
        if not broadcast_text.strip():
            st.warning("اكتب نص الرسالة أول.")
        elif not telegram_token and not telegram_chat_id and not ntfy_topic:
            st.error("لازم تعبّي بيانات تلغرام أو ntfy أول من الأعلى.")
        else:
            results = []
            if telegram_token and telegram_chat_id:
                results.append(("تلغرام", send_telegram_alert(telegram_token, telegram_chat_id, broadcast_text.strip())))
            if ntfy_topic:
                results.append(("ntfy", send_ntfy_alert(ntfy_topic, broadcast_text.strip(), title="📨 رسالة يدوية")))
            for name, ok in results:
                if ok:
                    st.success(f"✅ تم الإرسال عبر {name}.")
                else:
                    st.error(f"❌ فشل الإرسال عبر {name} — تأكد من صحة الإعدادات.")

    st.divider()
    ai_model = st.selectbox("نموذج الذكاء الاصطناعي", ["gpt-4o-mini", "gpt-4o"], index=0)

    st.divider()
    enable_shorts = st.checkbox(
        "🔻 تفعيل توصيات البيع على المكشوف (شورت)",
        value=False,
        help="مطفّي افتراضياً. البيع على المكشوف يحتاج حساب هامش (Margin Account) "
             "عند وسيطك، وخسارته المحتملة غير محدودة نظرياً (بعكس الشراء العادي). "
             "لا تفعّله إلا لو متأكد إن حسابك يدعمه وفاهم المخاطرة.",
    )

    st.divider()
    st.caption("⚠️ هذه المنصة أداة استرشادية تعليمية وليست توصية استثمارية. القرار والمسؤولية تقع على المتداول.")

# ----------------------------------------------------------------------
# رأس الصفحة
# ----------------------------------------------------------------------
st.title("📡 رادار بوش (Boosh Radar)")
st.caption("منصة تحليل الأخبار اللحظية بالذكاء الاصطناعي وتوجيه المتداول")

# ----------------------------------------------------------------------
# شريط حالة السوق الأمريكي (مفتوح / ما قبل الافتتاح / ما بعد الإغلاق / عطلة)
# ----------------------------------------------------------------------
market_status = get_market_status_v2()
status_icon = {
    "السوق مفتوح (تداول رسمي)": "🟢",
    "ما قبل الافتتاح (Pre-Market)": "🟡",
    "ما بعد الإغلاق (After-Hours)": "🟠",
}.get(market_status["status"], "🔴")

status_cols = st.columns([2, 2, 2])
status_cols[0].markdown(f"**{status_icon} حالة السوق:** {market_status['status']}")
status_cols[1].markdown(f"**🇸🇦 الوقت الآن (السعودية):** {market_status.get('now_saudi', market_status.get('now_et', '—'))}")
if market_status.get("next_holiday"):
    status_cols[2].markdown(
        f"**📅 أقرب عطلة:** {market_status['next_holiday']['name']} ({market_status['next_holiday']['date']})"
    )
elif market_status.get("detail"):
    status_cols[2].markdown(f"**ℹ️ ملاحظة:** {market_status['detail']}")

# ---- عدّاد تنازلي حي لفتح/إغلاق السوق + مقياس مزاج السوق ----
gauge_col1, gauge_col2 = st.columns([1, 1])

with gauge_col1:
    transition = get_next_market_transition()
    if "error" not in transition:
        st.markdown(f"**⏱️ {transition['label']}:**")
        components.html(f"""
        <div id="countdown-box" style="
            font-family:'Tajawal',sans-serif; font-size:2rem; font-weight:800;
            background:linear-gradient(100deg,#22D3A8,#4FD1FF);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            direction:ltr; text-align:right; letter-spacing:1px;">--:--:--</div>
        <script>
        const target = new Date("{transition['target_utc_iso']}").getTime();
        function tick() {{
            const now = new Date().getTime();
            let diff = Math.max(0, target - now);
            const h = String(Math.floor(diff / 3600000)).padStart(2,'0');
            const m = String(Math.floor((diff % 3600000) / 60000)).padStart(2,'0');
            const s = String(Math.floor((diff % 60000) / 1000)).padStart(2,'0');
            document.getElementById('countdown-box').innerText = h + ':' + m + ':' + s;
        }}
        tick();
        setInterval(tick, 1000);
        </script>
        """, height=55)

with gauge_col2:
    gauge = get_market_sentiment_gauge()
    st.markdown(f"**🧭 مقياس مزاج تحليلاتك (آخر 48 ساعة):**")
    _needle_angle = -90 + (gauge["score"] / 100) * 180
    components.html(f"""
    <div style="font-family:'Tajawal',sans-serif; text-align:center;">
      <svg width="180" height="100" viewBox="0 0 180 100">
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#F87171"/>
            <stop offset="50%" stop-color="#FBBF24"/>
            <stop offset="100%" stop-color="#34D399"/>
          </linearGradient>
        </defs>
        <path d="M 10 90 A 80 80 0 0 1 170 90" fill="none" stroke="url(#gaugeGrad)" stroke-width="14" stroke-linecap="round"/>
        <line id="needle" x1="90" y1="90" x2="90" y2="25" stroke="#F5F8FC" stroke-width="3"
              style="transform-origin:90px 90px; transform:rotate(0deg); transition: transform 1.2s cubic-bezier(.2,.8,.2,1);"/>
        <circle cx="90" cy="90" r="5" fill="#F5F8FC"/>
      </svg>
      <div style="color:#F5F8FC; font-weight:800; font-size:1.1rem; margin-top:-6px;">{gauge['score']}/100</div>
      <div style="color:#A9B2C3; font-size:0.82rem;">{gauge['label']}</div>
    </div>
    <script>
    setTimeout(() => {{
        document.getElementById('needle').style.transform = 'rotate({_needle_angle}deg)';
    }}, 150);
    </script>
    """, height=170)
    st.caption(f"مبني على {gauge['count']} تحليل محفوظ خلال آخر 48 ساعة — مقياس خاص بنشاطك، وليس مؤشراً رسمياً للسوق.")

with st.expander("🕐 أوقات التداول الكاملة بالسوق الأمريكي (بتوقيت السعودية)"):
    if "regular_hours_saudi" in market_status:
        st.write(f"**ما قبل الافتتاح (Pre-Market):** {market_status['pre_market_hours_saudi']} (توقيت السعودية)")
        st.write(f"**التداول الرسمي:** {market_status['regular_hours_saudi']} (توقيت السعودية)")
        st.write(f"**ما بعد الإغلاق (After-Hours):** {market_status['after_hours_saudi']} (توقيت السعودية)")
        st.caption(f"بتوقيت نيويورك (ET) للمرجعية: {market_status.get('regular_hours', '—')}")
    elif "regular_hours" in market_status:
        st.write(f"**ما قبل الافتتاح (Pre-Market):** {market_status['pre_market_hours']}")
        st.write(f"**التداول الرسمي:** {market_status['regular_hours']}")
        st.write(f"**ما بعد الإغلاق (After-Hours):** {market_status['after_hours']}")
    st.caption(
        "الأوقات تُحسب تلقائياً مع مراعاة التوقيت الصيفي/الشتوي لكل من نيويورك والسعودية. "
        "الأوقات والعطل معتمدة من الجدول الرسمي لبورصة نيويورك (NYSE) لعام 2026."

    )

(
    tab_dashboard,
    tab_monitor,
    tab_gainers,
    tab_search,
    tab_recommendations,
    tab_portfolio,
    tab_earnings,
    tab_alerts,
    tab_risk,
    tab_watch,
) = st.tabs(
    [
        "🗺️ لوحة الأخبار اليومية",
        "🚨 مراقبة لحظية",
        "🏆 الأكثر ربحاً",
        "🔍 تحليل سهم محدد",
        "📂 سجل التوصيات",
        "💼 محفظتي",
        "📅 تقويم الأرباح",
        "🔔 تنبيهات سعرية",
        "🎯 حاسبة المخاطر",
        "📋 متابعة سريعة",
    ]
)

# ========================================================================
# التبويب 1: لوحة الأخبار اليومية + خريطة الحرارة + فلترة عالية التأثير
# ========================================================================
with tab_dashboard:
    col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
    with col_a:
        st.subheader("الأخبار اللحظية للسوق")
    with col_b:
        news_category = st.selectbox("التصنيف", ["general", "merger", "forex", "crypto"], index=0)
    with col_c:
        only_high_impact = st.checkbox("عرض عالي التأثير فقط", value=True)
    with col_d:
        translate_dashboard = st.checkbox("🌐 ترجمة للعربية (مجانية)", value=True, key="translate_dashboard")

    if st.button("🔄 تحديث الأخبار الآن", type="primary"):
        st.session_state["refresh_news"] = True

    if not finnhub_key:
        st.warning("الرجاء إدخال مفتاح Finnhub من الشريط الجانبي لعرض الأخبار.")
    else:
        with st.spinner("جاري جلب الأخبار..."):
            news_items = fetch_market_news(finnhub_key, category=news_category, limit=50)

        if news_items and "error" in news_items[0]:
            st.error(f"خطأ في جلب الأخبار: {news_items[0]['error']}")
        elif not news_items:
            st.info("لا توجد أخبار متاحة حالياً.")
        else:
            display_items = filter_high_impact_news(news_items) if only_high_impact else news_items
            st.success(f"تم العثور على {len(display_items)} خبراً من أصل {len(news_items)}")

            # خريطة حرارية مبسطة: تصنيف مبدئي (بدون AI) لكل خبر حسب الكلمات المفتاحية
            display_items = display_items[:30]
            headlines_raw = [item.get("headline", "") for item in display_items]
            if translate_dashboard:
                with st.spinner("جاري ترجمة العناوين..."):
                    headlines_ar = get_arabic_translations(headlines_raw)
            else:
                headlines_ar = headlines_raw

            heat_rows = []
            for item, headline_display in zip(display_items, headlines_ar):
                classification = classify_news_impact(item.get("headline", ""))
                heat_rows.append({
                    "المصدر": item.get("source", "—"),
                    "العنوان": headline_display[:100],
                    "التصنيف الأولي": classification,
                    "الوقت": dt.datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d %H:%M") if item.get("datetime") else "—",
                    "الرابط": item.get("url", ""),
                })

            if heat_rows:
                df = pd.DataFrame(heat_rows)
                fig = px.bar(
                    df["التصنيف الأولي"].value_counts().reset_index(),
                    x="count", y="التصنيف الأولي", orientation="h",
                    color="التصنيف الأولي",
                    color_discrete_map={"مرشّح (High Impact)": "#e74c3c", "عادي (Low Impact)": "#95a5a6"},
                    title="توزيع الأخبار حسب مستوى التأثير الأولي",
                )
                fig.update_layout(showlegend=False, height=250, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={"الرابط": st.column_config.LinkColumn("الرابط")},
                    hide_index=True,
                )

                # تنبيه تلقائي (تلغرام و/أو ntfy) لأول خبر عالي التأثير في الجلسة الحالية
                high_impact_only = [r for r in heat_rows if r["التصنيف الأولي"] == "مرشّح (High Impact)"]
                if high_impact_only and not st.session_state.get("alert_sent_once"):
                    msg = f"🚨 خبر عالي التأثير:\n{high_impact_only[0]['العنوان']}\nالمصدر: {high_impact_only[0]['المصدر']}"
                    any_sent = False
                    if enable_telegram and telegram_token and telegram_chat_id:
                        any_sent = send_telegram_alert(telegram_token, telegram_chat_id, msg) or any_sent
                    if enable_ntfy and ntfy_topic:
                        any_sent = send_ntfy_alert(ntfy_topic, msg, title="🚨 خبر عالي التأثير", priority=4) or any_sent
                    st.session_state["alert_sent_once"] = True
                    if any_sent:
                        st.toast("تم إرسال التنبيه ✅")
            else:
                st.info("لا توجد أخبار مطابقة للفلترة الحالية.")

    st.divider()
    st.caption(
        "ملاحظة: التصنيف الظاهر في الجدول أعلاه هو تصنيف أولي سريع بالكلمات المفتاحية "
        "(بدون استهلاك رصيد الذكاء الاصطناعي). للتحليل العميق والتوصية الكاملة، استخدم تبويب "
        "'تحليل سهم محدد' لكل خبر تريد التعمق فيه."
    )

# ========================================================================
# التبويب 2: تحليل سهم محدد (بحث + سعر + خبر + AI + شارت + short squeeze)
# ========================================================================
with tab_search:
    st.subheader("ابحث عن سهم لتحليله بالكامل")
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol_input = st.text_input("رمز السهم (مثال: AAPL, NVDA, TSLA)", value="").upper().strip()
    with col2:
        analyze_btn = st.button("🔎 تحليل السهم", type="primary", use_container_width=True)

    # نحفظ الرمز بذاكرة الجلسة عشان النتائج ما تختفي لما يضغط المستخدم
    # على زر "حلّل هذا الخبر" جوه القائمة (اللي يسبب إعادة تحميل الصفحة)
    if analyze_btn and symbol_input:
        st.session_state["active_symbol"] = symbol_input

    active_symbol = st.session_state.get("active_symbol")

    if active_symbol:
        symbol_input = active_symbol
        with st.spinner(f"جاري جلب بيانات {symbol_input}..."):
            snapshot = fetch_stock_snapshot(symbol_input)

        if "error" in snapshot:
            st.error(f"تعذر جلب بيانات السهم: {snapshot['error']}")
        else:
            # ---- ملخص السعر ----
            price_cols = st.columns(5)
            price_valid = snapshot['current_price'] is not None and not pd.isna(snapshot['current_price'])
            price_cols[0].metric("السعر الحالي", f"${snapshot['current_price']:.2f}" if price_valid else "غير متوفر حالياً")
            change = snapshot.get("change_pct")
            price_cols[1].metric("التغير اليومي", f"{change:.2f}%" if change is not None else "—",
                                  delta=f"{change:.2f}%" if change is not None else None)
            price_cols[2].metric("القيمة السوقية", format_large_number(snapshot.get("market_cap")))
            price_cols[3].metric("متوسط حجم التداول", format_large_number(snapshot.get("avg_volume")))
            price_cols[4].metric("القطاع", snapshot.get("sector", "—"))

            # ---- فحص التوافق الشرعي (استرشادي) ----
            with st.spinner("جاري فحص التوافق الشرعي..."):
                shariah_info = check_shariah_compliance(symbol_input)
            shariah_status = shariah_info.get("status", "غير محدد")
            shariah_badge = {"متوافق تقريباً": "🟢", "غير متوافق": "🔴", "غير محدد": "⚪"}.get(shariah_status, "⚪")
            st.markdown(f"#### {shariah_badge} التوافق الشرعي: {shariah_status}")
            st.caption(shariah_info.get("reason", ""))
            if shariah_info.get("debt_ratio") is not None:
                sh1, sh2 = st.columns(2)
                sh1.metric("نسبة الدين/القيمة السوقية", f"{shariah_info['debt_ratio']}%")
                sh2.metric("نسبة النقد/القيمة السوقية", f"{shariah_info['cash_ratio']}%")
            st.caption(
                "⚠️ هذا تصنيف آلي استرشادي بمنهجية مالية شائعة (شبيهة بـ AAOIFI/Dow Jones Islamic Market)، "
                "وليس فتوى شرعية. راجع مصدراً شرعياً موثوقاً أو خدمة متخصصة قبل الاعتماد عليه بقرار استثماري."
            )

            # ---- فحص الشورت والفلوت ----
            st.markdown("#### 📉 فحص السيولة والضغط الشرائي")
            squeeze_info = check_short_squeeze_potential(snapshot)
            sq_cols = st.columns(3)
            sq_cols[0].metric("Float منخفض؟", "نعم ⚠️" if squeeze_info["low_float"] else "لا")
            sq_cols[1].metric("نسبة الشورت من الفلوت", f"{squeeze_info.get('short_percent_display', '—')}%")
            sq_cols[2].info(squeeze_info["squeeze_potential"])

            # ---- مستويات الدعم والمقاومة ----
            st.markdown("#### 📐 مستويات الدعم والمقاومة")
            with st.spinner("جاري حساب الدعم والمقاومة..."):
                sr_info = calculate_support_resistance(symbol_input)
            if "error" in sr_info:
                st.caption(sr_info["error"])
            else:
                sr_cols = st.columns(4)
                sr_cols[0].metric("دعم (20 يوم)", f"${sr_info['support_20d']:.2f}")
                sr_cols[1].metric("مقاومة (20 يوم)", f"${sr_info['resistance_20d']:.2f}")
                sr_cols[2].metric("دعم (50 يوم)", f"${sr_info['support_50d']:.2f}")
                sr_cols[3].metric("مقاومة (50 يوم)", f"${sr_info['resistance_50d']:.2f}")
                if sr_info.get("broke_resistance") or sr_info.get("broke_support"):
                    st.warning(sr_info["signal"])
                else:
                    st.info(sr_info["signal"])
            st.caption("حساب فني مبسط بناءً على أعلى/أدنى سعر خلال الفترة، وليس تحليلاً فنياً احترافياً كاملاً.")

            # ---- دخول السيولة اللحظي ----
            st.markdown("#### 💧 هل دخلت سيولة على السهم مؤخراً؟")
            with st.spinner("جاري فحص حجم التداول اللحظي..."):
                liquidity_info = check_liquidity_activity(symbol_input)
            if "error" in liquidity_info:
                st.caption(liquidity_info.get("status", "تعذر الفحص."))
            else:
                st.info(liquidity_info.get("status", "—"))
                if liquidity_info.get("spike_volume"):
                    lc1, lc2 = st.columns(2)
                    lc1.metric("حجم الشمعة عند الدخول", format_large_number(liquidity_info["spike_volume"]))
                    lc2.metric("متوسط الحجم (كل 5 دقائق)", format_large_number(liquidity_info["avg_volume_5m"]))
            st.caption("هذا تقدير آلي من حجم التداول الظاهر، وليس بيانات تدفق أوامر حقيقية (Level 2).")

            # ---- المؤشرات الفنية: RSI و MACD ----
            st.markdown("#### 📈 المؤشرات الفنية (RSI و MACD)")
            with st.spinner("جاري حساب المؤشرات الفنية..."):
                tech_info = calculate_technical_indicators(symbol_input)
            if "error" in tech_info:
                st.caption(tech_info["error"])
            else:
                tc1, tc2 = st.columns(2)
                tc1.metric("RSI (14)", tech_info["rsi"])
                tc1.caption(tech_info["rsi_label"])
                tc2.info(tech_info["macd_signal"])

            # ---- إجماع المحللين ----
            st.markdown("#### 🏦 إجماع المحللين (لو متوفر)")
            analyst_info = get_analyst_consensus(symbol_input)
            if "error" in analyst_info or not analyst_info.get("num_analysts"):
                st.caption("لا تتوفر بيانات إجماع محللين لهذا السهم (غالباً يتوفر بس للأسهم الكبيرة والمتوسطة).")
            else:
                ac1, ac2, ac3, ac4 = st.columns(4)
                ac1.metric("توصية المحللين", analyst_info.get("recommendation", "—"))
                ac2.metric("متوسط السعر المستهدف", f"${analyst_info['target_mean']:.2f}" if analyst_info.get("target_mean") else "—")
                ac3.metric("أعلى سعر مستهدف", f"${analyst_info['target_high']:.2f}" if analyst_info.get("target_high") else "—")
                ac4.metric("عدد المحللين", analyst_info.get("num_analysts", "—"))

            # ---- أسعار ما قبل الافتتاح وما بعد الإغلاق ----
            st.markdown("#### 🌙 التداول قبل وبعد السوق الأمريكي")
            ext_hours = fetch_extended_hours_data(symbol_input)
            if "error" in ext_hours:
                st.caption("تعذر جلب بيانات ما قبل/بعد السوق حالياً.")
            else:
                eh1, eh2, eh3 = st.columns(3)
                pre_price = ext_hours.get("pre_market_price")
                post_price = ext_hours.get("post_market_price")
                eh1.metric(
                    "قبل الافتتاح (Pre-Market)",
                    f"${pre_price:.2f}" if pre_price else "غير متوفر الآن",
                    delta=f"{ext_hours['pre_market_change_pct']:.2f}%" if ext_hours.get("pre_market_change_pct") else None,
                )
                eh2.metric("السعر الرسمي الحالي", f"${ext_hours.get('regular_market_price'):.2f}" if ext_hours.get("regular_market_price") else "—")
                eh3.metric(
                    "بعد الإغلاق (After-Hours)",
                    f"${post_price:.2f}" if post_price else "غير متوفر الآن",
                    delta=f"{ext_hours['post_market_change_pct']:.2f}%" if ext_hours.get("post_market_change_pct") else None,
                )
                st.caption("تظهر هذي الأسعار فقط خلال ساعات ما قبل الافتتاح أو ما بعد الإغلاق الفعلية، وتختفي أثناء التداول الرسمي.")

            # ---- خلاصة تساعد على اتخاذ القرار ----
            st.markdown("#### 🧭 خلاصة سريعة تساعدك على القرار")
            decision_points = []
            score = 0

            if snapshot.get("change_pct") is not None:
                if snapshot["change_pct"] > 2:
                    decision_points.append("✅ السهم في زخم صعودي قوي اليوم")
                    score += 1
                elif snapshot["change_pct"] < -2:
                    decision_points.append("⚠️ السهم في تراجع ملحوظ اليوم")
                    score -= 1

            if "error" not in sr_info:
                if sr_info.get("broke_resistance"):
                    decision_points.append("✅ اخترق مستوى مقاومة مهم مؤخراً")
                    score += 1
                elif sr_info.get("broke_support"):
                    decision_points.append("⚠️ كسر مستوى دعم مهم مؤخراً")
                    score -= 1

            if "error" not in tech_info:
                if tech_info["rsi"] <= 30:
                    decision_points.append("🟢 RSI يشير لتشبّع بيعي (منطقة ارتداد محتملة)")
                    score += 1
                elif tech_info["rsi"] >= 70:
                    decision_points.append("🔴 RSI يشير لتشبّع شرائي (منطقة تصحيح محتملة)")
                    score -= 1

            if squeeze_info.get("low_float") and squeeze_info.get("high_short_interest"):
                decision_points.append("🔥 فرصة Short Squeeze محتملة (فلوت منخفض + شورت مرتفع)")
                score += 1

            if "error" not in liquidity_info and "الآن" in liquidity_info.get("status", ""):
                decision_points.append("💧 دخلت سيولة ملحوظة على السهم الآن")
                score += 1

            if not decision_points:
                st.info("لا توجد إشارات قوية واضحة حالياً — السهم يتداول بشكل طبيعي بدون مؤشرات استثنائية.")
            else:
                for point in decision_points:
                    st.write(point)
                if score >= 2:
                    st.success("📈 الإشارات مجتمعة تميل نحو الإيجابية — قد تستحق مراقبة أقرب للدخول.")
                elif score <= -2:
                    st.error("📉 الإشارات مجتمعة تميل نحو السلبية — يُفضّل الحذر.")
                else:
                    st.info("الإشارات متضاربة أو محايدة — يُفضّل انتظار وضوح أكبر قبل القرار.")
            st.caption("هذي خلاصة آلية بسيطة من عدة مؤشرات، وليست توصية استثمارية نهائية.")

            # ---- اختبار تاريخي مبسّط (Backtest) ----
            st.markdown("#### 🕰️ اختبار تاريخي مبسّط (Backtest)")
            st.caption(
                "يحاكي دخول صفقة كل مرة كان السهم يخترق مقاومة 20 يوم تاريخياً، بوقف خسارة 3% وهدف 6%، "
                "ويحسب كم مرة كان هذا المنطق سينجح. ⚠️ يعتمد على السعر فقط بدون أخبار فعلية وقتها — تقريبي واسترشادي بس."
            )
            bt_months = st.select_slider("فترة الاختبار (بالأشهر)", options=[1, 3, 6, 12], value=3, key="bt_months")
            if st.button("🧪 شغّل الاختبار التاريخي", key="run_backtest"):
                with st.spinner("جاري تشغيل الاختبار على البيانات التاريخية..."):
                    bt_result = run_technical_backtest(symbol_input, months=bt_months)
                if "error" in bt_result:
                    st.info(bt_result["error"])
                else:
                    bt1, bt2, bt3, bt4 = st.columns(4)
                    bt1.metric("عدد الصفقات المحاكاة", bt_result["total_trades"])
                    bt2.metric("✅ ناجحة", bt_result["wins"])
                    bt3.metric("❌ خاسرة", bt_result["losses"])
                    bt4.metric("نسبة النجاح التاريخية", f"{bt_result['win_rate']}%")

            # ---- الشارت التفاعلي (TradingView Widget) ----
            st.markdown("#### 📊 الشارت التفاعلي (TradingView)")
            tv_html = f"""
            <div class="tradingview-widget-container">
              <div id="tv_chart_{symbol_input}"></div>
              <script src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget({{
                "width": "100%",
                "height": 500,
                "symbol": "{symbol_input}",
                "interval": "15",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "ar",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "allow_symbol_change": true,
                "studies": ["RSI@tv-basicstudies", "MACD@tv-basicstudies", "MASimple@tv-basicstudies"],
                "container_id": "tv_chart_{symbol_input}"
              }});
              </script>
            </div>
            """
            components.html(tv_html, height=520)

            # ---- الأخبار الخاصة بالسهم + تحليل AI ----
            st.markdown("#### 📰 آخر الأخبار والتحليل بالذكاء الاصطناعي")
            if not finnhub_key:
                st.warning("أدخل مفتاح Finnhub لعرض أخبار السهم.")
            else:
                with st.spinner("جاري جلب أخبار السهم..."):
                    company_news = fetch_company_news(finnhub_key, symbol_input, days_back=7, limit=10)

                if company_news and "error" in company_news[0]:
                    st.error(f"خطأ في جلب الأخبار: {company_news[0]['error']}")
                elif not company_news:
                    st.info("لا توجد أخبار حديثة لهذا السهم.")
                else:
                    news_headlines_raw = [n.get("headline", "بدون عنوان") for n in company_news]
                    with st.spinner("جاري ترجمة العناوين..."):
                        news_headlines_ar = get_arabic_translations(news_headlines_raw)

                    for idx, news in enumerate(company_news):
                        headline = news.get("headline", "بدون عنوان")
                        headline_display = news_headlines_ar[idx]
                        pre_class = classify_news_impact(headline)
                        badge = "🔴" if pre_class == "مرشّح (High Impact)" else "⚪"
                        with st.expander(f"{badge} {headline_display}"):
                            st.caption(
                                f"المصدر: {news.get('source', '—')} | "
                                f"{dt.datetime.fromtimestamp(news.get('datetime', 0)).strftime('%Y-%m-%d %H:%M') if news.get('datetime') else ''}"
                            )
                            if news.get("url"):
                                st.markdown(f"[رابط الخبر الأصلي]({news['url']})")

                            if not openai_key:
                                st.warning("أدخل مفتاح OpenAI لتفعيل التحليل الذكي لهذا الخبر.")
                            else:
                                if st.button("🤖 حلّل هذا الخبر بالذكاء الاصطناعي", key=f"analyze_{idx}"):
                                    with st.spinner("جاري التحليل..."):
                                        price_ctx = dict(sr_info) if "error" not in sr_info else None
                                        if price_ctx is not None:
                                            atr_info = calculate_atr(symbol_input)
                                            if "error" not in atr_info:
                                                price_ctx["atr_suggested_stop_distance"] = atr_info["suggested_stop_distance"]
                                                price_ctx["atr_suggested_target_distance"] = atr_info["suggested_target_distance"]
                                        analysis = analyze_news_with_ai(
                                            openai_key,
                                            headline=headline,
                                            summary=news.get("summary", ""),
                                            symbol=symbol_input,
                                            model=ai_model,
                                            price_context=price_ctx,
                                        )
                                    if "error" in analysis:
                                        st.error(analysis["error"])
                                    else:
                                        analysis = suppress_short_if_disabled(analysis, enable_shorts)
                                        analysis = enforce_neutral_wait(analysis)
                                        analysis = enforce_atr_stop_floor(analysis, symbol_input)
                                        if analysis.get("_stop_widened_by_atr"):
                                            st.info("🛡️ تم توسيع وقف الخسارة تلقائياً ليتماشى مع تقلب السهم الفعلي (كان أضيق من المعقول).")
                                        plan = analysis.get("trade_plan", {})

                                        confluence = None
                                        if "دخول" in plan.get("action", ""):
                                            with st.spinner("جاري حساب درجة تطابق الإشارات..."):
                                                confluence = calculate_confluence_score(
                                                    symbol_input, plan.get("action", ""),
                                                    analysis.get("sentiment", ""), analysis.get("is_likely_official", False),
                                                )

                                        if has_recent_open_recommendation(symbol_input, hours=24):
                                            st.info("ℹ️ فيه توصية مفتوحة لنفس السهم خلال آخر 24 ساعة — ما راح تُسجَّل هذي كتوصية جديدة (بس التحليل يظهر لك تحت).")
                                        else:
                                            save_recommendation(
                                                symbol_input, headline, analysis,
                                                created_by=st.session_state.get("username", "system"),
                                                confluence_score=confluence["score"] if confluence else None,
                                            )
                                            st.caption("✅ تم حفظ هذه التوصية في سجل التوصيات (تبويب 📂).")
                                        sentiment_color = {
                                            "إيجابي جداً": "green", "إيجابي": "green",
                                            "محايد": "gray",
                                            "سلبي": "red", "سلبي جداً": "red",
                                        }.get(analysis.get("sentiment", ""), "gray")

                                        st.markdown(f"**زبدة الخبر:** {analysis.get('summary', '—')}")
                                        c1, c2, c3, c4 = st.columns(4)
                                        c1.markdown(f"**المعنويات:** :{sentiment_color}[{analysis.get('sentiment', '—')}]")
                                        c2.markdown(f"**قوة التأثير:** {analysis.get('impact_level', '—')}")
                                        c3.markdown(f"**نسبة الثقة:** {analysis.get('confidence', '—')}%")
                                        c4.markdown(
                                            f"**مصدر موثوق؟** {'✅ نعم' if analysis.get('is_likely_official') else '⚠️ غير مؤكد'}"
                                        )

                                        # ---- درجة تطابق الإشارات (Confluence Score) ----
                                        if confluence:
                                            st.markdown("##### 🧭 درجة تطابق الإشارات (Confluence Score)")
                                            conf_color = "green" if confluence["score"] >= 75 else ("orange" if confluence["score"] >= 55 else "red")
                                            st.markdown(f"### :{conf_color}[{confluence['score']}%] — {confluence['verdict']}")
                                            if confluence["confirmations"]:
                                                st.markdown("**مؤكِّدات:**")
                                                for c in confluence["confirmations"]:
                                                    st.write(c)
                                            if confluence["conflicts"]:
                                                st.markdown("**تعارضات:**")
                                                for c in confluence["conflicts"]:
                                                    st.write(c)
                                            if confluence["unavailable"]:
                                                st.caption("غير متوفر: " + "، ".join(confluence["unavailable"]))
                                            if confluence.get("risk_warnings"):
                                                for w in confluence["risk_warnings"]:
                                                    st.warning(w)
                                            st.caption("⚠️ هذي درجة استرشادية إحصائية تجمع عدة مؤشرات، وليست ضماناً لنجاح الصفقة.")

                                        st.markdown("##### 🎯 خطة التداول المقترحة (استرشادية)")
                                        if plan.get("entry_price") and plan.get("stop_loss_price") and plan.get("target_price"):
                                            pc1, pc2, pc3 = st.columns(3)
                                            pc1.metric("📍 سعر الدخول", f"${plan['entry_price']}")
                                            pc2.metric("🛑 وقف الخسارة", f"${plan['stop_loss_price']}")
                                            pc3.metric("🎯 الهدف", f"${plan['target_price']}")
                                        st.write(f"**الإجراء:** {plan.get('action', '—')}")
                                        st.write(f"**منطقة الدخول:** {plan.get('entry_note', '—')}")
                                        st.write(f"**وقف الخسارة:** {plan.get('stop_loss_note', '—')}")
                                        st.write(f"**الهدف:** {plan.get('target_note', '—')}")
                                        st.write(f"**⏱️ مدة الصفقة التقريبية:** {plan.get('estimated_duration', '—')}")
                                        st.write(f"**شرط الخروج:** {plan.get('exit_condition', '—')}")
                                        st.warning(analysis.get("risk_warning", ""))

# ========================================================================
# التبويب 3: مراقبة لحظية للأخبار القوية + تحديث تلقائي كل دقيقة
# ========================================================================
with tab_monitor:
    st.subheader("🚨 مراقبة الأخبار القوية لحظياً")
    st.caption(
        "يفحص النظام قائمة أسهمك كل دقيقة بحثاً عن أخبار جديدة عالية التأثير، "
        "ويرسل تنبيهاً داخل التطبيق وعبر تلغرام (إن فعّلته)، مع ترتيب أقوى التوصيات تلقائياً."
    )

    mc1, mc2 = st.columns(2)
    with mc1:
        enable_monitor = st.checkbox("🔄 تفعيل المراقبة التلقائية (كل دقيقة)", value=False, key="enable_monitor")
    with mc2:
        enable_auto_ai = st.checkbox(
            "🤖 تحليل تلقائي بالذكاء الاصطناعي للأخبار الجديدة القوية",
            value=False, key="enable_auto_ai",
            help="تنبيه: هذا يستهلك رصيد OpenAI تلقائياً كل مرة يُرصد فيها خبر قوي جديد.",
        )

    min_confluence_for_alert = st.slider(
        "🎚️ الحد الأدنى لدرجة التطابق لإرسال تنبيه (تلغرام/ntfy)",
        min_value=0, max_value=95, value=0, step=5,
        help="0 = يرسل كل التوصيات. لو رفعته لـ 70 مثلاً، ما توصلك تنبيهات إلا للتوصيات اللي درجة تطابقها 70% فأعلى — تقلل الضجيج وتركّز على الأقوى.",
    )

    watch_scope = st.radio(
        "نطاق المراقبة",
        ["أسهم محددة", "السوق العام (كل الأسهم)", "💰 أقل من $10 (تحديث حي كل دقيقة)"],
        horizontal=True,
        key="watch_scope",
    )

    if watch_scope == "أسهم محددة":
        if "watch_symbols_input" not in st.session_state:
            st.session_state["watch_symbols_input"] = (
                "AAPL,NVDA,TSLA,MSFT,AMZN,GOOGL,META,AMD,NFLX,AVGO,"
                "INTC,PLTR,F,SOFI,RIVN,LCID,NIO,PLUG,SNAP,UBER,"
                "BA,DIS,PYPL,COIN,MARA,RIOT,MSTR,SMCI,ORCL,CRM,"
                "ADBE,QCOM,MU,CSCO,PFE,XOM,CVX,WMT,KO,PEP,"
                "JPM,BAC,GS,V,MA,T,VZ,GM,DAL,AAL"
            )

        preset_col1, preset_col2 = st.columns(2)
        with preset_col1:
            if st.button("📋 القائمة الافتراضية (الأكثر تداولاً)", use_container_width=True):
                st.session_state["watch_symbols_input"] = (
                    "AAPL,NVDA,TSLA,MSFT,AMZN,GOOGL,META,AMD,NFLX,AVGO,"
                    "INTC,PLTR,F,SOFI,RIVN,LCID,NIO,PLUG,SNAP,UBER,"
                    "BA,DIS,PYPL,COIN,MARA,RIOT,MSTR,SMCI,ORCL,CRM,"
                    "ADBE,QCOM,MU,CSCO,PFE,XOM,CVX,WMT,KO,PEP,"
                    "JPM,BAC,GS,V,MA,T,VZ,GM,DAL,AAL"
                )
        with preset_col2:
            if st.button("💰 أسهم أمريكية أقل من $10 (تقريبية)", use_container_width=True):
                st.session_state["watch_symbols_input"] = (
                    "NOK,GSAT,PLUG,FCEL,BLNK,CHPT,MVIS,NNDM,SNDL,TLRY,"
                    "CGC,ACB,OPEN,WOLF,RGTI,QUBT,BBAI,SOUN,JOBY,ACHR,"
                    "LAZR,GOEV,NKLA,RIVN,LCID,NIO,XPEV,VALE,KGC,HL,"
                    "NGD,PBR,DNA,CLNE,GEVO,UUUU,MARA,RIOT,HUT,BTBT,"
                    "CIFR,CLSK,SPCE,IQ,ASTS,OCGN,GPRO,BB,AMC,SIRI"
                )
                st.caption("⚠️ هذي قائمة تقريبية بناءً على أسعار معروفة تاريخياً — تأكد من الأسعار الحالية الفعلية من تبويب 'متابعة سريعة' لأن أسعار الأسهم الرخيصة تتقلب بسرعة وقد يتجاوز بعضها 10$ فعلياً الآن.")

        watch_symbols_text = st.text_input(
            "رموز الأسهم المراقَبة (مفصولة بفاصلة، بحد أقصى 50 رمزاً)",
            key="watch_symbols_input",
        )
        st.caption(
            "⚠️ Finnhub المجاني يسمح بـ 60 طلب بالدقيقة. مراقبة 50 سهم كل دقيقة تستهلك قريب من الحد بالكامل — "
            "لو ظهرت رسائل خطأ متقطعة، قلّل عدد الأسهم أو تجنب استخدام تبويبات ثانية بنفس الوقت."
        )
    else:
        st.caption(
            "⚠️ في وضع 'السوق العام'، النظام يفحص أهم الأخبار العامة بالسوق كامل بدل قائمة محددة. "
            "بعض الأخبار قد لا يظهر معها رمز سهم محدد (تُعرض حينها كـ 'خبر عام')."
        )
        watch_category = st.selectbox(
            "تصنيف الأخبار", ["general", "merger", "forex", "crypto"], index=0, key="watch_category",
        )

    if enable_monitor:
        st_autorefresh(interval=60_000, key="monitor_autorefresh")

    if "seen_news_fp" not in st.session_state:
        st.session_state["seen_news_fp"] = set()

    if enable_monitor:
        newly_found = None
        if not finnhub_key:
            st.warning("أدخل مفتاح Finnhub من الشريط الجانبي لتفعيل المراقبة.")
        elif watch_scope == "أسهم محددة":
            symbols = [s.strip().upper() for s in watch_symbols_text.split(",") if s.strip()][:50]
            newly_found = []
            fallback_candidates = []
            with st.spinner(f"جاري فحص {len(symbols)} سهم..."):
                for sym in symbols:
                    items = fetch_company_news(finnhub_key, sym, days_back=2, limit=10)
                    for item in items:
                        if "error" in item:
                            continue
                        item["_symbol"] = sym
                        fp = news_fingerprint(item)
                        if fp in st.session_state["seen_news_fp"]:
                            continue
                        st.session_state["seen_news_fp"].add(fp)
                        if classify_news_impact(item.get("headline", "")) == "مرشّح (High Impact)":
                            newly_found.append(item)
                        else:
                            fallback_candidates.append(item)

            # احتياطي: لو ما فيه أي خبر "قوي" اليوم، نحلل أحدث خبر عادي مرة وحدة
            # باليوم عشان تضمن توصية واحدة على الأقل يومياً (فقط لو التحليل التلقائي مفعّل)
            today_str = dt.date.today().isoformat()
            if (
                enable_auto_ai and openai_key and not newly_found
                and fallback_candidates
                and st.session_state.get("last_fallback_date") != today_str
            ):
                pick = max(fallback_candidates, key=lambda x: x.get("datetime", 0))
                newly_found.append(pick)
                st.session_state["last_fallback_date"] = today_str
        elif watch_scope == "السوق العام (كل الأسهم)":
            newly_found = []
            with st.spinner("جاري فحص أخبار السوق العام..."):
                items = fetch_market_news(finnhub_key, category=watch_category, limit=50)
                for item in items:
                    if "error" in item:
                        continue
                    fp = news_fingerprint(item)
                    if fp in st.session_state["seen_news_fp"]:
                        continue
                    st.session_state["seen_news_fp"].add(fp)
                    if classify_news_impact(item.get("headline", "")) == "مرشّح (High Impact)":
                        related = (item.get("related") or "").strip()
                        item["_symbol"] = related if related else "خبر عام"
                        newly_found.append(item)
        else:  # "💰 أقل من $10 (تحديث حي كل دقيقة)"
            newly_found = []
            with st.spinner("جاري جلب الأسهم الحية اللي سعرها الآن أقل من $10..."):
                cheap_stocks = fetch_stocks_under_price(max_price=10.0, limit=50)

            if cheap_stocks and "error" in cheap_stocks[0]:
                st.error(cheap_stocks[0]["error"])
            else:
                st.success(f"✅ تم التحقق الآن: {len(cheap_stocks)} سهم حقيقي سعره الحالي أقل من $10")
                df_cheap = pd.DataFrame(cheap_stocks).rename(columns={
                    "symbol": "الرمز", "name": "الاسم", "price": "السعر ($)",
                    "change_pct": "التغير %", "volume": "حجم التداول",
                })
                st.dataframe(df_cheap, use_container_width=True, hide_index=True)

                cheap_symbols = [row["symbol"] for row in cheap_stocks if row.get("symbol")]
                fallback_candidates = []
                with st.spinner(f"جاري فحص أخبار {len(cheap_symbols)} سهم..."):
                    for sym in cheap_symbols:
                        items = fetch_company_news(finnhub_key, sym, days_back=2, limit=10)
                        for item in items:
                            if "error" in item:
                                continue
                            item["_symbol"] = sym
                            fp = news_fingerprint(item)
                            if fp in st.session_state["seen_news_fp"]:
                                continue
                            st.session_state["seen_news_fp"].add(fp)
                            if classify_news_impact(item.get("headline", "")) == "مرشّح (High Impact)":
                                newly_found.append(item)
                            else:
                                fallback_candidates.append(item)

                today_str = dt.date.today().isoformat()
                if (
                    enable_auto_ai and openai_key and not newly_found
                    and fallback_candidates
                    and st.session_state.get("last_fallback_date_cheap") != today_str
                ):
                    pick = max(fallback_candidates, key=lambda x: x.get("datetime", 0))
                    newly_found.append(pick)
                    st.session_state["last_fallback_date_cheap"] = today_str

                # احتياطي ثانٍ: لو ما فيه أي خبر إطلاقاً (حتى عادي) عن هالأسهم
                # الصغيرة، نحلل أكثر سهم تداولاً بناءً على حركة السعر والمؤشرات
                # الفنية بس (بدون خبر)، عشان تضمن توصية على الأقل يومياً
                if (
                    enable_auto_ai and openai_key and not newly_found and not fallback_candidates
                    and cheap_stocks and st.session_state.get("last_fallback_date_cheap") != today_str
                ):
                    top_symbol = cheap_stocks[0]["symbol"]  # الأكثر تداولاً بالقائمة
                    newly_found.append({
                        "headline": f"لا يوجد خبر جديد على {top_symbol} — تحليل فني بناءً على حركة السعر والمؤشرات",
                        "summary": "لا توجد أخبار حديثة متوفرة لهذا السهم من المصدر. هذا تحليل مبني فقط على السعر، الدعم/المقاومة، RSI و MACD.",
                        "source": "تحليل فني تلقائي",
                        "datetime": 0,
                        "_symbol": top_symbol,
                        "_technical_only": True,
                    })
                    st.session_state["last_fallback_date_cheap"] = today_str

        if newly_found is not None:

            for item in newly_found:
                is_technical_only = item.get("_technical_only", False)
                headline_ar = item["headline"] if is_technical_only else get_arabic_translations([item["headline"]])[0]

                icon = "📊" if is_technical_only else "🚨"
                st.toast(f"{icon} {item['_symbol']}: {headline_ar[:60]}", icon=icon)

                msg_prefix = "📊 تحليل فني (لا يوجد خبر)" if is_technical_only else "🚨 خبر قوي جديد"
                news_msg = f"{msg_prefix} على {item['_symbol']}\n{headline_ar}\nالمصدر: {item.get('source', '—')}"
                if enable_telegram and telegram_token and telegram_chat_id:
                    send_telegram_alert(telegram_token, telegram_chat_id, news_msg)
                if enable_ntfy and ntfy_topic:
                    send_ntfy_alert(ntfy_topic, news_msg, title=f"{msg_prefix}: {item['_symbol']}", priority=4)

                if enable_auto_ai and openai_key:
                    price_ctx = None
                    if item["_symbol"] not in ("خبر عام",):
                        price_ctx = calculate_support_resistance(item["_symbol"])
                        if "error" not in price_ctx:
                            atr_info = calculate_atr(item["_symbol"])
                            if "error" not in atr_info:
                                price_ctx["atr_suggested_stop_distance"] = atr_info["suggested_stop_distance"]
                                price_ctx["atr_suggested_target_distance"] = atr_info["suggested_target_distance"]
                    analysis = analyze_news_with_ai(
                        openai_key, item["headline"], item.get("summary", ""), item["_symbol"],
                        model=ai_model, price_context=price_ctx,
                    )
                    if "error" not in analysis:
                        analysis = suppress_short_if_disabled(analysis, enable_shorts)
                        analysis = enforce_neutral_wait(analysis)
                        analysis = enforce_atr_stop_floor(analysis, item["_symbol"])
                        plan = analysis.get("trade_plan", {})

                        confluence = None
                        if "دخول" in plan.get("action", ""):
                            confluence = calculate_confluence_score(
                                item["_symbol"], plan.get("action", ""),
                                analysis.get("sentiment", ""), analysis.get("is_likely_official", False),
                            )

                        if has_recent_open_recommendation(item["_symbol"], hours=24):
                            st.caption(f"ℹ️ تم تجاهل حفظ توصية مكررة لـ {item['_symbol']} (فيه توصية مفتوحة خلال آخر 24 ساعة).")
                        else:
                            save_recommendation(
                                item["_symbol"], item["headline"], analysis,
                                created_by=st.session_state.get("username", "system"),
                                confluence_score=confluence["score"] if confluence else None,
                            )
                            confluence_line = f"🧭 درجة التطابق: {confluence['score']}% — {confluence['verdict']}\n" if confluence else ""
                            entry_p = plan.get("entry_price")
                            sl_p = plan.get("stop_loss_price")
                            tp_p = plan.get("target_price")
                            price_line = ""
                            if entry_p and sl_p and tp_p:
                                price_line = f"📍 دخول: ${entry_p} | وقف: ${sl_p} | هدف: ${tp_p}\n"
                            shariah_badge_line = ""
                            try:
                                _sh = check_shariah_compliance(item["_symbol"])
                                _sh_icon = {"متوافق تقريباً": "🟢", "غير متوافق": "🔴"}.get(_sh.get("status"), "⚪")
                                shariah_badge_line = f"{_sh_icon} التوافق الشرعي (استرشادي): {_sh.get('status', '—')}\n"
                            except Exception:
                                pass
                            rec_msg = (
                                f"🎯 توصية جديدة: {item['_symbol']}\n"
                                f"{shariah_badge_line}"
                                f"{confluence_line}"
                                f"الإجراء: {plan.get('action', '—')}\n"
                                f"{price_line}"
                                f"الدخول: {plan.get('entry_note', '—')}\n"
                                f"وقف الخسارة: {plan.get('stop_loss_note', '—')}\n"
                                f"الهدف: {plan.get('target_note', '—')}\n"
                                f"المدة التقريبية: {plan.get('estimated_duration', '—')}"
                            )
                            meets_threshold = (confluence is None) or (confluence["score"] >= min_confluence_for_alert)
                            if meets_threshold:
                                if enable_telegram and telegram_token and telegram_chat_id:
                                    send_telegram_alert(telegram_token, telegram_chat_id, rec_msg)
                                if enable_ntfy and ntfy_topic:
                                    send_ntfy_alert(ntfy_topic, rec_msg, title=f"🎯 توصية جديدة: {item['_symbol']}", priority=5)
                            else:
                                st.caption(f"🔇 توصية {item['_symbol']} تحت الحد الأدنى للتنبيه ({confluence['score']}% < {min_confluence_for_alert}%) — تم الحفظ بدون إرسال إشعار.")

            if newly_found:
                st.success(f"✅ تم رصد {len(newly_found)} خبراً قوياً جديداً في آخر دورة فحص.")
            else:
                st.caption(f"لا توجد أخبار قوية جديدة منذ آخر فحص. آخر تحديث: {dt.datetime.now().strftime('%H:%M:%S')}")

    st.divider()
    st.markdown("#### 🏅 أعلى توصية مرتبة الآن")
    top_recs = get_all_recommendations(limit=1, only_buy_signals=True)
    if top_recs:
        r = top_recs[0]
        st.info(
            f"**{r['symbol']}** — {r['action']} | المعنويات: {r['sentiment']} | "
            f"الثقة: {r['confidence']}% | ⏱️ المدة التقديرية: {r['estimated_duration']}"
        )
    else:
        st.caption("لا توجد توصيات محفوظة بعد. فعّل التحليل التلقائي أعلاه، أو حلّل خبراً يدوياً من تبويب 'تحليل سهم محدد'.")

# ========================================================================
# التبويب 4: الأسهم الأكثر ربحاً في السوق
# ========================================================================
with tab_gainers:
    st.subheader("🏆 الأسهم الأكثر ارتفاعاً في السوق اليوم")
    st.caption("⚠️ المصدر نقطة بيانات غير رسمية من ياهو فايننس، قد تتوقف أو تتغيّر دون سابق إنذار.")

    st.button("🔄 تحديث القائمة", key="refresh_gainers")  # الضغط يعيد تشغيل الصفحة ويجلب بيانات جديدة

    with st.spinner("جاري جلب القائمة..."):
        gainers = fetch_top_gainers(limit=15)
    if gainers and "error" in gainers[0]:
        st.error(gainers[0]["error"])
    elif gainers:
        st.dataframe(pd.DataFrame(gainers), use_container_width=True, hide_index=True)
    else:
        st.info("لا توجد بيانات متاحة حالياً.")

# ========================================================================
# التبويب 5: سجل التوصيات والصفقات (مرتّبة حسب قوة الإشارة)
# ========================================================================
with tab_recommendations:
    st.subheader("📂 سجل التوصيات والصفقات")
    st.caption("كل التوصيات اللي ولّدها الذكاء الاصطناعي (يدوياً أو تلقائياً)، مرتّبة من الأقوى إشارة للأضعف.")

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        if st.button("🔄 تحديث حالة الصفقات المفتوحة", use_container_width=True):
            with st.spinner("جاري مقارنة الصفقات المفتوحة بالأسعار الحالية..."):
                result = check_and_update_open_recommendations()
            st.success(
                f"✅ تم فحص {result['checked']} صفقة — "
                f"تحقق الهدف: {result['hit_target']} | ضرب وقف الخسارة: {result['hit_stop']} | لسا مفتوحة: {result['still_open']}"
            )
        if st.button("🧹 تنظيف التكرارات القديمة", use_container_width=True):
            with st.spinner("جاري فحص السجل التاريخي عن تكرارات..."):
                cleanup_result = cleanup_duplicate_recommendations(hours=24)
            st.success(f"✅ تم تصنيف {cleanup_result['marked_duplicates']} توصية كتكرار — استُبعدت من حساب نسبة النجاح.")
            st.rerun()

    win_stats = get_win_rate_stats()
    with rc2:
        if win_stats["total_closed"] > 0:
            wc1, wc2, wc3, wc4 = st.columns(4)
            wc1.metric("✅ تحقق الهدف", win_stats["hit_target"])
            wc2.metric("❌ ضرب وقف الخسارة", win_stats["hit_stop"])
            wc3.metric("🔓 لسا مفتوحة", win_stats["still_open"])
            wc4.metric("📊 نسبة النجاح الفعلية", f"{win_stats['win_rate']}%")

            if win_stats.get("avg_risk_reward") is not None:
                wc5, wc6 = st.columns(2)
                wc5.metric("⚖️ متوسط المخاطرة/العائد المخطط", f"1 : {win_stats['avg_risk_reward']}")
                expectancy = win_stats.get("expectancy_r")
                if expectancy is not None:
                    exp_label = "✅ إيجابي (مربح إحصائياً)" if expectancy > 0 else "⚠️ سلبي (خاسر إحصائياً)"
                    wc6.metric("🧮 التوقع الرياضي (لكل وحدة مخاطرة)", f"{expectancy}R", help=exp_label)
                st.caption(
                    "التوقع الرياضي (Expectancy) يجمع بين نسبة النجاح ونسبة المخاطرة/العائد بمعادلة واحدة — "
                    "رقم موجب يعني الاستراتيجية مربحة إحصائياً على المدى الطويل حتى لو نسبة النجاح أقل من 50%، "
                    "ورقم سالب يعني العكس حتى لو نسبة النجاح عالية."
                )
        else:
            st.caption("ما فيه صفقات مغلقة بعد — اضغط 'تحديث حالة الصفقات' للفحص، أو انتظر توصيات جديدة تتحقق مع الوقت.")

    # ---- دقة درجة التطابق الفعلية (Feedback Loop) ----
    st.divider()
    st.markdown("##### 🔁 دقة درجة التطابق التاريخية (مقارنة بالأداء الفعلي)")
    confluence_stats = get_confluence_accuracy_stats()
    if confluence_stats:
        st.dataframe(pd.DataFrame(confluence_stats), use_container_width=True, hide_index=True)
        st.caption(
            "هذا الجدول يقارن درجة التطابق اللي أعطاها النظام وقت التوصية بالنتيجة الفعلية اللي صارت — "
            "لو فئة '75-95%' فعلاً عندها أعلى نسبة نجاح، يعني درجة التطابق موثوقة. لو مو كذا، يعني الوزن يحتاج تعديل."
        )
    else:
        st.caption("لا توجد بيانات كافية بعد — يحتاج تراكم عدة صفقات مغلقة فيها درجة تطابق محفوظة (بعد آخر تحديث للمنصة).")

    st.caption(
        "💡 الحالة تُحدَّث فقط لما تضغط الزر أعلاه (ما تتحدث تلقائياً بالخلفية). "
        "الصفقة تُحسب 'تحقق الهدف' أو 'ضرب وقف الخسارة' بمقارنة السعر الحالي الفعلي بأرقام الدخول/الهدف/الوقف "
        "المحفوظة وقت التوصية — يحتاج هذا وجود أسعار رقمية محفوظة (تظهر فقط للتوصيات اللي حُللت بعد آخر تحديث للمنصة)."
    )

    st.divider()
    fc1, fc2, fc3 = st.columns([1, 1, 1])
    with fc1:
        only_buy = st.checkbox("عرض إشارات الدخول فقط (شراء/بيع)", value=False, key="only_buy_recs")
    with fc2:
        status_filter = st.selectbox(
            "تصفية حسب الحالة",
            ["الكل", "❌ ضرب وقف الخسارة فقط (الفاشلة)", "✅ تحقق الهدف فقط (الناجحة)", "🔓 مفتوحة فقط"],
            key="status_filter_recs",
        )
    with fc3:
        sort_choice = st.selectbox(
            "الترتيب",
            ["🕐 الأحدث أولاً (موصى به)", "💪 الأقوى إشارة أولاً"],
            key="sort_choice_recs",
            help="'الأحدث أولاً' يضمن ظهور أي توصية جديدة بالسجل فوراً بغض النظر عن قوتها.",
        )
    order_by = "score" if sort_choice.startswith("💪") else "time"
    recs = get_all_recommendations(limit=500, only_buy_signals=only_buy, order_by=order_by)

    if not recs:
        st.info("لا توجد توصيات محفوظة بعد.")
    else:
        status_icon_map = {
            "hit_target": "✅ تحقق الهدف",
            "hit_stop": "❌ ضرب وقف الخسارة",
        }
        for rec in recs:
            rec["الحالة"] = status_icon_map.get(rec.get("status"), "🔓 مفتوحة")

        if status_filter == "❌ ضرب وقف الخسارة فقط (الفاشلة)":
            recs = [r for r in recs if r.get("status") == "hit_stop"]
        elif status_filter == "✅ تحقق الهدف فقط (الناجحة)":
            recs = [r for r in recs if r.get("status") == "hit_target"]
        elif status_filter == "🔓 مفتوحة فقط":
            recs = [r for r in recs if r.get("status") not in ("hit_target", "hit_stop")]

        if not recs:
            st.info("لا توجد توصيات مطابقة لهذا الفلتر.")
            st.stop()

        df_recs = pd.DataFrame(recs)

        # حساب نسبة مسافة وقف الخسارة عن سعر الدخول (لتشخيص هل الوقف كان ضيق جداً)
        if "entry_price" in df_recs.columns and "stop_loss_price" in df_recs.columns:
            df_recs["stop_distance_pct"] = df_recs.apply(
                lambda r: round(abs(r["entry_price"] - r["stop_loss_price"]) / r["entry_price"] * 100, 2)
                if pd.notna(r.get("entry_price")) and pd.notna(r.get("stop_loss_price")) and r.get("entry_price")
                else None,
                axis=1,
            )

        display_cols = [
            "created_at", "symbol", "action", "الحالة", "sentiment", "confidence",
            "estimated_duration", "entry_price", "stop_loss_price", "target_price",
            "stop_distance_pct", "confluence_score",
            "entry_note", "stop_loss_note", "target_note", "score",
        ]
        display_cols = [c for c in display_cols if c in df_recs.columns]
        rename_map = {
            "created_at": "الوقت", "symbol": "الرمز", "action": "الإجراء",
            "sentiment": "المعنويات", "confidence": "الثقة %",
            "estimated_duration": "المدة التقديرية",
            "entry_price": "سعر الدخول ($)", "stop_loss_price": "وقف الخسارة ($)", "target_price": "الهدف ($)",
            "stop_distance_pct": "مسافة الوقف %", "confluence_score": "درجة التطابق",
            "entry_note": "ملاحظة الدخول",
            "stop_loss_note": "ملاحظة وقف الخسارة", "target_note": "ملاحظة الهدف", "score": "درجة الترتيب",
        }
        df_display = df_recs[display_cols].rename(columns=rename_map)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        csv_data = df_display.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ تحميل الجدول كملف Excel/CSV",
            data=csv_data,
            file_name=f"توصيات_finai_{dt.date.today().isoformat()}.csv",
            mime="text/csv",
        )

# ========================================================================
# التبويب 6: حاسبة إدارة المخاطر وحجم الصفقة
# ========================================================================
# ========================================================================
# التبويب الجديد: تنبيهات سعرية بدون الحاجة لخبر
# ========================================================================
with tab_alerts:
    st.subheader("🔔 تنبيهات سعرية")
    st.caption("حط سعر مستهدف لأي سهم، وتوصلك رسالة (تلغرام/ntfy) فور ما يوصل — بدون انتظار خبر أو تحليل.")

    with st.form("add_alert_form"):
        ac1, ac2, ac3 = st.columns([2, 2, 2])
        with ac1:
            alert_symbol = st.text_input("رمز السهم", placeholder="مثال: NVDA")
        with ac2:
            alert_price = st.number_input("السعر المستهدف ($)", min_value=0.0, step=0.5)
        with ac3:
            alert_direction = st.selectbox("الشرط", ["above", "below"], format_func=lambda x: "وصل فوق ⬆️" if x == "above" else "نزل تحت ⬇️")
        add_alert_submitted = st.form_submit_button("➕ إضافة تنبيه", type="primary")

    if add_alert_submitted:
        if not alert_symbol or alert_price <= 0:
            st.error("لازم تكتب رمز سهم وسعر أكبر من صفر.")
        else:
            add_price_alert(alert_symbol.strip().upper(), alert_price, alert_direction, created_by=st.session_state.get("username", "system"))
            st.success(f"✅ تم إضافة تنبيه: {alert_symbol.upper()} {'فوق' if alert_direction == 'above' else 'تحت'} ${alert_price}")
            st.rerun()

    st.divider()
    check_alerts_col1, check_alerts_col2 = st.columns([1, 3])
    with check_alerts_col1:
        check_alerts_now = st.button("🔄 فحص التنبيهات الآن", use_container_width=True)

    if check_alerts_now:
        with st.spinner("جاري فحص الأسعار الحالية..."):
            triggered = check_price_alerts()
        if triggered:
            for alert in triggered:
                direction_ar = "فوق" if alert["direction"] == "above" else "تحت"
                msg = f"🔔 تنبيه سعري تحقق: {alert['symbol']} وصل ${alert['current_price']:.2f} ({direction_ar} هدفك ${alert['target_price']})"
                st.success(msg)
                if enable_telegram and telegram_token and telegram_chat_id:
                    send_telegram_alert(telegram_token, telegram_chat_id, msg)
                if enable_ntfy and ntfy_topic:
                    send_ntfy_alert(ntfy_topic, msg, title=f"🔔 تنبيه سعري: {alert['symbol']}", priority=4)
        else:
            st.info("لا يوجد تنبيه تحقق حتى الآن.")

    st.divider()
    st.markdown("##### التنبيهات النشطة")
    active_alerts = get_active_price_alerts()
    if not active_alerts:
        st.caption("ما فيه تنبيهات نشطة حالياً.")
    else:
        for alert in active_alerts:
            acol1, acol2 = st.columns([5, 1])
            direction_ar = "وصل فوق ⬆️" if alert["direction"] == "above" else "نزل تحت ⬇️"
            acol1.write(f"**{alert['symbol']}** — {direction_ar} **${alert['target_price']}**")
            if acol2.button("🗑️ حذف", key=f"del_alert_{alert['id']}"):
                delete_price_alert(alert["id"])
                st.rerun()

    st.caption("💡 الفحص يدوي بالزر أعلاه حالياً (ما يشتغل تلقائياً بالخلفية). اضغطه كل ما تبي تتأكد من أسعار تنبيهاتك.")


# ========================================================================
# التبويب الجديد: محفظتي الفعلية (صفقاتك الحقيقية، مو توصيات AI)
# ========================================================================
with tab_portfolio:
    st.subheader("💼 محفظتي الفعلية")
    st.caption("سجّل هنا الصفقات اللي دخلتها فعلاً بأموالك الحقيقية (مو توصيات الذكاء الاصطناعي)، وتابع ربحها/خسارتها الحية.")

    with st.form("add_position_form"):
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            pos_symbol = st.text_input("رمز السهم", placeholder="مثال: AAPL")
        with pc2:
            pos_qty = st.number_input("عدد الأسهم", min_value=0.0, step=1.0)
        with pc3:
            pos_entry = st.number_input("سعر الدخول ($)", min_value=0.0, step=0.5)
        with pc4:
            pos_date = st.date_input("تاريخ الدخول", value=dt.date.today())
        add_position_submitted = st.form_submit_button("➕ أضف الصفقة لمحفظتي", type="primary")

    if add_position_submitted:
        if not pos_symbol or pos_qty <= 0 or pos_entry <= 0:
            st.error("لازم تعبّي رمز السهم وعدد الأسهم وسعر الدخول (أكبر من صفر).")
        else:
            add_portfolio_position(
                pos_symbol.strip().upper(), pos_qty, pos_entry, pos_date.isoformat(),
                created_by=st.session_state.get("username", "system"),
            )
            st.success(f"✅ تم إضافة {pos_qty} سهم من {pos_symbol.upper()} لمحفظتك.")
            st.rerun()

    st.divider()

    if st.button("🔄 تحديث أسعار المحفظة الحية"):
        with st.spinner("جاري جلب الأسعار الحالية..."):
            st.session_state["portfolio_pnl"] = calculate_portfolio_pnl()

    if "portfolio_pnl" not in st.session_state:
        st.session_state["portfolio_pnl"] = calculate_portfolio_pnl()

    pnl_data = st.session_state["portfolio_pnl"]

    if not pnl_data["positions"]:
        st.info("محفظتك فاضية حالياً — أضف أول صفقة من الفورم أعلاه.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💰 إجمالي المستثمر", f"${pnl_data['total_invested']:,.2f}")
        m2.metric("📊 القيمة الحالية", f"${pnl_data['total_current_value']:,.2f}")
        pnl_color = "normal" if pnl_data["total_pnl"] >= 0 else "inverse"
        m3.metric("الربح/الخسارة", f"${pnl_data['total_pnl']:,.2f}", delta=f"{pnl_data['total_pnl_pct']}%")
        m4.metric("عدد الصفقات", len(pnl_data["positions"]))

        st.divider()
        for pos in pnl_data["positions"]:
            with st.container(border=True):
                pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns([1.5, 1, 1, 1.5, 0.7])
                pcol1.markdown(f"**{pos['symbol']}** — {pos['quantity']} سهم")
                pcol2.caption(f"دخول: ${pos['entry_price']}")
                if pos.get("current_price") is not None:
                    pcol3.caption(f"حالي: ${pos['current_price']:.2f}")
                    pnl_sign = "🟢" if pos["pnl"] >= 0 else "🔴"
                    pcol4.markdown(f"{pnl_sign} ${pos['pnl']:,.2f} ({pos['pnl_pct']}%)")
                else:
                    pcol3.caption("السعر غير متوفر")
                if pcol5.button("🗑️", key=f"del_pos_{pos['id']}"):
                    delete_portfolio_position(pos["id"])
                    st.session_state.pop("portfolio_pnl", None)
                    st.rerun()

        st.caption("⚠️ الأسعار تُحدَّث فقط لما تضغط زر 'تحديث أسعار المحفظة الحية' أعلاه، وليست حية تلقائياً بالخلفية.")


# ========================================================================
# التبويب الجديد: تقويم الأرباح
# ========================================================================
with tab_earnings:
    st.subheader("📅 تقويم الأرباح")
    st.caption("اعرف بالضبط متى الشركات المهتم فيها بتعلن أرباحها — من أقوى محركات حركة السعر.")

    earnings_symbols_text = st.text_input(
        "رموز الأسهم (مفصولة بفاصلة)", value="AAPL,NVDA,TSLA,MSFT,AMZN,GOOGL,META,AMD", key="earnings_symbols",
    )
    if st.button("📅 اعرض تواريخ الأرباح", type="primary"):
        symbols_list = [s.strip().upper() for s in earnings_symbols_text.split(",") if s.strip()][:30]
        with st.spinner("جاري جلب تواريخ الأرباح..."):
            earnings_data = fetch_earnings_calendar(symbols_list)
        if not earnings_data:
            st.info("ما فيه تواريخ أرباح متوفرة لهذي الرموز حالياً.")
        else:
            df_earnings = pd.DataFrame(earnings_data).rename(columns={"symbol": "الرمز", "earnings_date": "تاريخ الأرباح المتوقع"})
            st.dataframe(df_earnings, use_container_width=True, hide_index=True)
            st.caption("⚠️ التواريخ تقديرية من ياهو فايننس، وممكن تتغيّر قبل الإعلان الرسمي من الشركة.")



    st.subheader("🎯 حاسبة إدارة المخاطر وتحديد حجم الصفقة")
    st.caption("أدخل بيانات محفظتك ونقاط الصفقة لمعرفة الحجم المناسب والمخاطرة الفعلية بالدولار.")

    r1, r2 = st.columns(2)
    with r1:
        account_balance = st.number_input("💰 رصيد المحفظة الإجمالي ($)", min_value=0.0, value=10000.0, step=100.0)
        risk_percent = st.slider("نسبة المخاطرة من المحفظة (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        entry_price = st.number_input("سعر الدخول ($)", min_value=0.0, value=100.0, step=0.5)
    with r2:
        stop_loss_price = st.number_input("سعر وقف الخسارة ($)", min_value=0.0, value=95.0, step=0.5)
        target_price = st.number_input("سعر الهدف ($) — اختياري", min_value=0.0, value=110.0, step=0.5)

    if st.button("احسب حجم الصفقة", type="primary"):
        result = calculate_position_size(account_balance, risk_percent, entry_price, stop_loss_price, target_price)
        if "error" in result:
            st.error(result["error"])
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("عدد الأسهم المقترح", f"{result['shares_to_buy']:,}")
            m2.metric("قيمة الصفقة الإجمالية", f"${result['total_position_value']:,}")
            m3.metric("نسبة الصفقة من المحفظة", f"{result['position_pct_of_account']}%")

            m4, m5, m6 = st.columns(3)
            m4.metric("أقصى خسارة محتملة", f"${result['max_loss_dollars']:,}", delta=f"-{risk_percent}% من المحفظة")
            if "potential_profit_dollars" in result:
                m5.metric("الربح المحتمل عند الهدف", f"${result['potential_profit_dollars']:,}")
                m6.metric("نسبة المخاطرة/العائد", f"1 : {result['risk_reward_ratio']}")

            if result.get("risk_reward_ratio") and result["risk_reward_ratio"] < 1.5:
                st.warning("⚠️ نسبة المخاطرة إلى العائد أقل من 1:1.5 — يُفضّل عادة صفقات بنسبة 1:2 أو أعلى.")
            elif result.get("risk_reward_ratio"):
                st.success("✅ نسبة المخاطرة إلى العائد جيدة نسبياً.")

# ========================================================================
# التبويب 4: قائمة متابعة سريعة (متعدد الأسهم دفعة واحدة)
# ========================================================================
with tab_watch:
    st.subheader("📋 قائمة المتابعة السريعة")
    st.caption("أدخل عدة رموز أسهم مفصولة بفاصلة لعرض لقطة سريعة عن كل منها.")

    symbols_text = st.text_input("مثال: AAPL, NVDA, TSLA, AMZN", value="AAPL, NVDA, TSLA")
    if st.button("عرض القائمة"):
        symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
        rows = []
        with st.spinner("جاري جلب البيانات..."):
            for sym in symbols:
                snap = fetch_stock_snapshot(sym)
                if "error" not in snap:
                    rows.append({
                        "الرمز": snap["symbol"],
                        "الاسم": snap["name"],
                        "السعر": snap["current_price"],
                        "التغير %": round(snap["change_pct"], 2) if snap["change_pct"] is not None else None,
                        "القطاع": snap["sector"],
                        "القيمة السوقية": format_large_number(snap.get("market_cap")),
                    })
        if rows:
            df_watch = pd.DataFrame(rows)
            st.dataframe(
                df_watch.style.map(
                    lambda v: "color: green" if isinstance(v, (int, float)) and v > 0 else (
                        "color: red" if isinstance(v, (int, float)) and v < 0 else ""
                    ),
                    subset=["التغير %"],
                ),
                use_container_width=True,
                hide_index=True,
            )
            csv_watch = df_watch.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ تحميل كملف Excel/CSV",
                data=csv_watch,
                file_name=f"متابعة_سريعة_{dt.date.today().isoformat()}.csv",
                mime="text/csv",
            )
        else:
            st.info("لم يتم العثور على بيانات لأي من الرموز المدخلة.")
