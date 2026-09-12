"""
Financial AI Agent — منصة تحليل الأخبار اللحظية وتوجيه المتداول
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
    يترجم قائمة نصوص للعربية مجاناً بالكامل (خدمة MyMemory) مع تخزين مؤقت
    بذاكرة الجلسة (session_state)، بحيث أي عنوان خبر تُرجم مرة ما يُعاد
    ترجمته مرة ثانية حتى لو تكرر ظهوره بتحديثات لاحقة.
    """
    if "translation_cache" not in st.session_state:
        st.session_state["translation_cache"] = {}
    cache = st.session_state["translation_cache"]

    if not texts:
        return list(texts)

    to_translate = [t for t in texts if t not in cache]
    if to_translate:
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


st.set_page_config(
    page_title="Financial AI Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# بوابة تسجيل الدخول — تظهر قبل أي محتوى آخر في التطبيق
# ----------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None

if not st.session_state["authenticated"]:
    st.title("🔐 تسجيل الدخول")
    st.caption("منصة Financial AI Agent — الدخول مقتصر على المستخدمين المصرّح لهم")

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
st.title("📈 Financial AI Agent")
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
    tab_search,
    tab_monitor,
    tab_gainers,
    tab_recommendations,
    tab_risk,
    tab_watch,
) = st.tabs(
    [
        "🗺️ لوحة الأخبار اليومية",
        "🔍 تحليل سهم محدد",
        "🚨 مراقبة لحظية",
        "🏆 الأكثر ربحاً",
        "📂 سجل التوصيات",
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
                                        price_ctx = sr_info if "error" not in sr_info else None
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
                                        save_recommendation(
                                            symbol_input, headline, analysis,
                                            created_by=st.session_state.get("username", "system"),
                                        )
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

                                        plan = analysis.get("trade_plan", {})
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
                                        st.caption("✅ تم حفظ هذه التوصية في سجل التوصيات (تبويب 📂).")

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
                    analysis = analyze_news_with_ai(
                        openai_key, item["headline"], item.get("summary", ""), item["_symbol"],
                        model=ai_model, price_context=price_ctx,
                    )
                    if "error" not in analysis:
                        analysis = suppress_short_if_disabled(analysis, enable_shorts)
                        save_recommendation(
                            item["_symbol"], item["headline"], analysis,
                            created_by=st.session_state.get("username", "system"),
                        )
                        plan = analysis.get("trade_plan", {})
                        entry_p = plan.get("entry_price")
                        sl_p = plan.get("stop_loss_price")
                        tp_p = plan.get("target_price")
                        price_line = ""
                        if entry_p and sl_p and tp_p:
                            price_line = f"📍 دخول: ${entry_p} | وقف: ${sl_p} | هدف: ${tp_p}\n"
                        rec_msg = (
                            f"🎯 توصية جديدة: {item['_symbol']}\n"
                            f"الإجراء: {plan.get('action', '—')}\n"
                            f"{price_line}"
                            f"الدخول: {plan.get('entry_note', '—')}\n"
                            f"وقف الخسارة: {plan.get('stop_loss_note', '—')}\n"
                            f"الهدف: {plan.get('target_note', '—')}\n"
                            f"المدة التقريبية: {plan.get('estimated_duration', '—')}"
                        )
                        if enable_telegram and telegram_token and telegram_chat_id:
                            send_telegram_alert(telegram_token, telegram_chat_id, rec_msg)
                        if enable_ntfy and ntfy_topic:
                            send_ntfy_alert(ntfy_topic, rec_msg, title=f"🎯 توصية جديدة: {item['_symbol']}", priority=5)

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

    only_buy = st.checkbox("عرض إشارات الدخول فقط (شراء/بيع)", value=False, key="only_buy_recs")
    recs = get_all_recommendations(limit=200, only_buy_signals=only_buy)

    if not recs:
        st.info("لا توجد توصيات محفوظة بعد.")
    else:
        df_recs = pd.DataFrame(recs)
        display_cols = [
            "created_at", "symbol", "action", "sentiment", "confidence",
            "estimated_duration", "entry_note", "stop_loss_note", "target_note", "score",
        ]
        display_cols = [c for c in display_cols if c in df_recs.columns]
        rename_map = {
            "created_at": "الوقت", "symbol": "الرمز", "action": "الإجراء",
            "sentiment": "المعنويات", "confidence": "الثقة %",
            "estimated_duration": "المدة التقديرية", "entry_note": "الدخول",
            "stop_loss_note": "وقف الخسارة", "target_note": "الهدف", "score": "درجة الترتيب",
        }
        df_display = df_recs[display_cols].rename(columns=rename_map)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

# ========================================================================
# التبويب 6: حاسبة إدارة المخاطر وحجم الصفقة
# ========================================================================
with tab_risk:
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
        else:
            st.info("لم يتم العثور على بيانات لأي من الرموز المدخلة.")
