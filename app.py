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

from auth import verify_login, get_display_name
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
    ai_model = st.selectbox("نموذج الذكاء الاصطناعي", ["gpt-4o-mini", "gpt-4o"], index=0)

    st.divider()
    st.caption("⚠️ هذه المنصة أداة استرشادية تعليمية وليست توصية استثمارية. القرار والمسؤولية تقع على المتداول.")

# ----------------------------------------------------------------------
# رأس الصفحة
# ----------------------------------------------------------------------
st.title("📈 Financial AI Agent")
st.caption("منصة تحليل الأخبار اللحظية بالذكاء الاصطناعي وتوجيه المتداول")

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
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        st.subheader("الأخبار اللحظية للسوق")
    with col_b:
        news_category = st.selectbox("التصنيف", ["general", "merger", "forex", "crypto"], index=0)
    with col_c:
        only_high_impact = st.checkbox("عرض عالي التأثير فقط", value=True)

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
            heat_rows = []
            for item in display_items[:30]:
                classification = classify_news_impact(item.get("headline", ""))
                heat_rows.append({
                    "المصدر": item.get("source", "—"),
                    "العنوان": item.get("headline", "")[:70],
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

                # تنبيه تلغرام تلقائي لأول خبر عالي التأثير في الجلسة الحالية
                if enable_telegram and telegram_token and telegram_chat_id:
                    high_impact_only = [r for r in heat_rows if r["التصنيف الأولي"] == "مرشّح (High Impact)"]
                    if high_impact_only and not st.session_state.get("telegram_sent_once"):
                        msg = f"🚨 خبر عالي التأثير:\n{high_impact_only[0]['العنوان']}\nالمصدر: {high_impact_only[0]['المصدر']}"
                        sent = send_telegram_alert(telegram_token, telegram_chat_id, msg)
                        st.session_state["telegram_sent_once"] = True
                        if sent:
                            st.toast("تم إرسال تنبيه تلغرام ✅")
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
            price_cols[0].metric("السعر الحالي", f"${snapshot['current_price']:.2f}" if snapshot['current_price'] else "—")
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
                    for idx, news in enumerate(company_news):
                        headline = news.get("headline", "بدون عنوان")
                        pre_class = classify_news_impact(headline)
                        badge = "🔴" if pre_class == "مرشّح (High Impact)" else "⚪"
                        with st.expander(f"{badge} {headline}"):
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
                                        analysis = analyze_news_with_ai(
                                            openai_key,
                                            headline=headline,
                                            summary=news.get("summary", ""),
                                            symbol=symbol_input,
                                            model=ai_model,
                                        )
                                    if "error" in analysis:
                                        st.error(analysis["error"])
                                    else:
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
        ["أسهم محددة", "السوق العام (كل الأسهم)"],
        horizontal=True,
        key="watch_scope",
    )

    if watch_scope == "أسهم محددة":
        watch_symbols_text = st.text_input(
            "رموز الأسهم المراقَبة (مفصولة بفاصلة، بحد أقصى 15 رمزاً لتفادي حدود Finnhub المجانية)",
            value="AAPL,NVDA,TSLA,MSFT,AMZN,GOOGL,META,AMD",
            key="watch_symbols",
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
            symbols = [s.strip().upper() for s in watch_symbols_text.split(",") if s.strip()][:15]
            newly_found = []
            with st.spinner(f"جاري فحص {len(symbols)} سهم..."):
                for sym in symbols:
                    items = fetch_company_news(finnhub_key, sym, days_back=1, limit=5)
                    for item in items:
                        if "error" in item:
                            continue
                        fp = news_fingerprint(item)
                        if fp in st.session_state["seen_news_fp"]:
                            continue
                        st.session_state["seen_news_fp"].add(fp)
                        if classify_news_impact(item.get("headline", "")) == "مرشّح (High Impact)":
                            item["_symbol"] = sym
                            newly_found.append(item)
        else:
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

        if newly_found is not None:

            for item in newly_found:
                st.toast(f"🚨 {item['_symbol']}: {item['headline'][:60]}", icon="🚨")

                if enable_telegram and telegram_token and telegram_chat_id:
                    send_telegram_alert(
                        telegram_token, telegram_chat_id,
                        f"🚨 خبر قوي جديد على {item['_symbol']}\n{item['headline']}\nالمصدر: {item.get('source', '—')}",
                    )

                if enable_auto_ai and openai_key:
                    analysis = analyze_news_with_ai(
                        openai_key, item["headline"], item.get("summary", ""), item["_symbol"], model=ai_model,
                    )
                    if "error" not in analysis:
                        save_recommendation(
                            item["_symbol"], item["headline"], analysis,
                            created_by=st.session_state.get("username", "system"),
                        )
                        if enable_telegram and telegram_token and telegram_chat_id:
                            plan = analysis.get("trade_plan", {})
                            msg = (
                                f"🎯 توصية جديدة: {item['_symbol']}\n"
                                f"الإجراء: {plan.get('action', '—')}\n"
                                f"الدخول: {plan.get('entry_note', '—')}\n"
                                f"وقف الخسارة: {plan.get('stop_loss_note', '—')}\n"
                                f"الهدف: {plan.get('target_note', '—')}\n"
                                f"المدة التقريبية: {plan.get('estimated_duration', '—')}"
                            )
                            send_telegram_alert(telegram_token, telegram_chat_id, msg)

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
