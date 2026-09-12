"""
utils.py
--------
كل الدوال المساعدة الخاصة بمنصة "Financial AI Agent":
- جلب الأخبار (Finnhub)
- جلب أسعار الأسهم (yfinance)
- تحليل الأخبار بالذكاء الاصطناعي (OpenAI)
- فحص السيولة/الشورت (Short Interest & Float)
- حاسبة إدارة المخاطر
- إرسال تنبيهات تلغرام
- فلترة الأخبار حسب القوة (High Impact Screener)
"""

import os
import json
import time
import hashlib
import sqlite3
import datetime as dt
from typing import List, Dict, Optional

import requests
import pandas as pd
import yfinance as yf

DB_PATH = os.path.join(os.path.dirname(__file__), "finai.db")

# مكتبة OpenAI (الإصدار الحديث SDK v1)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ----------------------------------------------------------------------
# 1) جلب الأخبار من Finnhub
# ----------------------------------------------------------------------

def fetch_market_news(api_key: str, category: str = "general", limit: int = 40) -> List[Dict]:
    """
    يجلب أخبار السوق العامة من Finnhub.
    category: general / forex / crypto / merger
    """
    if not api_key:
        return []
    url = "https://finnhub.io/api/v1/news"
    params = {"category": category, "token": api_key}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data[:limit] if isinstance(data, list) else []
    except Exception as e:
        return [{"error": str(e)}]


def fetch_company_news(api_key: str, symbol: str, days_back: int = 5, limit: int = 20) -> List[Dict]:
    """
    يجلب آخر أخبار سهم معيّن (Company News) من Finnhub.
    """
    if not api_key or not symbol:
        return []
    to_date = dt.date.today()
    from_date = to_date - dt.timedelta(days=days_back)
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol.upper(),
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            data.sort(key=lambda x: x.get("datetime", 0), reverse=True)
            return data[:limit]
        return []
    except Exception as e:
        return [{"error": str(e)}]


# ----------------------------------------------------------------------
# 2) جلب بيانات السهم (السعر، التغير، الشورت، الفلوت) عبر yfinance
# ----------------------------------------------------------------------

def fetch_stock_snapshot(symbol: str) -> Dict:
    """
    يرجع لقطة سريعة عن السهم: السعر الحالي، التغير %، الفلوت، نسبة الشورت،
    القيمة السوقية، القطاع، متوسط حجم التداول.
    """
    try:
        t = yf.Ticker(symbol)
        info = t.info if hasattr(t, "info") else {}
        hist = t.history(period="5d")

        current_price = None
        change_pct = None
        if not hist.empty:
            raw_price = float(hist["Close"].iloc[-1])
            if not pd.isna(raw_price):
                current_price = raw_price
                if len(hist) >= 2:
                    raw_prev = float(hist["Close"].iloc[-2])
                    if not pd.isna(raw_prev) and raw_prev != 0:
                        change_pct = ((current_price - raw_prev) / raw_prev) * 100

        # احتياطي: لو تاريخ الأسعار رجع فاضي أو NaN، جرب الأخذ من بيانات info مباشرة
        if current_price is None:
            fallback_price = info.get("currentPrice") or info.get("regularMarketPrice")
            if fallback_price and not pd.isna(fallback_price):
                current_price = float(fallback_price)
                prev_close_info = info.get("previousClose")
                if prev_close_info and not pd.isna(prev_close_info) and prev_close_info != 0:
                    change_pct = ((current_price - prev_close_info) / prev_close_info) * 100

        snapshot = {
            "symbol": symbol.upper(),
            "name": info.get("shortName") or info.get("longName") or symbol.upper(),
            "sector": info.get("sector", "غير محدد"),
            "current_price": current_price,
            "change_pct": change_pct,
            "market_cap": info.get("marketCap"),
            "avg_volume": info.get("averageVolume"),
            "float_shares": info.get("floatShares"),
            "shares_short": info.get("sharesShort"),
            "short_ratio": info.get("shortRatio"),
            "short_percent_of_float": info.get("shortPercentOfFloat"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        }
        return snapshot
    except Exception as e:
        return {"symbol": symbol.upper(), "error": str(e)}


def check_short_squeeze_potential(snapshot: Dict) -> Dict:
    """
    يقيّم احتمال Short Squeeze بناءً على float و short interest.
    هذا تقدير مبسّط وليس نصيحة استثمارية.
    """
    float_shares = snapshot.get("float_shares")
    short_pct = snapshot.get("short_percent_of_float")

    result = {"low_float": False, "high_short_interest": False, "squeeze_potential": "غير كافٍ للتقييم"}

    if float_shares:
        result["low_float"] = float_shares < 50_000_000  # فلوت أقل من 50 مليون سهم يعتبر منخفض نسبياً

    if short_pct:
        short_pct_display = short_pct * 100 if short_pct < 1 else short_pct
        result["short_percent_display"] = round(short_pct_display, 2)
        result["high_short_interest"] = short_pct_display > 15

    if result["low_float"] and result["high_short_interest"]:
        result["squeeze_potential"] = "🔥 احتمال صعود انفجاري (Short Squeeze) — فلوت منخفض + شورت مرتفع"
    elif result["high_short_interest"]:
        result["squeeze_potential"] = "⚠️ نسبة شورت مرتفعة، راقب السهم"
    elif result["low_float"]:
        result["squeeze_potential"] = "ℹ️ فلوت منخفض، تقلبات محتملة أعلى من المعتاد"
    else:
        result["squeeze_potential"] = "لا توجد إشارة قوية حالياً"

    return result


# ----------------------------------------------------------------------
# 3) محرك الذكاء الاصطناعي: تحليل الخبر + توصية تداول
# ----------------------------------------------------------------------

AI_SYSTEM_PROMPT = """أنت محلل مالي محترف متخصص في تحليل الأخبار اللحظية وأثرها على أسعار الأسهم.
مهمتك تحليل الخبر المُعطى وإرجاع تحليل دقيق وموجز بصيغة JSON فقط، بدون أي نص إضافي أو Markdown.

قد يُعطى لك أيضاً "سياق السعر الحالي" يتضمن السعر الحالي للسهم ومستويات دعم/مقاومة محسوبة فنياً.
إذا تم إعطاؤك هذا السياق، استخدمه لتحديد أرقام دخول ووقف خسارة وهدف تقريبية واقعية بناءً عليه
(وليس مجرد ملاحظات وصفية). إذا لم يتوفر سياق سعر، أعطِ ملاحظات وصفية بدون أرقام كما كان سابقاً.

أرجع حصراً كائن JSON بهذا الشكل بالضبط:
{
  "summary": "زبدة الخبر في جملتين قصيرتين بالعربية",
  "sentiment": "إيجابي جداً | إيجابي | محايد | سلبي | سلبي جداً",
  "impact_level": "عالي | متوسط | منخفض",
  "confidence": رقم من 0 إلى 100 يمثل ثقتك بهذا التحليل,
  "is_likely_official": true أو false (هل يبدو الخبر بياناً رسمياً/مصدراً موثوقاً أم إشاعة؟),
  "trade_plan": {
    "action": "دخول شراء | دخول بيع (شورت) | انتظار | تجنب",
    "entry_price": رقم تقريبي لسعر الدخول المقترح إذا توفر سياق السعر، وإلا اتركه null,
    "stop_loss_price": رقم تقريبي لسعر وقف الخسارة إذا توفر سياق السعر، وإلا اتركه null,
    "target_price": رقم تقريبي لسعر الهدف إذا توفر سياق السعر، وإلا اتركه null,
    "entry_note": "ملاحظة نصية موجزة عن منطقة الدخول المناسبة ومنطقها",
    "stop_loss_note": "ملاحظة موجزة عن منطق وقف الخسارة (مثلاً تحت مستوى الدعم)",
    "target_note": "ملاحظة موجزة عن الهدف المحتمل ومنطقه (مثلاً عند مستوى المقاومة)",
    "estimated_duration": "تقدير تقريبي جداً لمدة الصفقة حتى الوصول للهدف أو الخروج، اختر واحداً: 'قصيرة (خلال نفس يوم التداول)' أو 'قصيرة-متوسطة (1-3 أيام)' أو 'متوسطة (أسبوع تقريباً)' أو 'طويلة (أسابيع أو أكثر)'. هذا تقدير استرشادي فقط وليس وعداً بزمن دقيق",
    "exit_condition": "متى يجب الخروج من الصفقة (شرط واضح)"
  },
  "risk_warning": "تحذير مخاطرة قصير"
}

تنبيه: هذا تحليل استرشادي وليس توصية استثمارية مضمونة. كن واقعياً ولا تبالغ.
تنبيه إضافي: تقدير مدة الصفقة (estimated_duration) هو تخمين تقريبي جداً بناءً على طبيعة الخبر وسوابق مشابهة، وليس تنبؤاً دقيقاً بالوقت — الأسواق قد تتحرك أسرع أو أبطأ من أي تقدير بكثير.
تنبيه إضافي حول الأسعار: أي أرقام أسعار تعطيها هي تقديرات استرشادية مبنية على بيانات تاريخية ودعم/مقاومة، وليست ضماناً لتحرك السعر الفعلي."""


def translate_text_free(text: str, source_lang: str = "en", target_lang: str = "ar") -> str:
    """
    يترجم نصاً واحداً مجاناً بالكامل عبر خدمة MyMemory العامة (بدون مفتاح API
    وبدون أي تكلفة). مناسب لعناوين الأخبار القصيرة. يرجع النص الأصلي لو فشلت
    الترجمة لأي سبب.
    """
    if not text or not text.strip():
        return text
    try:
        r = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text[:490], "langpair": f"{source_lang}|{target_lang}"},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        translated = data.get("responseData", {}).get("translatedText")
        if translated and "MYMEMORY WARNING" not in translated.upper():
            return translated
        return text
    except Exception:
        return text


def translate_texts_free(texts: List[str], source_lang: str = "en", target_lang: str = "ar") -> List[str]:
    """يترجم قائمة نصوص مجاناً، نصاً نصاً (خدمة MyMemory لا تدعم الترجمة الدفعية)."""
    return [translate_text_free(t, source_lang, target_lang) for t in texts]


def translate_texts_to_arabic(
    openai_api_key: str, texts: List[str], model: str = "gpt-4o-mini"
) -> List[str]:
    """
    يترجم قائمة نصوص (عناوين أخبار عادةً) إلى العربية دفعة واحدة بطلب AI واحد
    بدل طلب منفصل لكل عنوان، توفيراً للتكلفة. يرجع النصوص الأصلية كما هي
    لو ما توفر مفتاح OpenAI أو صار خطأ.
    """
    if not openai_api_key or OpenAI is None or not texts:
        return texts

    client = OpenAI(api_key=openai_api_key)
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    system_prompt = (
        "أنت مترجم مالي محترف. تُرجم عناوين وأخبار مالية من الإنجليزية إلى العربية الفصحى "
        "بأسلوب صحفي مختصر وواضح، مع الحفاظ على أسماء الشركات والرموز كما هي بدون ترجمة. "
        "أرجع فقط كائن JSON بهذا الشكل: {\"translations\": [\"الترجمة 1\", \"الترجمة 2\", ...]} "
        "بنفس عدد وترتيب النصوص المُعطاة، بدون أي نص إضافي."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": numbered},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        translations = data.get("translations", [])
        if isinstance(translations, list) and len(translations) == len(texts):
            return translations
        return texts
    except Exception:
        return texts


def analyze_news_with_ai(
    openai_api_key: str,
    headline: str,
    summary: str,
    symbol: Optional[str] = None,
    model: str = "gpt-4o-mini",
    price_context: Optional[Dict] = None,
) -> Dict:
    """
    يرسل الخبر إلى OpenAI ويستقبل تحليلاً منظماً بصيغة JSON.
    price_context (اختياري): dict فيه current_price, support_20d, resistance_20d
    يُستخدم لتحديد نقاط دخول/خروج رقمية بدل ملاحظات وصفية فقط.
    """
    if not openai_api_key or OpenAI is None:
        return {"error": "مفتاح OpenAI غير متوفر أو المكتبة غير مثبتة."}

    client = OpenAI(api_key=openai_api_key)

    price_block = ""
    if price_context and "error" not in price_context:
        price_block = f"""
سياق السعر الحالي (استخدمه لتحديد أرقام دخول/خروج تقريبية):
السعر الحالي: {price_context.get('current_price')}
مستوى الدعم (20 يوم): {price_context.get('support_20d')}
مستوى المقاومة (20 يوم): {price_context.get('resistance_20d')}
"""

    user_content = f"""
السهم: {symbol or 'غير محدد'}
العنوان: {headline}
تفاصيل إضافية: {summary or 'لا يوجد'}
{price_block}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": f"فشل تحليل الذكاء الاصطناعي: {e}"}


# ----------------------------------------------------------------------
# 4) فلترة الأخبار حسب القوة (High-Impact Screener) - قواعد بسيطة
# ----------------------------------------------------------------------

HIGH_IMPACT_KEYWORDS = [
    # صفقات واندماجات
    "acquisition", "acquire", "acquires", "merger", "buyout", "takeover", "deal to buy",
    # تنظيمية وقانونية
    "fda approval", "fda", "sec investigation", "sec probe", "lawsuit", "antitrust",
    "regulatory", "settlement", "fine", "recall", "investigation", "probe",
    # أرباح وتوقعات
    "earnings beat", "earnings miss", "beats estimates", "misses estimates",
    "guidance", "raises guidance", "cuts guidance", "profit warning", "outlook",
    "quarterly results", "q1", "q2", "q3", "q4", "revenue beat", "revenue miss",
    # قرارات إدارية
    "bankruptcy", "chapter 11", "ceo resigns", "ceo steps down", "ceo fired",
    "layoffs", "job cuts", "restructuring", "spinoff", "ipo", "delisting",
    # حركة سعرية قوية
    "surges", "soars", "plunges", "tumbles", "crashes", "rallies", "sinks",
    "jumps", "spikes", "record high", "record low", "all-time high",
    # تحليل ومحللون
    "price target", "upgrade", "downgrade", "initiates coverage", "analyst",
    # اتفاقيات وعقود
    "partnership", "contract award", "government contract", "supply deal",
    "stock split", "dividend increase", "dividend cut", "buyback", "share repurchase",
    # تقنية وابتكار
    "data breach", "clinical trial", "patent", "chip shortage", "ai chip",
    "product launch", "recall of",
    # عربي
    "استحواذ", "اندماج", "إفلاس", "دعوى قضائية", "تحقيق", "أرباح",
    "توقعات", "شراكة", "عقد حكومي", "موافقة", "استقالة", "تراجع", "ارتفاع",
    "انهيار", "قفزة", "تخفيض", "رفع تصنيف", "خفض تصنيف", "استرداد أسهم",
]


def classify_news_impact(headline: str) -> str:
    """
    تصنيف أولي سريع (بدون AI) لمعرفة هل الخبر يستحق إرساله للتحليل العميق.
    """
    text = headline.lower()
    for kw in HIGH_IMPACT_KEYWORDS:
        if kw.lower() in text:
            return "مرشّح (High Impact)"
    return "عادي (Low Impact)"


def filter_high_impact_news(news_list: List[Dict]) -> List[Dict]:
    """يرجع فقط الأخبار المرشّحة كعالية التأثير حسب الكلمات المفتاحية."""
    result = []
    for n in news_list:
        headline = n.get("headline", "")
        if classify_news_impact(headline) == "مرشّح (High Impact)":
            n["_pre_classification"] = "High Impact"
            result.append(n)
    return result


# ----------------------------------------------------------------------
# 5) حاسبة إدارة المخاطر وحجم الصفقة (Position Sizer)
# ----------------------------------------------------------------------

def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float,
    target_price: Optional[float] = None,
) -> Dict:
    """
    يحسب حجم الصفقة المناسب بناءً على نسبة المخاطرة من المحفظة.
    """
    if entry_price <= 0 or stop_loss_price <= 0 or account_balance <= 0:
        return {"error": "الرجاء إدخال قيم أكبر من صفر."}

    risk_amount = account_balance * (risk_percent / 100)
    per_share_risk = abs(entry_price - stop_loss_price)

    if per_share_risk == 0:
        return {"error": "سعر الدخول ووقف الخسارة لا يمكن أن يكونا متساويين."}

    shares_to_buy = int(risk_amount / per_share_risk)
    total_position_value = shares_to_buy * entry_price
    max_loss_dollars = shares_to_buy * per_share_risk

    result = {
        "shares_to_buy": shares_to_buy,
        "total_position_value": round(total_position_value, 2),
        "max_loss_dollars": round(max_loss_dollars, 2),
        "risk_amount_allowed": round(risk_amount, 2),
        "per_share_risk": round(per_share_risk, 2),
        "position_pct_of_account": round((total_position_value / account_balance) * 100, 2),
    }

    if target_price:
        per_share_reward = abs(target_price - entry_price)
        result["potential_profit_dollars"] = round(shares_to_buy * per_share_reward, 2)
        result["risk_reward_ratio"] = round(per_share_reward / per_share_risk, 2) if per_share_risk else None

    return result


# ----------------------------------------------------------------------
# 6) إرسال تنبيه عبر تلغرام
# ----------------------------------------------------------------------

def send_telegram_alert(bot_token: str, chat_id: str, message: str) -> bool:
    """
    يرسل رسالة تلغرام. يدعم إرسال نفس الرسالة لأكثر من شخص إذا كانت
    chat_id تحتوي على عدة أرقام مفصولة بفاصلة، مثل: "982402036,1115974152"
    """
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    chat_ids = [c.strip() for c in chat_id.split(",") if c.strip()]
    if not chat_ids:
        return False
    all_sent = True
    for cid in chat_ids:
        try:
            r = requests.post(url, data={"chat_id": cid, "text": message}, timeout=10)
            if r.status_code != 200:
                all_sent = False
        except Exception:
            all_sent = False
    return all_sent


# ----------------------------------------------------------------------
# 7) أداة مساعدة: تنسيق الأرقام الكبيرة
# ----------------------------------------------------------------------

def format_large_number(num) -> str:
    if num is None:
        return "—"
    try:
        num = float(num)
    except (TypeError, ValueError):
        return "—"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000:
            return f"{num:,.2f}{unit}"
        num /= 1000
    return f"{num:,.2f}P"


# ----------------------------------------------------------------------
# 8) بصمة فريدة لكل خبر لمنع تكرار التنبيهات لنفس الخبر
# ----------------------------------------------------------------------

def news_fingerprint(item: Dict) -> str:
    """يولّد بصمة فريدة للخبر (لمنع إرسال نفس التنبيه أكثر من مرة)."""
    raw = f"{item.get('headline', '')}-{item.get('datetime', '')}-{item.get('source', '')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# 9) قاعدة بيانات التوصيات (SQLite) — لتخزين كل تحليل يُنتجه الذكاء الاصطناعي
#    وترتيبها لاحقاً حسب قوة الإشارة
# ----------------------------------------------------------------------

SENTIMENT_WEIGHTS = {
    "إيجابي جداً": 1.2,
    "إيجابي": 1.0,
    "محايد": 0.0,
    "سلبي": -1.0,
    "سلبي جداً": -1.2,
}


def init_db() -> None:
    """ينشئ جدول التوصيات إذا لم يكن موجوداً."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                headline TEXT,
                sentiment TEXT,
                impact_level TEXT,
                confidence REAL,
                is_likely_official INTEGER,
                action TEXT,
                entry_price REAL,
                stop_loss_price REAL,
                target_price REAL,
                entry_note TEXT,
                stop_loss_note TEXT,
                target_note TEXT,
                estimated_duration TEXT,
                exit_condition TEXT,
                score REAL,
                created_at TEXT,
                created_by TEXT
            )
        """)
        conn.commit()
        # إضافة الأعمدة الجديدة تلقائياً لو قاعدة البيانات قديمة (ترقية بدون فقدان بيانات)
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(recommendations)").fetchall()}
        for col in ["entry_price", "stop_loss_price", "target_price"]:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE recommendations ADD COLUMN {col} REAL")
        if "status" not in existing_cols:
            conn.execute("ALTER TABLE recommendations ADD COLUMN status TEXT")
        conn.commit()


def compute_score(sentiment: Optional[str], confidence) -> float:
    """
    يحسب درجة ترتيب للتوصية: كلما زادت الثقة وكانت المعنويات أقوى (إيجاباً أو سلباً)
    زادت الدرجة. يُستخدم هذا لترتيب "أفضل الفرص" في قائمة التوصيات.
    """
    weight = SENTIMENT_WEIGHTS.get(sentiment, 0.0)
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    return round(weight * conf, 2)


def save_recommendation(symbol: str, headline: str, analysis: Dict, created_by: str = "system") -> None:
    """يحفظ نتيجة تحليل الذكاء الاصطناعي كتوصية جديدة في قاعدة البيانات."""
    init_db()
    plan = analysis.get("trade_plan", {})
    score = compute_score(analysis.get("sentiment"), analysis.get("confidence"))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO recommendations
            (symbol, headline, sentiment, impact_level, confidence, is_likely_official,
             action, entry_price, stop_loss_price, target_price,
             entry_note, stop_loss_note, target_note, estimated_duration,
             exit_condition, score, created_at, created_by)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            symbol,
            headline,
            analysis.get("sentiment"),
            analysis.get("impact_level"),
            analysis.get("confidence"),
            1 if analysis.get("is_likely_official") else 0,
            plan.get("action"),
            plan.get("entry_price"),
            plan.get("stop_loss_price"),
            plan.get("target_price"),
            plan.get("entry_note"),
            plan.get("stop_loss_note"),
            plan.get("target_note"),
            plan.get("estimated_duration"),
            plan.get("exit_condition"),
            score,
            dt.datetime.now().isoformat(timespec="seconds"),
            created_by,
        ))
        conn.commit()


def get_all_recommendations(limit: int = 100, only_buy_signals: bool = False) -> List[Dict]:
    """
    يرجع كل التوصيات المحفوظة مرتّبة من الأقوى إشارة للأضعف.
    only_buy_signals=True يعرض فقط توصيات "دخول شراء" أو "دخول بيع (شورت)".
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM recommendations"
        if only_buy_signals:
            query += " WHERE action LIKE '%دخول%'"
        query += " ORDER BY score DESC, created_at DESC LIMIT ?"
        rows = conn.execute(query, (limit,)).fetchall()
        return [dict(r) for r in rows]


# ----------------------------------------------------------------------
# 10) الأسهم الأكثر ربحاً في السوق اليوم
# ----------------------------------------------------------------------

def fetch_top_gainers(limit: int = 15) -> List[Dict]:
    """
    يجلب أكثر الأسهم ارتفاعاً اليوم عبر نقطة بيانات ياهو فايننس غير الرسمية.
    ملاحظة مهمة: هذه النقطة (endpoint) غير موثّقة رسمياً من ياهو، وقد تتوقف
    أو يتغيّر شكلها دون سابق إنذار. إن توقفت، الحل هو الانتقال لمزوّد بيانات
    مدفوع مثل Polygon.io أو Financial Modeling Prep.
    """
    url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    params = {"formatted": "true", "scrIds": "day_gainers", "count": limit, "lang": "en-US"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        quotes = data["finance"]["result"][0]["quotes"]

        def _unwrap(val):
            if isinstance(val, dict):
                return val.get("raw", val.get("fmt"))
            return val

        rows = []
        for q in quotes[:limit]:
            rows.append({
                "الرمز": q.get("symbol"),
                "الاسم": q.get("shortName", q.get("symbol")),
                "السعر": _unwrap(q.get("regularMarketPrice")),
                "التغير %": _unwrap(q.get("regularMarketChangePercent")),
                "حجم التداول": _unwrap(q.get("regularMarketVolume")),
            })
        return rows
    except Exception as e:
        return [{"error": f"تعذر جلب القائمة (المصدر غير رسمي وقد يكون تغيّر): {e}"}]


def fetch_stocks_under_price(max_price: float = 10.0, limit: int = 50) -> List[Dict]:
    """
    يجلب ديناميكياً قائمة أسهم أمريكية حقيقية سعرها الحالي الفعلي أقل من
    max_price، مرتبة حسب حجم التداول (الأكثر سيولة أولاً)، عبر محرك فلترة
    ياهو فايننس غير الرسمي. يُنفَّذ هذا الطلب من جديد في كل مرة (كل دقيقة
    مع المراقبة التلقائية)، فيضمن إن كل رمز بالقائمة فعلاً تحت السعر المحدد
    وقت الفحص، بدل قائمة تخمينية ثابتة.
    ملاحظة: نقطة بيانات غير موثّقة رسمياً من ياهو، وقد تتوقف أو تتغيّر لاحقاً.
    """
    url = "https://query1.finance.yahoo.com/v1/finance/screener"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Content-Type": "application/json",
    }
    body = {
        "size": limit,
        "offset": 0,
        "sortType": "DESC",
        "sortField": "dayvolume",
        "quoteType": "EQUITY",
        "query": {
            "operator": "AND",
            "operands": [
                {"operator": "LT", "operands": ["intradayprice", max_price]},
                {"operator": "GT", "operands": ["intradayprice", 0.1]},
                {"operator": "EQ", "operands": ["region", "us"]},
            ],
        },
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=12)
        r.raise_for_status()
        data = r.json()
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        if not quotes:
            return [{"error": "ما رجعت الخدمة أي نتائج حالياً (قد تكون خارج أوقات التداول)."}]

        rows = []
        for q in quotes[:limit]:
            symbol = q.get("symbol")
            if not symbol:
                continue
            rows.append({
                "symbol": symbol,
                "name": q.get("shortName", symbol),
                "price": q.get("regularMarketPrice"),
                "change_pct": q.get("regularMarketChangePercent"),
                "volume": q.get("regularMarketVolume"),
            })
        return rows
    except Exception as e:
        return [{"error": f"تعذر جلب القائمة الديناميكية (المصدر غير رسمي وقد يكون تغيّر): {e}"}]


# ----------------------------------------------------------------------
# 11) فحص دخول السيولة/حجم التداول (متى دخلت سيولة على السهم)
# ----------------------------------------------------------------------

def check_liquidity_activity(symbol: str) -> Dict:
    """
    يفحص بيانات اليوم للسهم (شموع كل 5 دقائق) ويحدد متى آخر مرة دخلت فيها
    سيولة ملحوظة (حجم تداول أعلى من المعتاد بشكل واضح).
    هذا تقدير آلي بسيط بناءً على مقارنة حجم كل شمعة بمتوسط اليوم، وليس
    بيانات "Level 2" أو تدفق أوامر حقيقي.
    """
    try:
        t = yf.Ticker(symbol)
        intraday = t.history(period="1d", interval="5m")
        if intraday.empty or len(intraday) < 3:
            return {"status": "لا توجد بيانات كافية حالياً (قد يكون السوق مغلقاً)."}

        avg_vol = intraday["Volume"].mean()
        if avg_vol <= 0:
            return {"status": "لا توجد بيانات حجم كافية."}

        threshold = avg_vol * 1.8  # شمعة تعتبر "سيولة ملحوظة" لو حجمها أعلى من 1.8x المتوسط
        now = intraday.index[-1]

        spike_times = [idx for idx, vol in intraday["Volume"].items() if vol >= threshold]
        if not spike_times:
            return {
                "status": "لا توجد سيولة ملحوظة دخلت اليوم حتى الآن.",
                "last_candle_volume": int(intraday["Volume"].iloc[-1]),
                "avg_volume_5m": int(avg_vol),
            }

        last_spike = spike_times[-1]
        minutes_ago = (now - last_spike).total_seconds() / 60

        if minutes_ago <= 10:
            timing = "الآن (خلال آخر 10 دقائق)"
        elif minutes_ago <= 30:
            timing = "قبل شوي (خلال آخر نصف ساعة)"
        elif minutes_ago <= 120:
            timing = "من فترة قريبة (خلال آخر ساعتين)"
        else:
            timing = f"سابقاً اليوم (منذ حوالي {int(minutes_ago // 60)} ساعة)"

        return {
            "status": f"دخلت سيولة ملحوظة: {timing}",
            "spike_time": last_spike.strftime("%H:%M"),
            "minutes_ago": round(minutes_ago),
            "spike_volume": int(intraday.loc[last_spike, "Volume"]),
            "avg_volume_5m": int(avg_vol),
        }
    except Exception as e:
        return {"status": f"تعذر فحص السيولة: {e}"}


# ----------------------------------------------------------------------
# 12) بيانات ما قبل وبعد إغلاق السوق (Pre-Market / After-Hours)
# ----------------------------------------------------------------------

def fetch_extended_hours_data(symbol: str) -> Dict:
    """
    يجلب سعر السهم في تداول ما قبل الافتتاح (Pre-Market) وما بعد الإغلاق
    (After-Hours) إن كانت متوفرة من ياهو فايننس عبر yfinance.
    قد تكون بعض الحقول فارغة حسب توقيت الطلب (تتوفر فقط خارج ساعات التداول الرسمية).
    """
    try:
        t = yf.Ticker(symbol)
        info = t.info if hasattr(t, "info") else {}

        result = {
            "pre_market_price": info.get("preMarketPrice"),
            "pre_market_change_pct": info.get("preMarketChangePercent"),
            "post_market_price": info.get("postMarketPrice"),
            "post_market_change_pct": info.get("postMarketChangePercent"),
            "regular_market_price": info.get("regularMarketPrice") or info.get("currentPrice"),
        }
        return result
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------
# 13) أوقات تداول السوق الأمريكي وأيام العطل الرسمية
# ----------------------------------------------------------------------

# عطل بورصة نيويورك (NYSE) الرسمية لعام 2026 (بتوقيت شرق أمريكا ET)
US_MARKET_HOLIDAYS_2026 = {
    dt.date(2026, 1, 1): "رأس السنة الميلادية",
    dt.date(2026, 1, 19): "يوم مارتن لوثر كينغ",
    dt.date(2026, 2, 16): "يوم الرؤساء (واشنطن)",
    dt.date(2026, 4, 3): "الجمعة العظيمة (Good Friday)",
    dt.date(2026, 5, 25): "يوم الذكرى (Memorial Day)",
    dt.date(2026, 6, 19): "يوم جوونتينث (Juneteenth)",
    dt.date(2026, 7, 3): "عيد الاستقلال (يُحتفل به بدل 4 يوليو لأنه يوافق السبت)",
    dt.date(2026, 9, 7): "يوم العمال (Labor Day)",
    dt.date(2026, 11, 26): "عيد الشكر (Thanksgiving)",
    dt.date(2026, 12, 25): "عيد الميلاد",
}

# أوقات التداول بتوقيت شرق أمريكا (ET) — ملاحظة: لا تشمل فروقات التوقيت الصيفي/الشتوي تلقائياً
MARKET_HOURS_ET = {
    "pre_market_start": "04:00",
    "pre_market_end": "09:30",
    "regular_start": "09:30",
    "regular_end": "16:00",
    "after_hours_start": "16:00",
    "after_hours_end": "20:00",
}


def get_market_status() -> Dict:
    """
    يحدد الحالة الحالية للسوق الأمريكي: مفتوح، ما قبل الافتتاح، ما بعد الإغلاق،
    مغلق (عطلة نهاية أسبوع)، أو عطلة رسمية — بالاعتماد على توقيت شرق أمريكا.
    ملاحظة: الحساب يفترض توقيت ET بدون تعديل تلقائي دقيق للتوقيت الصيفي/الشتوي؛
    قد يختلف بساعة حسب موسم السنة.
    """
    now_utc = dt.datetime.utcnow()
    # تقريب بسيط لتوقيت شرق أمريكا (ET = UTC-5 شتاءً، UTC-4 صيفاً)
    # نستخدم UTC-4 (توقيت صيفي) لأغلب أشهر السنة كتقريب معقول
    now_et = now_utc - dt.timedelta(hours=4)
    today = now_et.date()
    weekday = now_et.weekday()  # 0=Monday ... 6=Sunday

    if today in US_MARKET_HOLIDAYS_2026:
        return {
            "status": "مغلق (عطلة رسمية)",
            "detail": US_MARKET_HOLIDAYS_2026[today],
            "now_et": now_et.strftime("%Y-%m-%d %H:%M"),
        }

    if weekday >= 5:  # سبت أو أحد
        return {
            "status": "مغلق (عطلة نهاية أسبوع)",
            "detail": "السوق يفتح يوم الاثنين القادم",
            "now_et": now_et.strftime("%Y-%m-%d %H:%M"),
        }

    current_time = now_et.strftime("%H:%M")
    if MARKET_HOURS_ET["pre_market_start"] <= current_time < MARKET_HOURS_ET["pre_market_end"]:
        status = "ما قبل الافتتاح (Pre-Market)"
    elif MARKET_HOURS_ET["regular_start"] <= current_time < MARKET_HOURS_ET["regular_end"]:
        status = "السوق مفتوح (تداول رسمي)"
    elif MARKET_HOURS_ET["after_hours_start"] <= current_time < MARKET_HOURS_ET["after_hours_end"]:
        status = "ما بعد الإغلاق (After-Hours)"
    else:
        status = "مغلق (خارج أوقات التداول)"

    # حساب أقرب عطلة رسمية قادمة
    upcoming_holidays = sorted([d for d in US_MARKET_HOLIDAYS_2026 if d >= today])
    next_holiday = None
    if upcoming_holidays:
        next_date = upcoming_holidays[0]
        next_holiday = {"date": next_date.strftime("%Y-%m-%d"), "name": US_MARKET_HOLIDAYS_2026[next_date]}

    return {
        "status": status,
        "now_et": now_et.strftime("%Y-%m-%d %H:%M"),
        "regular_hours": "9:30 صباحاً - 4:00 عصراً (بتوقيت شرق أمريكا ET)",
        "pre_market_hours": "4:00 - 9:30 صباحاً (ET)",
        "after_hours": "4:00 - 8:00 مساءً (ET)",
        "next_holiday": next_holiday,
    }


# ----------------------------------------------------------------------
# 14) إرسال إشعار عبر ntfy.sh (بديل مجاني لتلغرام/واتساب)
# ----------------------------------------------------------------------

def send_ntfy_alert(topic: str, message: str, title: str = "Financial AI Agent", priority: int = 3) -> bool:
    """
    يرسل إشعار فوري لكل المشتركين بقناة ntfy المحددة.
    نستخدم واجهة JSON الخاصة بـ ntfy (بدل الهيدرز مباشرة) لأنها تدعم
    النصوص العربية وغير اللاتينية بشكل صحيح.
    topic: اسم القناة السري (مثال: finai_alerts_boosh_2026_x7k9m)
    priority: من 1 (منخفض) إلى 5 (عاجل جداً)
    """
    if not topic or not message:
        return False
    try:
        r = requests.post(
            "https://ntfy.sh/",
            json={
                "topic": topic.strip(),
                "message": message,
                "title": title,
                "priority": priority,
                "tags": ["warning", "chart_with_upwards_trend"],
            },
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


# ----------------------------------------------------------------------
# 15) مستويات الدعم والمقاومة وكشف الاختراق
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 17) مؤشرات فنية: RSI و MACD
# ----------------------------------------------------------------------

def calculate_technical_indicators(symbol: str) -> Dict:
    """
    يحسب مؤشر القوة النسبية (RSI-14) وتقاطع MACD من بيانات يومية تاريخية.
    حساب قياسي معروف بدون مكتبات خارجية إضافية.
    """
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="6mo")
        if hist.empty or len(hist) < 35:
            return {"error": "بيانات غير كافية لحساب المؤشرات الفنية."}

        close = hist["Close"]

        # RSI (14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = float(rsi.iloc[-1])

        if latest_rsi >= 70:
            rsi_label = "🔴 تشبّع شرائي (Overbought) — قد يشهد تصحيحاً"
        elif latest_rsi <= 30:
            rsi_label = "🟢 تشبّع بيعي (Oversold) — قد يشهد ارتداداً"
        else:
            rsi_label = "⚪ منطقة محايدة"

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        macd_now, signal_now = float(macd_line.iloc[-1]), float(signal_line.iloc[-1])
        macd_prev, signal_prev = float(macd_line.iloc[-2]), float(signal_line.iloc[-2])

        if macd_prev <= signal_prev and macd_now > signal_now:
            macd_signal = "🟢 تقاطع إيجابي حديث (MACD فوق خط الإشارة) — إشارة صعودية محتملة"
        elif macd_prev >= signal_prev and macd_now < signal_now:
            macd_signal = "🔴 تقاطع سلبي حديث (MACD تحت خط الإشارة) — إشارة هبوطية محتملة"
        elif macd_now > signal_now:
            macd_signal = "⚪ MACD فوق خط الإشارة (زخم إيجابي مستمر)"
        else:
            macd_signal = "⚪ MACD تحت خط الإشارة (زخم سلبي مستمر)"

        return {
            "rsi": round(latest_rsi, 1),
            "rsi_label": rsi_label,
            "macd_signal": macd_signal,
        }
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------
# 18) إجماع المحللين (Analyst Consensus) من ياهو فايننس عبر yfinance
# ----------------------------------------------------------------------

def get_analyst_consensus(symbol: str) -> Dict:
    """
    يجلب متوسط سعر مستهدف من المحللين وتوصيتهم العامة (شراء/بيع/محايد)
    إن كانت متوفرة عبر ياهو فايننس. تتوفر عادة للأسهم الكبيرة والمتوسطة فقط.
    """
    try:
        t = yf.Ticker(symbol)
        info = t.info if hasattr(t, "info") else {}

        recommendation_map = {
            "strong_buy": "شراء قوي", "buy": "شراء", "hold": "محايد",
            "sell": "بيع", "strong_sell": "بيع قوي", "none": "غير متوفر",
        }
        rec_key = info.get("recommendationKey", "none")

        return {
            "target_mean": info.get("targetMeanPrice"),
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "recommendation": recommendation_map.get(rec_key, rec_key),
            "num_analysts": info.get("numberOfAnalystOpinions"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        }
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------
# 19) تتبع دقة أداء التوصيات فعلياً (Win Rate) — شفافية بدل الوعود
# ----------------------------------------------------------------------

def update_recommendation_status(rec_id: int, status: str) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE recommendations SET status = ? WHERE id = ?", (status, rec_id))
        conn.commit()


def check_and_update_open_recommendations() -> Dict:
    """
    يفحص كل التوصيات المفتوحة (status فارغ) اللي فيها أسعار دخول/وقف/هدف رقمية،
    يقارنها بالسعر الحالي الفعلي، ويحدّث حالتها تلقائياً (تحقق الهدف / ضرب وقف
    الخسارة / لسا مفتوحة). يرجع ملخص بعدد ما تحدّث.
    """
    init_db()
    updated = {"hit_target": 0, "hit_stop": 0, "still_open": 0, "checked": 0}
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM recommendations
            WHERE (status IS NULL OR status = '' OR status = 'open')
            AND entry_price IS NOT NULL AND stop_loss_price IS NOT NULL AND target_price IS NOT NULL
        """).fetchall()

    for row in rows:
        updated["checked"] += 1
        snap = fetch_stock_snapshot(row["symbol"])
        current_price = snap.get("current_price")
        if current_price is None:
            continue

        is_short = "بيع" in (row["action"] or "")
        new_status = None
        if is_short:
            if current_price <= row["target_price"]:
                new_status = "hit_target"
            elif current_price >= row["stop_loss_price"]:
                new_status = "hit_stop"
        else:
            if current_price >= row["target_price"]:
                new_status = "hit_target"
            elif current_price <= row["stop_loss_price"]:
                new_status = "hit_stop"

        if new_status:
            update_recommendation_status(row["id"], new_status)
            updated[new_status] += 1
        else:
            updated["still_open"] += 1

    return updated


def get_win_rate_stats() -> Dict:
    """يحسب نسبة نجاح التوصيات المغلقة (اللي تحقق فيها الهدف أو ضرب وقف الخسارة فعلياً)."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        hits = conn.execute("SELECT COUNT(*) FROM recommendations WHERE status = 'hit_target'").fetchone()[0]
        stops = conn.execute("SELECT COUNT(*) FROM recommendations WHERE status = 'hit_stop'").fetchone()[0]
        open_count = conn.execute(
            "SELECT COUNT(*) FROM recommendations WHERE status IS NULL OR status = '' OR status = 'open'"
        ).fetchone()[0]

    total_closed = hits + stops
    win_rate = round((hits / total_closed) * 100, 1) if total_closed > 0 else None
    return {
        "hit_target": hits, "hit_stop": stops, "still_open": open_count,
        "total_closed": total_closed, "win_rate": win_rate,
    }


def calculate_support_resistance(symbol: str) -> Dict:
    """
    يحسب مستويات دعم ومقاومة تقريبية للسهم بناءً على أعلى/أدنى سعر خلال
    آخر 20 و 50 يوم تداول، ويكشف هل السعر الحالي اخترق أحد هذي المستويات
    مؤخراً (آخر 3 أيام تداول). هذا حساب فني مبسط (Price Action) وليس
    تحليلاً فنياً احترافياً كاملاً (مثل Fibonacci أو Pivot Points الدقيقة).
    """
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="3mo")
        if hist.empty or len(hist) < 20:
            return {"error": "بيانات غير كافية لحساب الدعم والمقاومة."}

        current_price = float(hist["Close"].iloc[-1])

        support_20 = float(hist["Low"].tail(20).min())
        resistance_20 = float(hist["High"].tail(20).max())
        support_50 = float(hist["Low"].tail(50).min()) if len(hist) >= 50 else support_20
        resistance_50 = float(hist["High"].tail(50).max()) if len(hist) >= 50 else resistance_20

        recent = hist.tail(3)
        broke_resistance = bool((recent["Close"] > resistance_20 * 0.999).any()) and current_price >= resistance_20 * 0.995
        broke_support = bool((recent["Close"] < support_20 * 1.001).any()) and current_price <= support_20 * 1.005

        if broke_resistance:
            signal = "🚀 اختراق للأعلى: السعر كسر مستوى المقاومة القريب (20 يوم) مؤخراً"
        elif broke_support:
            signal = "⚠️ كسر للأسفل: السعر كسر مستوى الدعم القريب (20 يوم) مؤخراً"
        else:
            distance_to_resistance = ((resistance_20 - current_price) / current_price) * 100
            distance_to_support = ((current_price - support_20) / current_price) * 100
            signal = (
                f"يتداول بين الدعم والمقاومة — يبعد {distance_to_resistance:.1f}% عن المقاومة "
                f"و {distance_to_support:.1f}% فوق الدعم"
            )

        return {
            "current_price": round(current_price, 2),
            "support_20d": round(support_20, 2),
            "resistance_20d": round(resistance_20, 2),
            "support_50d": round(support_50, 2),
            "resistance_50d": round(resistance_50, 2),
            "signal": signal,
            "broke_resistance": broke_resistance,
            "broke_support": broke_support,
        }
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------
# 16) حالة السوق بتوقيت شرق أمريكا (ET) وتوقيت السعودية (AST) بدقة
#     باستخدام zoneinfo (يراعي التوقيت الصيفي/الشتوي تلقائياً)
# ----------------------------------------------------------------------

try:
    from zoneinfo import ZoneInfo
    _ET_ZONE = ZoneInfo("America/New_York")
    _SAUDI_ZONE = ZoneInfo("Asia/Riyadh")
    _ZONEINFO_OK = True
except Exception:
    _ZONEINFO_OK = False


def get_market_status_v2() -> Dict:
    """
    يحدد حالة السوق الأمريكي الحالية بدقة (مع مراعاة التوقيت الصيفي/الشتوي)،
    ويعرض الوقت الحالي بتوقيت نيويورك (ET) وبتوقيت السعودية (AST) معاً.
    """
    if not _ZONEINFO_OK:
        return get_market_status()  # احتياطي: النسخة التقريبية القديمة

    now_utc = dt.datetime.now(dt.timezone.utc)
    now_et = now_utc.astimezone(_ET_ZONE)
    now_saudi = now_utc.astimezone(_SAUDI_ZONE)
    today_et = now_et.date()
    weekday = now_et.weekday()

    result_base = {
        "now_et": now_et.strftime("%Y-%m-%d %H:%M"),
        "now_saudi": now_saudi.strftime("%Y-%m-%d %H:%M"),
    }

    if today_et in US_MARKET_HOLIDAYS_2026:
        result_base.update({"status": "مغلق (عطلة رسمية)", "detail": US_MARKET_HOLIDAYS_2026[today_et]})
        return result_base

    if weekday >= 5:
        result_base.update({"status": "مغلق (عطلة نهاية أسبوع)", "detail": "السوق يفتح يوم الاثنين القادم"})
        return result_base

    current_time = now_et.strftime("%H:%M")
    if MARKET_HOURS_ET["pre_market_start"] <= current_time < MARKET_HOURS_ET["pre_market_end"]:
        status = "ما قبل الافتتاح (Pre-Market)"
    elif MARKET_HOURS_ET["regular_start"] <= current_time < MARKET_HOURS_ET["regular_end"]:
        status = "السوق مفتوح (تداول رسمي)"
    elif MARKET_HOURS_ET["after_hours_start"] <= current_time < MARKET_HOURS_ET["after_hours_end"]:
        status = "ما بعد الإغلاق (After-Hours)"
    else:
        status = "مغلق (خارج أوقات التداول)"

    # نحسب أوقات الجلسات اليوم بتوقيت السعودية (تتحول تلقائياً حسب التوقيت الصيفي/الشتوي)
    def et_time_to_saudi_str(hhmm: str) -> str:
        h, m = map(int, hhmm.split(":"))
        et_dt = now_et.replace(hour=h, minute=m, second=0, microsecond=0)
        saudi_dt = et_dt.astimezone(_SAUDI_ZONE)
        return saudi_dt.strftime("%I:%M %p").lstrip("0")

    upcoming_holidays = sorted([d for d in US_MARKET_HOLIDAYS_2026 if d >= today_et])
    next_holiday = None
    if upcoming_holidays:
        next_date = upcoming_holidays[0]
        next_holiday = {"date": next_date.strftime("%Y-%m-%d"), "name": US_MARKET_HOLIDAYS_2026[next_date]}

    result_base.update({
        "status": status,
        "pre_market_hours_saudi": f"{et_time_to_saudi_str(MARKET_HOURS_ET['pre_market_start'])} - {et_time_to_saudi_str(MARKET_HOURS_ET['pre_market_end'])}",
        "regular_hours_saudi": f"{et_time_to_saudi_str(MARKET_HOURS_ET['regular_start'])} - {et_time_to_saudi_str(MARKET_HOURS_ET['regular_end'])}",
        "after_hours_saudi": f"{et_time_to_saudi_str(MARKET_HOURS_ET['after_hours_start'])} - {et_time_to_saudi_str(MARKET_HOURS_ET['after_hours_end'])}",
        "regular_hours": MARKET_HOURS_ET["regular_start"] + " - " + MARKET_HOURS_ET["regular_end"] + " (ET)",
        "pre_market_hours": MARKET_HOURS_ET["pre_market_start"] + " - " + MARKET_HOURS_ET["pre_market_end"] + " (ET)",
        "after_hours": MARKET_HOURS_ET["after_hours_start"] + " - " + MARKET_HOURS_ET["after_hours_end"] + " (ET)",
        "next_holiday": next_holiday,
    })
    return result_base


# ----------------------------------------------------------------------
# 20) حسابات مستخدمين ذاتية التسجيل (عبر رمز دعوة) — تُحفظ بقاعدة البيانات
# ----------------------------------------------------------------------

def init_users_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_users (
                username TEXT PRIMARY KEY,
                name TEXT,
                password_hash TEXT,
                created_at TEXT
            )
        """)
        conn.commit()


def create_app_user(username: str, name: str, password_hash: str) -> bool:
    """يسجّل مستخدماً جديداً. يرجع False لو اسم المستخدم محجوز مسبقاً."""
    init_users_db()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            conn.execute(
                "INSERT INTO app_users (username, name, password_hash, created_at) VALUES (?,?,?,?)",
                (username, name, password_hash, dt.datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def get_app_user(username: str) -> Optional[Dict]:
    init_users_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM app_users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def list_app_users() -> List[Dict]:
    init_users_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT username, name, created_at FROM app_users ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
