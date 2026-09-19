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
    except requests.exceptions.HTTPError as e:
        if r.status_code == 429:
            return [{"error": "تجاوزت حد الطلبات المجاني من Finnhub (60 طلب/دقيقة). قلّل عدد الأسهم المراقَبة أو انتظر دقيقة وجرب مرة ثانية."}]
        return [{"error": str(e)}]
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
        atr_block = ""
        if price_context.get("atr_suggested_stop_distance"):
            atr_block = f"""
مسافة وقف خسارة مقترحة فنياً (بناءً على تقلب السهم ATR): {price_context.get('atr_suggested_stop_distance')}
مسافة هدف مقترحة فنياً (بناءً على ATR): {price_context.get('atr_suggested_target_distance')}
استخدم هذي المسافات كمرجع لضبط وقف الخسارة والهدف بما يناسب تقلب السهم الفعلي (سهم متقلب يحتاج مسافة أكبر)."""
        price_block = f"""
سياق السعر الحالي (استخدمه لتحديد أرقام دخول/خروج تقريبية):
السعر الحالي: {price_context.get('current_price')}
مستوى الدعم (20 يوم): {price_context.get('support_20d')}
مستوى المقاومة (20 يوم): {price_context.get('resistance_20d')}
{atr_block}
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
        if "confluence_score" not in existing_cols:
            conn.execute("ALTER TABLE recommendations ADD COLUMN confluence_score REAL")
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


def has_recent_open_recommendation(symbol: str, hours: int = 24) -> bool:
    """
    يتحقق هل فيه توصية 'مفتوحة' (غير محسومة بعد) لنفس السهم خلال آخر عدد
    ساعات محدد. يُستخدم لمنع تسجيل نفس الإشارة كصفقة 'جديدة' لو أعيد
    تحليلها بالخطأ (مثلاً بعد إعادة تشغيل الجلسة أو التطبيق).
    """
    init_db()
    cutoff = (dt.datetime.now() - dt.timedelta(hours=hours)).isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT COUNT(*) FROM recommendations
            WHERE symbol = ? AND created_at >= ?
            AND (status IS NULL OR status = '' OR status = 'open')
        """, (symbol, cutoff)).fetchone()
    return row[0] > 0


def save_recommendation(symbol: str, headline: str, analysis: Dict, created_by: str = "system", confluence_score: Optional[float] = None) -> None:
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
             exit_condition, score, created_at, created_by, confluence_score)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            confluence_score,
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
    max_price، مرتبة حسب حجم التداول (الأكثر سيولة أولاً). يعتمد على نفس
    نقطة "القوائم الجاهزة" المستخدمة بدالة fetch_top_gainers (وهي الشغّالة
    فعلياً)، ويجمع نتائج عدة قوائم جاهزة (الأكثر تداولاً، الأكثر ربحاً،
    الأكثر خسارة، الشركات الصغيرة الصاعدة)، ثم يفلتر النتائج بنفسه محلياً
    للاحتفاظ فقط بالأسهم اللي سعرها الفعلي أقل من max_price وقت الفحص.

    يُنفَّذ هذا الطلب من جديد في كل مرة (كل دقيقة مع المراقبة التلقائية)،
    فيضمن إن كل رمز بالقائمة فعلاً تحت السعر المحدد وقت الفحص، بدل قائمة
    تخمينية ثابتة. ملاحظة: يعتمد على نقطة بيانات غير موثّقة رسمياً من
    ياهو، وقد تتوقف أو تتغيّر لاحقاً.
    """
    url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    screener_ids = ["most_actives", "day_gainers", "day_losers", "small_cap_gainers"]

    def _unwrap(val):
        if isinstance(val, dict):
            return val.get("raw", val.get("fmt"))
        return val

    collected = {}
    last_error = None
    for scr_id in screener_ids:
        try:
            params = {"formatted": "true", "scrIds": scr_id, "count": 100, "lang": "en-US"}
            r = requests.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            quotes = data["finance"]["result"][0]["quotes"]
            for q in quotes:
                symbol = q.get("symbol")
                if not symbol or symbol in collected:
                    continue
                price = _unwrap(q.get("regularMarketPrice"))
                if price is None or price <= 0.1 or price >= max_price:
                    continue
                collected[symbol] = {
                    "symbol": symbol,
                    "name": q.get("shortName", symbol),
                    "price": price,
                    "change_pct": _unwrap(q.get("regularMarketChangePercent")),
                    "volume": _unwrap(q.get("regularMarketVolume")) or 0,
                }
        except Exception as e:
            last_error = str(e)
            continue

    if not collected:
        return [{"error": f"تعذر جلب أي نتائج حالياً (المصدر غير رسمي وقد يكون تغيّر): {last_error or 'لا توجد نتائج مطابقة'}"}]

    rows = sorted(collected.values(), key=lambda x: x.get("volume", 0), reverse=True)
    return rows[:limit]


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
    """
    يحسب نسبة نجاح التوصيات المغلقة (اللي تحقق فيها الهدف أو ضرب وقف الخسارة فعلياً)،
    ومتوسط نسبة المخاطرة/العائد المخطط لها (Risk:Reward) لإعطاء صورة أدق
    من مجرد نسبة النجاح — صفقة رابحة كبيرة تعادل عدة صفقات خاسرة صغيرة.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        hits = conn.execute("SELECT COUNT(*) FROM recommendations WHERE status = 'hit_target'").fetchone()[0]
        stops = conn.execute("SELECT COUNT(*) FROM recommendations WHERE status = 'hit_stop'").fetchone()[0]
        open_count = conn.execute(
            "SELECT COUNT(*) FROM recommendations WHERE status IS NULL OR status = '' OR status = 'open'"
        ).fetchone()[0]

        closed_rows = conn.execute("""
            SELECT entry_price, stop_loss_price, target_price FROM recommendations
            WHERE status IN ('hit_target', 'hit_stop')
            AND entry_price IS NOT NULL AND stop_loss_price IS NOT NULL AND target_price IS NOT NULL
        """).fetchall()

    total_closed = hits + stops
    win_rate = round((hits / total_closed) * 100, 1) if total_closed > 0 else None

    rr_ratios = []
    for row in closed_rows:
        risk = abs(row["entry_price"] - row["stop_loss_price"])
        reward = abs(row["target_price"] - row["entry_price"])
        if risk > 0:
            rr_ratios.append(reward / risk)
    avg_rr = round(sum(rr_ratios) / len(rr_ratios), 2) if rr_ratios else None

    expectancy = None
    if avg_rr is not None and win_rate is not None:
        win_prob = win_rate / 100
        expectancy = round((win_prob * avg_rr) - (1 - win_prob), 2)

    return {
        "hit_target": hits, "hit_stop": stops, "still_open": open_count,
        "total_closed": total_closed, "win_rate": win_rate,
        "avg_risk_reward": avg_rr, "expectancy_r": expectancy,
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


# ----------------------------------------------------------------------
# 21) تنبيهات سعرية بدون الحاجة لخبر (Price Alerts)
# ----------------------------------------------------------------------

def init_price_alerts_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                target_price REAL,
                direction TEXT,
                created_by TEXT,
                created_at TEXT,
                triggered INTEGER DEFAULT 0,
                triggered_at TEXT
            )
        """)
        conn.commit()


def add_price_alert(symbol: str, target_price: float, direction: str, created_by: str = "system") -> None:
    """direction: 'above' (نبّهني إذا وصل فوق) أو 'below' (نبّهني إذا نزل تحت)."""
    init_price_alerts_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO price_alerts (symbol, target_price, direction, created_by, created_at) VALUES (?,?,?,?,?)",
            (symbol.upper(), target_price, direction, created_by, dt.datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()


def get_active_price_alerts() -> List[Dict]:
    init_price_alerts_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM price_alerts WHERE triggered = 0 ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_price_alert(alert_id: int) -> None:
    init_price_alerts_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM price_alerts WHERE id = ?", (alert_id,))
        conn.commit()


def check_price_alerts() -> List[Dict]:
    """
    يفحص كل التنبيهات السعرية النشطة، يقارنها بالسعر الحالي الفعلي،
    ويرجع قائمة بالتنبيهات اللي تحققت الآن (ويعلّمها كمُفعّلة بقاعدة البيانات
    عشان ما تتكرر التنبيهات لنفس الحدث).
    """
    init_price_alerts_db()
    alerts = get_active_price_alerts()
    triggered = []
    with sqlite3.connect(DB_PATH) as conn:
        for alert in alerts:
            snap = fetch_stock_snapshot(alert["symbol"])
            current_price = snap.get("current_price")
            if current_price is None:
                continue
            hit = (
                (alert["direction"] == "above" and current_price >= alert["target_price"])
                or (alert["direction"] == "below" and current_price <= alert["target_price"])
            )
            if hit:
                conn.execute(
                    "UPDATE price_alerts SET triggered = 1, triggered_at = ? WHERE id = ?",
                    (dt.datetime.now().isoformat(timespec="seconds"), alert["id"]),
                )
                alert["current_price"] = current_price
                triggered.append(alert)
        conn.commit()
    return triggered


# ----------------------------------------------------------------------
# 22) تقويم الأرباح (Earnings Calendar)
# ----------------------------------------------------------------------

def fetch_earnings_calendar(symbols: List[str]) -> List[Dict]:
    """
    يجلب تاريخ إعلان الأرباح القادم لكل رمز بقائمة معطاة، عبر yfinance.
    يرجع النتائج مرتبة من الأقرب موعداً للأبعد. الرموز اللي ما فيها معلومة
    متوفرة تُستبعد من النتيجة.
    """
    results = []
    for symbol in symbols:
        try:
            t = yf.Ticker(symbol)
            cal = t.calendar
            earnings_date = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if isinstance(ed, list) and ed:
                    earnings_date = ed[0]
                elif ed:
                    earnings_date = ed
            elif hasattr(cal, "empty") and not cal.empty:
                if "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"]
                    earnings_date = val.iloc[0] if hasattr(val, "iloc") else val

            if earnings_date is None:
                continue

            if hasattr(earnings_date, "strftime"):
                date_str = earnings_date.strftime("%Y-%m-%d")
                date_obj = earnings_date
            else:
                date_str = str(earnings_date)
                date_obj = None

            results.append({"symbol": symbol, "earnings_date": date_str, "_sort_key": date_obj or dt.date.max})
        except Exception:
            continue

    results.sort(key=lambda x: str(x["_sort_key"]))
    for r in results:
        r.pop("_sort_key", None)
    return results


# ----------------------------------------------------------------------
# 23) اختبار تاريخي مبسّط (Backtest) بناءً على السعر فقط (بدون أخبار)
# ----------------------------------------------------------------------

def run_technical_backtest(symbol: str, months: int = 3) -> Dict:
    """
    اختبار تاريخي مبسّط: يفحص آخر عدة أشهر من بيانات السهم، ويحاكي دخول
    صفقة كل مرة يخترق فيها السعر مقاومة 20 يوم (بافتراض وقف خسارة 3% وهدف
    6% من سعر الدخول)، ويحسب كم صفقة كانت ستنجح تاريخياً بهذا المنطق.

    ⚠️ هذا اختبار مبني على حركة السعر فقط (Price Action)، ولا يشمل تأثير
    الأخبار الفعلية وقتها — فهو تقريبي وأداة استرشادية لفهم سلوك السهم
    التاريخي العام، وليس ضماناً لأداء مستقبلي أو محاكاة دقيقة لمنطق
    الذكاء الاصطناعي المستخدم بالتحليل اللحظي.
    """
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=f"{months}mo")
        if hist.empty or len(hist) < 25:
            return {"error": "بيانات تاريخية غير كافية لإجراء الاختبار."}

        closes = hist["Close"].values
        highs = hist["High"].values
        lows = hist["Low"].values

        trades = []
        i = 20
        while i < len(closes) - 1:
            resistance_20 = highs[max(0, i - 20):i].max()
            if closes[i] > resistance_20:
                entry = closes[i]
                stop = entry * 0.97
                target = entry * 1.06
                outcome = None
                for j in range(i + 1, min(i + 15, len(closes))):
                    if lows[j] <= stop:
                        outcome = "loss"
                        break
                    if highs[j] >= target:
                        outcome = "win"
                        break
                if outcome:
                    trades.append(outcome)
                    i = j
                else:
                    i += 1
            else:
                i += 1

        if not trades:
            return {"error": "لم يُسجَّل أي اختراق مقاومة واضح خلال هذي الفترة لإجراء اختبار عليه."}

        wins = trades.count("win")
        losses = trades.count("loss")
        win_rate = round((wins / len(trades)) * 100, 1)

        return {
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "period_months": months,
        }
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------
# 24) درجة تطابق الإشارات (Confluence Score) — يجمع الفني + السيولة +
#     إجماع المحللين + مصداقية المصدر بنسبة واحدة واضحة قبل الدخول
# ----------------------------------------------------------------------

def calculate_confluence_score(symbol: str, action: str, sentiment: str, is_likely_official: bool) -> Dict:
    """
    يحسب درجة تطابق شاملة (0-95%) تجمع بين عدة مصادر مستقلة:
    - مصداقية مصدر الخبر (بيان رسمي أو إشاعة)
    - قوة المعنويات من تحليل الذكاء الاصطناعي
    - مؤشرات فنية (RSI, MACD)
    - اختراق دعم/مقاومة
    - دخول سيولة حديثة
    - إجماع المحللين (لو متوفر)

    كل عامل يُصنَّف كـ 'مؤكِّد' (يدعم الصفقة) أو 'متعارض' (يضعفها) أو
    'غير متوفر'، عشان يشوف المستخدم بالضبط وين الاتفاق ووين التضارب
    قبل ما يقرر يدخل. هذا مؤشر استرشادي إحصائي، وليس ضماناً لنجاح الصفقة.
    """
    is_buy = "شراء" in (action or "") or "دخول" in (action or "") and "بيع" not in (action or "")
    is_sell = "بيع" in (action or "") or "شورت" in (action or "")

    score = 50  # نقطة بداية محايدة
    confirmations = []
    conflicts = []
    unavailable = []
    risk_warnings = []

    # 1) مصداقية المصدر
    if is_likely_official:
        score += 8
        confirmations.append("✅ الخبر يبدو من مصدر رسمي موثوق (بيان صحفي رسمي)")
    else:
        conflicts.append("⚠️ الخبر قد يكون شائعة أو غير مؤكد رسمياً")
        score -= 5

    # 2) قوة المعنويات
    sentiment_map = {"إيجابي جداً": 15, "إيجابي": 8, "سلبي": -8, "سلبي جداً": -15, "محايد": 0}
    sent_points = sentiment_map.get(sentiment, 0)
    if is_sell:
        sent_points = -sent_points
    if sent_points > 0:
        score += sent_points
        confirmations.append(f"✅ معنويات الخبر ({sentiment}) تدعم اتجاه الصفقة")
    elif sent_points < 0:
        score += sent_points
        conflicts.append(f"⚠️ معنويات الخبر ({sentiment}) تتعارض مع اتجاه الصفقة")

    # 3) المؤشرات الفنية RSI/MACD
    tech = calculate_technical_indicators(symbol)
    if "error" in tech:
        unavailable.append("المؤشرات الفنية (RSI/MACD) غير متوفرة حالياً")
    else:
        rsi = tech["rsi"]
        if is_buy:
            if rsi <= 30:
                score += 10
                confirmations.append(f"✅ RSI ({rsi}) بمنطقة تشبّع بيعي — فرصة ارتداد تدعم الشراء")
            elif rsi >= 70:
                score -= 12
                conflicts.append(f"⚠️ RSI ({rsi}) بمنطقة تشبّع شرائي — مخاطرة تصحيح قريب")
        elif is_sell:
            if rsi >= 70:
                score += 10
                confirmations.append(f"✅ RSI ({rsi}) بمنطقة تشبّع شرائي — يدعم فرضية الهبوط")
            elif rsi <= 30:
                score -= 12
                conflicts.append(f"⚠️ RSI ({rsi}) بمنطقة تشبّع بيعي — مخاطرة ارتداد ضد الشورت")

        bullish_macd = "🟢" in tech.get("macd_signal", "")
        bearish_macd = "🔴" in tech.get("macd_signal", "")
        if is_buy and bullish_macd:
            score += 8
            confirmations.append("✅ MACD يعطي إشارة إيجابية حديثة")
        elif is_buy and bearish_macd:
            score -= 8
            conflicts.append("⚠️ MACD يعطي إشارة سلبية حديثة")
        elif is_sell and bearish_macd:
            score += 8
            confirmations.append("✅ MACD يعطي إشارة سلبية حديثة (يدعم الشورت)")
        elif is_sell and bullish_macd:
            score -= 8
            conflicts.append("⚠️ MACD يعطي إشارة إيجابية حديثة (يتعارض مع الشورت)")

    # 4) اختراق الدعم/المقاومة
    sr = calculate_support_resistance(symbol)
    if "error" in sr:
        unavailable.append("مستويات الدعم/المقاومة غير متوفرة حالياً")
    else:
        if is_buy and sr.get("broke_resistance"):
            score += 12
            confirmations.append("✅ السعر اخترق مستوى مقاومة مهم مؤخراً")
        elif is_buy and sr.get("broke_support"):
            score -= 12
            conflicts.append("⚠️ السعر كسر مستوى دعم مهم مؤخراً")
        elif is_sell and sr.get("broke_support"):
            score += 12
            confirmations.append("✅ السعر كسر مستوى دعم مهم (يدعم الشورت)")
        elif is_sell and sr.get("broke_resistance"):
            score -= 12
            conflicts.append("⚠️ السعر اخترق مقاومة للأعلى (يتعارض مع الشورت)")

    # 5) دخول سيولة حديثة
    liq = check_liquidity_activity(symbol)
    if "error" in liq or "status" not in liq:
        unavailable.append("بيانات السيولة اللحظية غير متوفرة حالياً")
    else:
        if "الآن" in liq["status"] or "قبل شوي" in liq["status"]:
            score += 7
            confirmations.append("✅ دخلت سيولة تداول ملحوظة على السهم مؤخراً")

    # 6) إجماع المحللين
    analyst = get_analyst_consensus(symbol)
    if "error" in analyst or not analyst.get("num_analysts"):
        unavailable.append("إجماع المحللين غير متوفر لهذا السهم")
    else:
        rec = analyst.get("recommendation", "")
        if is_buy and rec in ("شراء قوي", "شراء"):
            score += 10
            confirmations.append(f"✅ إجماع المحللين ({rec}) يدعم الشراء")
        elif is_buy and rec in ("بيع", "بيع قوي"):
            score -= 10
            conflicts.append(f"⚠️ إجماع المحللين ({rec}) يتعارض مع الشراء")
        elif is_sell and rec in ("بيع", "بيع قوي"):
            score += 10
            confirmations.append(f"✅ إجماع المحللين ({rec}) يدعم الشورت")
        elif is_sell and rec in ("شراء قوي", "شراء"):
            score -= 10
            conflicts.append(f"⚠️ إجماع المحللين ({rec}) يتعارض مع الشورت")

    # 7) اتجاه السوق العام (S&P 500)
    market = fetch_market_trend()
    if "error" in market:
        unavailable.append("اتجاه السوق العام غير متوفر حالياً")
    else:
        if is_buy and market["direction"] == "up":
            score += 7
            confirmations.append(f"✅ السوق العام بزخم صعودي ({market['change_pct']}% آخر 5 أيام) يدعم الشراء")
        elif is_buy and market["direction"] == "down":
            score -= 7
            conflicts.append(f"⚠️ السوق العام بزخم هبوطي ({market['change_pct']}% آخر 5 أيام) يضعف فرص الشراء")
        elif is_sell and market["direction"] == "down":
            score += 7
            confirmations.append(f"✅ السوق العام بزخم هبوطي ({market['change_pct']}% آخر 5 أيام) يدعم الشورت")
        elif is_sell and market["direction"] == "up":
            score -= 7
            conflicts.append(f"⚠️ السوق العام بزخم صعودي ({market['change_pct']}% آخر 5 أيام) يضعف فرص الشورت")

    # تحذيرات مخاطرة مستقلة عن درجة التطابق (لا تؤثر على النسبة، بس تنبيه إضافي)
    earnings = fetch_earnings_calendar([symbol])
    if earnings:
        try:
            ed = dt.datetime.strptime(earnings[0]["earnings_date"][:10], "%Y-%m-%d").date()
            days_away = (ed - dt.date.today()).days
            if 0 <= days_away <= 3:
                risk_warnings.append(f"📅 إعلان أرباح {symbol} خلال {days_away} يوم — تقلب حاد متوقع، مخاطرة إضافية على وقف الخسارة")
        except Exception:
            pass

    concentration = check_sector_concentration(symbol)
    if concentration.get("same_sector_count", 0) >= 2:
        risk_warnings.append(
            f"🏭 عندك {concentration['same_sector_count']} توصية مفتوحة أخرى بنفس قطاع '{concentration['sector']}' "
            f"({', '.join(concentration.get('same_sector_symbols', []))}) — تركّز مخاطرة على قطاع واحد"
        )

    score = max(5, min(95, round(score)))  # لا نعطي أبداً 0% أو 100% — ما فيه يقين مطلق بالأسواق

    if score >= 75:
        verdict = "🟢 تطابق قوي — عدة إشارات مستقلة تدعم بعضها"
    elif score >= 55:
        verdict = "🟡 تطابق متوسط — إشارات إيجابية أكثر من السلبية، لكن فيه تحفظات"
    elif score >= 40:
        verdict = "🟠 تطابق ضعيف/متضارب — إشارات متعارضة، كن حذراً"
    else:
        verdict = "🔴 تطابق سلبي — أغلب الإشارات تتعارض مع الصفقة"

    return {
        "score": score,
        "verdict": verdict,
        "confirmations": confirmations,
        "conflicts": conflicts,
        "unavailable": unavailable,
        "risk_warnings": risk_warnings,
    }


# ----------------------------------------------------------------------
# 25) اتجاه السوق العام (S&P 500) — عامل سابع لدرجة التطابق
# ----------------------------------------------------------------------

def fetch_market_trend() -> Dict:
    """
    يفحص اتجاه مؤشر S&P 500 (عبر صندوق SPY) خلال آخر 5 أيام تداول،
    لمعرفة هل السوق العام بزخم صعودي أو هبوطي — يفيد كفلتر إضافي
    (صفقة شراء بسوق هابط بقوة أضعف احتمالاً من نفس الصفقة بسوق صاعد).
    """
    try:
        t = yf.Ticker("SPY")
        hist = t.history(period="10d")
        if hist.empty or len(hist) < 5:
            return {"error": "بيانات السوق العام غير كافية حالياً."}
        recent = hist["Close"].tail(5)
        change_pct = ((recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]) * 100
        if change_pct > 1:
            direction = "up"
        elif change_pct < -1:
            direction = "down"
        else:
            direction = "flat"
        return {"direction": direction, "change_pct": round(float(change_pct), 2)}
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------
# 26) متوسط المدى الحقيقي (ATR) لتحديد وقف/هدف متكيّف مع التقلب
# ----------------------------------------------------------------------

def calculate_atr(symbol: str, period: int = 14) -> Dict:
    """
    يحسب متوسط المدى الحقيقي (ATR) للسهم — مقياس علمي لمدى تقلب السهم
    يومياً، يُستخدم لتحديد مسافة وقف خسارة/هدف مناسبة لطبيعة كل سهم
    (سهم متقلب زي MARA يحتاج مسافة أكبر من سهم هادئ زي KO).
    """
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="2mo")
        if hist.empty or len(hist) < period + 1:
            return {"error": "بيانات غير كافية لحساب ATR."}

        high, low, close = hist["High"], hist["Low"], hist["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        current_price = float(close.iloc[-1])
        atr_pct = (atr / current_price) * 100 if current_price else None

        return {
            "atr": round(float(atr), 2),
            "atr_pct": round(float(atr_pct), 2) if atr_pct else None,
            "suggested_stop_distance": round(float(atr) * 1.5, 2),
            "suggested_target_distance": round(float(atr) * 3, 2),
        }
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------
# 27) تحذير التركّز الزائد على قطاع واحد
# ----------------------------------------------------------------------

def check_sector_concentration(symbol: str) -> Dict:
    """
    يفحص هل فيه توصيات مفتوحة أخرى بنفس قطاع السهم المطلوب، لتنبيه
    المستخدم إنه معرّض لمخاطرة مركّزة على قطاع واحد بدل التنويع.
    """
    try:
        snap = fetch_stock_snapshot(symbol)
        target_sector = snap.get("sector")
        if not target_sector or target_sector == "غير محدد":
            return {"sector": None, "same_sector_count": 0}

        init_db()
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT DISTINCT symbol FROM recommendations
                WHERE (status IS NULL OR status = '' OR status = 'open')
                AND symbol != ?
            """, (symbol,)).fetchall()

        same_sector_symbols = []
        for row in rows:
            other_snap = fetch_stock_snapshot(row["symbol"])
            if other_snap.get("sector") == target_sector:
                same_sector_symbols.append(row["symbol"])

        return {
            "sector": target_sector,
            "same_sector_count": len(same_sector_symbols),
            "same_sector_symbols": same_sector_symbols,
        }
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------
# 28) إحصائية دقة درجة التطابق الفعلية (Feedback Loop)
# ----------------------------------------------------------------------

def get_confluence_accuracy_stats() -> List[Dict]:
    """
    يجمّع التوصيات المغلقة (تحقق الهدف/ضرب الوقف) حسب فئة درجة التطابق
    المحفوظة وقت التوصية، ويحسب نسبة النجاح الفعلية لكل فئة — عشان
    يشوف المستخدم هل درجة التطابق فعلاً متوافقة مع النتائج الحقيقية،
    بدل افتراض نظري ثابت.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT confluence_score, status FROM recommendations
            WHERE status IN ('hit_target', 'hit_stop')
            AND confluence_score IS NOT NULL
        """).fetchall()

    buckets = {"75-95% (قوي)": [], "55-74% (متوسط)": [], "40-54% (ضعيف)": [], "أقل من 40% (سلبي)": []}
    for row in rows:
        score = row["confluence_score"]
        won = 1 if row["status"] == "hit_target" else 0
        if score >= 75:
            buckets["75-95% (قوي)"].append(won)
        elif score >= 55:
            buckets["55-74% (متوسط)"].append(won)
        elif score >= 40:
            buckets["40-54% (ضعيف)"].append(won)
        else:
            buckets["أقل من 40% (سلبي)"].append(won)

    results = []
    for label, outcomes in buckets.items():
        if outcomes:
            results.append({
                "الفئة": label,
                "عدد الصفقات": len(outcomes),
                "نسبة النجاح الفعلية": f"{round((sum(outcomes) / len(outcomes)) * 100, 1)}%",
            })
    return results


# ----------------------------------------------------------------------
# 29) بيانات الشريط المتحرك (Ticker) لأهم الأسهم والمؤشرات
# ----------------------------------------------------------------------

def get_ticker_data() -> List[Dict]:
    """
    يجلب أسعار مجموعة أسهم ومؤشرات رئيسية دفعة واحدة (طلب واحد بدل عدة
    طلبات) لعرضها بشريط متحرك أعلى الصفحة، بأسلوب المنصات المالية
    العالمية. يستخدم yf.download بدل yf.Ticker المتكرر لتقليل زمن التحميل.
    """
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY", "QQQ", "BTC-USD"]
    try:
        data = yf.download(symbols, period="2d", group_by="ticker", progress=False, threads=True)
        rows = []
        for sym in symbols:
            try:
                closes = data[sym]["Close"].dropna()
                if len(closes) < 1:
                    continue
                current = float(closes.iloc[-1])
                if len(closes) >= 2:
                    prev = float(closes.iloc[-2])
                    change_pct = ((current - prev) / prev) * 100
                else:
                    change_pct = 0.0
                rows.append({"symbol": sym, "price": round(current, 2), "change_pct": round(change_pct, 2)})
            except Exception:
                continue
        return rows
    except Exception:
        return []


# ----------------------------------------------------------------------
# 30) فحص التوافق الشرعي (Shariah Compliance) — استرشادي وليس فتوى
# ----------------------------------------------------------------------

SHARIAH_EXCLUDED_KEYWORDS = [
    "bank", "insurance", "financial services", "casino", "gambling", "lottery",
    "brewer", "distiller", "winer", "wineries", "beverages - wineries", "tobacco",
    "cigarette", "adult", "pornograph", "pork", "resorts & casinos",
    "credit services", "mortgage", "capital markets",
]


def check_shariah_compliance(symbol: str) -> Dict:
    """
    فحص تقريبي استرشادي لمدى توافق السهم مع معايير مالية إسلامية شائعة
    (شبيهة بمنهجية AAOIFI ومؤشر Dow Jones الإسلامي)، عبر مرحلتين:
    1) فحص نشاط الشركة (القطاع/الصناعة) — استبعاد البنوك التقليدية،
       التأمين التقليدي، الكحول، القمار، التبغ، المحتوى الجنسي، لحم الخنزير.
    2) فحص نسب مالية: الدين إلى القيمة السوقية، والنقد وشبه النقد إلى
       القيمة السوقية — يجب أن تكون كل نسبة أقل من 33% تقريباً.

    ⚠️ هذا تصنيف آلي استرشادي بمنهجية شائعة واحدة من عدة منهجيات فقهية
    مختلفة، وليس فتوى شرعية. راجع مصدراً شرعياً موثوقاً أو خدمة متخصصة
    (مثل Zoya أو Islamicly) قبل الاعتماد عليه بقرار استثماري.
    """
    try:
        t = yf.Ticker(symbol)
        info = t.info if hasattr(t, "info") else {}

        sector = (info.get("sector") or "").lower()
        industry = (info.get("industry") or "").lower()
        combined = f"{sector} {industry}"

        for kw in SHARIAH_EXCLUDED_KEYWORDS:
            if kw in combined:
                return {
                    "status": "غير متوافق",
                    "reason": f"نشاط الشركة (القطاع: {info.get('sector', '—')} / الصناعة: {info.get('industry', '—')}) ضمن الأنشطة المستبعدة عادةً",
                    "stage": "نشاط الشركة",
                }

        market_cap = info.get("marketCap")
        total_debt = info.get("totalDebt")
        total_cash = info.get("totalCash")

        if not market_cap or market_cap <= 0:
            return {"status": "غير محدد", "reason": "بيانات القيمة السوقية غير متوفرة لإجراء فحص النسب المالية.", "stage": "بيانات ناقصة"}

        debt_ratio = (total_debt / market_cap) if total_debt else 0
        cash_ratio = (total_cash / market_cap) if total_cash else 0

        if debt_ratio > 0.33:
            return {
                "status": "غير متوافق",
                "reason": f"نسبة الدين إلى القيمة السوقية ({round(debt_ratio * 100, 1)}%) تتجاوز الحد الشائع (33%)",
                "stage": "نسب مالية", "debt_ratio": round(debt_ratio * 100, 1), "cash_ratio": round(cash_ratio * 100, 1),
            }
        if cash_ratio > 0.33:
            return {
                "status": "غير متوافق",
                "reason": f"نسبة النقد وشبه النقد إلى القيمة السوقية ({round(cash_ratio * 100, 1)}%) تتجاوز الحد الشائع (33%)",
                "stage": "نسب مالية", "debt_ratio": round(debt_ratio * 100, 1), "cash_ratio": round(cash_ratio * 100, 1),
            }

        return {
            "status": "متوافق تقريباً",
            "reason": "اجتاز فحص النشاط والنسب المالية الأساسية حسب المنهجية المستخدمة",
            "stage": "اجتاز الفحصين", "debt_ratio": round(debt_ratio * 100, 1), "cash_ratio": round(cash_ratio * 100, 1),
        }
    except Exception as e:
        return {"status": "غير محدد", "reason": f"تعذر إجراء الفحص: {e}", "stage": "خطأ"}
