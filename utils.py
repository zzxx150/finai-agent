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
            current_price = float(hist["Close"].iloc[-1])
            if len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[-2])
                change_pct = ((current_price - prev_close) / prev_close) * 100

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

أرجع حصراً كائن JSON بهذا الشكل بالضبط:
{
  "summary": "زبدة الخبر في جملتين قصيرتين بالعربية",
  "sentiment": "إيجابي جداً | إيجابي | محايد | سلبي | سلبي جداً",
  "impact_level": "عالي | متوسط | منخفض",
  "confidence": رقم من 0 إلى 100 يمثل ثقتك بهذا التحليل,
  "is_likely_official": true أو false (هل يبدو الخبر بياناً رسمياً/مصدراً موثوقاً أم إشاعة؟),
  "trade_plan": {
    "action": "دخول شراء | دخول بيع (شورت) | انتظار | تجنب",
    "entry_note": "ملاحظة نصية موجزة عن منطقة الدخول المناسبة (وليس سعراً دقيقاً لأنك لا تملك شارت لحظي)",
    "stop_loss_note": "ملاحظة موجزة عن منطق وقف الخسارة",
    "target_note": "ملاحظة موجزة عن الهدف المحتمل",
    "estimated_duration": "تقدير تقريبي جداً لمدة الصفقة حتى الوصول للهدف أو الخروج، اختر واحداً: 'قصيرة (خلال نفس يوم التداول)' أو 'قصيرة-متوسطة (1-3 أيام)' أو 'متوسطة (أسبوع تقريباً)' أو 'طويلة (أسابيع أو أكثر)'. هذا تقدير استرشادي فقط وليس وعداً بزمن دقيق",
    "exit_condition": "متى يجب الخروج من الصفقة (شرط واضح)"
  },
  "risk_warning": "تحذير مخاطرة قصير"
}

تنبيه: هذا تحليل استرشادي وليس توصية استثمارية مضمونة. كن واقعياً ولا تبالغ.
تنبيه إضافي: تقدير مدة الصفقة (estimated_duration) هو تخمين تقريبي جداً بناءً على طبيعة الخبر وسوابق مشابهة، وليس تنبؤاً دقيقاً بالوقت — الأسواق قد تتحرك أسرع أو أبطأ من أي تقدير بكثير."""


def analyze_news_with_ai(
    openai_api_key: str,
    headline: str,
    summary: str,
    symbol: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> Dict:
    """
    يرسل الخبر إلى OpenAI ويستقبل تحليلاً منظماً بصيغة JSON.
    """
    if not openai_api_key or OpenAI is None:
        return {"error": "مفتاح OpenAI غير متوفر أو المكتبة غير مثبتة."}

    client = OpenAI(api_key=openai_api_key)

    user_content = f"""
السهم: {symbol or 'غير محدد'}
العنوان: {headline}
تفاصيل إضافية: {summary or 'لا يوجد'}
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
    "acquisition", "acquire", "merger", "buyout", "fda approval", "fda",
    "earnings beat", "earnings miss", "guidance", "bankruptcy", "lawsuit",
    "sec investigation", "recall", "partnership", "contract award",
    "stock split", "dividend increase", "ceo resigns", "ceo steps down",
    "data breach", "clinical trial", "patent",
    "استحواذ", "اندماج", "إفلاس", "دعوى قضائية", "تحقيق", "أرباح",
    "توقعات", "شراكة", "عقد حكومي", "موافقة", "استقالة",
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
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


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
             action, entry_note, stop_loss_note, target_note, estimated_duration,
             exit_condition, score, created_at, created_by)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            symbol,
            headline,
            analysis.get("sentiment"),
            analysis.get("impact_level"),
            analysis.get("confidence"),
            1 if analysis.get("is_likely_official") else 0,
            plan.get("action"),
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
