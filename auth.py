"""
auth.py
-------
نظام تسجيل دخول بسيط ومستقل (بدون مكتبات خارجية معقدة) لدعم أكثر من مستخدم.
كلمات المرور تُحفظ كـ SHA-256 hash — إما داخل auth_config.yaml (حسابات ثابتة
مثل المدير)، أو بقاعدة البيانات finai.db (حسابات سجّلها المستخدمون بأنفسهم
عبر رمز دعوة). لا تُحفظ كلمات المرور كنص صريح أبداً في أي مكان.
"""

import hashlib
import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "auth_config.yaml")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("users", {})


def verify_login(username: str, password: str) -> bool:
    if not username or not password:
        return False

    # 1) تحقق من الحسابات الثابتة (auth_config.yaml)
    users = load_users()
    user = users.get(username)
    if user:
        return user.get("password_hash") == hash_password(password)

    # 2) تحقق من الحسابات ذاتية التسجيل (قاعدة البيانات)
    from utils import get_app_user  # استيراد محلي لتفادي أي تبعية دائرية
    db_user = get_app_user(username)
    if db_user:
        return db_user.get("password_hash") == hash_password(password)

    return False


def get_display_name(username: str) -> str:
    users = load_users()
    if username in users:
        return users[username].get("name", username)

    from utils import get_app_user
    db_user = get_app_user(username)
    if db_user:
        return db_user.get("name", username)

    return username
