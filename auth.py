"""
auth.py
-------
نظام تسجيل دخول بسيط ومستقل (بدون مكتبات خارجية معقدة) لدعم أكثر من مستخدم.
كلمات المرور تُحفظ كـ SHA-256 hash داخل auth_config.yaml — لا تُحفظ كنص صريح أبداً.
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
    users = load_users()
    user = users.get(username)
    if not user:
        return False
    return user.get("password_hash") == hash_password(password)


def get_display_name(username: str) -> str:
    users = load_users()
    return users.get(username, {}).get("name", username)
