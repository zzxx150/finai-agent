"""
شغّل هذا الملف لإنشاء كلمة مرور مشفّرة (hash) لمستخدم جديد أو لتغيير كلمة مرور موجودة:

    python generate_password.py

بعدها انسخ الناتج والصقه في auth_config.yaml في حقل password_hash
تحت اسم المستخدم المطلوب.
"""

import hashlib
import getpass

if __name__ == "__main__":
    pwd = getpass.getpass("اكتب كلمة المرور الجديدة (لن تظهر أثناء الكتابة): ")
    pwd_confirm = getpass.getpass("أكّد كلمة المرور مرة ثانية: ")

    if pwd != pwd_confirm:
        print("\n❌ كلمتا المرور غير متطابقتين. حاول مرة أخرى.")
    elif len(pwd) < 6:
        print("\n⚠️ يُفضّل استخدام كلمة مرور 6 أحرف على الأقل.")
    else:
        hashed = hashlib.sha256(pwd.encode("utf-8")).hexdigest()
        print("\n✅ الكلمة المشفّرة (انسخها بالكامل والصقها في auth_config.yaml):\n")
        print(hashed)
