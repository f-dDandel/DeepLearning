import sys
import io
import locale
import re

def safe_print(*args, **kwargs):
    """Обертка для print с обработкой Unicode"""
    text = ' '.join(str(arg) for arg in args)
    encoded = text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
    print(encoded, **kwargs)

def clean_text(text):
    """Фильтрует не-ASCII символы из текста"""
    return text.encode('ascii', errors='ignore').decode('ascii')

def setup_encoding():
    # Настройка кодировки
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    # Дополнительная страховка для Windows
    if sys.platform == "win32":
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')