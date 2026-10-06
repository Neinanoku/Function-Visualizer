# -*- coding: utf-8 -*-
"""
Полнота переводов интерфейса (иврит и русский):
  • каждая строка, которую app.py передаёт в T()/make_label/make_check/card/
    small_button/lbl, и каждая подсказка клавиатуры есть в STRINGS_HE и STRINGS_RU;
  • каждая строка tr("...") движка есть в ENGINE_STRINGS_*;
  • таблицы ошибок формулы совпадают по ключам, шаблонные ошибки переводятся;
  • T()/tr_err() на русском не возвращают английский текст.

Запуск:
    MPLBACKEND=Agg /usr/bin/python3.12 -m pytest tests/test_i18n.py -q
"""
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
    matplotlib.use = lambda *a, **k: None

import app  # noqa: E402

_APP_SRC = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
_ENGINE_SRC = open(os.path.join(ROOT, "function_visualizer.py"), encoding="utf-8").read()
_STR = r'"((?:[^"\\]|\\.)+)"'
# Несколько литералов подряд ("a" "b" через перенос строки) — один ключ
_STRS = r'((?:"(?:[^"\\]|\\.)+"\s*)+)'
_CALLS = re.compile(r'\b(?:T|make_label|make_check|card|small_button|lbl)\(\s*(?:[A-Za-z_.]+\s*,\s*)?' + _STRS)
_TR = re.compile(r"\btr\(\s*(?:" + _STR + r"|'((?:[^'\\]|\\.)+)')")


def _app_keys():
    keys = set()
    import ast
    for m in _CALLS.finditer(_APP_SRC):
        keys.add(ast.literal_eval("(" + m.group(1) + ")"))   # склеивает соседние литералы
    # Подсказки клавиатуры: (подпись, токен, подсказка)
    for rows in (app.Keypad.LEFT_BASIC, app.Keypad.LEFT_EXTRA, app.Keypad.RIGHT):
        for row in rows:
            for item in row:
                if item and len(item) >= 3 and item[2]:
                    keys.add(item[2])
    return keys


def _engine_keys():
    keys = set()
    for m in _TR.finditer(_ENGINE_SRC):
        keys.add(m.group(1) or m.group(2))
    return keys


@pytest.mark.parametrize("lang,table", [("he", app.STRINGS_HE), ("ru", app.STRINGS_RU)])
def test_ui_strings_complete(lang, table):
    keys = _app_keys()
    assert len(keys) > 60
    # Ключи без букв (числа, "f0", «×») и чисто-символьные переводить не нужно
    missing = sorted(k for k in keys if re.search(r"[A-Za-z]{2,}", k) and k not in table
                     and k not in ("Function Visualizer", "by Daniel"))
    assert missing == [], f"{lang}: нет перевода для {missing}"


@pytest.mark.parametrize("lang,table", [("he", app.ENGINE_STRINGS_HE), ("ru", app.ENGINE_STRINGS_RU)])
def test_engine_strings_complete(lang, table):
    keys = _engine_keys()
    assert len(keys) >= 10
    missing = sorted(k for k in keys if k not in table)
    assert missing == [], f"{lang}: нет перевода для {missing}"


def test_tables_have_same_keys():
    assert set(app.STRINGS_RU) == set(app.STRINGS_HE)
    assert set(app.ENGINE_STRINGS_RU) == set(app.ENGINE_STRINGS_HE)
    assert set(app.ERRORS_RU) == set(app.ERRORS_HE)
    assert set(app._ERR_PARTS_RU) == set(app._ERR_PARTS_HE) == set(app._ERR_EMPTY_RU)


def test_russian_lookup(monkeypatch):
    monkeypatch.setattr(app, "LANG", "ru")
    assert app.T("Ready") == "Готово"
    assert app.T("Region #{n}: check the point", n=2) == "Область №2: проверьте точку"
    assert app.T("no such key") == "no such key"
    assert app.tr_err("empty denominator") == "пустой знаменатель"
    assert app.tr_err("trailing operator in numerator") == "числитель: оператор в конце"
    assert app.tr_err("operator '+' without left operand") == "оператор '+' без левого операнда"
    assert app.tr_err("unknown name: foo") == "неизвестное имя: foo"
    assert app.tr_err("Syntax error") == "Синтаксическая ошибка"
    assert app.tr_err("weird message") == "weird message"
    assert app.has_local("Шаг X") and not app.has_local("Step X")
    assert app.ui_font("Шаг X", 9)[0] == app.RU_FONT and app.ui_font("f0", 9)[0] == app.UI_FONT
    assert app.he_display("Шаг X:") == "Шаг X:"                 # русский — без bidi
    monkeypatch.setattr(app, "LANG", "he")
    assert app.tr_err("empty denominator") == "מכנה ריק"
    monkeypatch.setattr(app, "LANG", "en")
    assert app.tr_err("empty denominator") == "empty denominator"
    assert app.T("Ready") == "Ready"


def test_russian_strings_are_russian():
    for k, v in app.STRINGS_RU.items():
        if re.search(r"[A-Za-z]{3,}", k) and k not in ("Function Visualizer project", "45deg ////", "135deg \\\\"):
            assert re.search(r"[А-Яа-яЁё]", v), (k, v)
