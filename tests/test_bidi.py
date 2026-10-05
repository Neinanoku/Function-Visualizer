# -*- coding: utf-8 -*-
"""
Bidi-раскладка ивритского интерфейса (function_visualizer.bidi_visual и
app.he_display / app.make_label):
  • нейтральные знаки (двоеточие, тире, многоточие, пробел, скобки) встают
    по правилам UAX#9 — «:X צעד», «…שמור תמונה», «— f1 / f2»;
  • числа и латиница сохраняют свой порядок, скобки зеркалятся;
  • модель Windows (GDI рисует абзац слева направо): строка, которую отдаёт
    he_display, после GDI-перестановки даёт ровно нужный визуальный порядок —
    проверяется на всех переводах интерфейса.

Запуск:
    MPLBACKEND=Agg /usr/bin/python3.12 -m pytest tests/test_bidi.py -q
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402
if not os.environ.get("DISPLAY"):
    # app.py при импорте переключает matplotlib на TkAgg; без дисплея это
    # невозможно — подменяем переключение, сам Tk здесь не нужен.
    matplotlib.use("Agg")
    matplotlib.use = lambda *a, **k: None

import function_visualizer as fv  # noqa: E402
import app  # noqa: E402

B = fv.bidi_visual


@pytest.mark.parametrize("logical,visual", [
    ("X:  מ / עד", "דע / מ  :X"),                      # двоеточие слева от X, пробелы на месте
    ("צעד X:", ":X דעצ"),
    ("שמור תמונה…", "…הנומת רומש"),                    # многоточие в конце = слева
    ("חיתוך עם X", "X םע ךותיח"),
    ("גודל תווית:", ":תיוות לדוג"),
    ("+ הוסף פונקציה", "היצקנופ ףסוה +"),
    ("מדמה פונקציות — Ariadna", "Ariadna — תויצקנופ המדמ"),
    ("f1 / f2 — אינדקסי (0, 1)", "(1 ,0) יסקדניא — f1 / f2"),   # тире слева от f1/f2, скобки зеркальны
    ("להסיר את כל 12 התוויות?", "?תויוותה 12 לכ תא ריסהל"),      # число не переворачивается
    ("מוכן  ·  נראה x: -5 … 5", "x: -5 … 5 הארנ  ·  ןכומ"),       # латинский блок читается слева направо
    ("משתנה y (למשוואות, למשל x²+y²=9)", "(x²+y²=9 לשמל ,תואוושמל) y הנתשמ"),
    ("שם לא מוכר: foo", "foo :רכומ אל םש"),
    ("hello world 12", "hello world 12"),               # без иврита — без изменений
    ("f1:", ":f1"),                                     # подпись без иврита в ивритском интерфейсе
    ("by Daniel", "by Daniel"),
    ("", ""),
])
def test_bidi_visual_rtl(logical, visual):
    assert B(logical, 'R') == visual


def test_bidi_visual_ltr_base_models_gdi():
    # Абзац слева направо: ивритское слово зеркалится, хвостовое многоточие остаётся справа
    assert B("שמור תמונה…", 'L') == "הנומת רומש…"
    assert B("X: מ / עד", 'L') == "X: דע / מ"
    assert B("abc", 'L') == "abc"


def test_bidi_multiline_and_marks():
    assert B("א\nב", 'R') == "א\nב"
    assert B("\u200fאב\u200e", 'R') == "בא"                 # маркеры направления удаляются


def _all_ui_strings():
    out = list(app.STRINGS_HE.values()) + list(app.ENGINE_STRINGS_HE.values()) + list(app.ERRORS_HE.values())
    out += [v.format(idx=3, n=2, path="C:/x.fvproj", err="boom", log="error.log", what="X מ", val="1,5", key="f0")
            for v in out if "{" in v]
    return [v for v in out if "{" not in v]


def test_windows_display_round_trip(monkeypatch):
    """На Windows he_display отдаёт строку E, для которой GDI(E) == V."""
    monkeypatch.setattr(app, "_TK_DOES_BIDI", True)
    for text in _all_ui_strings():
        for line in text.split("\n"):
            e = app.he_display(line)
            if not app.has_heb(line):
                assert e == line                      # без иврита строка не трогается
                continue
            v = B(line, 'R')
            assert B(e, 'L') == v, (line, e, v)


def test_linux_display_is_visual(monkeypatch):
    monkeypatch.setattr(app, "_TK_DOES_BIDI", False)
    assert app.he_display("צעד X:") == ":X דעצ"
    assert app.he_display("plain") == "plain"


def test_visual_runs_split_fonts(monkeypatch):
    runs = app._visual_runs("X:  מ / עד")
    assert runs == [("דע", True), (" / ", False), ("מ", True), ("  :X", False)]
    monkeypatch.setattr(app, "_TK_DOES_BIDI", True)
    assert app._widget_text("דע", True) == "עד"             # GDI зеркалит сам → логический порядок
    assert app._widget_text("  :X", False) == "  :X"
    assert app._widget_text("  ", False) == "\u00a0\u00a0"
    monkeypatch.setattr(app, "_TK_DOES_BIDI", False)
    assert app._widget_text("דע", True) == "דע"
