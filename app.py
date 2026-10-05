"""
Визуализатор функций — GUI (live-режим)
Зависимости: pip install matplotlib numpy sympy scipy pillow
Сборка .exe: build.bat  (PyInstaller, см. generate_spec.py)

Архитектура:
  app.py                 — окно Tk: слева панель настроек, справа ЖИВОЙ график
  math_editor.py         — 2-D редактор формул (дроби, степени, корни) + модель
  function_visualizer.py — движок построения/анализа графика (matplotlib + sympy)

Поток данных: любое изменение в панели → schedule_redraw() (debounce 250 мс)
→ _redraw(): собираем настройки → заполняем глобалы движка → fv.plot_function(fig)
рисует в встроенную фигуру. Тяжёлые символьные вычисления (sympy: пределы,
особые точки, точные корни) движок откладывает (SYMBOLIC_MODE='cached'),
а мы считаем их в фоновом потоке и перерисовываем, когда они готовы.
"""

import sys
import os
import json
import re
import math
import time
import queue
import threading
import collections
import traceback
import tkinter as tk
from tkinter import messagebox, filedialog, colorchooser

# ── DPI (Windows) ────────────────────────────────────────────
# Без этого при масштабе монитора 125–200 % Windows растягивает окно как
# картинку, и весь текст выглядит пикселизированным. Объявляем процесс
# per-monitor DPI-aware ДО создания первого окна (и до импорта Tk-бэкенда
# matplotlib); все пиксельные размеры интерфейса масштабируются через
# UI_SCALE (см. App.__init__), шрифты в пунктах Tk масштабирует сам.
if sys.platform == "win32":
    try:
        import ctypes
        try:
            _f = ctypes.windll.user32.SetProcessDpiAwarenessContext
            _f.argtypes = [ctypes.c_void_p]
            if not _f(ctypes.c_void_p(-4)):          # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
                raise OSError("SetProcessDpiAwarenessContext failed")
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PROCESS_PER_MONITOR_DPI_AWARE
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Фоновый поток sympy держит GIL подолгу; с интервалом переключения по
# умолчанию (5 мс) тысячи мелких numpy/scipy-вызовов главного потока ждут
# его каждый раз, и перерисовка во время «Refining labels…» замедляется в
# разы. Короткий интервал почти убирает этот эффект.
try:
    sys.setswitchinterval(0.0005)
except Exception:
    pass

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Шрифты, поставляемые с программой (папка fonts/) ─────────
def _bundled_font_dir():
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "fonts")


def _register_bundled_fonts():
    """
    TeX Gyre Schola (свободный клон Century Schoolbook) лежит в fonts/ и не
    требует установки: для matplotlib файлы добавляются в его менеджер шрифтов,
    для Tk на Windows — регистрируются как приватные шрифты процесса через
    GDI (AddFontResourceExW, FR_PRIVATE). На Linux/macOS Tk берёт шрифт из
    системы, если он установлен (пакет fonts-texgyre), иначе — запасной.
    """
    import glob
    paths = sorted(glob.glob(os.path.join(_bundled_font_dir(), "*.otf")) +
                   glob.glob(os.path.join(_bundled_font_dir(), "*.ttf")))
    if not paths:
        return []
    try:
        from matplotlib import font_manager
        for p in paths:
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            FR_PRIVATE = 0x10
            for p in paths:
                ctypes.windll.gdi32.AddFontResourceExW(p, FR_PRIVATE, 0)
        except Exception:
            pass
    return paths


BUNDLED_FONTS = _register_bundled_fonts()

# ── Импортируем движок и редактор ────────────────────────────
# (в exe они лежат как обычные модули PyInstaller — см. generate_spec.py)
import function_visualizer as fv
from math_editor import MathEditor, MathModel, IncompleteExpression
fv.apply_font_preset()          # пресет пересчитывается уже с учётом fonts/


# ═════════════════════════════════════════════════════════════
#  ЦВЕТА И СТИЛЬ — Light Theme
# ═════════════════════════════════════════════════════════════

APP_BG    = "#f4f6f9"
PANEL_BG  = "#e8ecf2"
CARD_BG   = "#ffffff"
ACCENT    = "#e94560"
TEXT      = "#1a1a2e"
SUBTEXT   = "#6b7280"
ENTRY_BG  = "#ffffff"
BORDER    = "#d1d5db"
BTN_SAVE  = "#e94560"
BTN_ADD   = "#2ab850"
BTN_DEL   = "#84878c"
BTN_BLUE  = "#2a7ae0"
KEY_BG    = "#ffffff"
KEY_BG2   = "#eef1f5"     # цифры
KEY_FG    = "#1a1a2e"
ERR_COLOR = "#d9363e"

FUNC_COLORS = fv.CURVE_COLORS  # берём из движка

LEFT_PANEL_WIDTH = 500        # при 96 dpi; реальные пиксели — через _px()
UI_SCALE = 1.0                # коэффициент DPI (1.5 при масштабе 150 %), задаётся в App.__init__


def _px(v):
    """Пиксельный размер, заданный для 96 dpi → реальные пиксели экрана."""
    return int(round(v * UI_SCALE))
REDRAW_DELAY_MS  = 250        # после правок в панели
ZOOM_DELAY_MS    = 120        # после колеса мыши (предпросмотр уже показан)
SYMBOLIC_TIMEOUT_S = 20.0     # сторож: пачка символьных заданий дольше этого — считается зависшей
MAX_GRID_LINES   = 2000       # span / step не больше этого (иначе сетка «съедает» рисунок)

# Шрифт интерфейса (панель, клавиатура, подсказки): TeX Gyre Schola из fonts/
# (под Windows Tk видит его как «TeXGyreSchola»); если недоступен — замены.
# Имя уточняется при старте (_resolve_ui_font), когда Tk уже может
# перечислить семейства шрифтов.
UI_FONT = "TeX Gyre Schola"
UI_FONT_FALLBACKS = ["TeX Gyre Schola", "TeXGyreSchola", "Century Schoolbook",
                     "Times New Roman", "Liberation Serif", "Cambria", "DejaVu Serif"]


def _resolve_ui_font(root):
    global UI_FONT, HE_FONT, RU_FONT
    try:
        import tkinter.font as tkfont
        fams = set(tkfont.families(root))
        UI_FONT = next((f for f in UI_FONT_FALLBACKS if f in fams), UI_FONT_FALLBACKS[-1])
        HE_FONT = next((f for f in HE_FONT_FALLBACKS if f in fams), HE_FONT_FALLBACKS[-1])
        RU_FONT = next((f for f in RU_FONT_FALLBACKS if f in fams), RU_FONT_FALLBACKS[-1])
    except Exception:
        pass
    return UI_FONT


# ═════════════════════════════════════════════════════════════
#  ЯЗЫК ИНТЕРФЕЙСА / RTL
# ═════════════════════════════════════════════════════════════
# LANG = "en" — английский интерфейс (панель слева, график справа);
# LANG = "he" — иврит (app_he.py): панель справа, график слева, ивритский
# текст шрифтом David (HE_FONT), цифры и латиница — прежним UI_FONT;
# LANG = "ru" — русский (app_ru.py): раскладка как в английском, русский
# текст шрифтом Century Schoolbook (RU_FONT; в TeX Gyre Schola кириллицы
# нет), цифры, латиница и формулы — прежним UI_FONT.
# Визуальный порядок символов считается по правилам bidi (fv.bidi_visual):
# двоеточия, тире, многоточия, пробелы и скобки встают на «ивритские» места
# («:X צעד», «…שמור תמונה», «— f1 / f2»). Смешанные подписи разбиваются на
# фрагменты визуальной строки: ивритские слова — David, всё остальное
# (цифры, латиница, знаки) — UI_FONT. Tk на Windows переставляет ивритские
# буквы сам (GDI, абзац слева направо) — ему отдаём логический порядок слов
# и строку, которую он нарисует в нужном виде; Tk на Linux/macOS не
# переставляет ничего — отдаём уже визуальный порядок.
LANG = "en"
HE_FONT = "David"
HE_FONT_FALLBACKS = ["David", "David CLM", "Frank Ruehl CLM", "Noto Serif Hebrew",
                     "FreeSerif", "DejaVu Sans"]
RU_FONT = "Century Schoolbook"
RU_FONT_FALLBACKS = ["Century Schoolbook", "Times New Roman", "Cambria", "Georgia",
                     "Liberation Serif", "DejaVu Serif"]
_CYR_RE = re.compile(r'[\u0400-\u04FF]')
# Русские слова с пробелами между ними — один фрагмент шрифта RU_FONT
_CYR_WORDS_RE = re.compile(r'[\u0400-\u04FF]+(?: +[\u0400-\u04FF]+)*')
_HEB_RUN_RE = re.compile(r'[\u0590-\u05FF][\u0590-\u05FF\s]*[\u0590-\u05FF]|[\u0590-\u05FF]')


def RTL():
    return LANG == "he"


def has_heb(text):
    return bool(text) and _HEB_RUN_RE.search(str(text)) is not None


def has_cyr(text):
    return bool(text) and _CYR_RE.search(str(text)) is not None


def local_font():
    """Шрифт «местного» письма текущего языка: David (иврит), Century Schoolbook (русский)."""
    return HE_FONT if LANG == "he" else RU_FONT


def has_local(text):
    """Есть ли в строке буквы «местного» письма текущего языка."""
    if LANG == "he":
        return has_heb(text)
    if LANG == "ru":
        return has_cyr(text)
    return False


STRINGS_HE = {
    # окно / карточки
    "Function Visualizer — Ariadna": "מדמה פונקציות — Ariadna",
    "Functions": "פונקציות", "+ Add function": "+ הוסף פונקציה",
    "View Window": "חלון תצוגה", "X:  from / to": "X:  מ / עד", "Y:  from / to": "Y:  מ / עד",
    "Tip: mouse wheel over the graph zooms, drag pans.": "טיפ: גלגלת העכבר מעל הגרף — זום, גרירה — הזזה.",
    "Grid": "סריג", "Show grid": "הצג סריג", "  Step X:": "צעד X:", "Step Y:": "צעד Y:",
    "Display on Graph": "הצגה על הגרף",
    "Asymptotes": "אסימפטוטות", "Holes": "חורים", "Extrema": "קיצון",
    "X-intercepts": "חיתוך עם X", "Y-intercepts": "חיתוך עם Y", "Intersections": "חיתוך בין פונקציות",
    "Show values": "הצגת ערכים", "Hide X labels": "הסתר תוויות X", "Hide Y labels": "הסתר תוויות Y",
    "Label size:": "גודל תווית:",
    "Area Fill": "צביעת שטח", "+ Add fill": "+ הוסף צביעה",
    "f1 / f2 — function indices (0, 1, …) or 'x' for the X axis":
        "f1 / f2 — אינדקסי פונקציות (0, 1, …) או 'x' עבור ציר X",
    "Graph Labels": "תוויות על הגרף",
    "Double-click empty space on the graph to add a label. Drag to move, scroll to rotate, "
    "right-click for options. Point labels can be dragged too.":
        "לחיצה כפולה על מקום ריק בגרף מוסיפה תווית. גרירה — הזזה, גלגלת — סיבוב, "
        "לחיצה ימנית — אפשרויות. גם תוויות נקודות ניתנות לגרירה.",
    "Clear all labels": "נקה את כל התוויות", "Reset point labels": "אפס תוויות נקודות",
    "  Save image…": "שמור תמונה…", "Reset view": "איפוס תצוגה",
    "Save project": "שמור פרויקט", "Open project": "פתח פרויקט",
    # строки функции / заливки
    "w:": "עובי:", "domain:": "תחום:", "Solid": "רציף", "Dashed": "מקווקו", "Dotted": "נקודות",
    "Pick color for f{idx}": "בחר צבע עבור f{idx}",
    "from:": "מ:", "to:": "עד:", "style:": "סגנון:", "borders": "קצוות", "density:": "צפיפות:",
    "45deg ////": "45° ////", "135deg \\\\": "135° \\\\", "Dots ....": "נקודות ....",
    # диалоги
    "Cannot delete": "לא ניתן למחוק", "At least one function is required.": "נדרשת לפחות פונקציה אחת.",
    "Remove all {n} label(s) added on the graph?": "להסיר את כל {n} התוויות שנוספו על הגרף?",
    "Save graph image": "שמור תמונת גרף", "PNG image": "תמונת PNG", "SVG vector": "וקטור SVG",
    "All files": "כל הקבצים", "Save error": "שגיאת שמירה", "Saved: {path}": "נשמר: {path}",
    "Function Visualizer project": "פרויקט Function Visualizer",
    "Project saved: {path}": "הפרויקט נשמר: {path}", "Open error": "שגיאת פתיחה",
    "Cannot open project:\n{err}": "לא ניתן לפתוח את הפרויקט:\n{err}",
    "Project loaded: {path}": "הפרויקט נטען: {path}",
    # статус
    "Ready": "מוכן", "Refining labels (symbolic analysis)…": "מעדן תוויות (ניתוח סימבולי)…",
    "Some functions are incomplete or invalid — hover the red field":
        "חלק מהפונקציות לא שלמות או שגויות — העבר את העכבר מעל השדה האדום",
    "Symbolic analysis timed out — some labels stay numeric":
        "הניתוח הסימבולי חרג מהזמן — חלק מהתוויות יישארו מספריות",
    "Plot error — previous graph restored (details: {log})":
        "שגיאת שרטוט — הגרף הקודם שוחזר (פרטים: {log})",
    "Zoom-out limit for the current grid step — increase Step X / Step Y":
        "הגבלת הרחקה עבור צעד הסריג הנוכחי — הגדל את צעד X / Y",
    "visible x": "נראה x", "visible y": "נראה y",
    # проверка настроек
    "{what}: cannot read '{val}'": "{what}: לא ניתן לקרוא '{val}'",
    "X from": "X מ", "X to": "X עד", "Y from": "Y מ", "Y to": "Y עד", "Step X": "צעד X", "Step Y": "צעד Y",
    "View window must be finite": "חלון התצוגה חייב להיות סופי",
    "View window: left < right and bottom < top required": "חלון תצוגה: נדרש שמאל < ימין ותחתון < עליון",
    "Grid step must be a positive finite number": "צעד הסריג חייב להיות מספר חיובי סופי",
    "Grid step is too small for this view window": "צעד הסריג קטן מדי עבור חלון תצוגה זה",
    "Fill #{n}: check parameters": "צביעה #{n}: בדוק פרמטרים",
    # подсказки клавиатуры
    "variable x": "משתנה x", "variable y (for equations, e.g. x²+y²=9)": "משתנה y (למשוואות, למשל x²+y²=9)",
    "square": "ריבוע", "power": "חזקה", "open parenthesis": "פתח סוגריים",
    "leave the parentheses": "צא מהסוגריים", "square root": "שורש ריבועי", "n-th root": "שורש n-י",
    "absolute value": "ערך מוחלט", "the number π": "המספר π", "the number e": "המספר e", "fraction": "שבר",
    "sine": "סינוס", "cosine": "קוסינוס", "tangent": "טנגנס", "cotangent": "קוטנגנס",
    "natural logarithm": "לוגריתם טבעי", "common (base-10) logarithm": "לוגריתם עשרוני",
    "logarithm with base a": "לוגריתם בבסיס a", "exponential": "אקספוננט",
    "more functions: arcsin, arccos, sinh…": "עוד פונקציות: arcsin, arccos, sinh…",
    "back to sin, cos, ln…": "חזרה ל-sin, cos, ln…",
    "cursor up (numerator)": "סמן למעלה (מונה)", "cursor down (denominator)": "סמן למטה (מכנה)",
    "inverse sine": "סינוס הפוך", "inverse cosine": "קוסינוס הפוך", "inverse tangent": "טנגנס הפוך",
    "secant": "סקנס", "cosecant": "קוסקנס", "hyperbolic sine": "סינוס היפרבולי",
    "hyperbolic cosine": "קוסינוס היפרבולי", "hyperbolic tangent": "טנגנס היפרבולי",
    "multiply": "כפל", "minus": "מינוס", "decimal point": "נקודה עשרונית",
    "equals (equation / x = c)": "שווה (משוואה / x = c)", "plus": "פלוס",
    "cursor left": "סמן שמאלה", "cursor right": "סמן ימינה", "delete": "מחק", "clear the field": "נקה את השדה",
}

# Подписи движка (меню и диалоги свободных подписей на графике)
ENGINE_STRINGS_HE = {
    "New label": "תווית חדשה", "Edit label": "עריכת תווית", "Label text:": "טקסט התווית:",
    "Label color": "צבע התווית", "Edit text": "ערוך טקסט", "Text color...": "צבע טקסט...",
    "Text size": "גודל טקסט", "Reset rotation": "אפס סיבוב", "Delete": "מחק",
    "Add label here": "הוסף תווית כאן", '"{key}" label color...': 'צבע תווית "{key}"...',
    "Label size": "גודל תווית", "Reset position": "אפס מיקום",
}

# Сообщения об ошибках формулы (редактор + движок)
ERRORS_HE = {
    "Syntax error": "שגיאת תחביר", "Invalid expression": "ביטוי לא תקין",
    "Division by zero": "חלוקה באפס",
    "Undefined value (division by zero or log base 1?)": "ערך לא מוגדר (חלוקה באפס או לוג בבסיס 1?)",
    "equation is an identity — every point satisfies it": "המשוואה היא זהות — כל נקודה מקיימת אותה",
    "equation has no solutions (contradiction)": "למשוואה אין פתרונות (סתירה)",
    "empty expression": "ביטוי ריק", "exponent without base": "מעריך ללא בסיס",
    "subscript is only allowed as a log base": "אינדקס תחתון מותר רק כבסיס לוגריתם",
}
_ERR_PARTS_HE = {
    "exponent": "מעריך", "denominator": "מכנה", "numerator": "מונה", "expression": "ביטוי",
    "log base": "בסיס הלוגריתם", "parentheses": "סוגריים", "root index": "מעריך השורש",
    "root": "שורש", "absolute value": "ערך מוחלט",
}
_ERR_PATTERNS_HE = [
    (re.compile(r"^empty (.+)$"), lambda m: f"{_ERR_PARTS_HE.get(m.group(1), m.group(1))} ריק"),
    (re.compile(r"^trailing operator in (.+)$"), lambda m: f"אופרטור בסוף {_ERR_PARTS_HE.get(m.group(1), m.group(1))}"),
    (re.compile(r"^lone '\.' in (.+)$"), lambda m: f"נקודה בודדת ב{_ERR_PARTS_HE.get(m.group(1), m.group(1))}"),
    (re.compile(r"^(.+) without argument$"), lambda m: f"{m.group(1)} ללא ארגומנט"),
    (re.compile(r"^bad number '(.+)'$"), lambda m: f"מספר שגוי '{m.group(1)}'"),
    (re.compile(r"^operator '(.+)' without left operand$"), lambda m: f"אופרטור '{m.group(1)}' ללא אופרנד שמאלי"),
    (re.compile(r"^unknown name: (.+)$"), lambda m: f"שם לא מוכר: {m.group(1)}"),
    (re.compile(r"^cannot parse the constant in (.+)$"), lambda m: f"לא ניתן לפענח את הקבוע ב-{m.group(1)}"),
]

STRINGS_RU = {
    # окно / карточки
    "Function Visualizer — Ariadna": "Визуализатор функций — Ariadna",
    "Functions": "Функции", "+ Add function": "+ Добавить функцию",
    "View Window": "Окно просмотра", "X:  from / to": "X:  от / до", "Y:  from / to": "Y:  от / до",
    "Tip: mouse wheel over the graph zooms, drag pans.":
        "Подсказка: колесо мыши над графиком — масштаб, перетаскивание — сдвиг.",
    "Grid": "Сетка", "Show grid": "Показывать сетку", "  Step X:": "  Шаг X:", "Step Y:": "Шаг Y:",
    "Display on Graph": "Показывать на графике",
    "Asymptotes": "Асимптоты", "Holes": "Выколотые точки", "Extrema": "Экстремумы",
    "X-intercepts": "Пересечения с осью X", "Y-intercepts": "Пересечения с осью Y",
    "Intersections": "Пересечения функций",
    "Show values": "Показывать значения", "Hide X labels": "Скрыть подписи X", "Hide Y labels": "Скрыть подписи Y",
    "Label size:": "Размер подписей:",
    "Area Fill": "Заливка области", "+ Add fill": "+ Добавить заливку",
    "f1 / f2 — function indices (0, 1, …) or 'x' for the X axis":
        "f1 / f2 — номера функций (0, 1, …) или 'x' для оси X",
    "Graph Labels": "Подписи на графике",
    "Double-click empty space on the graph to add a label. Drag to move, scroll to rotate, "
    "right-click for options. Point labels can be dragged too.":
        "Двойной щелчок по пустому месту графика добавляет подпись. Перетаскивание — перемещение, "
        "колесо — поворот, правая кнопка — меню. Подписи точек тоже можно перетаскивать.",
    "Clear all labels": "Удалить все подписи", "Reset point labels": "Сбросить подписи точек",
    "  Save image…": "  Сохранить картинку…", "Reset view": "Сбросить вид",
    "Save project": "Сохранить проект", "Open project": "Открыть проект",
    # строки функции / заливки
    "w:": "толщ.:", "domain:": "ОДЗ:", "Solid": "Сплошная", "Dashed": "Штриховая", "Dotted": "Пунктир",
    "Pick color for f{idx}": "Цвет для f{idx}",
    "from:": "от:", "to:": "до:", "style:": "стиль:", "borders": "границы", "density:": "плотность:",
    "45deg ////": "45° ////", "135deg \\\\": "135° \\\\", "Dots ....": "Точки ....",
    # диалоги
    "Cannot delete": "Нельзя удалить", "At least one function is required.": "Нужна хотя бы одна функция.",
    "Remove all {n} label(s) added on the graph?": "Удалить все добавленные подписи ({n})?",
    "Save graph image": "Сохранить картинку графика", "PNG image": "Картинка PNG", "SVG vector": "Вектор SVG",
    "All files": "Все файлы", "Save error": "Ошибка сохранения", "Saved: {path}": "Сохранено: {path}",
    "Function Visualizer project": "Проект Function Visualizer",
    "Project saved: {path}": "Проект сохранён: {path}", "Open error": "Ошибка открытия",
    "Cannot open project:\n{err}": "Не удалось открыть проект:\n{err}",
    "Project loaded: {path}": "Проект загружен: {path}",
    # статус
    "Ready": "Готово", "Refining labels (symbolic analysis)…": "Уточняю подписи (символьный анализ)…",
    "Some functions are incomplete or invalid — hover the red field":
        "Некоторые функции не дописаны или ошибочны — наведите мышь на красное поле",
    "Symbolic analysis timed out — some labels stay numeric":
        "Символьный анализ не уложился в отведённое время — часть подписей останется числовой",
    "Plot error — previous graph restored (details: {log})":
        "Ошибка построения — восстановлен предыдущий график (подробности: {log})",
    "Zoom-out limit for the current grid step — increase Step X / Step Y":
        "Предел отдаления для текущего шага сетки — увеличьте шаг X / Y",
    "visible x": "видно x", "visible y": "видно y",
    # проверка настроек
    "{what}: cannot read '{val}'": "{what}: не удаётся прочитать '{val}'",
    "X from": "X от", "X to": "X до", "Y from": "Y от", "Y to": "Y до", "Step X": "Шаг X", "Step Y": "Шаг Y",
    "View window must be finite": "Окно просмотра должно быть конечным",
    "View window: left < right and bottom < top required":
        "Окно просмотра: нужно левая < правая и нижняя < верхняя",
    "Grid step must be a positive finite number": "Шаг сетки должен быть положительным конечным числом",
    "Grid step is too small for this view window": "Шаг сетки слишком мал для этого окна просмотра",
    "Fill #{n}: check parameters": "Заливка №{n}: проверьте параметры",
    # подсказки клавиатуры
    "variable x": "переменная x", "variable y (for equations, e.g. x²+y²=9)": "переменная y (для уравнений, напр. x²+y²=9)",
    "square": "квадрат", "power": "степень", "open parenthesis": "открыть скобку",
    "leave the parentheses": "выйти из скобок", "square root": "квадратный корень", "n-th root": "корень n-й степени",
    "absolute value": "модуль", "the number π": "число π", "the number e": "число e", "fraction": "дробь",
    "sine": "синус", "cosine": "косинус", "tangent": "тангенс", "cotangent": "котангенс",
    "natural logarithm": "натуральный логарифм", "common (base-10) logarithm": "десятичный логарифм",
    "logarithm with base a": "логарифм по основанию a", "exponential": "экспонента",
    "more functions: arcsin, arccos, sinh…": "ещё функции: arcsin, arccos, sinh…",
    "back to sin, cos, ln…": "назад к sin, cos, ln…",
    "cursor up (numerator)": "курсор вверх (числитель)", "cursor down (denominator)": "курсор вниз (знаменатель)",
    "inverse sine": "арксинус", "inverse cosine": "арккосинус", "inverse tangent": "арктангенс",
    "secant": "секанс", "cosecant": "косеканс", "hyperbolic sine": "гиперболический синус",
    "hyperbolic cosine": "гиперболический косинус", "hyperbolic tangent": "гиперболический тангенс",
    "multiply": "умножить", "minus": "минус", "decimal point": "десятичная точка",
    "equals (equation / x = c)": "равно (уравнение / x = c)", "plus": "плюс",
    "cursor left": "курсор влево", "cursor right": "курсор вправо", "delete": "удалить", "clear the field": "очистить поле",
}

ENGINE_STRINGS_RU = {
    "New label": "Новая подпись", "Edit label": "Изменить подпись", "Label text:": "Текст подписи:",
    "Label color": "Цвет подписи", "Edit text": "Изменить текст", "Text color...": "Цвет текста...",
    "Text size": "Размер текста", "Reset rotation": "Сбросить поворот", "Delete": "Удалить",
    "Add label here": "Добавить подпись здесь", '"{key}" label color...': 'Цвет подписи "{key}"...',
    "Label size": "Размер подписи", "Reset position": "Сбросить положение",
}

ERRORS_RU = {
    "Syntax error": "Синтаксическая ошибка", "Invalid expression": "Некорректное выражение",
    "Division by zero": "Деление на ноль",
    "Undefined value (division by zero or log base 1?)":
        "Неопределённое значение (деление на ноль или логарифм по основанию 1?)",
    "equation is an identity — every point satisfies it": "уравнение — тождество: ему удовлетворяет каждая точка",
    "equation has no solutions (contradiction)": "у уравнения нет решений (противоречие)",
    "empty expression": "пустое выражение", "exponent without base": "показатель степени без основания",
    "subscript is only allowed as a log base": "нижний индекс допустим только как основание логарифма",
}
_ERR_PARTS_RU = {
    "exponent": "показатель степени", "denominator": "знаменатель", "numerator": "числитель",
    "expression": "выражение", "log base": "основание логарифма", "parentheses": "скобки",
    "root index": "показатель корня", "root": "корень", "absolute value": "модуль",
}
_ERR_EMPTY_RU = {
    "exponent": "пустой показатель степени", "denominator": "пустой знаменатель",
    "numerator": "пустой числитель", "expression": "пустое выражение",
    "log base": "пустое основание логарифма", "parentheses": "пустые скобки",
    "root index": "пустой показатель корня", "root": "пустой корень", "absolute value": "пустой модуль",
}
_ERR_PATTERNS_RU = [
    (re.compile(r"^empty (.+)$"), lambda m: _ERR_EMPTY_RU.get(m.group(1), f"пусто: {m.group(1)}")),
    (re.compile(r"^trailing operator in (.+)$"),
     lambda m: f"{_ERR_PARTS_RU.get(m.group(1), m.group(1))}: оператор в конце"),
    (re.compile(r"^lone '\.' in (.+)$"),
     lambda m: f"{_ERR_PARTS_RU.get(m.group(1), m.group(1))}: одинокая точка '.'"),
    (re.compile(r"^(.+) without argument$"), lambda m: f"{m.group(1)} без аргумента"),
    (re.compile(r"^bad number '(.+)'$"), lambda m: f"неверное число '{m.group(1)}'"),
    (re.compile(r"^operator '(.+)' without left operand$"), lambda m: f"оператор '{m.group(1)}' без левого операнда"),
    (re.compile(r"^unknown name: (.+)$"), lambda m: f"неизвестное имя: {m.group(1)}"),
    (re.compile(r"^cannot parse the constant in (.+)$"),
     lambda m: f"не удаётся разобрать константу: {_ERR_PARTS_RU.get(m.group(1), m.group(1))}"),
]

# Таблицы по языкам
_STRINGS = {"he": STRINGS_HE, "ru": STRINGS_RU}
_ENGINE_STRINGS = {"he": ENGINE_STRINGS_HE, "ru": ENGINE_STRINGS_RU}
_ERRORS = {"he": (ERRORS_HE, _ERR_PATTERNS_HE), "ru": (ERRORS_RU, _ERR_PATTERNS_RU)}


def T(text, **fmt):
    """Строка интерфейса на текущем языке (ключ — английская строка)."""
    out = _STRINGS.get(LANG, {}).get(text, text)
    return out.format(**fmt) if fmt else out


def tr_err(msg):
    """Перевод сообщения об ошибке формулы; неизвестное — как есть."""
    if LANG not in _ERRORS or not msg:
        return msg
    table, patterns = _ERRORS[LANG]
    if msg in table:
        return table[msg]
    for rx, fn in patterns:
        m = rx.match(msg)
        if m:
            return fn(m)
    return msg


# Ивритские фрагменты визуальной строки: буквы иврита и пробелы между словами.
# Знаки препинания, цифры и латиница в них не входят — они идут шрифтом UI_FONT.
_HEB_WORDS_RE = re.compile(r'[\u0590-\u05FF]+(?: +[\u0590-\u05FF]+)*')

# Tk на Windows (GDI) переставляет RTL-текст сам, но считает абзац направленным
# слева направо; Tk на Linux/macOS и matplotlib не переставляют ничего.
_TK_DOES_BIDI = sys.platform == "win32"


def he_display(text):
    """
    Строка для одиночного виджета/диалога (кнопка, заголовок окна, пункт меню,
    messagebox): визуальный порядок по правилам bidi для ивритского абзаца.
    На Windows возвращается строка, которую GDI (абзац слева направо) нарисует
    именно в этом визуальном порядке: bidi_visual(V, 'L') — обратное
    преобразование, т. к. перестановка отрезков одного уровня — инволюция.
    """
    if not has_heb(text):
        return text
    v = fv.bidi_visual(text, 'R')
    return fv.bidi_visual(v, 'L') if _TK_DOES_BIDI else v


def _visual_runs(text):
    """[(фрагмент, иврит?)] визуальной строки слева направо."""
    v = fv.bidi_visual(text, 'R')
    runs, pos = [], 0
    for m in _HEB_WORDS_RE.finditer(v):
        if m.start() > pos:
            runs.append((v[pos:m.start()], False))
        runs.append((m.group(0), True))
        pos = m.end()
    if pos < len(v):
        runs.append((v[pos:], False))
    return runs


def _widget_text(run, is_heb):
    """Текст для отдельного виджета: ивритский фрагмент на Windows отдаём в
    логическом порядке (GDI зеркалит его сам), иначе — уже визуальный."""
    if is_heb and _TK_DOES_BIDI:
        return run[::-1]
    if not is_heb and not run.strip():
        return run.replace(" ", "\u00a0")       # пробел между фрагментами
    return run


def ui_font(text, size, bold=False, italic=False):
    """Кортеж шрифта Tk: David для строк с ивритом, иначе UI_FONT."""
    fam = local_font() if has_local(text) else UI_FONT
    style = " ".join(w for w, on in (("bold", bold), ("italic", italic)) if on)
    return (fam, size, style) if style else (fam, size)


def side():
    """Сторона упаковки «начала строки»: left для LTR, right для RTL."""
    return "right" if RTL() else "left"


def oside():
    return "left" if RTL() else "right"


def anchor_start():
    return "e" if RTL() else "w"


def make_label(parent, text, size=9, color=SUBTEXT, bold=False, bg=None, **kw):
    """
    Подпись интерфейса. В английском режиме — обычный tk.Label. В иврите
    строка разбивается на фрагменты: ивритские — шрифт David, остальные
    (цифры, латиница, знаки) — UI_FONT; фрагменты упакованы справа налево.
    Возвращает виджет (Label или Frame) — у него есть .pack()/.grid().
    """
    bg = bg or (parent.cget("bg") if hasattr(parent, "cget") else CARD_BG)
    text = T(text)
    if not (RTL() and has_heb(text)):
        if RTL():
            # Подпись без иврита в ивритском интерфейсе («f1:» → «:f1»): знаки
            # по краям берут направление абзаца; без RTL-символов GDI ничего не
            # переставляет, так что строка одинакова для Windows и Linux.
            text = fv.bidi_visual(text, 'R')
        if LANG == "ru" and has_cyr(text):
            # Русский: слова — RU_FONT, цифры/латиница/знаки — UI_FONT, слева направо
            box = tk.Frame(parent, bg=bg)
            pos = 0
            runs = []
            for m in _CYR_WORDS_RE.finditer(text):
                if m.start() > pos:
                    runs.append((text[pos:m.start()], False))
                runs.append((m.group(0), True))
                pos = m.end()
            if pos < len(text):
                runs.append((text[pos:], False))
            for run, is_cyr in runs:
                if not is_cyr and not run.strip():
                    run = run.replace(" ", "\u00a0")
                fam = RU_FONT if is_cyr else UI_FONT
                tk.Label(box, text=run, bg=bg, fg=color, font=(fam, size, "bold") if bold else (fam, size),
                         padx=0, bd=0).pack(side="left")
            return box
        return tk.Label(parent, text=text, bg=bg, fg=color, font=ui_font(text, size, bold), **kw)
    box = tk.Frame(parent, bg=bg)
    for run, is_heb in _visual_runs(text):          # уже слева направо
        fam, sz = (HE_FONT, size + 1) if is_heb else (UI_FONT, size)
        tk.Label(box, text=_widget_text(run, is_heb), bg=bg, fg=color,
                 font=(fam, sz, "bold") if bold else (fam, sz),
                 padx=0, bd=0).pack(side="left")
    return box


def make_paragraph(parent, text, size=8, color=SUBTEXT, bg=None, width_px=300):
    """
    Многострочная подсказка. В английском режиме — Label с wraplength.
    В иврите переносим строки сами (bidi работает построчно) и каждую
    строку собираем через make_label, прижимая к правому краю.
    """
    bg = bg or parent.cget("bg")
    text = T(text)
    if not (RTL() and has_heb(text)):
        return tk.Label(parent, text=text, bg=bg, fg=color, font=ui_font(text, size),
                        wraplength=width_px, justify="left", anchor="w")
    import tkinter.font as tkfont
    f = tkfont.Font(family=HE_FONT, size=size + 1)
    lines, cur = [], ""
    for word in text.split(" "):
        cand = f"{cur} {word}" if cur else word
        if cur and f.measure(cand) > width_px:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    if cur:
        lines.append(cur)
    box = tk.Frame(parent, bg=bg)
    for ln in lines:
        make_label(box, ln, size=size, color=color, bg=bg).pack(anchor="e")
    return box


def make_check(parent, text, var, bg=None):
    """Чекбокс с подписью. Иврит/русский: индикатор + подпись из фрагментов
    (make_label); в иврите индикатор справа, подпись слева, в русском — наоборот."""
    bg = bg or parent.cget("bg")
    text = T(text)
    if not has_local(text):
        return tk.Checkbutton(parent, text=text, variable=var, bg=bg, fg=TEXT,
                              selectcolor=ENTRY_BG, activebackground=bg, activeforeground=TEXT,
                              font=ui_font(text, 10), bd=0, highlightthickness=0)
    box = tk.Frame(parent, bg=bg)
    cb = tk.Checkbutton(box, text="", variable=var, bg=bg, fg=TEXT, selectcolor=ENTRY_BG,
                        activebackground=bg, activeforeground=TEXT, bd=0, highlightthickness=0, padx=0)
    cb.pack(side=side())
    lbl = make_label(box, text, size=10, color=TEXT, bg=bg)
    lbl.pack(side=side(), padx=(0 if RTL() else 2, 2 if RTL() else 0))

    def toggle(_e=None):
        var.set(0 if var.get() else 1)
    for w in [lbl] + list(lbl.winfo_children()):
        w.bind("<Button-1>", toggle)
        w.configure(cursor="hand2")
    return box


def apply_engine_language():
    """Передаёт движку переводы его меню и настройки ивритского текста на графике."""
    if LANG == "he":
        fv.UI_TRANSLATIONS = dict(ENGINE_STRINGS_HE)
        fv.EXTRA_FONT_FAMILIES = [HE_FONT, "DejaVu Sans"]
        fv.BIDI_SIMPLE = True
        fv.UI_DISPLAY = he_display
    elif LANG == "ru":
        fv.UI_TRANSLATIONS = dict(ENGINE_STRINGS_RU)
        fv.EXTRA_FONT_FAMILIES = [RU_FONT, "DejaVu Sans"]
        fv.BIDI_SIMPLE = False
        fv.UI_DISPLAY = None
    else:
        fv.UI_TRANSLATIONS = {}
        fv.EXTRA_FONT_FAMILIES = []
        fv.BIDI_SIMPLE = False
        fv.UI_DISPLAY = None
    try:
        fv.apply_font_preset()
    except Exception:
        pass


def _init_ui_scale(root):
    """
    UI_SCALE = DPI экрана / 96 (на Windows с DPI-awareness Tk сообщает
    реальный DPI; при масштабе 150 % получаем 1.5). Переменная окружения
    FV_UI_SCALE задаёт коэффициент принудительно (и для отладки под Xvfb).
    Вызывается для каждого корневого окна (выбор языка, главное окно).
    """
    global UI_SCALE
    forced = os.environ.get("FV_UI_SCALE")
    try:
        if forced:
            UI_SCALE = float(forced)
            # шрифты в пунктах тоже должны вырасти — как сделал бы Tk при реальном DPI
            root.tk.call('tk', 'scaling', UI_SCALE * 96.0 / 72.0)
        else:
            UI_SCALE = root.winfo_fpixels('1i') / 96.0
    except Exception:
        UI_SCALE = 1.0
    UI_SCALE = max(0.75, min(4.0, UI_SCALE))
    return UI_SCALE


def _resource_path(name):
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def _log_dir():
    """Папка для error.log: %LOCALAPPDATA%/FuncVisualizer (Windows) или ~/.funcvisualizer."""
    base = os.environ.get("LOCALAPPDATA") if sys.platform == "win32" else None
    if base:
        d = os.path.join(base, "FuncVisualizer")
    else:
        d = os.path.join(os.path.expanduser("~"), ".funcvisualizer")
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:
        import tempfile
        return tempfile.gettempdir()


ERROR_LOG = os.path.join(_log_dir(), "error.log")


def log_exception(context=""):
    """
    Пишет текущий traceback в error.log (в exe без консоли stderr нет, и
    print_exc() пропал бы) и дублирует в stderr, если он есть.
    """
    text = traceback.format_exc()
    try:
        import datetime
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {context}\n{text}")
    except Exception:
        pass
    try:
        if sys.stderr is not None:
            sys.stderr.write(text)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ВИДЖЕТЫ
# ═════════════════════════════════════════════════════════════

def styled_entry(parent, width=14, **kw):
    if RTL():
        kw.setdefault("justify", "right")
    e = tk.Entry(parent, width=width, bg=ENTRY_BG, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 font=(UI_FONT, 10),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, **kw)
    return e


def live_entry(parent, width, initial, on_change):
    """Entry со StringVar: любое изменение текста → on_change()."""
    var = tk.StringVar(value=str(initial))
    e = styled_entry(parent, width=width, textvariable=var)
    e.var = var
    var.trace_add("write", lambda *_: on_change())
    return e


def styled_check(parent, text, var, **kw):
    return tk.Checkbutton(parent, text=text, variable=var,
                          bg=parent.cget("bg"), fg=TEXT,
                          selectcolor=ENTRY_BG, activebackground=CARD_BG,
                          activeforeground=TEXT, font=(UI_FONT, 10),
                          bd=0, highlightthickness=0, **kw)


def card(parent, title=""):
    """Карточка-секция с заголовком. Возвращает inner frame для контента."""
    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    inner = tk.Frame(outer, bg=CARD_BG, bd=0, relief="flat")
    inner.pack(fill="both", expand=True)
    if title:
        make_label(inner, title, size=10, color=ACCENT, bold=True, bg=CARD_BG).pack(
            anchor=anchor_start(), padx=10, pady=(8, 2))
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(0, 6))
    inner._outer = outer   # храним ссылку чтобы pack вызывать снаружи
    return inner


def card_pack(widget, **kw):
    """pack карточки через её outer frame."""
    widget._outer.pack(**kw)


def small_button(parent, text, command, bg=BTN_DEL, fg="white", **kw):
    text = T(text)
    opts = dict(bg=bg, fg=fg, relief="flat", font=ui_font(text, 9, bold=True),
                cursor="hand2", bd=0, padx=10, pady=4, activebackground=bg,
                activeforeground=fg, command=command)
    if "font" in kw and has_local(text):
        f = kw.pop("font")
        kw["font"] = ui_font(text, f[1], bold="bold" in f[2:])
    opts.update(kw)
    return tk.Button(parent, text=he_display(text), **opts)


class Tooltip:
    """Простая всплывающая подсказка для любого виджета."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        box = tk.Frame(self.tip, bg=BORDER, padx=1, pady=1)
        box.pack()
        inner = tk.Frame(box, bg="#fffbe6", padx=6, pady=3)
        inner.pack()
        make_label(inner, self.text, size=9, color=TEXT, bg="#fffbe6").pack()

    def _hide(self, _event=None):
        if self.tip is not None:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None


# ═════════════════════════════════════════════════════════════
#  ЭКРАННАЯ МАТЕМАТИЧЕСКАЯ КЛАВИАТУРА
# ═════════════════════════════════════════════════════════════

class Keypad(tk.Frame):
    """
    Клавиатура ввода формул. Единственный способ ввода в поле функции —
    у поля нет доступа к физической клавиатуре. Каждая кнопка шлёт токен
    (см. math_editor.MathModel.insert_token) активному редактору через on_token.
    """

    # (подпись, токен, подсказка)
    LEFT_BASIC = [
        [("x", "x", "variable x"), ("y", "y", "variable y (for equations, e.g. x²+y²=9)"),
         ("a²", "sq", "square"), ("aᵇ", "^", "power")],
        [("(", "(", "open parenthesis"), (")", ")", "leave the parentheses"),
         ("√", "sqrt", "square root"), ("ⁿ√", "root", "n-th root")],
        [("|a|", "abs", "absolute value"), ("π", "pi", "the number π"),
         ("e", "e", "the number e"), ("a/b", "/", "fraction")],
        [("sin", "sin", "sine"), ("cos", "cos", "cosine"), ("tan", "tan", "tangent"),
         ("cot", "cot", "cotangent")],
        [("ln", "ln", "natural logarithm"), ("log", "log", "common (base-10) logarithm"),
         ("logₐ", "logb", "logarithm with base a"), ("eˣ", "exp", "exponential")],
        [("fn ▸", "__page__", "more functions: arcsin, arccos, sinh…"),
         ("↑", "up", "cursor up (numerator)"), ("↓", "down", "cursor down (denominator)"),
         ("", None, None)],
    ]
    LEFT_EXTRA = [
        [("arcsin", "asin", "inverse sine"), ("arccos", "acos", "inverse cosine"),
         ("arctan", "atan", "inverse tangent"), ("sec", "sec", "secant")],
        [("csc", "csc", "cosecant"), ("sinh", "sinh", "hyperbolic sine"),
         ("cosh", "cosh", "hyperbolic cosine"), ("tanh", "tanh", "hyperbolic tangent")],
    ]
    RIGHT = [
        [("7", "7", None), ("8", "8", None), ("9", "9", None), ("÷", "/", "fraction")],
        [("4", "4", None), ("5", "5", None), ("6", "6", None), ("×", "*", "multiply")],
        [("1", "1", None), ("2", "2", None), ("3", "3", None), ("−", "-", "minus")],
        [("0", "0", None), (".", ".", "decimal point"), ("=", "=", "equals (equation / x = c)"),
         ("+", "+", "plus")],
        [("←", "left", "cursor left"), ("→", "right", "cursor right"),
         ("⌫", "backspace", "delete"), ("C", "clear", "clear the field")],
    ]

    KEY_MIN_W = 46   # минимальная ширина кнопки, px при 96 dpi

    def __init__(self, parent, on_token, **kw):
        super().__init__(parent, bg=CARD_BG, **kw)
        self.on_token = on_token
        self._page = 0
        self._left_rows = []

        left = tk.Frame(self, bg=CARD_BG)
        left.pack(side="left", padx=(0, 10))
        right = tk.Frame(self, bg=CARD_BG)
        right.pack(side="left")

        # Левый блок: строки 0..2 постоянные, 3..4 переключаемые (fn ▸), 5 — навигация
        self._left = left
        self._build_left()
        self._build_block(right, self.RIGHT, numeric=True)

    # ── построение ───────────────────────────────────────────
    def _build_left(self):
        for w in self._left.winfo_children():
            w.destroy()
        rows = list(self.LEFT_BASIC[:3])
        if self._page == 0:
            rows += self.LEFT_BASIC[3:5]
        else:
            rows += self.LEFT_EXTRA
        rows += [self.LEFT_BASIC[5]]
        self._build_block(self._left, rows, numeric=False)

    def _build_block(self, parent, rows, numeric):
        for r, row in enumerate(rows):
            for c, (label, token, tip) in enumerate(row):
                if token is None:
                    tk.Frame(parent, bg=CARD_BG, width=_px(self.KEY_MIN_W), height=_px(28)).grid(row=r, column=c)
                    continue
                is_digit = numeric and (label.isdigit() or label == ".")
                bg = KEY_BG2 if is_digit else KEY_BG
                fnt = (UI_FONT, 10, "bold") if is_digit else (UI_FONT, 10)
                if len(label) > 4:            # arcsin, arccos… — чуть мельче, чтобы не слипались
                    fnt = (UI_FONT, 9)
                if token == "__page__":
                    label = "fn ▸" if self._page == 0 else "◂ back"
                    tip = "more functions: arcsin, arccos, sinh…" if self._page == 0 else "back to sin, cos, ln…"
                    fnt = (UI_FONT, 9)
                # width=1 + minsize колонки: кнопки одинаковой ширины в пикселях
                # независимо от шрифта; длинные подписи (arcsin, logₐ) сами
                # расширяют свою колонку. Рамка — отдельный Frame в 1 px: Tk на
                # Windows не рисует highlight-рамку у кнопок без фокуса.
                holder = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
                b = tk.Button(holder, text=label, bg=bg, fg=KEY_FG, font=fnt,
                              relief="flat", bd=0, cursor="hand2", width=1, padx=4, pady=3,
                              activebackground="#dde3ea", activeforeground=KEY_FG,
                              highlightthickness=0, takefocus=0,
                              command=lambda t=token: self._press(t))
                b.pack(fill="both", expand=True)
                holder.grid(row=r, column=c, padx=1, pady=1, sticky="nsew")
                parent.grid_columnconfigure(c, minsize=_px(self.KEY_MIN_W))
                if tip:
                    Tooltip(b, T(tip))

    # ── нажатие ──────────────────────────────────────────────
    def _press(self, token):
        if token == "__page__":
            self._page = 1 - self._page
            self._build_left()
            return
        self.on_token(token)


# ═════════════════════════════════════════════════════════════
#  СТРОКА ОДНОЙ ФУНКЦИИ
# ═════════════════════════════════════════════════════════════

class FuncRow:
    LINESTYLES = {"Solid": "-", "Dashed": "--", "Dotted": ":"}
    LW_VALUES  = [0.5, 1.0, 1.5, 1.8, 2.5, 3.5, 5.0, 7.0]

    def __init__(self, parent, idx, on_delete, on_change, on_focus):
        self.idx   = idx
        self.color = FUNC_COLORS[idx % len(FUNC_COLORS)]
        self.on_change = on_change

        self.frame = tk.Frame(parent, bg=PANEL_BG)
        self.frame.pack(fill="x", padx=4, pady=(3, 0))

        S, S2 = side(), oside()        # начало строки / конец строки (RTL-aware)

        # Индекс функции
        self.idx_label = tk.Label(self.frame, text=f"f{idx}", width=2,
                                  bg=PANEL_BG, fg=SUBTEXT, font=(UI_FONT, 9, "bold"))
        self.idx_label.pack(side=S, padx=(6, 2))

        # Кликабельный цветной квадрат — выбор цвета
        self.dot = tk.Button(self.frame, bg=self.color, activebackground=self.color,
                             width=2, height=1, relief="flat", bd=0,
                             cursor="hand2", command=self._pick_color)
        self.dot.pack(side=S, padx=(2, 6))

        # Поле формулы — 2-D редактор без клавиатуры (ввод с экранной клавиатуры)
        self.editor = MathEditor(self.frame, font_size=_px(15), on_change=lambda _e: on_change(),
                                 on_focus=lambda e: on_focus(self), width=_px(260), height=_px(40),
                                 error_font=(local_font(), 10) if LANG != "en" else None)
        self.editor.pack(side=S, padx=2, fill="x", expand=True)

        # Кнопка удалить
        tk.Button(self.frame, text="✕", bg=BTN_DEL, fg="white", activebackground=BTN_DEL,
                  relief="flat", font=(UI_FONT, 9, "bold"), cursor="hand2", bd=0, padx=6,
                  command=lambda: on_delete(self)).pack(side=S2, padx=4)

        # ── Вторая строка: стиль линии + область определения ──
        self.frame2 = tk.Frame(parent, bg=PANEL_BG)
        self.frame2.pack(fill="x", padx=4, pady=(0, 3))

        tk.Label(self.frame2, text="", width=2, bg=PANEL_BG).pack(side=S, padx=(6, 2))

        # Ширина линии — кнопки ▼/▲
        make_label(self.frame2, "w:", size=8, bg=PANEL_BG).pack(side=S, padx=(2, 1))
        self._lw_idx = 3   # default = 1.8
        self._lw_label = tk.Label(self.frame2, text=f"{self.get_linewidth()}", width=3,
                                  bg=ENTRY_BG, fg=TEXT, font=(UI_FONT, 9), relief="flat",
                                  highlightthickness=1, highlightbackground=BORDER)
        self._lw_label.pack(side=S)
        btn_f = dict(bg=PANEL_BG, fg=TEXT, relief="flat", bd=0, font=(UI_FONT, 8),
                     cursor="hand2", padx=2, activebackground=PANEL_BG)
        tk.Button(self.frame2, text="▼", command=lambda: self._lw_step(-1), **btn_f).pack(side=S)
        tk.Button(self.frame2, text="▲", command=lambda: self._lw_step(+1), **btn_f).pack(side=S, padx=(0, 4))

        # Тип линии (подписи пунктов — на языке интерфейса)
        self._style_names = {he_display(T(k)): v for k, v in self.LINESTYLES.items()}
        self.linestyle_var = tk.StringVar(value=he_display(T(list(self.LINESTYLES.keys())[0])))
        self.linestyle_var.trace_add("write", lambda *_: on_change())
        om = tk.OptionMenu(self.frame2, self.linestyle_var, *self._style_names.keys())
        om.config(bg=ENTRY_BG, fg=TEXT, activebackground=CARD_BG, activeforeground=TEXT,
                  relief="flat", font=ui_font(T("Solid"), 8), bd=0, highlightthickness=1,
                  highlightbackground=BORDER, width=7)
        om["menu"].config(bg=ENTRY_BG, fg=TEXT, activebackground=ACCENT, font=ui_font(T("Solid"), 9))
        om.pack(side=S, padx=(2, 6))

        # Область определения: функция строится только на [from, to]
        make_label(self.frame2, "domain:", size=8, bg=PANEL_BG).pack(side=S, padx=(2, 4))
        self.dom_from = live_entry(self.frame2, 6, "-inf", on_change)
        self.dom_from.pack(side=S, padx=2)
        tk.Label(self.frame2, text="…", bg=PANEL_BG, fg=SUBTEXT,
                 font=(UI_FONT, 8)).pack(side=S)
        self.dom_to = live_entry(self.frame2, 6, "inf", on_change)
        self.dom_to.pack(side=S, padx=2)

    # ── ширина ───────────────────────────────────────────────
    def _lw_step(self, d):
        self._lw_idx = max(0, min(len(self.LW_VALUES) - 1, self._lw_idx + d))
        self._lw_label.config(text=f"{self.get_linewidth()}")
        self.on_change()

    def _pick_color(self):
        result = colorchooser.askcolor(color=self.color,
                                       title=he_display(T("Pick color for f{idx}", idx=self.idx)))
        if result and result[1]:
            self.set_color(result[1])
            self.on_change()

    def set_color(self, color):
        self.color = color
        self.dot.config(bg=color, activebackground=color)

    def set_linewidth(self, lw):
        try:
            self._lw_idx = min(range(len(self.LW_VALUES)),
                               key=lambda i: abs(self.LW_VALUES[i] - float(lw)))
        except Exception:
            self._lw_idx = 3
        self._lw_label.config(text=f"{self.get_linewidth()}")

    def update_idx(self, new_idx):
        self.idx = new_idx
        self.idx_label.config(text=f"f{new_idx}")

    def get_color(self):
        return self.color

    def get_linewidth(self):
        return self.LW_VALUES[self._lw_idx]

    def get_linestyle(self):
        return self._style_names.get(self.linestyle_var.get(), "-")

    def _style_key(self):
        """Английское имя стиля (для проекта), независимо от языка интерфейса."""
        v = self.get_linestyle()
        return next((k for k, s in self.LINESTYLES.items() if s == v), "Solid")

    def is_empty(self):
        return self.editor.model.is_empty()

    def get(self):
        """Строка для движка (sympy). Бросает IncompleteExpression."""
        return self.editor.get_sympy()

    def get_domain(self):
        """Возвращает (from, to) области определения; по умолчанию (-inf, inf)."""
        try:
            d_from = fv.parse_number(self.dom_from.get(), default=float("-inf"))
        except Exception:
            d_from = float("-inf")
        try:
            d_to = fv.parse_number(self.dom_to.get(), default=float("inf"))
        except Exception:
            d_to = float("inf")
        if d_from > d_to:
            d_from, d_to = d_to, d_from
        return (d_from, d_to)

    # ── сериализация (проект) ────────────────────────────────
    def to_dict(self):
        return {
            "model": self.editor.model.to_json(),
            "text": self.editor.model.to_display(),
            "color": self.color,
            "width": self.get_linewidth(),
            "style": self._style_key(),
            "domain": [self.dom_from.get(), self.dom_to.get()],
        }

    def from_dict(self, d):
        # Приоритет: дерево формулы ("model"); если его нет/оно пустое, а есть
        # текст — разбираем текст (проект, написанный руками или другой программой).
        model = None
        m = d.get("model")
        if isinstance(m, dict) and m.get("root"):
            try:
                model = MathModel.from_json(m)
                if model.is_empty():
                    model = None
            except Exception:
                model = None
        if model is None and d.get("text"):
            try:
                model = MathModel.from_text(str(d["text"]))
            except Exception:
                model = None
        self.editor.set_model(model if model is not None else MathModel())
        self.set_color(d.get("color", self.color))
        self.set_linewidth(d.get("width", 1.8))
        if d.get("style") in self.LINESTYLES:
            self.linestyle_var.set(T(d["style"]))
        dom = d.get("domain", ["-inf", "inf"])
        self.dom_from.var.set(str(dom[0]))
        self.dom_to.var.set(str(dom[1]))

    def destroy(self):
        self.frame.destroy()
        self.frame2.destroy()


# ═════════════════════════════════════════════════════════════
#  СТРОКА ЗАЛИВКИ
# ═════════════════════════════════════════════════════════════

class FillRow:
    STYLES = ["45deg ////", "135deg \\\\\\\\", "Dots ...."]

    def __init__(self, parent, on_delete, on_change):
        self.frame = tk.Frame(parent, bg=PANEL_BG)
        self.frame.pack(fill="x", padx=4, pady=3)

        S, S2 = side(), oside()

        def lbl(t, parent=None):
            make_label(parent or self.frame, t, size=9, bg=PANEL_BG).pack(side=S, padx=(4, 1))

        lbl("f1:")
        self.f1 = live_entry(self.frame, 3, "0", on_change); self.f1.pack(side=S)
        lbl("f2:")
        self.f2 = live_entry(self.frame, 4, "x", on_change); self.f2.pack(side=S)
        lbl("from:")
        self.x_from = live_entry(self.frame, 6, "-3", on_change); self.x_from.pack(side=S)
        lbl("to:")
        self.x_to = live_entry(self.frame, 6, "3", on_change); self.x_to.pack(side=S)

        lbl("style:")
        self._style_names = [he_display(T(x)) for x in self.STYLES]
        self.style_var = tk.StringVar(value=self._style_names[0])
        self.style_var.trace_add("write", lambda *_: on_change())
        om = tk.OptionMenu(self.frame, self.style_var, *self._style_names)
        om.config(bg=ENTRY_BG, fg=TEXT, activebackground=CARD_BG, activeforeground=TEXT,
                  relief="flat", font=ui_font(self._style_names[-1], 9), bd=0,
                  highlightthickness=0, width=9)
        om["menu"].config(bg=ENTRY_BG, fg=TEXT, activebackground=ACCENT,
                          font=ui_font(self._style_names[-1], 9))
        om.pack(side=S, padx=4)

        tk.Button(self.frame, text="✕", bg=BTN_DEL, fg="white", activebackground=BTN_DEL,
                  relief="flat", font=(UI_FONT, 9, "bold"), cursor="hand2", bd=0, padx=6,
                  command=lambda: on_delete(self)).pack(side=S2, padx=(10, 4))

        # вторая строка: границы + плотность
        self.frame2 = tk.Frame(parent, bg=PANEL_BG)
        self.frame2.pack(fill="x", padx=4, pady=(0, 3))

        self.borders_var = tk.IntVar(value=1)
        self.borders_var.trace_add("write", lambda *_: on_change())
        make_check(self.frame2, "borders", self.borders_var, bg=PANEL_BG).pack(side=S, padx=4)

        lbl("density:", self.frame2)
        self.density_var = tk.IntVar(value=100)
        self._density_ind = tk.Label(self.frame2, text="100", width=3,
                                     bg=PANEL_BG, fg=TEXT, font=(UI_FONT, 9, "bold"))

        def _on_density(val):
            try:
                self._density_ind.config(text=str(int(float(val))))
            except Exception:
                pass
            on_change()

        tk.Scale(self.frame2, from_=0, to=100, orient="horizontal",
                 variable=self.density_var, length=_px(110), resolution=5,
                 bg=CARD_BG, fg=TEXT, troughcolor=ACCENT, activebackground=BTN_DEL,
                 highlightthickness=0, bd=0, sliderrelief="flat", showvalue=False,
                 command=_on_density, font=(UI_FONT, 7)).pack(side=S)
        self._density_ind.pack(side=S, padx=(4, 0))

    def get(self):
        """Returns tuple (f1, f2, x_from, x_to, style, borders, density) or None on error."""
        try:
            f1 = int(self.f1.get().strip())
            f2_raw = self.f2.get().strip()
            f2 = f2_raw if f2_raw.lower() in ("x", "") else int(f2_raw)
            x_from  = fv.parse_number(self.x_from.get())
            x_to    = fv.parse_number(self.x_to.get())
            style   = self._style_names.index(self.style_var.get())
            borders = bool(self.borders_var.get())
            # density: 0%→step=0.05 (редко),  100%→step=0.01 (густо)
            pct     = self.density_var.get() / 100.0
            density = 0.05 - pct * 0.04
            return (f1, f2, x_from, x_to, style, borders, density)
        except Exception:
            return None

    def to_dict(self):
        return {"f1": self.f1.get(), "f2": self.f2.get(), "from": self.x_from.get(),
                "to": self.x_to.get(), "style": self.STYLES[self._style_names.index(self.style_var.get())],
                "borders": self.borders_var.get(), "density": self.density_var.get()}

    def from_dict(self, d):
        self.f1.var.set(str(d.get("f1", "0")))
        self.f2.var.set(str(d.get("f2", "x")))
        self.x_from.var.set(str(d.get("from", "-3")))
        self.x_to.var.set(str(d.get("to", "3")))
        if d.get("style") in self.STYLES:
            self.style_var.set(self._style_names[self.STYLES.index(d["style"])])
        self.borders_var.set(int(d.get("borders", 1)))
        self.density_var.set(int(d.get("density", 100)))
        self._density_ind.config(text=str(self.density_var.get()))

    def destroy(self):
        self.frame.destroy()
        self.frame2.destroy()


# ═════════════════════════════════════════════════════════════
#  МГНОВЕННЫЙ ПРЕДПРОСМОТР ПРИ ПАНОРАМИРОВАНИИ / ЗУМЕ
# ═════════════════════════════════════════════════════════════

class FramePreview:
    """
    Пока идёт настоящая перерисовка (100–300 мс), показываем поверх холста
    снимок последнего кадра, сдвинутый (пан) или масштабированный (зум)
    вокруг курсора — как в Desmos. Снимок берётся из буфера Agg, кладётся
    в tk.Label поверх виджета холста и просто перемещается; скрывается
    сразу после синхронной отрисовки нового кадра.
    """

    def __init__(self, canvas):
        self.canvas = canvas
        self.widget = canvas.get_tk_widget()
        self.base = None        # PIL.Image последнего кадра
        self.photo = None
        self.label = None
        self.size = (0, 0)

    def snapshot(self):
        """Снимок текущего кадра; False, если буфер недоступен."""
        try:
            from PIL import Image
            import numpy as np
            buf = np.asarray(self.canvas.buffer_rgba())
            self.base = Image.fromarray(buf, "RGBA").convert("RGB")
            self.size = self.base.size
            return True
        except Exception:
            self.base = None
            return False

    @property
    def active(self):
        return self.label is not None

    def _show(self, img, x=0, y=0):
        try:
            from PIL import ImageTk
            self.photo = ImageTk.PhotoImage(img)
            if self.label is None:
                self.label = tk.Label(self.widget, image=self.photo, bd=0,
                                      highlightthickness=0, bg="white")
            else:
                self.label.configure(image=self.photo)
            self.label.place(in_=self.widget, x=int(round(x)), y=int(round(y)))
            self.label.lift()
        except Exception:
            self.hide()

    def show_shift(self, dx, dy):
        """Пан: снимок, сдвинутый на (dx, dy) пикселей (y вниз), на белом фоне
        во всю площадь холста — старый кадр из-под него не выглядывает."""
        if self.base is None:
            return
        try:
            from PIL import Image
            W, H = self.size
            img = Image.new("RGB", (W, H), "white")
            img.paste(self.base, (int(round(dx)), int(round(dy))))
            self._show(img, 0, 0)
        except Exception:
            self._show(self.base, dx, dy)

    def show_zoom(self, px, py, scale):
        """
        Зум: snapshot, масштабированный в 1/scale раз вокруг точки (px, py)
        (scale < 1 — приближение). Для приближения вырезаем область и
        растягиваем, для отдаления — сжимаем и кладём на белый фон.
        """
        if self.base is None:
            return
        try:
            from PIL import Image
            W, H = self.size
            if scale <= 0:
                return
            if scale < 1.0:
                box = (px - px * scale, py - py * scale,
                       px + (W - px) * scale, py + (H - py) * scale)
                img = self.base.crop(tuple(int(round(v)) for v in box)).resize((W, H), Image.BILINEAR)
            else:
                w2, h2 = max(1, int(round(W / scale))), max(1, int(round(H / scale)))
                small = self.base.resize((w2, h2), Image.BILINEAR)
                img = Image.new("RGB", (W, H), "white")
                img.paste(small, (int(round(px - px / scale)), int(round(py - py / scale))))
            self._show(img, 0, 0)
        except Exception:
            self.hide()

    def hide(self):
        if self.label is not None:
            try:
                self.label.place_forget()
                self.label.destroy()
            except Exception:
                pass
            self.label = None
        self.photo = None
        self.base = None


# ═════════════════════════════════════════════════════════════
#  ГЛАВНОЕ ОКНО
# ═════════════════════════════════════════════════════════════

class App(tk.Tk):
    PROJECT_EXT = ".fvproj"

    def __init__(self):
        super().__init__()
        _resolve_ui_font(self)
        self._init_dpi_scale()
        apply_engine_language()
        self.title(he_display(T("Function Visualizer — Ariadna")))
        self.configure(bg=APP_BG)
        self.resizable(True, True)
        self.minsize(_px(980), _px(640))
        self.geometry(f"{_px(1320)}x{_px(820)}")

        self.func_rows = []
        self.fill_rows = []
        self.active_row = None
        self._logo_img = None
        self._redraw_job = None
        self._drawing = False
        self._redraw_wanted = False
        self._loading = False
        self._pan = None
        self._zoom = None                 # сессия зума колесом: {'px','py','scale'}
        self._tl_job = None
        self._incomplete_rows = 0
        self._preview = None              # FramePreview (создаётся после холста)
        self._last_good = None            # последние настройки, которые построились без ошибок

        # Фоновая символика: ОДИН рабочий поток, пачки заданий (новые — первыми),
        # множество ключей «в работе» (чтобы не считать одно и то же дважды) и
        # сторож по времени (см. _poll_worker).
        self._worker_queue = queue.Queue()
        self._jobs_lock = threading.Lock()
        self._job_batches = collections.deque(maxlen=3)
        self._jobs_event = threading.Event()
        self._running_keys = set()
        self._batch_seq = 0
        self._active_batch = None         # (seq, keys, started_at)
        self._worker = None

        # Иконка окна и панели задач
        try:
            self.iconbitmap(_resource_path("icon.ico"))
        except Exception:
            pass

        # Фигура matplotlib, живущая в правой части окна
        self.fig = Figure(figsize=(7, 7), dpi=100, facecolor="white")

        self._build_ui()
        self._add_func()
        self._setup_clipboard_shortcuts()
        self._connect_graph_events()
        self._start_worker_thread()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_job = self.after(200, self._poll_worker)
        self.schedule_redraw()

    def _init_dpi_scale(self):
        _init_ui_scale(self)

    # ── Ctrl+C/X/V/A независимо от раскладки клавиатуры ─────
    def _setup_clipboard_shortcuts(self):
        """
        На Windows стандартные бинды Tk вида <Control-c> завязаны на keysym
        (символ), а не на физическую клавишу. Если раскладка не английская,
        физическая клавиша C/X/V/A при зажатом Ctrl присылает Tk совсем
        другой keysym, и <Control-c> не срабатывает. event.keycode — код
        физической клавиши, одинаков независимо от раскладки, поэтому ловим
        по нему и генерируем нужное virtual event. Когда раскладка английская,
        встроенный бинд уже срабатывает сам — не дублируем.
        (Действует на обычные числовые поля; поле формулы клавиатуры не имеет.)
        """
        if sys.platform != "win32":
            return      # коды клавиш ниже — Windows virtual-key codes; на X11/macOS
                        # стандартные бинды Tk работают и так
        KEYCODE_TO_ACTION = {
            67: ("<<Copy>>",  ("c", "C")),
            88: ("<<Cut>>",   ("x", "X")),
            86: ("<<Paste>>", ("v", "V")),
        }

        def handler(event):
            if event.state & 0x0001:      # Shift — не наш случай
                return None
            widget = event.widget
            if not isinstance(widget, (tk.Entry, tk.Text)):
                return None
            action = KEYCODE_TO_ACTION.get(event.keycode)
            if action is not None:
                virtual_event, latin_keysyms = action
                if event.keysym in latin_keysyms:
                    return None
                widget.event_generate(virtual_event)
                return "break"
            if event.keycode == 65:   # A — выделить всё
                try:
                    widget.select_range(0, 'end')
                    widget.icursor('end')
                except Exception:
                    pass
                return "break"
            return None

        self.bind_all("<Control-KeyPress>", handler)

    # ══════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── Колонка настроек (слева; в иврите — справа) ─────
        left = tk.Frame(self, bg=APP_BG, width=_px(LEFT_PANEL_WIDTH))
        left.pack(side=side(), fill="y")
        left.pack_propagate(False)

        # ── График (справа; в иврите — слева) ───────────────
        right = tk.Frame(self, bg=APP_BG)
        right.pack(side=side(), fill="both", expand=True)

        self._build_header(left)
        self._build_settings(left)
        self._build_graph_pane(right)

    def _build_header(self, parent):
        hdr = tk.Frame(parent, bg=APP_BG)
        hdr.pack(fill="x")
        try:
            from PIL import Image, ImageTk
            img = Image.open(_resource_path("ariadna-logo1-trnsp.png")).convert("RGBA")
            h = _px(64)
            w = int(img.width * h / img.height)
            img = img.resize((w, h), Image.LANCZOS)
            self._logo_img = ImageTk.PhotoImage(img)
            tk.Label(hdr, image=self._logo_img, bg=APP_BG).pack(side=side(), padx=(16, 10), pady=6)
        except Exception:
            tk.Label(hdr, text="ariadna", bg=APP_BG, fg=ACCENT,
                     font=(UI_FONT, 18, "bold")).pack(side=side(), padx=16, pady=6)

        title_frame = tk.Frame(hdr, bg=APP_BG)
        title_frame.pack(side=side(), padx=(0, 10))
        tk.Label(title_frame, text="Function Visualizer", bg=APP_BG, fg=TEXT,
                 font=(UI_FONT, 16, "bold")).pack(anchor=anchor_start())
        tk.Label(title_frame, text="by Daniel", bg=APP_BG, fg=SUBTEXT,
                 font=(UI_FONT, 9)).pack(anchor=anchor_start())
        tk.Frame(parent, bg=ACCENT, height=3).pack(fill="x")

    def _build_settings(self, parent):
        # Прокручиваемая область
        canvas = tk.Canvas(parent, bg=APP_BG, highlightthickness=0)
        scroll = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=APP_BG)
        self._settings_canvas = canvas
        win = canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_frame.bind("<Configure>",
                               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        # Внутренний фрейм всегда во всю ширину канваса
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=oside(), fill="y")          # полоса прокрутки у внешнего края
        canvas.pack(side=side(), fill="both", expand=True)

        # Колесо прокручивает настройки только когда курсор над левой панелью
        def _wheel(e):
            if not self._pointer_in(parent):
                return
            if getattr(e, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(e, "num", None) == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(-1 * int(e.delta / 120), "units")
        self.bind_all("<MouseWheel>", _wheel)
        self.bind_all("<Button-4>", _wheel)
        self.bind_all("<Button-5>", _wheel)

        p = self.scroll_frame

        # ── Functions ────────────────────────────────────────
        self.func_card = card(p, "Functions")
        card_pack(self.func_card, fill="x", padx=12, pady=(12, 4))
        self.func_list = tk.Frame(self.func_card, bg=PANEL_BG)
        self.func_list.pack(fill="x", padx=8, pady=4)
        small_button(self.func_card, "+ Add function", self._add_func,
                     bg=BTN_ADD).pack(anchor=anchor_start(), padx=10, pady=(0, 6))

        # Экранная клавиатура — внутри карточки функций
        tk.Frame(self.func_card, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(0, 6))
        self.keypad = Keypad(self.func_card, self._on_key)
        self.keypad.pack(anchor=anchor_start(), padx=10, pady=(0, 8))

        # ── View window ──────────────────────────────────────
        lim = card(p, "View Window")
        card_pack(lim, fill="x", padx=12, pady=4)
        g = tk.Frame(lim, bg=CARD_BG)
        g.pack(fill="x", padx=10, pady=(0, 10))

        def lim_row(label, defaults):
            r = tk.Frame(g, bg=CARD_BG)
            r.pack(fill="x", pady=2)
            if LANG != "en":
                make_label(r, label, size=9, bg=CARD_BG).pack(side=side(), padx=(0, 8))
            else:
                tk.Label(r, text=label, bg=CARD_BG, fg=SUBTEXT,
                         font=(UI_FONT, 9), width=12, anchor="e").pack(side="left")
            entries = []
            for d in defaults:
                e = live_entry(r, 8, d, self.schedule_redraw)
                e.pack(side=side(), padx=4)
                entries.append(e)
            return entries

        self.xlim_l, self.xlim_r = lim_row("X:  from / to", ["-5", "5"])
        self.ylim_b, self.ylim_t = lim_row("Y:  from / to", ["-5", "5"])
        make_label(g, "Tip: mouse wheel over the graph zooms, drag pans.", size=8,
                   bg=CARD_BG).pack(anchor=anchor_start(), padx=4, pady=(4, 0))

        # ── Grid ─────────────────────────────────────────────
        grid_c = card(p, "Grid")
        card_pack(grid_c, fill="x", padx=12, pady=4)
        gr = tk.Frame(grid_c, bg=CARD_BG)
        gr.pack(fill="x", padx=10, pady=(0, 10))

        self.grid_var = self._live_int(1)
        make_check(gr, "Show grid", self.grid_var).pack(side=side(), padx=4)
        make_label(gr, "  Step X:", size=9, bg=CARD_BG).pack(side=side())
        self.xgrid_e = live_entry(gr, 5, "1", self.schedule_redraw)
        self.xgrid_e.pack(side=side(), padx=4)
        make_label(gr, "Step Y:", size=9, bg=CARD_BG).pack(side=side())
        self.ygrid_e = live_entry(gr, 5, "1", self.schedule_redraw)
        self.ygrid_e.pack(side=side(), padx=4)

        # ── Display options ───────────────────────────────────
        disp = card(p, "Display on Graph")
        card_pack(disp, fill="x", padx=12, pady=4)
        dg = tk.Frame(disp, bg=CARD_BG)
        dg.pack(fill="x", padx=10, pady=(0, 10))

        self.v_asimp = self._live_int(1)
        self.v_disc  = self._live_int(1)
        self.v_extr  = self._live_int(1)
        self.v_xtag  = self._live_int(1)
        self.v_ytag  = self._live_int(1)
        self.v_inter = self._live_int(1)
        self.v_show_values = self._live_int(1)
        self.v_xhide = self._live_int(0)
        self.v_yhide = self._live_int(0)

        checks = [
            ("Asymptotes",    self.v_asimp),
            ("Holes",         self.v_disc),
            ("Extrema",       self.v_extr),
            ("X-intercepts",  self.v_xtag),
            ("Y-intercepts",  self.v_ytag),
            ("Intersections", self.v_inter),
            ("Show values",   self.v_show_values),
            ("Hide X labels", self.v_xhide),
            ("Hide Y labels", self.v_yhide),
        ]
        grid_f = tk.Frame(dg, bg=CARD_BG)
        grid_f.pack(fill="x")
        # Три колонки; если самая длинная переведённая подпись не влезает
        # (русские названия длиннее английских) — две колонки.
        cols = 3
        if LANG != "en":
            try:
                import tkinter.font as tkfont
                f = tkfont.Font(family=local_font(), size=10)
                widest = max(f.measure(T(t)) for t, _ in checks) + _px(44)   # + индикатор и отступы
                if widest * 3 > _px(LEFT_PANEL_WIDTH - 60):
                    cols = 2
            except Exception:
                pass
        for i, (txt, var) in enumerate(checks):
            col = (cols - 1 - i % cols) if RTL() else (i % cols)   # в иврите колонки справа налево
            make_check(grid_f, txt, var).grid(row=i // cols, column=col, sticky=anchor_start(),
                                              padx=6, pady=1)
        if RTL():
            grid_f.pack_configure(anchor="e")

        # Размер текста
        row3 = tk.Frame(dg, bg=CARD_BG); row3.pack(fill="x", pady=(6, 2))
        make_label(row3, "Label size:", size=9, bg=CARD_BG).pack(side=side(), padx=(6, 4))
        self.font_size_var = tk.IntVar(value=10)
        tk.Scale(row3, from_=6, to=20, orient="horizontal", variable=self.font_size_var,
                 length=_px(160), bg=CARD_BG, fg=TEXT, troughcolor=ACCENT, activebackground=BTN_DEL,
                 highlightthickness=0, bd=0, sliderrelief="flat", font=(UI_FONT, 8),
                 command=lambda _v: self.schedule_redraw()).pack(side=side())
        tk.Label(row3, textvariable=self.font_size_var, bg=CARD_BG, fg=ACCENT,
                 font=(UI_FONT, 9, "bold"), width=3).pack(side=side(), padx=2)


        # ── Fill ─────────────────────────────────────────────
        self.fill_card = card(p, "Area Fill")
        card_pack(self.fill_card, fill="x", padx=12, pady=4)
        make_label(self.fill_card, "f1 / f2 — function indices (0, 1, …) or 'x' for the X axis",
                   size=8, bg=CARD_BG).pack(anchor=anchor_start(), padx=10)
        self.fill_list = tk.Frame(self.fill_card, bg=PANEL_BG)
        self.fill_list.pack(fill="x", padx=8, pady=4)
        small_button(self.fill_card, "+ Add fill", self._add_fill,
                     bg=BTN_BLUE).pack(anchor=anchor_start(), padx=10, pady=(0, 8))

        # ── Graph Labels ─────────────────────────────────────
        labels_card = card(p, "Graph Labels")
        card_pack(labels_card, fill="x", padx=12, pady=4)
        hint = T("Double-click empty space on the graph to add a label. "
                 "Drag to move, scroll to rotate, right-click for options. "
                 "Point labels can be dragged too.")
        make_paragraph(labels_card, hint, size=8, color=SUBTEXT, bg=CARD_BG,
                       width_px=_px(LEFT_PANEL_WIDTH - 60)).pack(anchor=anchor_start(), fill="x",
                                                                 padx=10, pady=(0, 6))
        bl = tk.Frame(labels_card, bg=CARD_BG)
        bl.pack(anchor=anchor_start(), padx=10, pady=(0, 8))
        small_button(bl, "Clear all labels", self._clear_labels).pack(side=side())
        small_button(bl, "Reset point labels", self._reset_point_labels).pack(side=side(), padx=6)

        tk.Frame(p, bg=APP_BG, height=12).pack()

    def _live_int(self, value):
        var = tk.IntVar(value=value)
        var.trace_add("write", lambda *_: self.schedule_redraw())
        return var

    def _build_graph_pane(self, parent):
        bar = tk.Frame(parent, bg=APP_BG)
        bar.pack(fill="x", padx=10, pady=(10, 4))

        small_button(bar, "  Save image…", self._save_image, bg=BTN_SAVE,
                     font=(UI_FONT, 11, "bold"), padx=16, pady=7).pack(side=side())
        small_button(bar, "Reset view", self._reset_view, bg=BTN_DEL,
                     pady=7).pack(side=side(), padx=(8, 0))
        small_button(bar, "Open project", self._open_project, bg=BTN_BLUE,
                     pady=7).pack(side=oside())
        small_button(bar, "Save project", self._save_project, bg=BTN_BLUE,
                     pady=7).pack(side=oside(), padx=(0, 6))

        # Холст matplotlib
        frame = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        w = self.canvas.get_tk_widget()
        w.configure(bg="white", highlightthickness=0)
        w.pack(fill="both", expand=True)
        # add="+": НЕ затирать собственный обработчик <Configure> бэкенда
        # matplotlib (он подгоняет размер фигуры под виджет) — иначе при
        # увеличении окна фигура остаётся прежней и справа/снизу белое поле.
        w.bind("<Configure>", self._on_canvas_resize, add="+")
        # Запрашиваемый размер холста держим маленьким: бэкенд при показе
        # (<Map>) выставляет его равным фигуре (7×7 дюймов ≈ 700+ px), и pack
        # в невысоком окне выдавливает строку состояния за край. Реальный
        # размер задаёт expand=True, фигура подгоняется под него.
        w.bind("<Map>", lambda _e: w.after_idle(self._shrink_canvas_request), add="+")
        self._shrink_canvas_request()
        self._preview = FramePreview(self.canvas)

        # Строка состояния. В иврите и русском текст собирается из фрагментов
        # (местный шрифт + UI_FONT для цифр) в status_box; self.status (Label)
        # хранит полный текст (его читают тесты) и тогда не показывается.
        self.status = tk.Label(parent, text="", bg=APP_BG, fg=SUBTEXT,
                               font=(UI_FONT, 9), anchor="w")
        if LANG != "en":
            self.status_box = tk.Frame(parent, bg=APP_BG)
            self.status_box.pack(fill="x", padx=12, pady=(0, 8))
        else:
            self.status_box = None
            self.status.pack(fill="x", padx=12, pady=(0, 8))

    # ── утилиты ──────────────────────────────────────────────
    def _pointer_in(self, widget):
        try:
            x, y = self.winfo_pointerxy()
            wx, wy = widget.winfo_rootx(), widget.winfo_rooty()
            return wx <= x < wx + widget.winfo_width() and wy <= y < wy + widget.winfo_height()
        except Exception:
            return False

    def _set_status(self, text, color=SUBTEXT):
        self.status.config(text=text, fg=color)
        if self.status_box is not None:
            for w in self.status_box.winfo_children():
                w.destroy()
            make_label(self.status_box, text, size=9, color=color, bg=APP_BG).pack(side=side())

    # ══════════════════════════════════════════════════════════
    #  Клавиатура → активный редактор
    # ══════════════════════════════════════════════════════════
    def _on_key(self, token):
        row = self.active_row
        if row is None or row not in self.func_rows:
            if not self.func_rows:
                return
            self._set_active(self.func_rows[0])
            row = self.active_row
        row.editor.insert_token(token)

    def _set_active(self, row):
        if row is self.active_row and row in self.func_rows:
            row.editor.set_active(True)
            return
        for r in self.func_rows:
            r.editor.set_active(r is row)
        self.active_row = row

    # ══════════════════════════════════════════════════════════
    #  Свободные подписи
    # ══════════════════════════════════════════════════════════
    def _clear_labels(self):
        if not fv.FREE_TEXTS:
            return
        if not messagebox.askyesno(he_display(T("Clear all labels")),
                                   he_display(T("Remove all {n} label(s) added on the graph?", n=len(fv.FREE_TEXTS)))):
            return
        fv.FREE_TEXTS.clear()
        self.schedule_redraw()

    def _reset_point_labels(self):
        """Вернуть все перетащенные подписи точек в положение по умолчанию."""
        fv.reset_annotation_offsets()
        self.schedule_redraw()

    # ══════════════════════════════════════════════════════════
    #  Функции / заливки
    # ══════════════════════════════════════════════════════════
    def _add_func(self):
        idx = len(self.func_rows)
        row = FuncRow(self.func_list, idx, self._del_func,
                      on_change=self.schedule_redraw, on_focus=self._set_active)
        self.func_rows.append(row)
        self._set_active(row)
        self.schedule_redraw()
        return row

    def _del_func(self, row):
        if len(self.func_rows) <= 1:
            messagebox.showwarning(he_display(T("Cannot delete")),
                                   he_display(T("At least one function is required.")))
            return
        row.destroy()
        self.func_rows.remove(row)
        for i, r in enumerate(self.func_rows):
            r.update_idx(i)
        if self.active_row is row:
            self._set_active(self.func_rows[0])
        self.schedule_redraw()

    def _add_fill(self):
        row = FillRow(self.fill_list, self._del_fill, on_change=self.schedule_redraw)
        self.fill_rows.append(row)
        self.schedule_redraw()
        return row

    def _del_fill(self, row):
        row.destroy()
        self.fill_rows.remove(row)
        self.schedule_redraw()

    # ══════════════════════════════════════════════════════════
    #  ЖИВАЯ ПЕРЕРИСОВКА
    # ══════════════════════════════════════════════════════════
    def schedule_redraw(self, *_, delay=None):
        """Отложенная перерисовка: множество изменений подряд → одна отрисовка."""
        if self._loading:
            return
        if self._redraw_job is not None:
            try:
                self.after_cancel(self._redraw_job)
            except Exception:
                pass
        self._redraw_job = self.after(REDRAW_DELAY_MS if delay is None else delay, self._redraw)

    def redraw_now(self):
        """Немедленная перерисовка (минуя debounce) — после пана/зума."""
        if self._redraw_job is not None:
            try:
                self.after_cancel(self._redraw_job)
            except Exception:
                pass
            self._redraw_job = None
        self._redraw()

    def _collect(self):
        """
        Собирает настройки из панели. Возвращает dict для движка или
        бросает ValueError с сообщением для строки состояния.
        """
        funcs, colors, widths, styles, domains = [], [], [], [], []
        self._incomplete_rows = 0
        for r in self.func_rows:
            if r.is_empty():
                r.editor.set_error(None)
                funcs.append("")       # пустая строка = нет кривой (индексы стабильны)
            else:
                try:
                    funcs.append(r.get())
                    r.editor.set_error(None)
                except IncompleteExpression as ex:
                    r.editor.set_error(he_display(tr_err(str(ex))))
                    self._incomplete_rows += 1
                    funcs.append("")
            colors.append(r.get_color())
            widths.append(r.get_linewidth())
            styles.append(r.get_linestyle())
            domains.append(r.get_domain())

        def num(entry, what):
            try:
                return fv.parse_number(entry.get())
            except Exception:
                raise ValueError(T("{what}: cannot read '{val}'", what=T(what), val=entry.get()))

        xl = num(self.xlim_l, "X from"); xr = num(self.xlim_r, "X to")
        yb = num(self.ylim_b, "Y from"); yt = num(self.ylim_t, "Y to")
        if not all(map(lambda v: v == v and abs(v) != float("inf"), (xl, xr, yb, yt))):
            raise ValueError(T("View window must be finite"))
        if xl >= xr or yb >= yt:
            raise ValueError(T("View window: left < right and bottom < top required"))
        xg = num(self.xgrid_e, "Step X"); yg = num(self.ygrid_e, "Step Y")
        if not (math.isfinite(xg) and math.isfinite(yg)) or xg <= 0 or yg <= 0:
            raise ValueError(T("Grid step must be a positive finite number"))
        if (xr - xl) / xg > MAX_GRID_LINES or (yt - yb) / yg > MAX_GRID_LINES:
            raise ValueError(T("Grid step is too small for this view window"))

        fills = []
        for i, fr in enumerate(self.fill_rows):
            entry = fr.get()
            if entry is None:
                raise ValueError(T("Fill #{n}: check parameters", n=i + 1))
            fills.append(entry)

        return dict(funcs=funcs, colors=colors, widths=widths, styles=styles, domains=domains,
                    xl=xl, xr=xr, yb=yb, yt=yt, xg=xg, yg=yg, fills=fills)

    def _apply_to_engine(self, s):
        fv.FUNCS        = s["funcs"]
        fv.FUNC_DOMAINS = s["domains"]
        fv.CURVE_COLORS = s["colors"]
        fv.CURVE_WIDTHS = s["widths"]
        fv.CURVE_STYLES = s["styles"]
        fv.X_LIM_L, fv.X_LIM_R = s["xl"], s["xr"]
        fv.Y_LIM_B, fv.Y_LIM_T = s["yb"], s["yt"]
        fv.GRID   = self.grid_var.get()
        fv.X_GRID = s["xg"]
        fv.Y_GRID = s["yg"]
        fv.ASIMP  = self.v_asimp.get()
        fv.DISC   = self.v_disc.get()
        fv.EXTR   = self.v_extr.get()
        fv.X_TAG  = self.v_xtag.get()
        fv.Y_TAG  = self.v_ytag.get()
        fv.INTER  = self.v_inter.get()
        fv.SHOW_VALUES = self.v_show_values.get()
        fv.X_HIDE = self.v_xhide.get()
        fv.Y_HIDE = self.v_yhide.get()
        fv.FONT_SIZE = self.font_size_var.get()
        fv.FILL   = s["fills"]

    def _redraw(self):
        self._redraw_job = None
        if self._drawing or self._pan is not None:
            # уже рисуем (вложенное событие) или пользователь тянет график —
            # перерисуем, когда это закончится
            self._redraw_wanted = True
            return
        self._drawing = True
        try:
            try:
                settings = self._collect()
            except ValueError as ex:
                self._set_status(str(ex), ERR_COLOR)
                if self._preview is not None:
                    self._preview.hide()
                self._zoom = None
                return
            self._apply_to_engine(settings)

            fv.SYMBOLIC_MODE = 'cached'
            try:
                result = fv.plot_function(self.fig)
            except Exception:
                log_exception("plot_function")
                self._set_status(T("Plot error — previous graph restored (details: {log})", log=ERROR_LOG),
                                 ERR_COLOR)
                self._restore_last_good()
                return
            self._last_good = settings

            errors = result.get('errors', {}) or {}
            for idx, r in enumerate(self.func_rows):
                if idx in errors and not r.is_empty():
                    r.editor.set_error(he_display(tr_err(errors[idx])))
            # Синхронная отрисовка: новый кадр готов сразу, и предпросмотр
            # (сдвинутый/масштабированный старый кадр) можно убрать без «моргания».
            try:
                self.canvas.draw()
            except Exception:
                log_exception("canvas.draw")
                self.canvas.draw_idle()
            self._zoom = None
            if self._preview is not None:
                self._preview.hide()

            if result.get('pending'):
                self._submit_jobs(fv.take_pending_jobs())
                self._set_status(T("Refining labels (symbolic analysis)…"))
            elif errors or self._incomplete_rows:
                self._set_status(T("Some functions are incomplete or invalid — hover the red field"),
                                 ERR_COLOR)
            else:
                self._set_status(T("Ready") + self._visible_range_note(result.get('ax')))
        finally:
            self._drawing = False
            if self._redraw_wanted and self._pan is None:
                self._redraw_wanted = False
                self.schedule_redraw()

    def _visible_range_note(self, ax):
        """«· visible x −8.33…8.33» — когда холст шире/выше окна и видно больше, чем задано."""
        try:
            lims = self._current_limits()
            if ax is None or lims is None:
                return ""
            xl, xr = ax.get_xlim()
            yb, yt = ax.get_ylim()
            def g(v):
                return f"{v:.4g}".replace("-", "\u2212")
            if abs(xr - xl - (lims[1] - lims[0])) > 1e-9 * max(1.0, abs(xr - xl)):
                return f"  ·  {T('visible x')}: {g(xl)} … {g(xr)}"
            if abs(yt - yb - (lims[3] - lims[2])) > 1e-9 * max(1.0, abs(yt - yb)):
                return f"  ·  {T('visible y')}: {g(yb)} … {g(yt)}"
        except Exception:
            pass
        return ""

    def _restore_last_good(self):
        """После сбоя построения возвращаем на холст последний удачный график."""
        if not self._last_good:
            return
        try:
            self._apply_to_engine(self._last_good)
            fv.plot_function(self.fig)
            self.canvas.draw_idle()
        except Exception:
            log_exception("restore_last_good")

    # ── фоновый поток для sympy ──────────────────────────────
    def _start_worker_thread(self):
        t = threading.Thread(target=self._worker_loop, name="symbolic-worker", daemon=True)
        self._worker = t
        t.start()

    def _submit_jobs(self, jobs):
        """Ставит пачку отложенных заданий движка в очередь (новые — первыми)."""
        with self._jobs_lock:
            jobs = [(k, fn) for k, fn in jobs if k not in self._running_keys]
            if not jobs:
                return False
            self._batch_seq += 1
            self._job_batches.append((self._batch_seq, jobs))
            self._jobs_event.set()
        return True

    def _worker_loop(self):
        me = threading.current_thread()
        while True:
            self._jobs_event.wait()
            if self._worker is not me:          # нас заменил сторож — выходим
                return
            with self._jobs_lock:
                if not self._job_batches:
                    self._jobs_event.clear()
                    continue
                seq, jobs = self._job_batches.pop()   # самая свежая пачка — первой
                keys = [k for k, _ in jobs]
                self._running_keys.update(keys)
                self._active_batch = (seq, keys, time.time())
            try:
                fv.run_jobs(jobs)
            except Exception:
                log_exception("symbolic worker")
            finally:
                with self._jobs_lock:
                    self._running_keys.difference_update(keys)
                    if self._active_batch is not None and self._active_batch[0] == seq:
                        self._active_batch = None
                self._worker_queue.put(("done", seq))
            if self._worker is not me:
                return

    def _poll_worker(self):
        """Главный поток: результаты рабочего потока + сторож по времени."""
        got = False
        try:
            while True:
                self._worker_queue.get_nowait()
                got = True
        except queue.Empty:
            pass
        if got:
            self.schedule_redraw()

        # Сторож: sympy иногда «уходит в себя» на минуты (simplify громоздких
        # радикалов и т.п.). Тогда помечаем невычисленные ключи пачки как
        # неудачные (движок оставит численные подписи и не поставит их снова),
        # запускаем новый рабочий поток, а зависший оставляем доживать daemon'ом.
        with self._jobs_lock:
            ab = self._active_batch
        if ab is not None and time.time() - ab[2] > SYMBOLIC_TIMEOUT_S:
            seq, keys, _ = ab
            try:
                fv.mark_jobs_failed(keys, TimeoutError("symbolic analysis timed out"))
            except Exception:
                pass
            with self._jobs_lock:
                self._running_keys.difference_update(keys)
                self._active_batch = None
            self._start_worker_thread()
            self._set_status(T("Symbolic analysis timed out — some labels stay numeric"), ERR_COLOR)
            self.schedule_redraw()

        self._poll_job = self.after(150, self._poll_worker)

    def _shrink_canvas_request(self):
        try:
            w = self.canvas.get_tk_widget()
            small = (_px(320), _px(240))
            if (w.winfo_reqwidth(), w.winfo_reqheight()) != small:
                w.configure(width=small[0], height=small[1])
        except Exception:
            pass

    def _on_canvas_resize(self, _event):
        self._shrink_canvas_request()
        # Пересчитать поля фигуры под новый размер (с задержкой)
        if self._tl_job is not None:
            try:
                self.after_cancel(self._tl_job)
            except Exception:
                pass
        self._tl_job = self.after(200, self._relayout)

    def _relayout(self):
        """После изменения размера холста — полная перерисовка: штриховка и
        смещения подписей считаются от размера осей в пикселях."""
        self._tl_job = None
        try:
            if self._preview is not None:
                self._preview.hide()
        except Exception:
            pass
        self.schedule_redraw(delay=1)

    # ══════════════════════════════════════════════════════════
    #  МЫШЬ НА ГРАФИКЕ: зум колесом, панорамирование перетаскиванием
    # ══════════════════════════════════════════════════════════
    def _connect_graph_events(self):
        c = self.canvas
        c.mpl_connect('scroll_event', self._on_graph_scroll)
        c.mpl_connect('button_press_event', self._on_graph_press)
        c.mpl_connect('motion_notify_event', self._on_graph_motion)
        c.mpl_connect('button_release_event', self._on_graph_release)

    def _graph_hit_any(self, event):
        """Есть ли под курсором интерактивный объект движка (подпись, метка оси)."""
        try:
            for d in fv._ACTIVE_DRAGGABLES:
                contains, _ = d.ann.contains(event)
                if contains:
                    return True
            m = fv._active_free_text_manager
            if m is not None and m._find_hit(event) is not None:
                return True
            a = fv._active_axis_label_manager
            if a is not None and a._hit(event) is not None:
                return True
        except Exception:
            pass
        return False

    def _current_limits(self):
        try:
            return (fv.parse_number(self.xlim_l.get()), fv.parse_number(self.xlim_r.get()),
                    fv.parse_number(self.ylim_b.get()), fv.parse_number(self.ylim_t.get()))
        except Exception:
            return None

    def _set_limits(self, xl, xr, yb, yt, delay=None, immediate=False):
        def fmt(v):
            # 10 значащих цифр: обратное чтение из поля не теряет точность
            # при зуме/панорамировании вдали от начала координат
            s = f"{v:.10g}"
            return s
        self._loading = True      # одна перерисовка вместо четырёх
        try:
            self.xlim_l.var.set(fmt(xl)); self.xlim_r.var.set(fmt(xr))
            self.ylim_b.var.set(fmt(yb)); self.ylim_t.var.set(fmt(yt))
        finally:
            self._loading = False
        if immediate:
            self.redraw_now()
        else:
            self.schedule_redraw(delay=delay)

    def _on_graph_scroll(self, event):
        if event.inaxes is None or event.xdata is None:
            return
        try:
            m = fv._active_free_text_manager
            if m is not None and m._find_hit(event) is not None:
                return        # колесо над свободной подписью — поворот (движок)
        except Exception:
            pass
        lims = self._current_limits()
        if lims is None:
            return
        xl, xr, yb, yt = lims
        factor = 1 / 1.2 if event.button == 'up' else 1.2
        cx, cy = event.xdata, event.ydata
        if (xr - xl) * factor < 1e-6 or (xr - xl) * factor > 1e6:
            return
        nxl = cx - (cx - xl) * factor
        nxr = cx + (xr - cx) * factor
        nyb = cy - (cy - yb) * factor
        nyt = cy + (yt - cy) * factor
        # Не отдаляем дальше, чем позволяет шаг сетки (иначе _collect откажет,
        # а поля «уедут» от реального окна)
        try:
            xg = fv.parse_number(self.xgrid_e.get()); yg = fv.parse_number(self.ygrid_e.get())
            if (nxr - nxl) / xg > MAX_GRID_LINES or (nyt - nyb) / yg > MAX_GRID_LINES:
                self._set_status(T("Zoom-out limit for the current grid step — increase Step X / Step Y"),
                                 ERR_COLOR)
                return
        except Exception:
            pass
        # Мгновенный предпросмотр: масштабируем последний кадр вокруг курсора;
        # настоящая перерисовка придёт через ZOOM_DELAY_MS и заменит его.
        self._zoom_preview(event, factor)
        self._set_limits(nxl, nxr, nyb, nyt, delay=ZOOM_DELAY_MS)

    def _zoom_preview(self, event, factor):
        pv = self._preview
        if pv is None:
            return
        try:
            if self._zoom is None or not pv.active:
                if not pv.snapshot():
                    self._zoom = None
                    return
                W, H = pv.size
                self._zoom = {"px": float(event.x), "py": float(H - event.y), "scale": 1.0}
            z = self._zoom
            z["scale"] *= factor
            pv.show_zoom(z["px"], z["py"], z["scale"])
        except Exception:
            self._zoom = None

    def _on_graph_press(self, event):
        self._pan = None
        if event.button != 1 or event.inaxes is None or event.dblclick:
            return
        if self._graph_hit_any(event):
            return
        lims = self._current_limits()
        if lims is None:
            return
        self._pan = {"x0": event.xdata, "y0": event.ydata, "px": event.x, "py": event.y,
                     "lims": lims, "moved": False, "ax": event.inaxes, "preview": False}

    def _on_graph_motion(self, event):
        p = self._pan
        if p is None or event.x is None:
            return
        if not p["moved"] and abs(event.x - p["px"]) < 4 and abs(event.y - p["py"]) < 4:
            return
        p["moved"] = True
        ax = p["ax"]
        # переводим сдвиг в пикселях в единицы данных (по масштабу осей на момент захвата)
        inv = ax.transData.inverted()
        x1, y1 = inv.transform((event.x, event.y))
        x0, y0 = inv.transform((p["px"], p["py"]))
        dx, dy = x1 - x0, y1 - y0
        xl, xr, yb, yt = p["lims"]
        p["cur"] = (xl - dx, xr - dx, yb - dy, yt - dy)
        # Предпросмотр: сдвигаем снимок кадра целиком (оси, сетка, подписи —
        # всё едет вместе), никакой перерисовки до отпускания кнопки
        pv = self._preview
        if pv is not None:
            if not p["preview"]:
                p["preview"] = pv.snapshot()
                if not p["preview"]:
                    self._zoom = None
            if p["preview"]:
                pv.show_shift(event.x - p["px"], -(event.y - p["py"]))
                return
        # запасной вариант (нет снимка): дешёвый сдвиг осей
        ax.set_xlim(xl - dx, xr - dx)
        ax.set_ylim(yb - dy, yt - dy)
        self.canvas.draw_idle()

    def _on_graph_release(self, event):
        p = self._pan
        self._pan = None
        if p is None or not p["moved"]:
            if self._redraw_wanted:
                self._redraw_wanted = False
                self.schedule_redraw()
            return
        cur = p.get("cur")
        if cur is None:
            if self._preview is not None:
                self._preview.hide()
            return
        # Сразу строим новый кадр; предпросмотр остаётся на экране, пока он
        # не готов, и убирается внутри _redraw после синхронного draw()
        self._set_limits(*cur, immediate=True)

    def _reset_view(self):
        self._set_limits(-5, 5, -5, 5)

    # ══════════════════════════════════════════════════════════
    #  СОХРАНЕНИЕ
    # ══════════════════════════════════════════════════════════
    def _save_image(self):
        path = filedialog.asksaveasfilename(
            title=he_display(T("Save graph image")),
            defaultextension=".png",
            filetypes=[(he_display(T("PNG image")), "*.png"), (he_display(T("SVG vector")), "*.svg"),
                       ("PDF", "*.pdf"), (he_display(T("All files")), "*.*")],
            initialfile="graph.png")
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            self._set_status(T("Saved: {path}", path=path), BTN_ADD)
        except Exception as ex:
            messagebox.showerror(he_display(T("Save error")), str(ex))

    # ── проект (все настройки + подписи) ─────────────────────
    def _project_dict(self):
        return {
            "version": 2,
            "functions": [r.to_dict() for r in self.func_rows],
            "view": {"x": [self.xlim_l.get(), self.xlim_r.get()],
                     "y": [self.ylim_b.get(), self.ylim_t.get()]},
            "grid": {"show": self.grid_var.get(), "x": self.xgrid_e.get(), "y": self.ygrid_e.get()},
            "display": {
                "asymptotes": self.v_asimp.get(), "holes": self.v_disc.get(),
                "extrema": self.v_extr.get(), "x_intercepts": self.v_xtag.get(),
                "y_intercepts": self.v_ytag.get(), "intersections": self.v_inter.get(),
                "show_values": self.v_show_values.get(), "hide_x": self.v_xhide.get(),
                "hide_y": self.v_yhide.get(), "font_size": self.font_size_var.get(),
            },
            "fills": [f.to_dict() for f in self.fill_rows],
            "free_texts": [dict(t) for t in fv.FREE_TEXTS],
            "axis_labels": fv.AXIS_LABELS,
            "annotation_offsets": [[list(k), list(v)] for k, v in fv.ANNOTATION_OFFSETS.items()],
        }

    def _save_project(self):
        path = filedialog.asksaveasfilename(
            title=he_display(T("Save project")), defaultextension=self.PROJECT_EXT,
            filetypes=[(he_display(T("Function Visualizer project")), "*" + self.PROJECT_EXT), ("JSON", "*.json")],
            initialfile="graph" + self.PROJECT_EXT)
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._project_dict(), f, ensure_ascii=False, indent=2)
            self._set_status(T("Project saved: {path}", path=path), BTN_ADD)
        except Exception as ex:
            messagebox.showerror(he_display(T("Save error")), str(ex))

    def _open_project(self):
        path = filedialog.askopenfilename(
            title=he_display(T("Open project")),
            filetypes=[(he_display(T("Function Visualizer project")), "*" + self.PROJECT_EXT),
                       ("JSON", "*.json"), (he_display(T("All files")), "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.load_project(data)
            self._set_status(T("Project loaded: {path}", path=path), BTN_ADD)
        except Exception as ex:
            messagebox.showerror(he_display(T("Open error")), he_display(T("Cannot open project:\n{err}", err=ex)))

    def load_project(self, data):
        self._loading = True
        try:
            for r in list(self.func_rows):
                r.destroy()
            self.func_rows = []
            for r in list(self.fill_rows):
                r.destroy()
            self.fill_rows = []

            for fd in data.get("functions", []) or [{}]:
                row = self._add_func()
                row.from_dict(fd)
            if not self.func_rows:
                self._add_func()
            for fd in data.get("fills", []):
                self._add_fill().from_dict(fd)

            view = data.get("view", {})
            x = view.get("x", ["-5", "5"]); y = view.get("y", ["-5", "5"])
            self.xlim_l.var.set(str(x[0])); self.xlim_r.var.set(str(x[1]))
            self.ylim_b.var.set(str(y[0])); self.ylim_t.var.set(str(y[1]))
            grid = data.get("grid", {})
            self.grid_var.set(int(grid.get("show", 1)))
            self.xgrid_e.var.set(str(grid.get("x", "1"))); self.ygrid_e.var.set(str(grid.get("y", "1")))
            d = data.get("display", {})
            self.v_asimp.set(int(d.get("asymptotes", 1))); self.v_disc.set(int(d.get("holes", 1)))
            self.v_extr.set(int(d.get("extrema", 1))); self.v_xtag.set(int(d.get("x_intercepts", 1)))
            self.v_ytag.set(int(d.get("y_intercepts", 1))); self.v_inter.set(int(d.get("intersections", 1)))
            self.v_show_values.set(int(d.get("show_values", 1)))
            self.v_xhide.set(int(d.get("hide_x", 0))); self.v_yhide.set(int(d.get("hide_y", 0)))
            self.font_size_var.set(int(d.get("font_size", 10)))

            fv.FREE_TEXTS.clear()
            for t in data.get("free_texts", []):
                if isinstance(t, dict) and "text" in t:
                    fv.FREE_TEXTS.append(dict(t))
            al = data.get("axis_labels", {}) or {}
            for key in ("x", "y"):
                st = fv.AXIS_LABELS[key]
                st.update({'dx': 0.0, 'dy': 0.0, 'color': None, 'fontsize': None})
                src = al.get(key)
                if isinstance(src, dict):
                    try:
                        st['dx'] = float(src.get('dx', 0.0) or 0.0)
                        st['dy'] = float(src.get('dy', 0.0) or 0.0)
                    except Exception:
                        pass
                    if src.get('color'):
                        st['color'] = str(src['color'])
            fv.reset_annotation_offsets()
            for item in data.get("annotation_offsets", []):
                try:
                    k, v = item
                    fv.ANNOTATION_OFFSETS[tuple(k)] = tuple(v)
                except Exception:
                    pass
            self._set_active(self.func_rows[0])
        finally:
            self._loading = False
        self.schedule_redraw()

    def _on_close(self):
        for job in (self._redraw_job, getattr(self, "_poll_job", None), self._tl_job):
            try:
                if job is not None:
                    self.after_cancel(job)
            except Exception:
                pass
        self.destroy()


# ═════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════
#  ВЫБОР ЯЗЫКА ПРИ ЗАПУСКЕ
# ═════════════════════════════════════════════════════════════

# (код, название на самом языке, клавиша)
LANGUAGES = [("en", "English", "1"), ("he", "עברית", "2"), ("ru", "Русский", "3")]
_CHOOSER_PROMPTS = [("en", "Choose language"), ("he", "בחר שפה"), ("ru", "Выберите язык")]


def _lang_font(code, size, bold=False):
    fam = {"en": UI_FONT, "he": HE_FONT, "ru": RU_FONT}[code]
    return (fam, size + (1 if code == "he" else 0), "bold") if bold else (fam, size + (1 if code == "he" else 0))


class LanguageChooser(tk.Tk):
    """
    Стартовое окно: на каком языке открыть программу. Три кнопки — каждая
    своим шрифтом (Schola / David / Century Schoolbook). Клавиши 1, 2, 3
    выбирают язык, Esc закрывает программу. Результат — в .choice.
    """

    def __init__(self):
        super().__init__()
        _resolve_ui_font(self)
        _init_ui_scale(self)
        self.choice = None
        self.title("Function Visualizer")
        self.configure(bg=APP_BG)
        self.resizable(False, False)
        try:
            self.iconbitmap(_resource_path("icon.ico"))
        except Exception:
            pass

        outer = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        outer.pack(padx=_px(18), pady=_px(18))
        box = tk.Frame(outer, bg=CARD_BG, padx=_px(28), pady=_px(18))
        box.pack()
        tk.Label(box, text="Function Visualizer", bg=CARD_BG, fg=TEXT,
                 font=(UI_FONT, 15, "bold")).pack(pady=(0, _px(2)))
        tk.Label(box, text="ariadna", bg=CARD_BG, fg=ACCENT,
                 font=(UI_FONT, 10, "bold")).pack(pady=(0, _px(10)))
        for code, prompt in _CHOOSER_PROMPTS:
            tk.Label(box, text=he_display(prompt), bg=CARD_BG, fg=SUBTEXT,
                     font=_lang_font(code, 9)).pack()
        tk.Frame(box, bg=BORDER, height=1).pack(fill="x", pady=_px(10))
        self._buttons = {}
        for code, name, key in LANGUAGES:
            b = tk.Button(box, text=he_display(name), font=_lang_font(code, 12, bold=True),
                          bg=BTN_BLUE, fg="white", activebackground=ACCENT, activeforeground="white",
                          relief="flat", bd=0, cursor="hand2", width=14, pady=_px(6),
                          command=lambda c=code: self._pick(c))
            b.pack(fill="x", pady=_px(3))
            self._buttons[code] = b
            self.bind(key, lambda _e, c=code: self._pick(c))
            self.bind(f"<KP_{key}>", lambda _e, c=code: self._pick(c))
        self.bind("<Escape>", lambda _e: self.destroy())
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        # По центру экрана
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        x = max(0, (self.winfo_screenwidth() - w) // 2)
        y = max(0, (self.winfo_screenheight() - h) // 2 - _px(40))
        self.geometry(f"+{x}+{y}")
        self.lift()
        self.focus_force()

    def _pick(self, code):
        self.choice = code
        self.destroy()


def choose_language():
    """Показывает окно выбора языка; возвращает код языка или None (закрыли окно)."""
    win = LanguageChooser()
    win.mainloop()
    return win.choice


def main(lang=None):
    """
    Точка входа. Без аргумента — сначала окно выбора языка (единый exe);
    lang='en'/'he'/'ru' — сразу на этом языке (app_he.py, app_ru.py).
    """
    global LANG
    if lang is None:
        lang = choose_language()
        if lang is None:
            return
    LANG = lang
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
