"""
Умный визуализатор функций
Зависимости: matplotlib, numpy, sympy, scipy
Установка: pip install matplotlib numpy sympy scipy

Движок построения графика. Используется двумя способами:
  • standalone — `python function_visualizer.py` (plot_function() без аргументов
    открывает окно matplotlib);
  • из GUI (app.py) — plot_function(fig) рисует в переданную Figure, которая
    встроена в окно Tk (FigureCanvasTkAgg) и перерисовывается «вживую».

Тяжёлые символьные вычисления sympy (singularities / limit / solveset /
simplify) проходят через кэш sym_cached(). В режиме SYMBOLIC_MODE='cached'
(GUI) незакэшированный запрос НЕ считается на месте, а откладывается
(SymbolicPending) — график строится сразу по численным методам, а GUI
досчитывает отложенные задания в фоновом потоке (run_jobs) и перерисовывает.
"""

import logging
import math
import re
import functools
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sympy import (
    sympify, symbols, lambdify, diff, limit, oo, nan, zoo,
    singularities, solveset, pi, E, Rational, Integer, S, Abs, sign,
    log, exp, nsimplify, simplify, count_ops, expand_log, Function,
    Add, Mul, Pow,
)
from sympy.sets.sets import FiniteSet, Union, Complement, Intersection
from sympy.sets.fancysets import ImageSet
from sympy.sets.conditionset import ConditionSet
from scipy.signal import argrelextrema
from scipy.optimize import brentq, minimize_scalar

# Шрифт графика: список с запасными вариантами (Calibri есть только на
# Windows; Carlito — его метрический аналог в Linux; DejaVu Sans — везде).
# matplotlib принимает список семейств и берёт первый доступный.
# ── Шрифт графика и формат подписей ───────────────────────────
# GRAPH_FONT — пресет шрифта (см. FONT_PRESETS): 'schola' — TeX Gyre Schola
# (свободный клон Century Schoolbook, файлы лежат в папке fonts/ и регистрируются
# приложением при старте), 'century' — Century Schoolbook, 'times' — Times New
# Roman + STIX, 'latex' — Computer Modern, 'serif' — DejaVu Serif, 'sans' — Calibri.
# MATHTEXT_LABELS — подписи точек/делений через mathtext matplotlib:
# \sqrt{5}, \frac{\pi}{2} рисуются как настоящие корни и дроби; False —
# обычные unicode-строки (√5, π/2).
GRAPH_FONT = 'schola'
MATHTEXT_LABELS = True

FONT_PRESETS = {
    # имя: (список семейств для обычного текста, mathtext.fontset)
    # Два имени Schola: «TeX Gyre Schola» (typographic family, fontconfig/Linux)
    # и «TeXGyreSchola» (legacy family из name-таблицы, Windows GDI).
    'schola':  (['TeX Gyre Schola', 'TeXGyreSchola', 'Century Schoolbook', 'Times New Roman',
                 'Liberation Serif', 'DejaVu Serif', 'serif'], 'custom'),
    'times':   (['Times New Roman', 'Cambria', 'Liberation Serif', 'DejaVu Serif', 'serif'], 'stix'),
    'latex':   (['CMU Serif', 'Latin Modern Roman', 'Times New Roman', 'Liberation Serif',
                 'DejaVu Serif', 'serif'], 'cm'),
    'century': (['Century Schoolbook', 'Century', 'TeX Gyre Schola', 'Times New Roman',
                 'Cambria', 'Liberation Serif', 'DejaVu Serif', 'serif'], 'custom'),
    'serif':   (['DejaVu Serif', 'serif'], 'dejavuserif'),
    'sans':    (['Calibri', 'Carlito', 'DejaVu Sans', 'sans-serif'], 'dejavusans'),
}


# ── Локализация подписей интерфейса движка (меню/диалоги свободных подписей)
# Приложение кладёт сюда словарь {английская строка: перевод}; tr() возвращает
# перевод или исходную строку. EXTRA_FONT_FAMILIES — дополнительные семейства
# (например, шрифт с ивритом) в конец списка font.family: matplotlib берёт из
# них глифы, которых нет в основном шрифте. BIDI_SIMPLE — простой bidi для
# свободных подписей (matplotlib не переставляет RTL-текст): зеркалим
# ивритские фрагменты.
UI_TRANSLATIONS = {}
EXTRA_FONT_FAMILIES = []
BIDI_SIMPLE = False
# Приложение может положить сюда функцию «строка → как показывать» (для
# ивритской версии: визуальный порядок под Tk/Windows); tr() применяет её
# к переведённым подписям меню и диалогов.
UI_DISPLAY = None
_HEB_RUN_RE = re.compile(r'[\u0590-\u05FF][\u0590-\u05FF\s]*[\u0590-\u05FF]|[\u0590-\u05FF]')


def tr(text, **fmt):
    out = UI_TRANSLATIONS.get(text, text)
    if fmt:
        out = out.format(**fmt)
    return UI_DISPLAY(out) if UI_DISPLAY else out


# ── Упрощённый алгоритм Unicode Bidi (UAX#9) ───────────────────────────────
# matplotlib и Tk на Linux вообще не переставляют RTL-текст, а Tk на Windows
# (GDI) переставляет его, считая абзац направленным слева направо. Поэтому
# визуальный порядок символов считаем сами: правила W1–W7 (числа), N1–N2
# (нейтральные знаки — пробелы, двоеточия, тире, многоточия, скобки берут
# направление окружения или абзаца), уровни, перестановка L2, зеркальные
# скобки. Явные вложения (LRE/RLE/…) и маркеры направления удаляются.
import unicodedata as _ud

_BIDI_MIRROR = {'(': ')', ')': '(', '[': ']', ']': '[', '{': '}', '}': '{', '<': '>', '>': '<',
                '«': '»', '»': '«', '‹': '›', '›': '‹', '⟨': '⟩', '⟩': '⟨', '≤': '≥', '≥': '≤'}
_BIDI_STRIP = {'LRE', 'RLE', 'PDF', 'LRO', 'RLO', 'LRI', 'RLI', 'FSI', 'PDI', 'BN'}
_BIDI_STRIP_CHARS = {'\u200e', '\u200f', '\u061c'}
_BIDI_NI = {'B', 'S', 'WS', 'ON'}


def _bidi_class(c):
    b = _ud.bidirectional(c) or 'ON'
    if b == 'AL':
        return 'R'
    if b == 'AN':
        return 'EN'
    return b


def bidi_visual(text, base='R'):
    """
    Строка в визуальном порядке (слева направо) для абзаца с направлением
    base: 'R' — иврит (по умолчанию), 'L' — латиница. Многострочный текст
    обрабатывается построчно. Для строки без RTL-символов при base='L'
    возвращается она же.
    """
    if not text:
        return text
    if '\n' in text:
        return '\n'.join(bidi_visual(line, base) for line in text.split('\n'))
    chars = [c for c in text if c not in _BIDI_STRIP_CHARS and _ud.bidirectional(c) not in _BIDI_STRIP]
    if not chars:
        return ''
    n = len(chars)
    cls = [_bidi_class(c) for c in chars]
    # W1: диакритика наследует класс предыдущего символа
    for i in range(n):
        if cls[i] == 'NSM':
            cls[i] = cls[i - 1] if i else base
    # W4: одиночный разделитель между двумя цифрами — часть числа (1.5, 1:2)
    for i in range(1, n - 1):
        if cls[i] in ('ES', 'CS') and cls[i - 1] == 'EN' and cls[i + 1] == 'EN':
            cls[i] = 'EN'
    # W5: терминаторы (%, °, $) рядом с числом — часть числа
    i = 0
    while i < n:
        if cls[i] == 'ET':
            j = i
            while j < n and cls[j] == 'ET':
                j += 1
            if (i > 0 and cls[i - 1] == 'EN') or (j < n and cls[j] == 'EN'):
                for k in range(i, j):
                    cls[k] = 'EN'
            i = j
        else:
            i += 1
    # W6: оставшиеся разделители — нейтральные
    for i in range(n):
        if cls[i] in ('ES', 'ET', 'CS'):
            cls[i] = 'ON'
    # W7: число после латиницы ведёт себя как латиница
    last_strong = base
    for i in range(n):
        if cls[i] in ('L', 'R'):
            last_strong = cls[i]
        elif cls[i] == 'EN' and last_strong == 'L':
            cls[i] = 'L'
    # N1/N2: нейтральные между одинаковыми направлениями берут его, иначе — направление абзаца
    def strong(c):
        return 'R' if c == 'EN' else c
    i = 0
    while i < n:
        if cls[i] in _BIDI_NI:
            j = i
            while j < n and cls[j] in _BIDI_NI:
                j += 1
            before = strong(cls[i - 1]) if i > 0 else base
            after = strong(cls[j]) if j < n else base
            d = before if before == after else base
            for k in range(i, j):
                cls[k] = d
            i = j
        else:
            i += 1
    # Уровни (I1/I2)
    e = 1 if base == 'R' else 0
    lev = []
    for c in cls:
        if e == 0:
            lev.append(0 if c == 'L' else 1 if c == 'R' else 2)
        else:
            lev.append(1 if c == 'R' else 2)
    # Зеркальные скобки на нечётных уровнях
    for i in range(n):
        if lev[i] % 2 == 1:
            chars[i] = _BIDI_MIRROR.get(chars[i], chars[i])
    # L2: от наибольшего уровня до наименьшего нечётного переворачиваем отрезки
    odd = [l for l in lev if l % 2 == 1]
    if odd:
        for k in range(max(lev), min(odd) - 1, -1):
            i = 0
            while i < n:
                if lev[i] >= k:
                    j = i
                    while j < n and lev[j] >= k:
                        j += 1
                    chars[i:j] = chars[i:j][::-1]
                    lev[i:j] = lev[i:j][::-1]
                    i = j
                else:
                    i += 1
    return ''.join(chars)


def bidi_display(text):
    """Визуальный порядок для свободной подписи с ивритом (matplotlib не
    переставляет RTL-текст сам); без BIDI_SIMPLE или без иврита — как есть."""
    if not text or not BIDI_SIMPLE or not _HEB_RUN_RE.search(text):
        return text
    return bidi_visual(text, 'R')


def _font_available(name):
    try:
        from matplotlib import font_manager
        font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
        return True
    except Exception:
        return False


def apply_font_preset(name=None):
    """Применяет пресет шрифта к rcParams (вызывается при смене настройки)."""
    global GRAPH_FONT
    name = name or GRAPH_FONT
    families, fontset = FONT_PRESETS.get(name, FONT_PRESETS['schola'])
    GRAPH_FONT = name if name in FONT_PRESETS else 'schola'
    fam_list = list(families)
    if EXTRA_FONT_FAMILIES:
        # запасные семейства (иврит и т.п.) — перед generic 'serif'
        generic = [f for f in fam_list if f in ('serif', 'sans-serif')]
        fam_list = [f for f in fam_list if f not in generic] + \
                   [f for f in EXTRA_FONT_FAMILIES if f not in fam_list] + generic
    plt.rcParams['font.family'] = fam_list
    if fontset == 'custom':
        # «Свой» шрифт для математики: берём первое установленное семейство
        # (Century Schoolbook; на машине без него — Schola/Times); недостающие
        # глифы (√, π…) подставляет STIX (mathtext.fallback).
        fam = next((f for f in families if f not in ('serif', 'sans-serif')
                    and _font_available(f)), None)
        if fam is None or fam not in families[:3]:
            # нет ни Schola, ни Century Schoolbook — Times-подобный STIX
            fontset = 'stix'
        else:
            plt.rcParams['mathtext.fontset'] = 'custom'
            plt.rcParams['mathtext.rm'] = fam
            plt.rcParams['mathtext.it'] = f'{fam}:italic'
            plt.rcParams['mathtext.bf'] = f'{fam}:bold'
            try:
                plt.rcParams['mathtext.fallback'] = 'stix'
            except Exception:
                pass
    if fontset != 'custom':
        plt.rcParams['mathtext.fontset'] = fontset


apply_font_preset(GRAPH_FONT)
# Не засорять консоль предупреждениями «findfont: Font family not found».
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ──────────────────────────────────────────────
#  ПАРАМЕТРЫ — меняй только эту секцию
# ──────────────────────────────────────────────

FUNCS  = [
    "sin(x) / x",
    # "x**2",       # добавь ещё функции сюда
]

# Область определения для каждой функции: список (from, to) той же длины,
# что FUNCS. По умолчанию (или если короче) — (-inf, inf), т.е. вся область.
# Функция строится только на пересечении [from, to] с view window.
FUNC_DOMAINS = []

GRID   = 1    # 1 — показывать сетку, 0 — нет
X_GRID = 1    # шаг сетки по оси X
Y_GRID = 1    # шаг сетки по оси Y

X_LIM_L = -5  # левый предел по X
X_LIM_R =  5  # правый предел по X
Y_LIM_B =  -5  # нижний предел по Y
Y_LIM_T =   5  # верхний предел по Y

ASIMP  = 1    # 1 — показывать асимптоты (гориз. и верт.),  0 — нет
DISC   = 1    # 1 — показывать устранимые разрывы (holes),   0 — нет
EXTR   = 1    # 1 — показывать экстремумы,             0 — нет
X_TAG  = 1    # 1 — показывать пересечения с осью X,   0 — нет
Y_TAG  = 1    # 1 — показывать пересечения с осью Y,   0 — нет
INTER  = 1    # 1 — показывать пересечения между функциями, 0 — нет
SHOW_VALUES = 1  # 1 — показывать координаты точек,    0 — только точки без подписей

X_HIDE = 0    # 1 — скрыть числа делений на оси X,     0 — показывать
Y_HIDE = 0    # 1 — скрыть числа делений на оси Y,     0 — показывать

FONT_SIZE = 10  # размер текста на графике (точки/подписи/деления)

# Заливка площади
# Формат: (f1, f2, x_от, x_до, стиль)
#
# f1  — индекс первой функции (из FUNCS)
# f2  — индекс второй функции (из FUNCS), или "x" чтобы считать до оси X
# x_от, x_до — диапазон по X
# стиль — 0: штриховка 45°,  1: штриховка 135°,  2: точки
#
# Примеры:
#   (0, "x", -3, 3, 0)   — площадь между функцией 0 и осью X, штриховка 45°
#   (0, 1, -3, 3, 1)     — площадь между функцией 0 и функцией 1, штриховка 135°
FILL = [
    # (0, "x", -3, 3, 0),
]

# ──────────────────────────────────────────────
#  КОД — не трогай ниже этой строки
# ──────────────────────────────────────────────
def _preprocess_func_str(s):
    """
    Преобразует удобные пользователю записи логарифмов в форму sympy:
      log_2(x)   -> log(x, 2)
      log_10(x)  -> log(x, 10)
      log2(x)    -> log(x, 2)
      log10(x)   -> log(x, 10)
      lg(x)      -> log(x, 10)   (десятичный логарифм)
    Основание может быть числом или выражением (log_e, log_pi и т.п.).
    ln(x) и обычный log(x) (натуральный) не трогаются.
    """
    import re
    if not s:
        return s

    # log_<base>(...)  где base — число/идентификатор (e, pi, 2, 10, ...)
    # Заменяем на log(<...>, <base>). Берём сбалансированные скобки простым
    # нежадным подходом для одного уровня вложенности аргумента.
    def repl_underscore(m):
        base = m.group(1)
        return f'log__BASE__{base}__('

    # сначала пометим log_<base>(  -> спец-маркер с захватом аргумента ниже
    s = re.sub(r'log_([A-Za-z0-9.]+)\s*\(', repl_underscore, s)

    # log2( / log10(  -> тоже маркер
    s = re.sub(r'log2\s*\(',  'log__BASE__2__(',  s)
    s = re.sub(r'log10\s*\(', 'log__BASE__10__(', s)
    # lg( -> десятичный
    s = re.sub(r'(?<![A-Za-z])lg\s*\(', 'log__BASE__10__(', s)

    # Теперь раскрываем маркеры log__BASE__<b>__( ... ) -> log( ... , <b> )
    # с корректным учётом вложенных скобок.
    out = []
    i = 0
    marker = 'log__BASE__'
    while i < len(s):
        j = s.find(marker, i)
        if j == -1:
            out.append(s[i:])
            break
        out.append(s[i:j])
        # читаем основание до '__('
        k = j + len(marker)
        end_base = s.find('__(', k)
        base = s[k:end_base]
        # находим парную закрывающую скобку для '(' в позиции end_base+2
        open_paren = end_base + 2
        depth = 1
        p = open_paren + 1
        while p < len(s) and depth > 0:
            if s[p] == '(':
                depth += 1
            elif s[p] == ')':
                depth -= 1
            p += 1
        inner = s[open_paren + 1:p - 1]
        out.append(f'log(({inner}), {base})')
        i = p
    return ''.join(out)


def _parse_equation_input(s):
    """
    Классифицирует строку из поля функции:
      • 'y = ln(x)'          -> ('func', 'ln(x)')      обычная функция y=f(x)
      • 'ln(x)'              -> ('func', 'ln(x)')       без префикса — тоже функция
      • 'x = e'  / 'x = 2pi' -> ('vline', <число>)      вертикальная линия x=c
      • 'y = 3'              -> ('func', '3')            горизонталь как обычная функция
      • 'x**2+y**2=25'       -> ('implicit', ('x**2+y**2', '25'))   неявная кривая
      • 'y**2 = x'           -> ('implicit', ('y**2', 'x'))         (в правой части есть x → неявная)
    Правило: 'y=<выражение без y>' — обычная функция; 'x=<константа без x,y>' —
    вертикаль; всё остальное с '=' — неявная кривая F(x,y)=G(x,y).
    """
    if s is None:
        return ('func', '')
    raw = str(s).strip()

    import re
    def _has_var(expr, var):
        # есть ли переменная var (x или y) как отдельный идентификатор
        return re.search(r'(?<![A-Za-z_])' + var + r'(?![A-Za-z_])', expr) is not None

    if '=' not in raw:
        # Нет '=' — подразумеваем 'y = <выражение>'. Если в выражении есть y
        # (например 'cos(y)'), это уже неявное уравнение y = f(...,y);
        # иначе — обычная функция y=f(x).
        if _has_var(raw.lower(), 'y'):
            return ('implicit', ('y', raw))
        return ('func', raw)

    left, _, right = raw.partition('=')
    left_n = left.strip().lower()
    right = right.strip()

    # y = f(x): слева ровно 'y', справа НЕТ y -> обычная функция
    if left_n == 'y' and not _has_var(right.lower(), 'y'):
        return ('func', right)

    # x = c: слева ровно 'x', справа константа (нет ни x, ни y) -> вертикаль
    if left_n == 'x' and not _has_var(right.lower(), 'x') and not _has_var(right.lower(), 'y'):
        try:
            c = parse_number(right)
            return ('vline', c)
        except Exception:
            return ('vline', None)

    # Всё остальное с '=' — неявная кривая F(x,y) = G(x,y)
    return ('implicit', (left.strip(), right))



# ══════════════════════════════════════════════════════════════
#  КЭШ СИМВОЛЬНЫХ ВЫЧИСЛЕНИЙ И ОТЛОЖЕННЫЕ ЗАДАНИЯ
# ══════════════════════════════════════════════════════════════

# 'compute' — считать всё сразу (standalone, тесты);
# 'cached'  — поток GUI: незакэшированные символьные запросы откладываются,
#             график строится численно, задания досчитывает run_jobs().
SYMBOLIC_MODE = 'compute'


class SymbolicPending(Exception):
    """Символьный результат ещё не вычислен (отложен в очередь заданий)."""


_SYM_CACHE = {}          # key -> (ok: bool, value | exception)
_PENDING = {}            # key -> fn   (dict — дедупликация по ключу, порядок сохраняется)
_SYM_LOCK = threading.Lock()
_SYM_TLS = threading.local()        # force_compute=True внутри run_jobs
_DRAW_STATE = {'pending': False}    # было ли что-то отложено в текущем построении


@functools.lru_cache(maxsize=1024)
def _ekey(expr):
    """str(expr) для ключей кэша — печать sympy дорогая, запоминаем."""
    return str(expr)


def sym_cached(key, fn):
    """
    Возвращает закэшированный результат fn() по ключу key (хэшируемый кортеж
    строк). Если результата нет:
      • режим 'compute' (или вызов из run_jobs) — считает, кэширует
        (значение ИЛИ исключение) и возвращает / бросает;
      • режим 'cached' — ставит (key, fn) в очередь и бросает SymbolicPending.
    Исключение sympy (NotImplementedError и т.п.) тоже кэшируется, чтобы не
    повторять заведомо провальный расчёт при каждой перерисовке.
    """
    with _SYM_LOCK:
        hit = _SYM_CACHE.get(key)
    if hit is not None:
        ok, payload = hit
        if ok:
            return payload
        raise payload
    force = getattr(_SYM_TLS, 'force_compute', False)
    if SYMBOLIC_MODE == 'cached' and not force:
        with _SYM_LOCK:
            _PENDING.setdefault(key, fn)
            _DRAW_STATE['pending'] = True
        raise SymbolicPending(key)
    return _compute_job(key, fn)


def _compute_job(key, fn):
    try:
        val = fn()
    except SymbolicPending:
        raise
    except Exception as exc:
        with _SYM_LOCK:
            _SYM_CACHE[key] = (False, exc)
        raise
    with _SYM_LOCK:
        _SYM_CACHE[key] = (True, val)
    return val


def take_pending_jobs():
    """Забирает (и очищает) очередь отложенных заданий: список (key, fn)."""
    with _SYM_LOCK:
        jobs = list(_PENDING.items())
        _PENDING.clear()
    return jobs


def run_jobs(jobs):
    """
    Выполняет задания (key, fn) с семантикой sym_cached. Безопасно вызывать из
    рабочего потока: внутри принудительно включён режим вычисления, поэтому
    вложенные sym_cached() тоже считаются, а не откладываются снова.
    """
    prev = getattr(_SYM_TLS, 'force_compute', False)
    _SYM_TLS.force_compute = True
    try:
        for key, fn in jobs:
            with _SYM_LOCK:
                if key in _SYM_CACHE:
                    continue
            try:
                _compute_job(key, fn)
            except Exception:
                pass
    finally:
        _SYM_TLS.force_compute = prev


def mark_jobs_failed(keys, exc=None):
    """
    Помечает ещё не вычисленные ключи как неудачные (сторож GUI по времени):
    дальнейшие sym_cached() по ним сразу бросают исключение → численный
    fallback, и задания больше не ставятся в очередь. Уже готовые результаты
    не трогаем; если зависший поток всё же досчитает — значение перекроет метку.
    """
    if exc is None:
        exc = TimeoutError("symbolic analysis timed out")
    with _SYM_LOCK:
        for k in keys:
            _SYM_CACHE.setdefault(k, (False, exc))
            _PENDING.pop(k, None)


def clear_symbolic_cache():
    with _SYM_LOCK:
        _SYM_CACHE.clear()
        _PENDING.clear()
    _NUMPY_FUNC_CACHE.clear()
    _IMPLICIT_FUNC_CACHE.clear()
    _ENUM_CACHE.clear()
    _FINITE_FLOATS.clear()
    _ekey.cache_clear()


# ══════════════════════════════════════════════════════════════
#  РАЗБОР ВЫРАЖЕНИЙ И ЧИСЛЕННЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════

_X_SYM, _Y_SYM = symbols('x y')


class nroot(Function):
    """
    Корень n-й степени nroot(b, n) с вещественным значением для отрицательных
    b при нечётном n (как ∛(−8) = −2). Для чётного n — обычная степень 1/n;
    для нецелого n остаётся невычисленным (считается численно, см. _np_nroot).
    """
    nargs = 2

    @classmethod
    def eval(cls, b, n):
        if n.is_Integer and n > 0:
            if n.is_odd:
                return sign(b) * Abs(b) ** Rational(1, n)
            return b ** Rational(1, n)
        return None


def _np_nroot(b, n):
    b = np.asarray(b, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(all='ignore'):
        odd = np.mod(n, 2) == 1
        return np.where(odd,
                        np.sign(b) * np.power(np.abs(b), 1.0 / n),
                        np.power(b, 1.0 / n))


def _np_cot(t):
    return 1.0 / np.tan(t)


def _np_sec(t):
    return 1.0 / np.cos(t)


def _np_csc(t):
    return 1.0 / np.sin(t)


def _np_acot(t):
    return np.arctan(1.0 / t)


# Модули для lambdify: явные numpy-реализации + всё остальное из numpy.
_LAMBDIFY_MODULES = [
    {'sqrt': np.sqrt, 'log': np.log, 'ln': np.log,
     'exp': np.exp, 'sin': np.sin, 'cos': np.cos,
     'tan': np.tan, 'abs': np.abs, 'Abs': np.abs, 'pi': np.pi,
     'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
     'cot': _np_cot, 'sec': _np_sec, 'csc': _np_csc, 'acot': _np_acot,
     'sign': np.sign, 'nroot': _np_nroot, 'cbrt': np.cbrt},
    'numpy',
]


def _parse_transformations():
    from sympy.parsing.sympy_parser import (
        standard_transformations, implicit_multiplication_application,
        convert_xor)
    # Неявное умножение (2x, ex, 2pi) и ^ как степень.
    return standard_transformations + (implicit_multiplication_application,
                                       convert_xor)


def _parse_local_dict(with_y=False):
    """
    Словарь имён для parse_expr: 'e' — число Эйлера, 'pi' — π, 'x' (и 'y') —
    переменные; дополнительные имена функций (русские tg/ctg, arc-формы,
    nroot/cbrt/root).
    """
    from sympy import asin, acos, atan, acot, tan, cot, sinh, cosh, tanh
    d = {'e': E, 'pi': pi, 'x': _X_SYM,
         # **_kw: parse_expr(evaluate=False) передаёт evaluate=… в каждый вызов
         'nroot': nroot, 'cbrt': lambda b, **_kw: nroot(b, 3),
         'root': nroot,
         'arcsin': asin, 'arccos': acos, 'arctan': atan, 'arctg': atan,
         'arcctg': acot, 'tg': tan, 'ctg': cot, 'sh': sinh, 'ch': cosh,
         'th': tanh, 'sgn': sign}
    if with_y:
        d['y'] = _Y_SYM
    return d


def parse_exact(text):
    """
    Разбирает пользовательский текст константы (например 'x = 2pi' → '2pi')
    в ТОЧНОЕ выражение sympy без свободных символов (2π, √2/2, e, 3/4 …).
    Возвращает None, если разобрать не удалось или остались переменные.
    """
    if text is None:
        return None
    t = str(text).strip().lower().replace('π', 'pi').replace('∞', 'oo')
    if not t:
        return None
    try:
        from sympy.parsing.sympy_parser import parse_expr
        expr = parse_expr(_preprocess_func_str(t),
                          transformations=_parse_transformations(),
                          local_dict=_parse_local_dict())
        expr = sympify(expr)
        if expr.free_symbols:
            return None
        if expr.has(oo, -oo, zoo, nan):
            return None
        return expr
    except Exception:
        return None


def _unknown_names_msg(src, extra_symbols, local_dict):
    """
    Сообщение «unknown name: …». Неявное умножение режет незнакомое слово
    на буквы (foo → f·o·o), поэтому ищем в исходной строке целые
    идентификаторы, которых нет среди известных имён sympy / local_dict.
    """
    import re as _re
    import sympy as _sp
    known = set(local_dict) | {'x', 'y', 'e', 'pi', 'oo', 'inf', 'lg', 'log2', 'log10'}
    bad = []
    for ident in _re.findall(r'[A-Za-z_][A-Za-z_0-9]*', str(src)):
        low = ident.lower()
        if low in known or ident in known:
            continue
        if hasattr(_sp, ident) or hasattr(_sp, low):
            continue
        if ident not in bad:
            bad.append(ident)
    if not bad:
        bad = sorted(str(s) for s in extra_symbols)
    return "unknown name: " + ', '.join(bad)


_NUMPY_FUNC_CACHE = {}      # func_str -> (expr, expr_raw, f)
_IMPLICIT_FUNC_CACHE = {}   # (lhs, rhs) -> (H, h)
_FUNC_CACHE_LIMIT = 256


def build_numpy_func(func_str):
    """
    Строит (expr, expr_raw, f) для строки функции: sympy-выражение,
    его невычисленную форму (evaluate=False — для поиска особых точек
    вида (x²−1)/(x−1)) и numpy-функцию f(x). Результат мемоизируется по
    строке: в live-режиме функция вызывается при каждой перерисовке.
    Ошибки разбора НЕ кэшируются (бросаются вызывающему).
    """
    hit = _NUMPY_FUNC_CACHE.get(func_str)
    if hit is not None:
        return hit
    from sympy.parsing.sympy_parser import parse_expr
    src = _preprocess_func_str(func_str)
    x = _X_SYM
    transformations = _parse_transformations()
    local_dict = _parse_local_dict()

    expr_raw = parse_expr(src, evaluate=False,
                          transformations=transformations, local_dict=local_dict)
    expr = parse_expr(src, transformations=transformations,
                      local_dict=local_dict)
    expr = sympify(expr)
    expr_raw = sympify(expr_raw)
    extra = expr.free_symbols - {x}
    if extra:
        raise NameError(_unknown_names_msg(src, extra, local_dict))
    f = lambdify(x, expr, modules=_LAMBDIFY_MODULES)
    if len(_NUMPY_FUNC_CACHE) >= _FUNC_CACHE_LIMIT:
        _NUMPY_FUNC_CACHE.clear()
    _NUMPY_FUNC_CACHE[func_str] = (expr, expr_raw, f)
    return expr, expr_raw, f


def build_implicit_func(lhs_str, rhs_str):
    """
    Строит numpy-функцию H(x, y) = F(x, y) - G(x, y) для неявной кривой
    F(x,y) = G(x,y). Кривая — это множество точек, где H = 0, поэтому её
    можно нарисовать как линию уровня 0 (contour). Поддерживает pi/e,
    неявное умножение, ^ как степень — так же, как build_numpy_func.
    Возвращает (H_sympy, h), h(X, Y) работает с массивами (meshgrid).
    Результат мемоизируется по паре строк.
    """
    key = (lhs_str, rhs_str)
    hit = _IMPLICIT_FUNC_CACHE.get(key)
    if hit is not None:
        return hit
    from sympy.parsing.sympy_parser import parse_expr
    x, y = _X_SYM, _Y_SYM
    transformations = _parse_transformations()
    local_dict = _parse_local_dict(with_y=True)

    lhs = parse_expr(_preprocess_func_str(lhs_str),
                     transformations=transformations, local_dict=local_dict)
    rhs = parse_expr(_preprocess_func_str(rhs_str),
                     transformations=transformations, local_dict=local_dict)
    H = sympify(lhs - rhs)
    extra = H.free_symbols - {x, y}
    if extra:
        raise NameError(_unknown_names_msg(lhs_str + ' ' + rhs_str, extra, local_dict))
    h = lambdify((x, y), H, modules=_LAMBDIFY_MODULES)
    if len(_IMPLICIT_FUNC_CACHE) >= _FUNC_CACHE_LIMIT:
        _IMPLICIT_FUNC_CACHE.clear()
    _IMPLICIT_FUNC_CACHE[key] = (H, h)
    return H, h


def _eval_array(f, xs):
    """
    Вычисляет f на всём массиве xs одним вызовом (векторно). Если функция
    не векторизуется (исключение, неверная форма результата, скалярный
    результат, не совпадающий с поточечным) — поточечный запасной цикл.
    Неконечные и комплексные (с заметной мнимой частью) значения → NaN.
    """
    xs = np.asarray(xs, dtype=float)
    try:
        with np.errstate(all='ignore'):
            out = f(xs)
            out = np.asarray(out)
            if np.iscomplexobj(out):
                re_, im_ = out.real, out.imag
                out = np.where(np.abs(im_) <= 1e-9 * np.maximum(1.0, np.abs(re_)),
                               re_, np.nan)
            if out.ndim == 0 and xs.size > 1:
                # Скаляр на массив: либо константа (ок), либо функция «съела»
                # массив через float() и вернула NaN — проверяем по точкам.
                v = float(out)
                for xi in (xs[0], xs[xs.size // 2], xs[-1]):
                    p = _safe_val(f, xi)
                    same = (np.isnan(v) and np.isnan(p)) or (np.isfinite(v) and p == v)
                    if not same:
                        raise ValueError("not vectorizable")
            out = np.array(np.broadcast_to(out, xs.shape), dtype=float)
            out[~np.isfinite(out)] = np.nan
            return out
    except Exception:
        pass
    with np.errstate(all='ignore'):
        return np.array([_safe_val(f, xi) for xi in xs], dtype=float)


_MAX_ROOTS_1D = 60   # больше корней на одной оси — вырождение, не подписываем
_MAX_GRID_LINES = 5000   # защита от «бесконечной» сетки при слишком мелком шаге


def _scan_roots_1d(g, lo, hi, n=2000, xtol=1e-12):
    """
    Находит корни функции одной переменной g на [lo, hi]: сканирует мелкую
    сетку, ищет смену знака между соседними узлами и уточняет корень через
    brentq. Возвращает отсортированный список уникальных корней.
    Используется для пересечений неявной кривой с осями: g(t)=H(t,0) для оси
    X и g(t)=H(0,t) для оси Y.
    """
    ts = np.linspace(lo, hi, n)
    vals = _eval_array(g, ts)
    roots = []

    # Вырожденный случай (g ≡ 0 или знакосмена почти в каждом узле, как у
    # y = y или sin(1/x) у нуля): «корней» были бы тысячи, подписи не имеют
    # смысла, а их точная идентификация заняла бы минуты — возвращаем пусто.
    finite = np.isfinite(vals)
    if not np.any(finite):
        return []
    if np.all(vals[finite] == 0.0):
        return []
    n_changes = int(np.sum((vals[:-1] * vals[1:] < 0) & finite[:-1] & finite[1:]))
    n_zeros = int(np.sum(vals[finite] == 0.0))
    if n_changes + n_zeros > _MAX_ROOTS_1D:
        return []

    def _add(r):
        if not any(abs(r - e) < 1e-6 for e in roots):
            roots.append(r)

    va, vb = vals[:-1], vals[1:]
    with np.errstate(invalid='ignore', over='ignore'):
        both = finite[:-1] & finite[1:]
        zero_at = np.where(both & (va == 0.0))[0]
        cross = np.where(both & (va * vb < 0))[0]
    for i in zero_at:
        _add(ts[i])
    for i in cross:
        try:
            r = brentq(lambda t: _safe_scalar(g, t), ts[i], ts[i + 1], xtol=xtol)
            if np.isfinite(r):
                _add(r)
        except Exception:
            pass
    # правый конец
    if len(vals) and vals[-1] == 0.0:
        _add(ts[-1])
    return sorted(roots)


def _safe_scalar(g, t):
    try:
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            v = g(t)
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _dedup_points(pts, tol=1e-3):
    """Убирает почти совпадающие точки (x,y)."""
    out = []
    for p in pts:
        if not any(abs(p[0]-q[0]) < tol and abs(p[1]-q[1]) < tol for q in out):
            out.append(p)
    return out


def intersect_func_implicit(f, h, x_lo, x_hi, y_lo, y_hi):
    """
    Пересечения явной функции y=f(x) и неявной кривой H(x,y)=0:
    ищем корни одномерной g(x) = H(x, f(x)) на [x_lo, x_hi], затем оставляем
    те точки, чья y=f(x) попадает в окно.
    """
    def g(t):
        # работает и со скаляром (brentq), и с массивом (_scan_roots_1d → _eval_array)
        t_arr = np.asarray(t, dtype=float)
        with np.errstate(all='ignore'):
            yv = _eval_array(f, np.atleast_1d(t_arr))
            out = np.asarray(h(np.atleast_1d(t_arr), yv), dtype=float)
            out = np.array(np.broadcast_to(out, yv.shape), dtype=float)
            out[~np.isfinite(yv)] = np.nan
        return float(out[0]) if t_arr.ndim == 0 else out
    pts = []
    for xr in _scan_roots_1d(g, x_lo, x_hi):
        yv = _safe_scalar(f, xr)
        if np.isfinite(yv) and y_lo - 1e-9 <= yv <= y_hi + 1e-9:
            pts.append((xr, yv))
    return _dedup_points(pts)


def intersect_implicit_implicit(h1, h2, x_lo, x_hi, y_lo, y_hi, n=240):
    """
    Пересечения двух неявных кривых H1(x,y)=0 и H2(x,y)=0.
    Идея: на сетке считаем знаки H1 и H2; в каждой ячейке, где ОБЕ функции
    меняют знак (то есть через ячейку проходят обе кривые), берём центр
    ячейки как стартовое приближение и уточняем 2D-методом Ньютона по
    системе {H1=0, H2=0}. Затем дедуп. Ловит любое число точек (в т.ч.
    2 пересечения двух окружностей).
    """
    xs = np.linspace(x_lo, x_hi, n)
    ys = np.linspace(y_lo, y_hi, n)
    X, Y = np.meshgrid(xs, ys)
    with np.errstate(all='ignore'):
        Z1 = np.asarray(h1(X, Y), dtype=float)
        Z2 = np.asarray(h2(X, Y), dtype=float)

    def sign_change_cell(Z, i, j):
        a, b, c, d = Z[i, j], Z[i, j+1], Z[i+1, j], Z[i+1, j+1]
        vals = [v for v in (a, b, c, d) if np.isfinite(v)]
        if len(vals) < 2:
            return False
        return (min(vals) <= 0 <= max(vals))

    def newton(x0, y0):
        # 2D Newton на систему H1=H2=0 с численным якобианом
        x, y = x0, y0
        eps = 1e-6
        for _ in range(40):
            f1 = _safe_scalar(lambda _: h1(x, y), 0.0)
            f2 = _safe_scalar(lambda _: h2(x, y), 0.0)
            if not (np.isfinite(f1) and np.isfinite(f2)):
                return None
            if abs(f1) < 1e-11 and abs(f2) < 1e-11:
                return (x, y)
            j11 = (_safe_scalar(lambda _: h1(x+eps, y), 0.0) - f1) / eps
            j12 = (_safe_scalar(lambda _: h1(x, y+eps), 0.0) - f1) / eps
            j21 = (_safe_scalar(lambda _: h2(x+eps, y), 0.0) - f2) / eps
            j22 = (_safe_scalar(lambda _: h2(x, y+eps), 0.0) - f2) / eps
            det = j11*j22 - j12*j21
            if abs(det) < 1e-14:
                return None
            dx = ( j22*f1 - j12*f2) / det
            dy = (-j21*f1 + j11*f2) / det
            x -= dx; y -= dy
            if not (np.isfinite(x) and np.isfinite(y)):
                return None
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                break
        f1 = _safe_scalar(lambda _: h1(x, y), 0.0)
        f2 = _safe_scalar(lambda _: h2(x, y), 0.0)
        if np.isfinite(f1) and np.isfinite(f2) and abs(f1) < 1e-6 and abs(f2) < 1e-6:
            return (x, y)
        return None

    seeds = []
    for i in range(n-1):
        for j in range(n-1):
            if sign_change_cell(Z1, i, j) and sign_change_cell(Z2, i, j):
                seeds.append(((xs[j]+xs[j+1])/2.0, (ys[i]+ys[i+1])/2.0))

    pts = []
    for (sx, sy) in seeds:
        r = newton(sx, sy)
        if r is None:
            continue
        rx, ry = r
        if x_lo - 1e-9 <= rx <= x_hi + 1e-9 and y_lo - 1e-9 <= ry <= y_hi + 1e-9:
            pts.append((rx, ry))
    return _dedup_points(pts, tol=1e-2)


def _get_x(expr_sympy):
    free = expr_sympy.free_symbols
    return next((s for s in free if str(s) == 'x'), symbols('x'))


def _safe_val(f, x):
    try:
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            v = float(f(x))
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan



# ══════════════════════════════════════════════════════════════
#  МНОЖЕСТВА SYMPY → ВЕЩЕСТВЕННЫЕ ТОЧКИ В ОКНЕ
# ══════════════════════════════════════════════════════════════

_ENUM_CACHE = {}        # (set, lo, hi) -> [(float, exact), ...]
_FINITE_FLOATS = {}     # FiniteSet -> [(float | None, exact), ...]
_ENUM_CACHE_LIMIT = 2000


def _real_float(c):
    """float вещественного числа sympy или None (символы / комплексное / ∞)."""
    try:
        if c.free_symbols:
            return None
        if c.has(oo, -oo, zoo, nan):
            return None
        cv = complex(c.evalf(15))
        if abs(cv.imag) > 1e-9 * max(1.0, abs(cv.real)):
            return None
        v = cv.real
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _finite_elements(sset):
    """Все элементы конечных множеств внутри (вложенных) Union/Complement."""
    out = []
    try:
        if isinstance(sset, FiniteSet):
            out.extend(list(sset.args))
        elif isinstance(sset, (Union, Complement, Intersection)):
            for a in sset.args:
                out.extend(_finite_elements(a))
    except Exception:
        pass
    return out


def _enumerate_real_set(sset, lo, hi, nice=None, n_range=60, max_count=400):
    """
    Перечисляет вещественные элементы множества sympy, попадающие в [lo, hi]:
    FiniteSet, Union, Complement, Intersection, ImageSet над Integers
    (линейный образ a·n+b — по диапазону n, покрывающему окно; иначе
    n ∈ [-n_range, n_range]). ConditionSet / EmptySet / прочее → [].
    nice — словарь «элемент → его упрощённая форма» (см. exact_candidates).
    Возвращает список пар (float, точное_значение).
    """
    if sset is None:
        return []
    # Ключ — сам объект множества (хэш sympy кэширует), а не str(sset):
    # печать больших множеств стоила десятки миллисекунд на каждую перерисовку.
    try:
        ck = (sset, float(lo), float(hi))
        hit = _ENUM_CACHE.get(ck)
    except Exception:
        ck, hit = None, None
    if hit is not None:
        return list(hit)
    eps = 1e-9 * max(1.0, abs(hi - lo))
    out = []

    def _add(c, v=None):
        if nice:
            c = nice.get(c, c)
        if v is None:
            v = _real_float(c)
        if v is None:
            return
        if lo - eps <= v <= hi + eps:
            if not any(abs(v - w) < 1e-9 for w, _ in out):
                out.append((v, c))

    def _finite_floats(fs):
        # float каждого элемента конечного множества считаем один раз
        # (evalf — самая дорогая часть), окно фильтруем уже по числам
        try:
            hit_f = _FINITE_FLOATS.get(fs)
        except Exception:
            hit_f = None
        if hit_f is None:
            hit_f = [(_real_float(nice.get(c, c) if nice else c), c) for c in fs.args]
            try:
                if len(_FINITE_FLOATS) >= _ENUM_CACHE_LIMIT:
                    _FINITE_FLOATS.clear()
                _FINITE_FLOATS[fs] = hit_f
            except Exception:
                pass
        return hit_f

    try:
        if isinstance(sset, FiniteSet):
            for v, c in _finite_floats(sset):
                if v is not None:
                    _add(c, v)
        elif isinstance(sset, Union):
            for a in sset.args:
                out.extend(_enumerate_real_set(a, lo, hi, nice, n_range, max_count))
        elif isinstance(sset, Complement):
            base, excl = sset.args
            items = _enumerate_real_set(base, lo, hi, nice, n_range, max_count)
            excluded = _enumerate_real_set(excl, lo, hi, nice, n_range, max_count)
            out.extend(p for p in items
                       if not any(abs(p[0] - q[0]) < 1e-9 for q in excluded))
        elif isinstance(sset, Intersection):
            parts = [a for a in sset.args if a != S.Reals]
            lists = [_enumerate_real_set(a, lo, hi, nice, n_range, max_count)
                     for a in parts]
            lists = [l for l, a in zip(lists, parts)
                     if not isinstance(a, ConditionSet)]
            if lists:
                cur = lists[0]
                for other in lists[1:]:
                    cur = [p for p in cur if any(abs(p[0] - q[0]) < 1e-9 for q in other)]
                out.extend(cur)
        elif isinstance(sset, ImageSet):
            lamb = sset.lamda
            bases = sset.base_sets
            if (len(bases) == 1 and bases[0] == S.Integers
                    and len(lamb.variables) == 1):
                nsym = lamb.variables[0]
                e = lamb.expr
                done = False
                try:
                    poly = e.as_poly(nsym)
                    if poly is not None and poly.degree() == 1:
                        a = float(poly.coeff_monomial(nsym))
                        b = float(poly.coeff_monomial(1))
                        if a != 0.0 and math.isfinite(a) and math.isfinite(b):
                            k1 = (lo - b) / a
                            k2 = (hi - b) / a
                            k_lo = int(math.ceil(min(k1, k2) - 1e-9))
                            k_hi = int(math.floor(max(k1, k2) + 1e-9))
                            if k_hi - k_lo + 1 <= max_count:
                                for k in range(k_lo, k_hi + 1):
                                    # значение — арифметикой (без evalf), точная форма — подстановкой
                                    _add(e.subs(nsym, Integer(k)), a * k + b)
                            done = True
                except Exception:
                    done = False
                if not done:
                    for k in range(-n_range, n_range + 1):
                        _add(e.subs(nsym, Integer(k)))
        # ConditionSet, EmptySet, Interval и прочее — нечего перечислять
    except Exception:
        pass

    out.sort(key=lambda p: p[0])
    if ck is not None:
        if len(_ENUM_CACHE) >= _ENUM_CACHE_LIMIT:
            _ENUM_CACHE.clear()
        _ENUM_CACHE[ck] = list(out)
    return out


# ══════════════════════════════════════════════════════════════
#  РАЗРЫВЫ И АСИМПТОТЫ
# ══════════════════════════════════════════════════════════════

def find_discontinuities_numerical(f, expr_sympy, expr_raw, x_lim_l, x_lim_r,
                                   n=10000, exact_out=None):
    """
    Точки разрыва функции в окне [x_lim_l, x_lim_r] (список float).
    Сначала — символьно (sympy singularities, через кэш sym_cached; в режиме
    'cached' результат может быть отложен), затем, если ничего не найдено, —
    численно: границы NaN-областей и резкие скачки на мелкой сетке.
    exact_out (dict) получает точные значения точек: round(px, 8) -> sympy.
    """
    disc = []
    x_sym = _get_x(expr_raw) if expr_raw.free_symbols else _get_x(expr_sympy)

    tried = set()
    for expr_candidate in [expr_raw, expr_sympy]:
        key = ('sing', _ekey(expr_candidate))
        if key in tried:
            continue
        tried.add(key)
        try:
            sset = sym_cached(key, lambda e=expr_candidate: singularities(e, x_sym))
            pts = _enumerate_real_set(sset, x_lim_l, x_lim_r)
        except Exception:
            # SymbolicPending / NotImplementedError и т.п. — пробуем дальше
            continue
        for px, pe in pts:
            if not any(abs(px - d) < 0.01 for d in disc):
                pr = round(px, 8)
                disc.append(pr)
                if exact_out is not None:
                    exact_out[pr] = pe
        if disc:
            break

    if not disc:
        xs = np.linspace(x_lim_l, x_lim_r, n)
        ys = _eval_array(f, xs)

        bad = ~np.isfinite(ys)
        transitions = np.where(np.diff(bad.astype(int)) != 0)[0]
        for idx in transitions:
            xc = (xs[idx] + xs[idx + 1]) / 2
            probes = [_safe_val(f, xc + d) for d in [-1e-5, 0, 1e-5]]
            if not all(np.isfinite(v) for v in probes):
                if not any(abs(xc - d) < 0.05 for d in disc):
                    disc.append(round(xc, 6))

        dy = np.abs(np.diff(ys))
        threshold = max(50.0, 20 * np.nanstd(dy)) if np.any(np.isfinite(dy)) else 50.0
        jumps = np.where(dy > threshold)[0]
        for idx in jumps:
            xc = (xs[idx] + xs[idx + 1]) / 2
            vl = abs(_safe_val(f, xc - 1e-4))
            vr = abs(_safe_val(f, xc + 1e-4))
            if (vl > 20 or vr > 20) and not any(abs(xc - d) < 0.05 for d in disc):
                disc.append(round(xc, 6))

    disc.sort()
    return disc


def classify_discontinuities(f, expr_sympy, disc_pts, y_lim_b, y_lim_t,
                             exact_pts=None, exact_out=None):
    """
    Делит точки разрыва на вертикальные асимптоты и устранимые разрывы
    (holes). Пределы слева/справа считаются sympy через кэш (в точной точке,
    если она известна из exact_pts), при неудаче/отложенном результате —
    численно по окрестности. Возвращает (vasymps, removable), где removable —
    список (x, y_предела). exact_out[x] = (x_точное | None, y_точное | None).
    """
    x_sym = _get_x(expr_sympy)
    vasymps = []
    removable = []

    for dp in disc_pts:
        lim_val = None
        lim_exact = None
        pt_exact = exact_pts.get(dp) if exact_pts else None
        pt_arg = pt_exact if pt_exact is not None else dp
        try:
            key_base = ('lim', _ekey(expr_sympy), str(pt_arg))
            # ВАЖНО: точку связываем параметром по умолчанию — в режиме
            # 'cached' лямбда выполняется позже, когда переменная цикла уже
            # указывает на последнюю точку.
            lv_p = sym_cached(key_base + ('+',),
                              lambda p=pt_arg: limit(expr_sympy, x_sym, p, '+'))
            lv_m = sym_cached(key_base + ('-',),
                              lambda p=pt_arg: limit(expr_sympy, x_sym, p, '-'))
            if lv_p in (oo, -oo, zoo) or lv_m in (oo, -oo, zoo):
                vasymps.append(dp)
                continue
            if lv_p == lv_m and lv_p.is_real:
                lim_val = float(lv_p)
                if not lv_p.is_Float:
                    lim_exact = lv_p
        except Exception:
            pass

        if lim_val is None:
            eps_vals = [_safe_val(f, dp + d) for d in [1e-6, -1e-6, 1e-5, -1e-5]]
            eps_vals = [v for v in eps_vals if np.isfinite(v)]
            if not eps_vals:
                vasymps.append(dp)
                continue
            if max(abs(v) for v in eps_vals) > 200:
                vasymps.append(dp)
                continue
            lim_val = round(sum(eps_vals) / len(eps_vals), 6)

        if lim_val is not None and y_lim_b <= lim_val <= y_lim_t:
            removable.append((dp, lim_val))
            if exact_out is not None:
                exact_out[dp] = (pt_exact, lim_exact)
        else:
            vasymps.append(dp)

    return vasymps, removable


def find_horizontal_asymptotes(f, expr_sympy, x_lim_l, x_lim_r, y_lim_b, y_lim_t):
    hasymps = []
    x_sym = _get_x(expr_sympy)

    for direction in [oo, -oo]:
        val = None
        try:
            lv = sym_cached(('lim', _ekey(expr_sympy), str(direction), ''),
                            lambda d=direction: limit(expr_sympy, x_sym, d))
            if lv not in (oo, -oo, zoo, nan) and lv.is_real:
                val = float(lv)
        except Exception:
            pass

        if val is None:
            x_span = x_lim_r - x_lim_l
            sign_ = 1 if direction == oo else -1
            far_points = [sign_ * x_span * k for k in [100, 1000, 10000, 100000]]
            probes = [_safe_val(f, xp) for xp in far_points]
            probes = [v for v in probes if np.isfinite(v)]
            if len(probes) >= 3:
                diffs = [abs(probes[i+1] - probes[i]) for i in range(len(probes)-1)]
                if all(d < 1e-4 for d in diffs):
                    val = probes[-1]

        if val is not None:
            val = round(val, 8)
            if y_lim_b <= val <= y_lim_t and val not in hasymps:
                hasymps.append(val)

    return hasymps


def detect_const_context(func_str):
    """
    Определяет, какие «красивые» константы уместны для данной функции:
      • π — если есть тригонометрия (sin/cos/tan/asin/acos/atan/...) ИЛИ
            пользователь сам явно написал pi/π в выражении.
      • e — если есть экспонента/логарифм (exp/log/ln/lg) ИЛИ пользователь
            сам явно написал e в выражении.
    Возвращает (allow_pi, allow_e). Это не даёт, например, точке пересечения
    чисто экспоненциальной функции e^x-5e «прилипнуть» к 5π/6.
    """
    if not func_str:
        return (True, True)
    s = str(func_str).lower()
    import re
    trig = re.search(r'\b(a?sin|a?cos|a?tan|sinh|cosh|tanh|cot|sec|csc)\b', s) is not None
    has_pi_literal = ('pi' in s) or ('π' in s)
    explog = re.search(r'\b(exp|ln|lg)\b', s) is not None or 'log' in s
    # одиночная буква e как литерал (не часть слова): 'e', '2e', 'e^x' и т.п.
    has_e_literal = re.search(r'(?<![A-Za-z])e(?![A-Za-z])', s) is not None
    allow_pi = trig or has_pi_literal
    allow_e  = explog or has_e_literal
    # Чисто алгебраическая функция (без тригонометрии, exp/log и без явных
    # pi/e) — ни π, ни e не имеют здесь смысла, поэтому отключаем оба.
    # Так точки вроде вершины параболы при x≈1.6 и пересечения e^x-5e не
    # «прилипают» к π/e. (Целочисленная привязка работает всегда.)
    return (allow_pi, allow_e)


def snap_to_nice(val, tol_pct=0.02, allow_pi=True, allow_e=True):
    if val == 0:
        return 0.0

    PI = math.pi
    E  = math.e
    PI_E_TOL = 0.004   # 0.4 % — узкий допуск для долей π и кратных e

    # Собираем ВСЕХ подходящих кандидатов (целое, доля π, кратное e), затем
    # выбираем ближайшего. Важно именно сравнивать их между собой, а НЕ брать
    # первого сработавшего: у дальних экстремумов (5π/2≈7.854, 7π/2≈10.996…)
    # ближайшее целое попадает в 2%-коридор и раньше перехватывало π-долю —
    # отсюда «8» вместо «5π/2». Теперь побеждает тот, кто реально ближе.
    candidates = []  # (относительная_ошибка, точное_значение)

    # Целое
    nearest_int = round(val)
    if nearest_int != 0:
        rel = abs(val - nearest_int) / abs(nearest_int)
        if rel <= tol_pct:
            candidates.append((rel, float(nearest_int)))

    # Доли π: k·π/den, den ∈ {1,2,3,4,6}
    if allow_pi:
        for den in (1, 2, 3, 4, 6):
            num = round(val / PI * den)
            if num == 0:
                continue
            exact = num / den * PI
            if abs(exact) > 1e-12:
                rel = abs(val - exact) / abs(exact)
                if rel <= PI_E_TOL:
                    candidates.append((rel, exact))

    # Кратные e
    if allow_e:
        n_e = round(val / E)
        if n_e != 0:
            exact_e = n_e * E
            rel = abs(val - exact_e) / abs(exact_e)
            if rel <= PI_E_TOL:
                candidates.append((rel, exact_e))

    if candidates:
        # ближайший по относительной ошибке
        return min(candidates, key=lambda c: c[0])[1]
    return val


def fmt_num(val, tol=1e-4, tex=False):
    """
    Подпись деления оси / десятичное число: кратные π (знаменатели 1,2,3,4,6)
    и e — символьно, иначе {val:g}. tex=True — фрагмент mathtext.
    """
    import math
    from math import gcd
    PI = math.pi
    E  = math.e
    PI_S = '\\pi' if tex else 'π'

    # Кратные π с «хорошими» знаменателями: 1, 2, 3, 4, 6.
    # Проверяем их от большего знаменателя к меньшему и берём первое
    # достаточно точное совпадение в наиболее простой форме.
    ratio_pi = val / PI
    best = None
    for den in (1, 2, 3, 4, 6):
        num = round(ratio_pi * den)
        if num == 0:
            continue
        approx = (num / den) * PI
        if abs(val - approx) / (abs(val) + 1e-12) < tol:
            g = gcd(abs(num), den)
            sn, sd = num // g, den // g
            # Кандидат найден; сохраняем самый простой (наименьший знаменатель)
            if best is None or sd < best[1]:
                best = (sn, sd)
    if best is not None:
        num, den = best
        sign = '-' if num < 0 else ''
        coef = '' if abs(num) == 1 else str(abs(num))
        if den == 1:
            return f'{sign}{coef}{PI_S}'
        if tex:
            return f'{sign}\\frac{{{coef}{PI_S}}}{{{den}}}'
        return f'{sign}{coef}π/{den}'

    ratio_e = val / E
    n_e = round(ratio_e)
    if abs(n_e) > 0 and abs(val - n_e * E) / (abs(val) + 1e-12) < tol:
        if n_e ==  1: return  'e'
        if n_e == -1: return '-e'
        return f'{n_e}e'

    return _dec_tex(f'{val:g}') if tex else f'{val:g}'


def _dec_tex(s):
    """Десятичная строка → фрагмент mathtext (1e-05 → 1\\cdot10^{-5})."""
    if 'e' in s or 'E' in s:
        m, _, ex = s.lower().partition('e')
        try:
            return f'{m}\\cdot10^{{{int(ex)}}}'
        except ValueError:
            return s
    return s


@functools.lru_cache(maxsize=4096)
def _mathtext_ok(text):
    """Проверка, что строка разбирается mathtext (иначе рисуем как обычный текст)."""
    try:
        from matplotlib.mathtext import MathTextParser
        MathTextParser('path').parse(text, dpi=72, prop=None)
        return True
    except Exception:
        return False


def tick_label(v):
    """Подпись деления оси (mathtext при MATHTEXT_LABELS)."""
    return math_label(fmt_num(v, tex=True)) if MATHTEXT_LABELS else fmt_num(v)


_TEX_PLAIN_RE = re.compile(r'^[0-9.,()+\-\\ e]*$')


def math_label(frag):
    """
    Оборачивает фрагмент в $…$ (если MATHTEXT_LABELS и строка корректна).
    Фрагменты из одних цифр/знаков/скобок (например «(2.76, 3.76)» или «-4»)
    рисуем ОБЫЧНЫМ текстом тем же serif-шрифтом с настоящим минусом:
    раскладка mathtext стоит ~5 мс на подпись, а выглядит так же.
    """
    if not MATHTEXT_LABELS:
        return frag
    if '\\' not in frag.replace('\\ ', ' ') and _TEX_PLAIN_RE.match(frag):
        return frag.replace('\\ ', ' ').replace('-', '\u2212')
    t = f'${frag}$'
    return t if _mathtext_ok(t) else frag


def parse_number(s, default=None):
    """
    Парсит строку с числом ИЛИ математическим выражением в float.
    Поддерживает: pi/π, e, exp(1), 2*pi, pi/2, sqrt(2), 3+2, -inf, inf и т.п.
    Это позволяет вводить «pi», «2pi», «pi/4» в любые числовые поля
    (сетка, домен, окно просмотра, диапазон заливки), а не только в функции.

    При пустой строке возвращает default (если задан), иначе бросает ValueError.
    """
    if s is None:
        if default is not None:
            return default
        raise ValueError("empty")
    t = str(s).strip().lower()
    if t == "":
        if default is not None:
            return default
        raise ValueError("empty")

    # Бесконечности — отдельно (sympy их тоже понимает, но так быстрее и явнее)
    t_clean = t.replace(" ", "")
    if t_clean in ("inf", "+inf", "infinity", "+infinity", "∞", "+∞"):
        return float("inf")
    if t_clean in ("-inf", "-infinity", "-∞"):
        return float("-inf")

    # Нормализуем удобные пользователю обозначения
    t = t.replace("π", "pi").replace("∞", "oo")
    # «2pi», «3e», «2pi/3» — неявное умножение разрешит sympy (implicit_multiplication)
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application, convert_xor)
        transformations = (standard_transformations +
                           (implicit_multiplication_application, convert_xor))
        # E — символ Эйлера в sympy; 'e' пользователя приводим к нему.
        # Делаем это через локальный словарь, чтобы 'e' не считался символом.
        from sympy import E as _E, pi as _pi
        expr = parse_expr(t, transformations=transformations,
                          local_dict={'e': _E, 'pi': _pi})
        val = float(expr.evalf())
        if not np.isfinite(val):
            # выражения вроде oo
            if expr.is_infinite:
                return float("inf") if expr.is_positive else float("-inf")
        return val
    except Exception:
        # Последняя попытка — обычный float
        return float(t_clean)


def prettify_math_text(text):
    """
    Заменяет в пользовательском тексте 'pi' -> 'π' и одиночное 'e' -> 'e'
    (символ как есть), а также 'inf' -> '∞', чтобы подписи вроде
    '(pi, 2pi)' отображались как '(π, 2π)'.

    Аккуратно: заменяет 'pi' только как отдельный математический токен
    (не внутри слов типа 'epic', 'spin'), опираясь на границы из
    не-буквенных символов. 'e' само по себе НЕ трогаем как букву (слишком
    часто встречается в обычном тексте); конвертируем только связки вида
    '2e', '3e' (число + e), где это явно число Эйлера.
    """
    import re
    if not text:
        return text

    # pi / π как отдельный токен (с возможным числовым коэффициентом перед ним
    # мы не трогаем — само 'pi' превратится в 'π', а '2pi' станет '2π').
    # Граница слева: начало строки или не-буква; справа: конец или не-буква.
    text = re.sub(r'(?<![A-Za-z])pi(?![A-Za-z])', 'π', text)

    # inf -> ∞ (как отдельный токен)
    text = re.sub(r'(?<![A-Za-z])inf(?![A-Za-z])', '∞', text)

    # 'число + e' как число Эйлера: '2e' -> '2e' (оставляем e-символ),
    # но только если это не часть слова и не похоже на экспоненту в духе
    # '1e5'. Требуем, чтобы после e не шла цифра.
    # Здесь по сути ничего не меняем визуально (e и так 'e'), поэтому
    # оставляем как есть — намеренно не трогаем 'e', чтобы не ломать слова.
    return text



# ══════════════════════════════════════════════════════════════
#  ТОЧНЫЕ ПОДПИСИ: символьные кандидаты, идентификация, формат
# ══════════════════════════════════════════════════════════════
#
# Вместо старого «прилипания» численных значений к π/e с допуском 2 %
# (snap_to_nice) подписи теперь получаются так:
#   1. точные решения sympy (solveset f=0, f'=0, f1−f2=0; пределы для holes)
#      сопоставляются с точными численными результатами (brentq 1e-12,
#      minimize_scalar);
#   2. если решений нет — идентификация числа (nsimplify с допуском 1e-9),
#      принимаемая ТОЛЬКО если результат простой и ПРОВЕРЕН (подстановка
#      даёт ноль);
#   3. иначе — десятичная запись.

def _nice_const(c):
    """Приводит константу к более читаемой форме: log(5·e) → 1 + log(5)."""
    try:
        if c.has(log):
            c2 = expand_log(c, force=True)
            if count_ops(c2) <= count_ops(c) + 1:
                return c2
    except Exception:
        pass
    return c


def exact_candidates(expr, x_sym, kind, x_lo=None, x_hi=None):
    """
    Точные вещественные кандидаты в окне [x_lo, x_hi]:
      kind='roots'    — solveset(expr = 0),
      kind='critical' — solveset(expr' = 0).
    Считается через sym_cached (в режиме 'cached' может бросить
    SymbolicPending). Возвращает список пар (float, sympy). Любая неудача
    sympy (NotImplementedError, ConditionSet …) → [].
    """
    if x_lo is None:
        x_lo = X_LIM_L
    if x_hi is None:
        x_hi = X_LIM_R
    key = ('roots' if kind == 'roots' else 'crit', _ekey(expr), str(x_sym))

    def job():
        target = expr if kind == 'roots' else diff(expr, x_sym)
        sset = solveset(target, x_sym, S.Reals)
        nice = {}
        for c in _finite_elements(sset):
            n_ = _nice_const(c)
            if n_ is not c:
                nice[c] = n_
        return sset, nice

    try:
        sset, nice = sym_cached(key, job)
    except SymbolicPending:
        raise
    except Exception:
        return []
    try:
        return _enumerate_real_set(sset, x_lo, x_hi, nice=nice)
    except Exception:
        return []


_SIMPLE_INT_MAX = 1000
_SIMPLE_DEN_MAX = 24


def _is_simple(c):
    """
    «Простое» выражение: count_ops ≤ 6, все целые ≤ 1000, знаменатели
    рациональных ≤ 24, степени — только целые |k| ≤ 4 или 1/2, 1/3, 1/4
    над целыми/рациональными/π/e; из функций — только log от рационального.
    """
    try:
        if c.free_symbols or c.is_Float:
            return False
        if count_ops(c) > 6:
            return False
        for a in c.atoms():
            if a.is_Float:
                return False
            if a.is_Integer:
                if abs(int(a)) > _SIMPLE_INT_MAX:
                    return False
            elif a.is_Rational:
                if abs(a.p) > _SIMPLE_INT_MAX or a.q > _SIMPLE_DEN_MAX:
                    return False
            elif a.is_NumberSymbol:
                if a not in (pi, E):
                    return False
            elif a.is_Number:
                return False
        for p in c.atoms(Pow):
            base, ex = p.args
            if not (base.is_Rational or base in (pi, E)):
                return False
            if ex.is_Integer:
                if abs(int(ex)) > 4:
                    return False
            elif ex.is_Rational:
                if ex.p != 1 or ex.q not in (2, 3, 4):
                    return False
                if not base.is_Rational or base < 0:
                    return False
            else:
                return False
        for e_ in c.atoms(exp):
            if not e_.args[0].is_Integer or abs(int(e_.args[0])) > 4:
                return False
        for fn in c.atoms(Function):
            if isinstance(fn, exp):
                continue
            if fn.func is log and len(fn.args) == 1 and fn.args[0].is_Rational \
                    and fn.args[0] > 0:
                continue
            return False
        return True
    except Exception:
        return False


def _safe_verify(verify, c):
    try:
        return bool(verify(c))
    except Exception:
        return False


def _is_zero_at(expr, sym, c, tol=1e-9):
    """Проверка: expr(sym=c) == 0 (точно или численно с высокой точностью)."""
    try:
        v = expr.subs(sym, c)
        if v == 0:
            return True
        if v.free_symbols or v.has(oo, -oo, zoo, nan):
            return False
        cv = complex(v.evalf(30))
        return abs(cv) < tol
    except Exception:
        return False


def _identify_by_nsimplify(v, verify, allow_consts, tol):
    if abs(v) < 1e-12:
        c = Integer(0)
        return c if (verify is None or _safe_verify(verify, c)) else None
    bases = [()]
    if allow_consts:
        bases += [(pi,), (E,)]
    found = []
    for consts in bases:
        try:
            c = nsimplify(v, list(consts), tolerance=tol)
        except Exception:
            continue
        if c is None:
            continue
        c = sympify(c)
        if not _is_simple(c):
            continue
        cv = _real_float(c)
        if cv is None or abs(cv - v) > tol * max(1.0, abs(v)):
            continue
        if not any(c == d for _, d in found):
            found.append((count_ops(c), c))
    found.sort(key=lambda t: t[0])
    for _, c in found:
        if verify is None or _safe_verify(verify, c):
            return c
    return None


def identify_value(v, candidates=(), verify=None, allow_consts=True,
                   tol=1e-9, cache_key=None):
    """
    Точное значение для числа v (float) или None:
      1) ближайший кандидат c (sympy или пара (float, sympy)) с
         |float(c) − v| ≤ 1e-6·max(1, |v|) → c;
      2) иначе nsimplify(v, [π] / [e] / без констант, tolerance=tol) —
         принимается ТОЛЬКО если результат простой (_is_simple) И verify(c)
         (если задан) возвращает True;
      3) иначе None.
    cache_key (кортеж строк) — шаг 2 выполняется через sym_cached, т.е.
    в режиме 'cached' откладывается (тогда возвращается None).
    """
    try:
        v = float(v)
    except Exception:
        return None
    if not np.isfinite(v):
        return None

    best, best_d = None, None
    for c in candidates:
        if isinstance(c, tuple):
            cv, ce = c
        else:
            ce = c
            cv = _real_float(c)
            if cv is None:
                continue
        d = abs(cv - v)
        if d <= 1e-6 * max(1.0, abs(v)) and (best_d is None or d < best_d):
            best, best_d = ce, d
    if best is not None:
        return best

    def job():
        return _identify_by_nsimplify(v, verify, allow_consts, tol)

    if cache_key is not None:
        key = ('ident',) + tuple(str(k) for k in cache_key) + (f"{v:.9g}",)
        try:
            return sym_cached(key, job)
        except Exception:
            return None
    try:
        return job()
    except Exception:
        return None


def exact_y_of(expr, x_sym, cx):
    """
    Точное значение expr в точной точке cx: подстановка, при необходимости —
    simplify через кэш (в режиме 'cached' может быть отложено → None).
    Возвращает sympy-число, которое умеет форматировать fmt_sym, иначе None.
    """
    try:
        e = sympify(expr).subs(x_sym, cx)
        if e.free_symbols or e.has(oo, -oo, zoo, nan):
            return None
        if fmt_sym(e) is not None:
            return e
        # simplify громоздких выражений (корни Кардано и т.п.) может длиться
        # минуты, а «простым» результат всё равно не станет — не пытаемся.
        if count_ops(e) > 40:
            return None
        s = sym_cached(('simp', _ekey(e)), lambda: simplify(e))
        if s.free_symbols or s.has(oo, -oo, zoo, nan):
            return None
        return s if fmt_sym(s) is not None else None
    except Exception:
        return None


def _exact_matches(c, v, rel=1e-6):
    """Точное значение c согласуется с численным v."""
    cv = _real_float(c)
    return cv is not None and abs(cv - v) <= rel * max(1.0, abs(v))


_SUP_TRANS = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
_ROOT_SIGNS = {2: '√', 3: '∛', 4: '∜'}
_FMT_INT_MAX = 10 ** 6


def _sup(k):
    return str(int(k)).translate(_SUP_TRANS)


def _fmt_base(b, tex=False):
    """
    Строка для иррационального множителя (π, e, eⁿ, √n, ∛n, ln(q)) или None.
    tex=True — фрагмент mathtext (\\pi, e^{2}, \\sqrt{5}, \\sqrt[3]{2}, \\ln(5)).
    """
    def powtxt(base_s, k):
        return f'{base_s}^{{{int(k)}}}' if tex else base_s + _sup(k)

    if b is pi:
        return '\\pi' if tex else 'π'
    if b is E:
        return 'e'
    if isinstance(b, exp):
        k = b.args[0]
        if k.is_Integer and 1 <= int(k) <= 9:
            return 'e' if int(k) == 1 else powtxt('e', k)
        return None
    if b.is_Pow:
        base, ex = b.args
        if base is pi and ex.is_Integer and 2 <= int(ex) <= 9:
            return powtxt('\\pi' if tex else 'π', ex)
        if base is E and ex.is_Integer and 2 <= int(ex) <= 9:
            return powtxt('e', ex)
        if (base.is_Integer and int(base) > 1 and int(base) <= _FMT_INT_MAX
                and ex.is_Rational and ex.p == 1 and ex.q in _ROOT_SIGNS):
            if tex:
                idx = '' if ex.q == 2 else f'[{ex.q}]'
                return f'\\sqrt{idx}{{{int(base)}}}'
            return _ROOT_SIGNS[ex.q] + str(int(base))
        return None
    if b.func is log and len(b.args) == 1:
        a = b.args[0]
        if a.is_Rational and a > 0 and a != 1 \
                and abs(a.p) <= _FMT_INT_MAX and a.q <= _FMT_INT_MAX:
            arg = str(a.p) if a.q == 1 else (f'\\frac{{{a.p}}}{{{a.q}}}' if tex else f'{a.p}/{a.q}')
            return ('\\ln(' if tex else 'ln(') + arg + ')'
        return None
    return None


def _fmt_term(t, tex=False):
    """
    Разбирает слагаемое c·B₁·B₂/(q·B₃): возвращает (знак, числитель,
    знаменатель_строка, q) или None. q — целый знаменатель коэффициента.
    """
    c, rest = t.as_coeff_Mul()
    if not c.is_Rational or c == 0:
        return None
    if abs(c.p) > _FMT_INT_MAX or c.q > _FMT_INT_MAX:
        return None
    num_f, den_f = [], []
    if rest != 1:
        for fac in Mul.make_args(rest):
            if fac.is_Pow and fac.args[1].is_negative:
                s = _fmt_base(fac.args[0] ** (-fac.args[1]), tex)
                target = den_f
            elif isinstance(fac, exp) and fac.args[0].is_negative:
                s = _fmt_base(exp(-fac.args[0]), tex)
                target = den_f
            else:
                s = _fmt_base(fac, tex)
                target = num_f
            if s is None:
                return None
            target.append(s)
    if len(num_f) > 2 or len(den_f) > 1:
        return None
    sign_ = -1 if c < 0 else 1
    p, q = abs(c.p), c.q
    num = ('' if (p == 1 and num_f) else str(p)) + ''.join(num_f)
    return sign_, num, den_f, q


def _join_den(q, den_f, tex=False):
    parts = ([str(q)] if q > 1 else []) + list(den_f)
    if not parts:
        return ''
    s = ''.join(parts)
    if tex:
        return s
    return '(' + s + ')' if len(parts) > 1 else s


def _frac_txt(num, den, tex):
    """num/den: в tex — \\frac{num}{den}, иначе num/den."""
    return f'\\frac{{{num}}}{{{den}}}' if tex else f'{num}/{den}'


_SUB_TRANS = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
_LOG_QUOT_RE = re.compile(r'ln\((\d+)\)/ln\((\d+)\)')
_LOG_QUOT_TEX_RE = re.compile(r'\\frac\{\\ln\((\d+)\)\}\{\\ln\((\d+)\)\}')


def fmt_sym(expr):
    """
    Компактная unicode-запись точного числа: 3, -1/2, π, 2π/3, -π/2, e, 2e,
    e², √2, 2√3, √2/2, -√3/2, ∛2, 1+√2, (1+√13)/2, 1+ln(5), 2/π, log₂(3) …
    Возвращает None, если выражение не из этого класса (вызывающий печатает
    десятичную запись).
    """
    out = _fmt_sym_core(expr, tex=False)
    if out is None:
        return None
    # ln(a)/ln(b) — это log_b(a): так короче и привычнее (корень 2^x = 3 → log₂(3))
    return _LOG_QUOT_RE.sub(lambda m: f"log{m.group(2).translate(_SUB_TRANS)}({m.group(1)})", out)


def fmt_sym_tex(expr):
    """
    То же, что fmt_sym, но фрагмент mathtext (без $): \\sqrt{5},
    \\frac{\\pi}{2}, \\frac{1+\\sqrt{13}}{2}, \\log_{2}(3), e^{2} …
    """
    out = _fmt_sym_core(expr, tex=True)
    if out is None:
        return None
    return _LOG_QUOT_TEX_RE.sub(lambda m: f"\\log_{{{m.group(2)}}}({m.group(1)})", out)


def _fmt_sym_core(expr, tex=False):
    try:
        expr = sympify(expr)
        if expr.free_symbols or expr.is_Float or expr.has(oo, -oo, zoo, nan):
            return None
        if expr.is_Integer:
            n = int(expr)
            return str(n) if abs(n) <= _FMT_INT_MAX else None
        if expr.is_Rational:
            if abs(expr.p) > _FMT_INT_MAX or expr.q > _FMT_INT_MAX:
                return None
            sgn = '-' if expr.p < 0 else ''
            return sgn + _frac_txt(abs(expr.p), expr.q, tex)
        terms = Add.make_args(expr)
        if len(terms) > 3:
            return None
        parsed = []
        for t in terms:
            pt = _fmt_term(t, tex)
            if pt is None:
                return None
            parsed.append(pt)

        def term_str(pt, lead):
            sign_, num, den_f, q = pt
            if q > 1 or den_f:
                body = _frac_txt(num, _join_den(q, den_f, tex), tex)
            else:
                body = num
            if lead:
                return ('-' if sign_ < 0 else '') + body
            return ('-' if sign_ < 0 else '+') + body

        if len(parsed) == 1:
            return term_str(parsed[0], True)

        # Порядок: рациональные слагаемые первыми; если первое отрицательное,
        # а есть положительное — положительное вперёд (√13−1, а не −1+√13).
        parsed.sort(key=lambda pt: 0 if pt[1].isdigit() else 1)
        if parsed[0][0] < 0:
            for i, pt in enumerate(parsed):
                if pt[0] > 0:
                    parsed.insert(0, parsed.pop(i))
                    break

        # Общий целый знаменатель: 1/2 + √13/2 → (1+√13)/2
        qs = [pt[3] for pt in parsed]
        if all(not pt[2] for pt in parsed) and any(q > 1 for q in qs):
            d = 1
            for q in qs:
                d = d * q // math.gcd(d, q)
            if d <= _FMT_INT_MAX:
                inner = ''
                for i, (sign_, num, den_f, q) in enumerate(parsed):
                    mult = d // q
                    # num — либо число, либо [число]·база
                    k = 0
                    while k < len(num) and num[k].isdigit():
                        k += 1
                    digits, tail = num[:k], num[k:]
                    coef = (int(digits) if digits else 1) * mult
                    body = ('' if (coef == 1 and tail) else str(coef)) + tail
                    inner += ('-' if sign_ < 0 else ('' if i == 0 else '+')) + body
                return _frac_txt(inner, d, tex) if tex else f'({inner})/{d}'

        return ''.join(term_str(pt, i == 0) for i, pt in enumerate(parsed))
    except Exception:
        return None


def fmt_exact_or(v, c=None, dec=None, tex=False):
    """
    Текст координаты: точная форма c (если задана, форматируется и
    согласуется с v), иначе десятичная запись dec(v) (по умолчанию fmt_num).
    tex=True — фрагмент mathtext.
    """
    if c is not None:
        try:
            if _exact_matches(c, v):
                s = fmt_sym_tex(c) if tex else fmt_sym(c)
                if s is not None:
                    return s
        except Exception:
            pass
    return (dec or fmt_num)(v)



# ══════════════════════════════════════════════════════════════
#  ЧИСЛЕННЫЙ ПОИСК ТОЧЕК
# ══════════════════════════════════════════════════════════════

def _refine_extremum(f, x_vals, idx, is_max):
    """
    Уточняет экстремум, найденный на сетке в узле idx, одномерной
    минимизацией на скобке [x[idx-1], x[idx+1]] (minimize_scalar, bounded,
    xatol=1e-10): x точен до ~1e-7, y — до ~1e-12. Если уточнение не
    удалось или дало худшее значение — остаётся узел сетки.
    """
    i0 = max(0, idx - 1)
    i1 = min(len(x_vals) - 1, idx + 1)
    a, b = float(x_vals[i0]), float(x_vals[i1])
    sgn = -1.0 if is_max else 1.0
    y_grid = _safe_val(f, float(x_vals[idx]))

    def obj(t):
        v = _safe_val(f, t)
        return sgn * v if np.isfinite(v) else 1e300

    try:
        res = minimize_scalar(obj, bounds=(a, b), method='bounded',
                              options={'xatol': 1e-10, 'maxiter': 300})
        xr = float(res.x)
        yr = _safe_val(f, xr)
        if (np.isfinite(yr) and a <= xr <= b
                and (not np.isfinite(y_grid) or sgn * yr <= sgn * y_grid + 1e-12)):
            return xr, yr
    except Exception:
        pass
    return float(x_vals[idx]), float(y_grid)


def find_extrema_numerical(f, x_vals, y_vals, y_lim_b, y_lim_t, disc_pts_x,
                           allow_pi=True, allow_e=True):
    """
    Локальные экстремумы по сетке (argrelextrema) с последующим уточнением
    (_refine_extremum). Возвращает список (x, y) float — точных численно;
    подпись (точная форма) подбирается уже в plot_function.
    Параметры allow_pi/allow_e сохранены для совместимости и не используются.
    """
    finite_mask = np.isfinite(y_vals)
    yf = np.where(finite_mask, y_vals, np.nan)

    # Адаптивный order: пропорционален доле конечных точек
    n_finite = int(np.sum(finite_mask))
    n_total  = len(y_vals)
    density  = n_finite / max(n_total, 1)
    order    = max(3, int(10 * density))

    maxima_idx = argrelextrema(np.nan_to_num(yf, nan=-1e18), np.greater, order=order)[0]
    minima_idx = argrelextrema(np.nan_to_num(yf, nan=+1e18), np.less,    order=order)[0]

    # Порог near_disc: 1% от диапазона X, но не более 0.1
    # (был 0.3 — слишком много, отсекал экстремумы рядом с разрывом)
    x_span = x_vals[-1] - x_vals[0]
    disc_tol = min(0.1, x_span * 0.01)

    def near_disc(xv):
        return any(abs(xv - d) < disc_tol for d in disc_pts_x)

    # Исключаем точки у границ NaN-областей (артефакты nan_to_num)
    nan_edges = set()
    for i in range(1, len(finite_mask)):
        if finite_mask[i] != finite_mask[i-1]:
            nan_edges.add(i-1)
            nan_edges.add(i)

    extrema = []
    seen = []
    for idx, is_max in ([(i, True) for i in maxima_idx] +
                        [(i, False) for i in minima_idx]):
        if idx in nan_edges:
            continue
        # Крайние узлы окна — не экстремумы (функция продолжается за окном)
        if idx == 0 or idx == len(x_vals) - 1:
            continue
        ex, ey = x_vals[idx], y_vals[idx]
        if not np.isfinite(ey):
            continue
        if not (y_lim_b <= ey <= y_lim_t):
            continue
        if near_disc(ex):
            continue
        if any(abs(ex - s) < 0.2 for s in seen):
            continue
        seen.append(ex)
        ex, ey = _refine_extremum(f, x_vals, idx, is_max)
        if not (y_lim_b <= ey <= y_lim_t):
            continue
        extrema.append((float(ex), float(ey)))
    return extrema


def find_x_intercepts(f, x_vals, y_vals, disc_pts, allow_pi=True, allow_e=True):
    """
    Нули функции: смена знака на сетке → brentq (xtol=1e-12); точные нули
    в узлах сетки (y == 0) берутся как есть. Возвращает отсортированные float.
    """
    zeros = []
    seen = []
    disc_x = [d[0] if isinstance(d, tuple) else d for d in disc_pts]
    x_span = x_vals[-1] - x_vals[0]
    disc_tol = min(0.1, x_span * 0.01)

    def is_near_disc(xv):
        return any(abs(xv - d) < disc_tol for d in disc_x)

    def add_zero(xz):
        if not is_near_disc(xz) and not any(abs(xz - s) < 0.1 for s in seen):
            seen.append(xz)
            zeros.append(float(xz))

    def fscalar(t):
        v = _safe_val(f, t)
        return v if np.isfinite(v) else float('nan')

    n = len(y_vals)

    def isolated_zero(i):
        """Точный ноль в узле i — корень, только если соседние узлы ненулевые
        (иначе это плато f ≡ 0, и подписывать каждый узел бессмысленно)."""
        if i > 0:
            yp = y_vals[i - 1]
            if not np.isfinite(yp) or yp == 0.0:
                return False
        if i + 1 < n:
            yn = y_vals[i + 1]
            if not np.isfinite(yn) or yn == 0.0:
                return False
        return True

    if n >= 2:
        ya, yb = y_vals[:-1], y_vals[1:]
        with np.errstate(invalid='ignore', over='ignore'):
            finite = np.isfinite(ya) & np.isfinite(yb)
            zero_at = np.where(finite & (ya == 0.0))[0]
            cross = np.where(finite & (ya * yb < 0))[0]
        for i in zero_at:
            if isolated_zero(int(i)):
                add_zero(x_vals[i])
        for i in cross:
            try:
                xz = brentq(fscalar, x_vals[i], x_vals[i + 1], xtol=1e-12)
                add_zero(xz)
            except Exception:
                pass
    if n and np.isfinite(y_vals[-1]) and y_vals[-1] == 0.0 and isolated_zero(n - 1):
        add_zero(x_vals[-1])

    return sorted(zeros)


def find_y_intercept(f, disc_pts, y_lim_b, y_lim_t):
    if not any(abs(d) < 1e-9 for d in disc_pts):
        val = _safe_val(f, 0)
        if np.isfinite(val) and y_lim_b <= val <= y_lim_t:
            return round(val, 6)
    probes = [_safe_val(f, eps) for eps in [1e-7, -1e-7, 1e-6, -1e-6]]
    probes = [v for v in probes if np.isfinite(v)]
    if probes and max(probes) - min(probes) < 1e-4:
        val = sum(probes) / len(probes)
        if y_lim_b <= val <= y_lim_t:
            return round(val, 6)
    return None




def make_y_array(f, x_vals, disc_pts, y_lim_b, y_lim_t, tol=0.05):
    """Вычисляет y (векторно, см. _eval_array); маскирует окрестность разрывов и выбросы."""
    y_span = y_lim_t - y_lim_b
    ys = _eval_array(f, x_vals)
    for dp in disc_pts:
        ys[np.abs(x_vals - dp) < tol] = np.nan
    ys[np.abs(ys) > y_span * 20] = np.nan
    return ys


def find_intersections(f1, f2, x_vals, y1_vals, y2_vals, disc_pts_x):
    """
    Находит точки пересечения двух функций:
    1. Знакосмены (y1-y2) → Брент (обычные пересечения)
    2. Локальные минимумы |y1-y2| близкие к 0 → касательные точки
    Возвращает список (x, y) float без округления.
    """
    diff_arr = y1_vals - y2_vals
    points = []
    seen = []

    disc_tol = min(0.1, (x_vals[-1] - x_vals[0]) * 0.01)

    def near_disc(xv):
        return any(abs(xv - d) < disc_tol for d in disc_pts_x)

    def add_point(xz):
        if near_disc(xz):
            return
        if any(abs(xz - s) < 0.05 for s in seen):
            return
        yz = _safe_val(f1, xz)
        if np.isfinite(yz):
            seen.append(xz)
            points.append((float(xz), float(yz)))

    def diff_f(x):
        v1 = _safe_val(f1, x)
        v2 = _safe_val(f2, x)
        return v1 - v2 if (np.isfinite(v1) and np.isfinite(v2)) else float('nan')

    abs_diff = np.abs(diff_arr)

    if len(diff_arr) >= 2:
        da, db = diff_arr[:-1], diff_arr[1:]
        with np.errstate(invalid='ignore', over='ignore'):
            finite = np.isfinite(da) & np.isfinite(db)
            zero_at = np.where(finite & (da == 0.0))[0]
            cross = np.where(finite & (da * db < 0))[0]
        # Точное совпадение в узле сетки — только изолированное (соседние
        # узлы ненулевые), иначе кривые совпадают на отрезке и подписывать
        # каждый узел бессмысленно
        for i in zero_at:
            i = int(i)
            prev_ok = (i == 0) or (np.isfinite(diff_arr[i - 1]) and diff_arr[i - 1] != 0.0)
            if prev_ok and diff_arr[i + 1] != 0.0:
                add_point(x_vals[i])
        # Случай 1: знакосмена → обычное пересечение
        for i in cross:
            try:
                xz = brentq(diff_f, x_vals[i], x_vals[i + 1], xtol=1e-12)
                add_point(xz)
            except Exception:
                pass

    # Случай 2: локальный минимум |diff| → касательная точка
    # Ищем точки где |diff| минимально и очень мало
    finite_mask = np.isfinite(abs_diff)
    if np.sum(finite_mask) > 3:
        # Адаптивный порог: 0.1% от максимального |diff|
        threshold = max(np.nanmax(abs_diff) * 0.001, 1e-6)
        local_mins = argrelextrema(
            np.where(finite_mask, abs_diff, np.inf), np.less, order=3)[0]
        for idx in local_mins:
            if abs_diff[idx] < threshold:
                # Уточняем минимум через minimize_scalar
                try:
                    i0 = max(0, idx - 5)
                    i1 = min(len(x_vals) - 1, idx + 5)
                    res = minimize_scalar(
                        lambda x: abs(diff_f(x)),
                        bounds=(x_vals[i0], x_vals[i1]),
                        method='bounded',
                        options={'xatol': 1e-10}
                    )
                    if res.fun < threshold:
                        add_point(res.x)
                except Exception:
                    add_point(x_vals[idx])

    return points



# ══════════════════════════════════════════════════════════════
#  DRAGGABLE ANNOTATIONS
# ══════════════════════════════════════════════════════════════

# Пользовательские смещения подписей точек, переживающие перерисовку
# (в live-режиме график перестраивается при каждом изменении — перетащенная
# подпись не должна «прыгать» назад). Ключ: (текст, round(px,3), round(py,3)),
# значение: (dx, dy) в ПУНКТАХ относительно стандартного положения.
ANNOTATION_OFFSETS = {}


def reset_annotation_offsets():
    ANNOTATION_OFFSETS.clear()


class DraggableAnnotation:
    """Позволяет перетаскивать текстбокс аннотации мышью."""

    # Общий «замок» захвата на время одного события press: когда несколько
    # подписей лежат друг на друге (например (1,0) и (1.1,0)), без этого
    # КАЖДАЯ из них захватывала бы один и тот же клик и они двигались бы
    # вместе. Здесь первый объект, чей бокс содержит курсор, «забирает»
    # это событие себе, а остальные при том же press видят, что оно уже
    # занято, и не реагируют. Ключ — сам объект события (у каждого клика
    # он свой), чтобы замок автоматически «сбрасывался» на следующем клике.
    # Держим сам объект события (сильная ссылка): пока он жив, его адрес
    # не может достаться новому событию. Раньше ключом был id(event), а
    # Python переиспользует id освобождённых объектов — следующий клик
    # получал тот же id, считался «уже захваченным» и перетаскивание
    # переставало работать после первого раза.
    _claimed_event = None    # последнее захваченное событие press

    def __init__(self, annotation, key=None, home=None, pt_scale=None):
        # key      — ключ в ANNOTATION_OFFSETS (None — не запоминать);
        # home     — стандартное положение текста (в координатах данных);
        # pt_scale — (единиц данных на пункт по X, по Y).
        self.ann   = annotation
        self.press = None
        self.fig   = annotation.figure
        self.key   = key
        self.home  = home
        self.pt_scale = pt_scale
        self.ann.set_picker(True)
        canvas = self.fig.canvas
        self.cidpress   = canvas.mpl_connect('button_press_event',   self.on_press)
        self.cidrelease = canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion  = canvas.mpl_connect('motion_notify_event',  self.on_motion)

    def disconnect(self):
        """Отписывается от событий мыши (перед перерисовкой графика)."""
        try:
            canvas = self.fig.canvas
            for cid in (self.cidpress, self.cidrelease, self.cidmotion):
                canvas.mpl_disconnect(cid)
        except Exception:
            pass
        self.press = None

    def on_press(self, event):
        if event.inaxes != self.ann.axes:
            return
        # Если этот же клик уже захвачен другой подписью — не реагируем,
        # чтобы перетаскивалась ровно одна точка, а не все под курсором.
        if DraggableAnnotation._claimed_event is event:
            return
        # Проверяем попадание курсора в бокс аннотации
        contains, _ = self.ann.contains(event)
        if not contains:
            return
        # Забираем это событие себе (замок до следующего клика)
        DraggableAnnotation._claimed_event = event
        # Запоминаем начальное положение текста (в координатах данных)
        self.press = (event.xdata, event.ydata,
                      self.ann.get_position())  # (mouse_x, mouse_y, ann_xy)

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ann.axes:
            return
        mx0, my0, (ax0, ay0) = self.press
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - mx0
        dy = event.ydata - my0
        # Переводим аннотацию в абсолютные координаты данных
        self.ann.xyann = (ax0 + dx, ay0 + dy)
        self.ann.set_position((ax0 + dx, ay0 + dy))
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.press is not None:
            self._store_offset()
        self.press = None
        self.fig.canvas.draw_idle()

    def _store_offset(self):
        """Запоминает смещение от стандартного положения (в пунктах)."""
        if self.key is None or self.home is None or not self.pt_scale:
            return
        try:
            x, y = self.ann.get_position()
            sx, sy = self.pt_scale
            dx = (x - self.home[0]) / sx if sx else 0.0
            dy = (y - self.home[1]) / sy if sy else 0.0
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                ANNOTATION_OFFSETS.pop(self.key, None)
            else:
                ANNOTATION_OFFSETS[self.key] = (float(dx), float(dy))
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════
#  СВОБОДНЫЕ ПОДПИСИ НА ГРАФИКЕ (добавляются прямо на холсте)
# ══════════════════════════════════════════════════════════════

# Персистентный список свободных подписей — хранится на уровне модуля,
# чтобы переживать повторные построения графика (Plot Graph), как и
# _ACTIVE_DRAGGABLES. Каждый элемент — dict с координатами в данных,
# текстом, поворотом и стилем.
FREE_TEXTS = []

# Держим ссылку на текущий менеджер тут же, на уровне модуля — иначе
# объект (и его подписки на события мыши через mpl_connect) может быть
# собран сборщиком мусора, ровно как было с DraggableAnnotation раньше.
_active_free_text_manager = None


class FreeTextManager:
    """
    Интерактивные подписи прямо на графике:
      • двойной клик / правый клик по пустому месту — добавить подпись
      • левый клик + перетаскивание по существующей подписи — переместить
      • двойной клик по существующей подписи — изменить текст
      • колесо мыши над подписью — повернуть
      • правый клик по подписи — контекстное меню (изменить / повернуть
        в исходное положение / удалить)
    """

    ROTATE_STEP = 10  # градусов за один щелчок колеса

    def __init__(self, fig, ax, font_size=11):
        self.fig = fig
        self.ax = ax
        self.font_size = font_size
        self.entries = []   # [(record_dict, Text_artist), ...]
        self.drag = None

        # Фича: при каждом построении графика ВСЕ подписи получают текущий
        # Label size (font_size). Любой ранее заданный вручную размер
        # сбрасывается — пользователь может снова поменять его после.
        for record in FREE_TEXTS:
            record['fontsize'] = font_size
            self.entries.append((record, self._create_artist(record)))

        self.cid_press   = fig.canvas.mpl_connect('button_press_event',   self.on_press)
        self.cid_motion  = fig.canvas.mpl_connect('motion_notify_event',  self.on_motion)
        self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_scroll  = fig.canvas.mpl_connect('scroll_event',         self.on_scroll)

    def disconnect(self):
        """Отписывается от событий мыши (перед перерисовкой графика)."""
        try:
            canvas = self.fig.canvas
            for cid in (self.cid_press, self.cid_motion,
                        self.cid_release, self.cid_scroll):
                canvas.mpl_disconnect(cid)
        except Exception:
            pass
        self.drag = None

    # ── Отрисовка ──────────────────────────────────────────
    def _create_artist(self, record):
        return self.ax.text(
            record['x'], record['y'], bidi_display(prettify_math_text(record['text'])),
            color=record.get('color', '#000000'),
            fontsize=record.get('fontsize', self.font_size),
            rotation=record.get('rotation', 0),
            ha='center', va='center',
            picker=True,
            zorder=12,
        )

    def _find_hit(self, event):
        for record, artist in self.entries:
            contains, _ = artist.contains(event)
            if contains:
                return record, artist
        return None

    def _tk_root(self):
        try:
            return self.fig.canvas.get_tk_widget().winfo_toplevel()
        except Exception:
            return None

    # ── Мышь ────────────────────────────────────────────────
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        hit = self._find_hit(event)

        if event.button == 3:   # правая кнопка — контекстное меню
            self.drag = None
            self._show_context_menu(event, hit)
            return

        if event.button != 1:
            return

        if hit is not None:
            record, artist = hit
            if event.dblclick:
                self.drag = None
                self._edit_text(record, artist)
                return
            self.drag = (record, artist, event.xdata, event.ydata,
                         record['x'], record['y'])
        elif event.dblclick:
            self._add_text_at(event.xdata, event.ydata)

    def on_motion(self, event):
        if self.drag is None or event.inaxes != self.ax:
            return
        record, artist, mx0, my0, x0, y0 = self.drag
        if event.xdata is None or event.ydata is None:
            return
        new_x = x0 + (event.xdata - mx0)
        new_y = y0 + (event.ydata - my0)
        artist.set_position((new_x, new_y))
        record['x'], record['y'] = new_x, new_y
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        self.drag = None

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        hit = self._find_hit(event)
        if hit is None:
            return
        record, artist = hit
        step = self.ROTATE_STEP if event.button == 'up' else -self.ROTATE_STEP
        new_rotation = (record.get('rotation', 0) + step) % 360
        artist.set_rotation(new_rotation)
        record['rotation'] = new_rotation
        self.fig.canvas.draw_idle()

    # ── Добавление / редактирование / удаление ──────────────
    def _add_text_at(self, x, y):
        from tkinter import simpledialog
        text = simpledialog.askstring(tr("New label"), tr("Label text:"),
                                       parent=self._tk_root())
        if not text:
            return
        record = {'text': text, 'x': x, 'y': y, 'rotation': 0,
                  'color': '#000000', 'fontsize': self.font_size}
        FREE_TEXTS.append(record)
        self.entries.append((record, self._create_artist(record)))
        self.fig.canvas.draw_idle()

    def _edit_text(self, record, artist):
        from tkinter import simpledialog
        new_text = simpledialog.askstring(
            tr("Edit label"), tr("Label text:"),
            initialvalue=record['text'], parent=self._tk_root())
        if new_text is None:
            return
        if new_text == "":
            self._delete(record, artist)
            return
        record['text'] = new_text
        artist.set_text(bidi_display(prettify_math_text(new_text)))
        self.fig.canvas.draw_idle()

    def _delete(self, record, artist):
        try:
            artist.remove()
        except Exception:
            pass
        self.entries = [(r, a) for r, a in self.entries if r is not record]
        if record in FREE_TEXTS:
            FREE_TEXTS.remove(record)
        self.fig.canvas.draw_idle()

    def _reset_rotation(self, record, artist):
        record['rotation'] = 0
        artist.set_rotation(0)
        self.fig.canvas.draw_idle()

    def _set_color(self, record, artist):
        from tkinter import colorchooser
        initial = record.get('color', '#000000')
        result = colorchooser.askcolor(
            color=initial, title=tr("Label color"), parent=self._tk_root())
        if result is None or result[1] is None:
            return
        record['color'] = result[1]
        artist.set_color(result[1])
        self.fig.canvas.draw_idle()

    def _set_size(self, record, artist, size):
        record['fontsize'] = size
        artist.set_fontsize(size)
        self.fig.canvas.draw_idle()

    def _build_size_submenu(self, parent_menu, record, artist):
        import tkinter as tk
        sub = tk.Menu(parent_menu, tearoff=0)
        for size in (8, 10, 12, 14, 16, 20, 24, 28):
            sub.add_command(
                label=str(size),
                command=lambda s=size: self._set_size(record, artist, s))
        return sub

    def _show_context_menu(self, event, hit):
        import tkinter as tk
        root = self._tk_root()
        if root is None or event.guiEvent is None:
            return
        menu = tk.Menu(root, tearoff=0)
        if hit is not None:
            record, artist = hit
            menu.add_command(label=tr("Edit text"),
                              command=lambda: self._edit_text(record, artist))
            menu.add_command(label=tr("Text color..."),
                              command=lambda: self._set_color(record, artist))
            menu.add_cascade(label=tr("Text size"),
                              menu=self._build_size_submenu(menu, record, artist))
            menu.add_command(label=tr("Reset rotation"),
                              command=lambda: self._reset_rotation(record, artist))
            menu.add_separator()
            menu.add_command(label=tr("Delete"),
                              command=lambda: self._delete(record, artist))
        else:
            if event.xdata is None or event.ydata is None:
                return
            menu.add_command(
                label=tr("Add label here"),
                command=lambda: self._add_text_at(event.xdata, event.ydata))
        try:
            menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
        finally:
            menu.grab_release()


# ══════════════════════════════════════════════════════════════
#  ПОДПИСИ ОСЕЙ "x" / "y" — тоже интерактивные
# ══════════════════════════════════════════════════════════════

# Персистентные пользовательские настройки подписей осей: смещение от
# "домашнего" положения (в пунктах), цвет и размер. Переживают повторное
# построение графика. dx/dy — это пользовательский сдвиг поверх стандартного
# положения; домашнее положение (само по себе) при этом не меняется.
AXIS_LABELS = {
    'x': {'dx': 0.0, 'dy': 0.0, 'color': None, 'fontsize': None},
    'y': {'dx': 0.0, 'dy': 0.0, 'color': None, 'fontsize': None},
}

_active_axis_label_manager = None


class AxisLabelManager:
    """
    Делает подписи осей "x" и "y" интерактивными:
      • левый клик + перетаскивание — переместить подпись
      • правый клик — контекстное меню: цвет, размер, сбросить положение
    Удалять/добавлять эти подписи нельзя (в отличие от свободных).
    "Домашнее" положение фиксировано; Reset position возвращает к нему.
    """

    def __init__(self, fig, ax, artists, home_offsets, default_fontsize):
        # artists:      {'x': Text, 'y': Text}
        # home_offsets: {'x': (dx_pts, dy_pts), 'y': (...)} — стандартное
        #               смещение в пунктах от точки привязки (xy).
        self.fig = fig
        self.ax = ax
        self.artists = artists
        self.home = home_offsets
        self.default_fontsize = default_fontsize
        self.drag = None

        # Применяем сохранённые пользовательские настройки.
        # ВАЖНО (фича "размер от Label size"): размер при каждом построении
        # сбрасывается на текущий Label size — артист уже создан с этим
        # размером, поэтому просто очищаем сохранённый пользовательский
        # размер. Цвет и смещение (позиция) при этом сохраняются.
        for key, art in artists.items():
            st = AXIS_LABELS[key]
            st['fontsize'] = None
            if st['color']:
                art.set_color(st['color'])
            self._apply_offset(key)
            art.set_picker(True)

        self.cid_press   = fig.canvas.mpl_connect('button_press_event',   self.on_press)
        self.cid_motion  = fig.canvas.mpl_connect('motion_notify_event',  self.on_motion)
        self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)

    def disconnect(self):
        """Отписывается от событий мыши (перед перерисовкой графика)."""
        try:
            canvas = self.fig.canvas
            for cid in (self.cid_press, self.cid_motion, self.cid_release):
                canvas.mpl_disconnect(cid)
        except Exception:
            pass
        self.drag = None

    def _apply_offset(self, key):
        # Итоговое смещение = домашнее + пользовательский сдвиг (в пунктах)
        from matplotlib.transforms import ScaledTranslation
        hx, hy = self.home[key]
        st = AXIS_LABELS[key]
        self.artists[key].set_transform(
            self.ax.transData +
            ScaledTranslation((hx + st['dx']) / 72.0,
                              (hy + st['dy']) / 72.0,
                              self.fig.dpi_scale_trans))

    def _hit(self, event):
        for key, art in self.artists.items():
            contains, _ = art.contains(event)
            if contains:
                return key
        return None

    def _tk_root(self):
        try:
            return self.fig.canvas.get_tk_widget().winfo_toplevel()
        except Exception:
            return None

    def on_press(self, event):
        # ВАЖНО: НЕ требуем event.inaxes == self.ax. Подписи "x"/"y" стоят
        # у самых краёв осей и смещены наружу, поэтому физически находятся
        # ЗА прямоугольником осей — клик по ним приходит с inaxes=None.
        # contains() ниже работает в экранных координатах и не зависит от
        # границ осей, поэтому хит-тестим артисты напрямую.
        if event.x is None or event.y is None:
            return
        key = self._hit(event)
        if key is None:
            return
        if event.button == 3:
            self.drag = None
            self._menu(event, key)
            return
        if event.button == 1:
            self.drag = (key, event.x, event.y,
                         AXIS_LABELS[key]['dx'], AXIS_LABELS[key]['dy'])

    def on_motion(self, event):
        if self.drag is None:
            return
        key, px0, py0, dx0, dy0 = self.drag
        if event.x is None or event.y is None:
            return
        # Перемещение мыши в пикселях -> в пункты (1 пункт = dpi/72 пикселей)
        scale = 72.0 / self.fig.dpi
        AXIS_LABELS[key]['dx'] = dx0 + (event.x - px0) * scale
        AXIS_LABELS[key]['dy'] = dy0 + (event.y - py0) * scale
        self._apply_offset(key)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        self.drag = None

    # ── Контекстное меню ───────────────────────────────────
    def _set_color(self, key):
        from tkinter import colorchooser
        art = self.artists[key]
        result = colorchooser.askcolor(
            color=art.get_color(), title=f'"{key}" label color',
            parent=self._tk_root())
        if result is None or result[1] is None:
            return
        AXIS_LABELS[key]['color'] = result[1]
        art.set_color(result[1])
        self.fig.canvas.draw_idle()

    def _set_size(self, key, size):
        AXIS_LABELS[key]['fontsize'] = size
        self.artists[key].set_fontsize(size)
        self.fig.canvas.draw_idle()

    def _reset_position(self, key):
        AXIS_LABELS[key]['dx'] = 0.0
        AXIS_LABELS[key]['dy'] = 0.0
        self._apply_offset(key)
        self.fig.canvas.draw_idle()

    def _menu(self, event, key):
        import tkinter as tk
        root = self._tk_root()
        if root is None or event.guiEvent is None:
            return
        menu = tk.Menu(root, tearoff=0)
        menu.add_command(label=tr('"{key}" label color...', key=key),
                          command=lambda: self._set_color(key))
        sub = tk.Menu(menu, tearoff=0)
        for size in (8, 10, 12, 14, 16, 20, 24, 28):
            sub.add_command(label=str(size),
                            command=lambda s=size: self._set_size(key, s))
        menu.add_cascade(label=tr("Label size"), menu=sub)
        menu.add_separator()
        menu.add_command(label=tr("Reset position"),
                          command=lambda: self._reset_position(key))
        try:
            menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
        finally:
            menu.grab_release()


# Цвета для кривых (по порядку функций в FUNCS)
CURVE_COLORS = [
    "#000000",  # чёрный
    "#e05c2a",  # оранжевый
    "#2a7ae0",  # синий
    "#2ab850",  # зелёный
    "#9b2ae0",  # фиолетовый
    "#e0b02a",  # жёлтый
    "#e02a6a",  # малиновый
    "#2acce0",  # голубой
]

CURVE_WIDTHS = []   # ширина линий (пусто = 1.8 для всех)
CURVE_STYLES = []   # тип линий: "-", "--", ":" (пусто = "-" для всех)


# Держим ссылки на текущие DraggableAnnotation на уровне МОДУЛЯ, а не внутри
# plot_function(). matplotlib подписывает bound-методы (self.on_press и т.д.)
# через weakref — если на сам объект DraggableAnnotation не остаётся ни одной
# сильной ссылки, сборщик мусора может удалить его, и подписка на события
# тихо умирает. Раньше единственной сильной ссылкой был локальный список
# внутри plot_function(), который "жил" только пока эта функция не вернулась
# (а она блокируется в plt.show() лишь при первом построении графика за сессию —
# при повторном построении plt.show() может вернуться сразу, и список тут же
# терялся вместе со всеми подписками на drag).
_ACTIVE_DRAGGABLES = []


def _short_error(exc):
    """Короткое человекочитаемое описание ошибки одной кривой."""
    from tokenize import TokenError
    try:
        from sympy import SympifyError
    except Exception:          # pragma: no cover
        SympifyError = ()
    if isinstance(exc, (SyntaxError, TokenError, SympifyError, IndexError)):
        return "Syntax error"
    if isinstance(exc, NameError):
        msg = str(exc).strip()
        return msg if msg else "Unknown name"
    if isinstance(exc, TypeError):
        return "Invalid expression"
    if isinstance(exc, ZeroDivisionError):
        return "Division by zero"
    if isinstance(exc, KeyError) and any(k in str(exc) for k in ('ComplexInfinity', 'zoo', 'NaN', 'nan')):
        return "Undefined value (division by zero or log base 1?)"
    if isinstance(exc, ValueError):
        msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ''
        return (msg or "Invalid value")[:80]
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ''
    name = type(exc).__name__
    out = f"{name}: {msg}" if msg else name
    return out[:80]


def _vline_exact(func_str):
    """Точное значение c из строки 'x = c' (parse_exact правой части)."""
    try:
        _, _, right = str(func_str).partition('=')
        return parse_exact(right)
    except Exception:
        return None


def _disconnect_previous():
    """Отписывает интерактивные объекты предыдущего построения."""
    global _active_free_text_manager, _active_axis_label_manager
    for d in _ACTIVE_DRAGGABLES:
        try:
            d.disconnect()
        except Exception:
            pass
    _ACTIVE_DRAGGABLES.clear()
    if _active_free_text_manager is not None:
        try:
            _active_free_text_manager.disconnect()
        except Exception:
            pass
        _active_free_text_manager = None
    if _active_axis_label_manager is not None:
        try:
            _active_axis_label_manager.disconnect()
        except Exception:
            pass
        _active_axis_label_manager = None


# View Window «вписывается» в холст по МЕНЬШЕЙ стороне, а по большей стороне
# показывается дополнительный диапазон (масштаб тот же). Так сетка остаётся
# квадратной, пока диапазоны X и Y равны, а при увеличении окна оси не
# растягиваются, а удлиняются в сторону роста окна. False — старое
# поведение: окно растягивается точно на холст.
EXTEND_TO_CANVAS = True


def effective_limits(w_px, h_px):
    """
    Видимые пределы для холста w_px×h_px при текущем View Window: окно
    занимает меньшую сторону холста целиком, лишнее место по большей
    стороне добавляет диапазон симметрично с обеих сторон.
    """
    xl, xr, yb, yt = X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T
    try:
        w_px = float(w_px)
        h_px = float(h_px)
        if w_px > h_px > 0:
            extra = (xr - xl) * (w_px / h_px - 1.0) / 2.0
            xl, xr = xl - extra, xr + extra
        elif h_px > w_px > 0:
            extra = (yt - yb) * (h_px / w_px - 1.0) / 2.0
            yb, yt = yb - extra, yt + extra
    except Exception:
        pass
    return xl, xr, yb, yt


def plot_function(fig=None):
    """
    Строит график по текущим настройкам модуля.
      fig=None — standalone: своя фигура 6×6, plt.show().
      fig задана — очищает её, рисует в fig.add_subplot(111), plt.show() НЕ
      вызывает (GUI с встроенным холстом). При EXTEND_TO_CANVAS пределы
      расширяются под пропорции холста (см. effective_limits).
    Возвращает {'ax': ax, 'errors': {индекс_функции: сообщение},
                'pending': были ли отложены символьные вычисления}.
    Ошибка одной кривой не роняет весь график: она попадает в errors,
    остальные кривые рисуются.
    """
    global X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T
    saved = (X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T)
    if fig is not None and EXTEND_TO_CANVAS:
        try:
            w_px = fig.get_figwidth() * fig.dpi
            h_px = fig.get_figheight() * fig.dpi
            X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T = effective_limits(w_px, h_px)
        except Exception:
            X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T = saved
    try:
        return _plot_function_impl(fig)
    finally:
        # Настройки пользователя (View Window) не трогаем — вернуть как было
        X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T = saved


def _plot_function_impl(fig=None):
    global CURVE_WIDTHS, CURVE_STYLES, CURVE_COLORS
    global _active_free_text_manager, _active_axis_label_manager

    _DRAW_STATE['pending'] = False
    _disconnect_previous()
    errors = {}

    x_span = X_LIM_R - X_LIM_L
    y_span = Y_LIM_T - Y_LIM_B

    ASYM_COLOR   = "#555555"
    GRID_COLOR   = "#cccccc"
    LABEL_COLOR  = "#000000"

    _scale_geom = math.sqrt(abs(x_span * y_span))  # оставляем для совместимости
    _FS       = max(4, min(24, FONT_SIZE))     # зажимаем в разумные пределы
    AXIS_FS   = max(6, int(_FS * 1.1))
    TICK_FS   = max(4, int(_FS * 0.9))

    standalone = fig is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig.clear()
        ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    ax.set_xlim(X_LIM_L, X_LIM_R)
    ax.set_ylim(Y_LIM_B, Y_LIM_T)

    # Область построения всегда занимает всю фигуру; пределы уже подогнаны
    # под пропорции холста (effective_limits), поэтому при равных диапазонах
    # X и Y клетки сетки квадратные, а при разных — график растянут так,
    # как задано View Window.
    ax.set_aspect('auto')
    try:
        # Границы осей совпадают с границами холста: полей нет вовсе
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    except Exception:
        pass
    # Размер осей в пикселях — для штриховки под 45° на ЭКРАНЕ и смещений
    # подписей в пунктах при неравных масштабах по X и Y.
    try:
        _bb = ax.get_position()
        _ax_w_px = max(1.0, _bb.width * fig.get_figwidth() * fig.dpi)
        _ax_h_px = max(1.0, _bb.height * fig.get_figheight() * fig.dpi)
    except Exception:
        _ax_w_px, _ax_h_px = 600.0, 600.0

    # ── Сетка ───────────────────────────────────
    def _multiples_in(lim_lo, lim_hi, step):
        """Все кратные step на [lim_lo, lim_hi] — считаем от границ окна, а не
        от нуля: раньше индексы шли в пределах ±(ширина окна/шаг), и дальше
        ~12 единиц от начала координат сетка «пропадала»."""
        if not (step > 0) or not (math.isfinite(lim_lo) and math.isfinite(lim_hi)):
            return []
        k_lo = int(math.ceil(lim_lo / step - 1e-9))
        k_hi = int(math.floor(lim_hi / step + 1e-9))
        if k_hi < k_lo or k_hi - k_lo > _MAX_GRID_LINES:
            return []
        return [round(k * step, 10) for k in range(k_lo, k_hi + 1)]

    def make_ticks_for_grid(lim_lo, lim_hi, step):
        return _multiples_in(lim_lo, lim_hi, step)

    if GRID:
        x_grid_vals = [v for v in make_ticks_for_grid(X_LIM_L, X_LIM_R, X_GRID)
                       if X_LIM_L + X_GRID - 1e-9 <= v <= X_LIM_R - X_GRID + 1e-9]
        y_grid_vals = [v for v in make_ticks_for_grid(Y_LIM_B, Y_LIM_T, Y_GRID)
                       if Y_LIM_B + Y_GRID - 1e-9 <= v <= Y_LIM_T - Y_GRID + 1e-9]
        ax.xaxis.set_major_locator(ticker.FixedLocator(x_grid_vals))
        ax.yaxis.set_major_locator(ticker.FixedLocator(y_grid_vals))
        ax.grid(True, which='major', color=GRID_COLOR, linewidth=0.6,
                linestyle='-', zorder=1)
    else:
        ax.grid(False)

    # Позиции осей
    x_axis_y = max(Y_LIM_B, min(Y_LIM_T, 0.0))
    y_axis_x = max(X_LIM_L, min(X_LIM_R, 0.0))

    # ── Оси со стрелками ────────────────────────
    ax.annotate("", xy=(X_LIM_R, x_axis_y), xytext=(X_LIM_L, x_axis_y),
                arrowprops=dict(arrowstyle='->', color=LABEL_COLOR,
                                lw=1.2, mutation_scale=12), zorder=4)
    ax.annotate("", xy=(y_axis_x, Y_LIM_T), xytext=(y_axis_x, Y_LIM_B),
                arrowprops=dict(arrowstyle='->', color=LABEL_COLOR,
                                lw=1.2, mutation_scale=12), zorder=4)

    # Подписи осей — ВНУТРИ области построения (у полей нулевая ширина):
    # «x» над стрелкой у правого края, «y» справа от стрелки у верхнего.
    _axis_label_x = ax.text(X_LIM_R, x_axis_y, "$x$" if MATHTEXT_LABELS else "x",
                ha='right', va='bottom', fontsize=AXIS_FS, color=LABEL_COLOR,
                clip_on=False, zorder=6)
    _axis_label_y = ax.text(y_axis_x, Y_LIM_T, "$y$" if MATHTEXT_LABELS else "y",
                ha='left', va='top', fontsize=AXIS_FS, color=LABEL_COLOR,
                clip_on=False, zorder=6)
    # Домашние смещения (в пунктах) от концов стрелок.
    _axis_home_offsets = {'x': (-5.0, 4.0), 'y': (5.0, -4.0)}

    # ── Деления на осях ─────────────────────────
    def make_ticks(lim_lo, lim_hi, step):
        # деления — кратные шага, отступающие от краёв не меньше чем на шаг, кроме нуля
        return [v for v in _multiples_in(lim_lo + step - 1e-9, lim_hi - step + 1e-9, step)
                if abs(v) > 1e-9]

    for v in make_ticks(X_LIM_L, X_LIM_R, X_GRID):
        if not X_HIDE:
            ax.plot(v, x_axis_y, '|', color=LABEL_COLOR, markersize=4, markeredgewidth=0.8, zorder=5)
            ax.annotate(tick_label(v), xy=(v, x_axis_y),
                        xytext=(0, -6), textcoords='offset points',
                        ha='center', va='top', fontsize=TICK_FS, color=LABEL_COLOR)

    for v in make_ticks(Y_LIM_B, Y_LIM_T, Y_GRID):
        if not Y_HIDE:
            ax.plot(y_axis_x, v, '_', color=LABEL_COLOR, markersize=4, markeredgewidth=0.8, zorder=5)
            ax.annotate(tick_label(v), xy=(y_axis_x, v),
                        xytext=(-6, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=TICK_FS, color=LABEL_COLOR)

    # ── Вспомогательные функции подписи ──────────
    TEX = bool(MATHTEXT_LABELS)

    def fmt2(v):
        """Десятичная запись координаты (2 знака); кратные π/e — через fmt_num."""
        if abs(v) < 1e-12:
            return '0'
        # fmt_num распознаёт кратные π (знаменатели 1,2,3,4,6) и e; численные
        # значения теперь точны до ~1e-10, поэтому допуск узкий.
        s = fmt_num(v, tol=1e-9, tex=TEX)
        if 'π' in s or 'pi' in s or 'e' in s:
            return s
        s = f'{v:.2f}'
        s = s.rstrip('0').rstrip('.')
        if s in ('-0', '', '-'):
            s = '0'
        return s

    def coord(v, c=None):
        """Текст одной координаты: точная форма c (если есть и согласуется), иначе fmt2."""
        return fmt_exact_or(v, c, dec=fmt2, tex=TEX)

    def pt_label(xs, ys):
        """Подпись точки из двух готовых фрагментов координат."""
        if TEX:
            return math_label(f"({xs},\\ {ys})")
        return f"({xs}, {ys})"

    def label_xy(px, py, cx=None, cy=None):
        return pt_label(coord(px, cx), coord(py, cy))


    draggables = _ACTIVE_DRAGGABLES

    # Множество уже подписанных точек (округлённые координаты), чтобы одна и
    # та же точка не подписывалась дважды — например (1, 0) у (x-1)(x+2) и у
    # (x-1). Сбрасывается на каждое построение. Ключ — округлённые (x, y).
    _annotated_points = set()
    _marked_points = set()

    def _point_key(px, py):
        return (round(float(px), 3), round(float(py), 3))

    def mark_point(px, py, color, **kw):
        """Рисует кружок-маркер точки, но только если такой ещё не рисовали."""
        key = _point_key(px, py)
        if key in _marked_points:
            return False
        _marked_points.add(key)
        opts = dict(markersize=4, zorder=9)
        opts.update(kw)
        ax.plot(px, py, 'o', color=color, **opts)
        return True

    # Переводим offset points → единицы данных для xytext
    # чтобы drag работал в системе координат данных
    _fig_w_in, _fig_h_in = fig.get_size_inches()
    pt_to_x = x_span / (_ax_w_px / fig.dpi * 72)      # единиц данных в одном пункте
    pt_to_y = y_span / (_ax_h_px / fig.dpi * 72)

    def annotate_point(px, py, label, above, color, side=None):
        # Дедупликация: если такая точка уже подписана — пропускаем.
        key = _point_key(px, py)
        if key in _annotated_points:
            return None
        _annotated_points.add(key)

        near_y = abs(px - y_axis_x) < x_span * 0.12
        BBOX   = dict(boxstyle='round,pad=0.3', fc='none',
                      ec='none', alpha=0.95)

        if side is not None:
            dx_pt = 8 if side == 'right' else -8
            ha    = 'left' if side == 'right' else 'right'
            txt_x = px + dx_pt * pt_to_x
            txt_y = py + 6 * pt_to_y
        else:
            dy_pt = 8 if above else -8
            va    = 'bottom' if above else 'top'
            txt_x = px
            txt_y = py + dy_pt * pt_to_y

        ha  = ha  if side is not None else 'center'
        va  = 'bottom' if (side is not None or above) else 'top'

        # Пользовательское смещение (перетащенная ранее подпись)
        home = (txt_x, txt_y)
        off_key = (str(label), round(float(px), 3), round(float(py), 3))
        off = ANNOTATION_OFFSETS.get(off_key)
        if off:
            try:
                txt_x = home[0] + float(off[0]) * pt_to_x
                txt_y = home[1] + float(off[1]) * pt_to_y
            except Exception:
                txt_x, txt_y = home

        ann = ax.annotate(
            label,
            xy=(px, py),
            xytext=(txt_x, txt_y),
            xycoords='data',
            textcoords='data',
            ha=ha, va=va,
            fontsize=_FS, color=color,
            bbox=BBOX,
            annotation_clip=False,
            zorder=10,
        )
        draggables.append(DraggableAnnotation(ann, key=off_key, home=home,
                                              pt_scale=(pt_to_x, pt_to_y)))
        return ann

    def choose_side(px, py, f):
        # Зондируем на фиксированное расстояние в данных: 15% x_span
        probe = x_span * 0.15
        pts_r = [_safe_val(f, px + probe * k) for k in [0.3, 0.6, 1.0]]
        pts_l = [_safe_val(f, px - probe * k) for k in [0.3, 0.6, 1.0]]
        dist_r = np.nanmean([abs(v - py) for v in pts_r if np.isfinite(v)] or [0])
        dist_l = np.nanmean([abs(v - py) for v in pts_l if np.isfinite(v)] or [0])
        return 'right' if dist_r >= dist_l else 'left'

    ON_Y_AXIS_TOL = x_span * 0.05
    x_vals = np.linspace(X_LIM_L, X_LIM_R, 6000)

    # ── Точные подписи: обёртки, никогда не бросающие исключений ──
    def cands(expr, sym, kind, lo, hi):
        try:
            return exact_candidates(expr, sym, kind, lo, hi)
        except Exception:        # SymbolicPending (флаг pending уже выставлен) и пр.
            return []

    def exact_root(v, expr, sym, cand_list, tag):
        return identify_value(v, cand_list,
                              verify=lambda c: _is_zero_at(expr, sym, c),
                              cache_key=(tag, _ekey(expr), str(sym)))

    def exact_crit(v, expr, sym, cand_list):
        return identify_value(v, cand_list,
                              verify=lambda c: _is_zero_at(diff(expr, sym), sym, c),
                              tol=1e-7, cache_key=('crit', _ekey(expr), str(sym)))

    # ══════════════════════════════════════════════
    #  ЦИКЛ ПО ФУНКЦИЯМ
    # ══════════════════════════════════════════════
    all_vasymps  = set()
    all_hasymps  = set()
    func_data    = []   # [(f, y_vals, color), ...] — для заливки после цикла
    # Параллельный список метаданных кривой (та же длина/порядок, что func_data)
    # для расчёта пересечений между разными типами кривых. Каждый элемент —
    # dict: {'kind','color', и колбэки}. kind ∈ 'func' | 'implicit' | 'vline' | 'error'.
    curve_meta   = []

    def draw_vline(func_idx, func_str, payload, color, lw, ls):
        cx = payload
        if cx is None or not np.isfinite(cx):
            raise ValueError(f"cannot parse the constant in '{func_str}'")
        cx_exact = _vline_exact(func_str)
        # Учитываем домен функции (если задан) как ограничение по Y:
        # домен для вертикали трактуем как диапазон Y, на котором
        # её рисовать. По умолчанию — весь видимый Y.
        y_lo_line, y_hi_line = Y_LIM_B, Y_LIM_T
        if func_idx < len(FUNC_DOMAINS):
            d_from, d_to = FUNC_DOMAINS[func_idx]
            if d_from != float("-inf"):
                y_lo_line = max(y_lo_line, d_from)
            if d_to != float("inf"):
                y_hi_line = min(y_hi_line, d_to)
        if X_LIM_L <= cx <= X_LIM_R and y_lo_line < y_hi_line:
            ax.plot([cx, cx], [y_lo_line, y_hi_line],
                    color=color, linewidth=lw, linestyle=ls, zorder=5)
            # Подпись точки пересечения с осью X: (c, 0)
            if X_TAG and Y_LIM_B <= 0 <= Y_LIM_T:
                if SHOW_VALUES:
                    annotate_point(cx, 0.0, pt_label(coord(cx, cx_exact), '0'),
                                   above=True, color=color)
                mark_point(cx, 0, color, markersize=5, zorder=8)
        func_data.append((None, None, color))
        curve_meta.append({'kind': 'vline', 'color': color,
                           'x': cx, 'x_exact': cx_exact})

    def draw_implicit(func_idx, lhs_str, rhs_str, color, lw, ls):
        H_sym, h = build_implicit_func(lhs_str, rhs_str)
        # Сетка по видимому окну. Плотность подобрана как компромисс
        # гладкость/скорость; contour сам интерполирует линию уровня 0.
        N = 600
        xs = np.linspace(X_LIM_L, X_LIM_R, N)
        ys = np.linspace(Y_LIM_B, Y_LIM_T, N)
        X, Y = np.meshgrid(xs, ys)
        with np.errstate(all='ignore'):
            Z = h(X, Y)
            Z = np.asarray(Z, dtype=float)
            Z = np.broadcast_to(Z, X.shape)
        # Вырожденные уравнения: тождество (y = y, x + 1 = x + 1 → H ≡ 0 —
        # верно в КАЖДОЙ точке плоскости) или противоречие (H — ненулевая
        # константа, x = x + 1). Рисовать нечего; сообщаем об этом как об
        # ошибке строки, чтобы не вешать интерфейс тысячами «корней».
        try:
            H_const = (not H_sym.free_symbols)
        except Exception:
            H_const = False
        Zf = Z[np.isfinite(Z)]
        if H_const or Zf.size == 0 or np.all(Zf == 0.0):
            if H_const and H_sym != 0:
                raise ValueError("equation has no solutions (contradiction)")
            raise ValueError("equation is an identity — every point satisfies it")
        # Рисуем линию уровня 0 => кривую F−G=0
        # linestyles matplotlib ожидает 'solid'/'dashed'/'dotted'
        ls_map = {'-': 'solid', '--': 'dashed', ':': 'dotted', '-.': 'dashdot'}
        ax.contour(X, Y, Z, levels=[0], colors=[color],
                   linewidths=lw, linestyles=[ls_map.get(ls, 'solid')],
                   zorder=5)

        # ── Пересечения неявной кривой с осями ──────────
        # Ось X: корни H(x, 0)=0 -> точки (x, 0)
        # Ось Y: корни H(0, y)=0 -> точки (0, y)
        if X_TAG and (Y_LIM_B <= 0 <= Y_LIM_T):
            gx = lambda t: h(t, 0.0)
            Hx0 = H_sym.subs(_Y_SYM, 0)
            cx_list = cands(Hx0, _X_SYM, 'roots', X_LIM_L, X_LIM_R)
            for xr in _scan_roots_1d(gx, X_LIM_L, X_LIM_R):
                xe = exact_root(xr, Hx0, _X_SYM, cx_list, 'root')
                mark_point(xr, 0.0, color, markersize=5, zorder=8)
                if SHOW_VALUES:
                    annotate_point(xr, 0.0, pt_label(coord(xr, xe), '0'),
                                   above=True, color=color)
        if Y_TAG and (X_LIM_L <= 0 <= X_LIM_R):
            gy = lambda t: h(0.0, t)
            H0y = H_sym.subs(_X_SYM, 0)
            cy_list = cands(H0y, _Y_SYM, 'roots', Y_LIM_B, Y_LIM_T)
            for yr in _scan_roots_1d(gy, Y_LIM_B, Y_LIM_T):
                ye = exact_root(yr, H0y, _Y_SYM, cy_list, 'root')
                mark_point(0.0, yr, color, markersize=5, zorder=8)
                if SHOW_VALUES:
                    annotate_point(0.0, yr, pt_label('0', coord(yr, ye)),
                                   above=True, color=color, side='right')
        func_data.append((None, None, color))
        curve_meta.append({'kind': 'implicit', 'color': color, 'h': h, 'H': H_sym})

    def draw_func(func_idx, func_str, color, lw, ls):
        if not str(func_str).strip():
            raise SyntaxError("empty expression")      # напр. 'y=' без правой части
        expr, expr_raw, f = build_numpy_func(func_str)
        x_sym = _get_x(expr)

        disc_exact = {}
        all_disc = (find_discontinuities_numerical(f, expr, expr_raw, X_LIM_L, X_LIM_R,
                                                   exact_out=disc_exact)
                    if (DISC or ASIMP) else [])
        removable_exact = {}
        vasymps, removable = classify_discontinuities(
            f, expr, all_disc, Y_LIM_B, Y_LIM_T,
            exact_pts=disc_exact, exact_out=removable_exact)
        hasymps = (find_horizontal_asymptotes(f, expr, X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T)
                   if ASIMP else [])

        disc_pts_x = [d[0] if isinstance(d, tuple) else d for d in all_disc]

        # Разрыв в самой линии графика делаем только там, где он математически
        # обязателен — на вертикальных асимптотах (функция там не определена).
        # На устранимых разрывах (holes) разрыв линии рисуем ТОЛЬКО если включено
        # отображение Holes — иначе галочка Asymptotes не должна влиять на holes.
        removable_x = [r[0] for r in removable]
        mask_pts_x  = list(vasymps) + (removable_x if DISC else [])
        y_vals      = make_y_array(f, x_vals, mask_pts_x, Y_LIM_B, Y_LIM_T)

        # Ограничение области определения (domain) этой функции: вне
        # [d_from, d_to] делаем y = NaN, поэтому кривая там не рисуется,
        # а экстремумы/нули (они берутся из y_vals) автоматически тоже
        # не считаются за пределами домена. View window при этом не меняется.
        if func_idx < len(FUNC_DOMAINS):
            d_from, d_to = FUNC_DOMAINS[func_idx]
            if d_from != float("-inf") or d_to != float("inf"):
                outside = (x_vals < d_from) | (x_vals > d_to)
                y_vals = y_vals.copy()
                y_vals[outside] = np.nan

        extrema      = (find_extrema_numerical(f, x_vals, y_vals, Y_LIM_B, Y_LIM_T, disc_pts_x)
                        if EXTR else [])
        x_intercepts = (find_x_intercepts(f, x_vals, y_vals, disc_pts_x)
                        if X_TAG else [])
        y_intercept  = find_y_intercept(f, disc_pts_x, Y_LIM_B, Y_LIM_T) if Y_TAG else None
        # y-пересечение (x=0) скрываем, если 0 вне домена функции
        if y_intercept is not None and func_idx < len(FUNC_DOMAINS):
            d_from, d_to = FUNC_DOMAINS[func_idx]
            if not (d_from <= 0 <= d_to):
                y_intercept = None

        # Точные кандидаты (в режиме 'cached' могут быть отложены → [])
        root_cands = cands(expr, x_sym, 'roots', X_LIM_L, X_LIM_R) if (X_TAG and x_intercepts) else []
        crit_cands = cands(expr, x_sym, 'critical', X_LIM_L, X_LIM_R) if (EXTR and extrema) else []

        # Асимптоты — рисуем только новые (чтобы не дублировать)
        if ASIMP:
            for ya in hasymps:
                if ya not in all_hasymps:
                    ax.axhline(ya, color=ASYM_COLOR, linewidth=1.0, linestyle='--', zorder=3)
                    all_hasymps.add(ya)
            for xa in vasymps:
                if xa not in all_vasymps:
                    ax.axvline(xa, color=ASYM_COLOR, linewidth=1.0, linestyle='--', zorder=3)
                    all_vasymps.add(xa)

        # ── Кривая ──────────────────────────────
        ax.plot(x_vals, y_vals, color=color, linewidth=lw, linestyle=ls, zorder=5)

        # Сохраняем для заливки после цикла
        func_data.append((f, y_vals, color))
        _dom = FUNC_DOMAINS[func_idx] if func_idx < len(FUNC_DOMAINS) else (float('-inf'), float('inf'))
        curve_meta.append({'kind': 'func', 'color': color, 'f': f, 'domain': _dom,
                           'expr': expr, 'x_sym': x_sym, 'disc': list(disc_pts_x)})

        # ── Точки разрыва ───────────────────────
        if DISC:
            for (dp, ylim) in removable:
                dp_e, yl_e = removable_exact.get(dp, (None, None))
                ax.plot(dp, ylim, 'o', color=color, markersize=5,
                        markerfacecolor='white', markeredgewidth=1.5, zorder=8)
                if SHOW_VALUES:
                    txt = label_xy(dp, ylim, dp_e, yl_e)
                    if abs(dp - y_axis_x) < ON_Y_AXIS_TOL:
                        annotate_point(dp, ylim, txt,
                                       above=True, color=color, side=choose_side(dp, ylim, f))
                    else:
                        annotate_point(dp, ylim, txt,
                                       above=(ylim >= x_axis_y), color=color)

        # ── Экстремумы ───────────────────────────
        if EXTR:
            for ex, ey in extrema:
                mark_point(ex, ey, color)
                if SHOW_VALUES:
                    ex_e = exact_crit(ex, expr, x_sym, crit_cands)
                    ey_e = exact_y_of(expr, x_sym, ex_e) if ex_e is not None else None
                    txt = label_xy(ex, ey, ex_e, ey_e)
                    if abs(ex - y_axis_x) < ON_Y_AXIS_TOL:
                        annotate_point(ex, ey, txt,
                                       above=True, color=color, side=choose_side(ex, ey, f))
                    else:
                        annotate_point(ex, ey, txt,
                                       above=(ey >= x_axis_y), color=color)

        # ── Пересечения с осью X ─────────────────
        if X_TAG and Y_LIM_B <= 0 <= Y_LIM_T:      # ось X в окне
            for xi in x_intercepts:
                mark_point(xi, x_axis_y, color)
                if SHOW_VALUES:
                    xi_e = exact_root(xi, expr, x_sym, root_cands, 'root')
                    txt = pt_label(coord(xi, xi_e), '0')
                    if abs(xi - y_axis_x) < ON_Y_AXIS_TOL:
                        annotate_point(xi, x_axis_y, txt,
                                       above=True, color=color, side=choose_side(xi, x_axis_y, f))
                    else:
                        annotate_point(xi, x_axis_y, txt,
                                       above=True, color=color)

        # ── Пересечение с осью Y ─────────────────
        is_removable_at_0 = any(abs(d - y_axis_x) < 1e-9 for d in removable_x)
        # ось Y должна быть в окне: иначе точка (0, y) рисовалась бы на прижатой к краю оси
        if (Y_TAG and y_intercept is not None and not is_removable_at_0
                and X_LIM_L <= 0 <= X_LIM_R):
            mark_point(y_axis_x, y_intercept, color)
            if SHOW_VALUES:
                yi_e = exact_y_of(expr, x_sym, Integer(0))
                annotate_point(y_axis_x, y_intercept, pt_label('0', coord(y_intercept, yi_e)),
                               above=True, color=color,
                               side=choose_side(y_axis_x, y_intercept, f))

    for func_idx, func_str in enumerate(FUNCS):
        color = CURVE_COLORS[func_idx % len(CURVE_COLORS)]
        lw = CURVE_WIDTHS[func_idx] if func_idx < len(CURVE_WIDTHS) else 1.8
        ls = CURVE_STYLES[func_idx] if func_idx < len(CURVE_STYLES) else "-"
        n_before = len(func_data)
        try:
            if func_str is None or not str(func_str).strip():
                raise _EmptyFunction()
            kind, payload = _parse_equation_input(func_str)
            if kind == 'vline':
                draw_vline(func_idx, func_str, payload, color, lw, ls)
            elif kind == 'implicit':
                draw_implicit(func_idx, payload[0], payload[1], color, lw, ls)
            else:
                draw_func(func_idx, payload, color, lw, ls)
        except _EmptyFunction:
            pass
        except SymbolicPending:
            # Не должно происходить (все символьные вызовы обёрнуты), но на
            # всякий случай: кривая пропускается до следующей перерисовки.
            pass
        except Exception as exc:
            errors[func_idx] = _short_error(exc)
        # placeholder, чтобы индексы заливки (f1/f2) совпадали с FUNCS
        if len(func_data) == n_before:
            func_data.append((None, None, color))
            curve_meta.append({'kind': 'error', 'color': color})

    # ══════════════════════════════════════════════
    #  ПЕРЕСЕЧЕНИЯ МЕЖДУ ФУНКЦИЯМИ
    # ══════════════════════════════════════════════
    if INTER and len(curve_meta) >= 2:
        def blend(c1, c2):
            c1 = c1.lstrip('#'); c2 = c2.lstrip('#')
            r = (int(c1[0:2],16) + int(c2[0:2],16)) // 2
            g = (int(c1[2:4],16) + int(c2[2:4],16)) // 2
            b = (int(c1[4:6],16) + int(c2[4:6],16)) // 2
            return f'#{r:02x}{g:02x}{b:02x}'

        def emit(ix, iy, pt_color, probe_f=None, cx=None, cy=None):
            # Допуск на границе окна: точка (3, 5) при Y_LIM_T = 5 не должна
            # пропадать из-за 5.000000000000037 от brentq
            eps_x = 1e-9 * max(1.0, X_LIM_R - X_LIM_L)
            eps_y = 1e-9 * max(1.0, Y_LIM_T - Y_LIM_B)
            if not (Y_LIM_B - eps_y <= iy <= Y_LIM_T + eps_y
                    and X_LIM_L - eps_x <= ix <= X_LIM_R + eps_x):
                return
            ix = min(max(ix, X_LIM_L), X_LIM_R)
            iy = min(max(iy, Y_LIM_B), Y_LIM_T)
            mark_point(ix, iy, pt_color, markersize=5, zorder=11)
            if SHOW_VALUES:
                above = iy >= x_axis_y
                txt = label_xy(ix, iy, cx, cy)
                if probe_f is not None and abs(ix - y_axis_x) < ON_Y_AXIS_TOL:
                    annotate_point(ix, iy, txt,
                                   above=above, color=pt_color,
                                   side=choose_side(ix, iy, probe_f))
                else:
                    annotate_point(ix, iy, txt,
                                   above=above, color=pt_color)

        def pair_zero(H1, H2, cx, cy):
            """Точка (cx, cy) лежит на обеих неявных кривых."""
            try:
                sub = {_X_SYM: cx, _Y_SYM: cy}
                for H in (H1, H2):
                    v = H.subs(sub)
                    if v == 0:
                        continue
                    if v.free_symbols:
                        return False
                    if abs(complex(v.evalf(30))) > 1e-9:
                        return False
                return True
            except Exception:
                return False

        for i in range(len(curve_meta)):
            for j in range(i + 1, len(curve_meta)):
                mi, mj = curve_meta[i], curve_meta[j]
                ki, kj = mi['kind'], mj['kind']
                pt_color = blend(mi['color'], mj['color'])
                try:
                    # ── функция × функция (как раньше) ──
                    if ki == 'func' and kj == 'func':
                        yi = func_data[i][1]; yj = func_data[j][1]
                        disc_ij = list(mi.get('disc', [])) + list(mj.get('disc', []))
                        pts = find_intersections(mi['f'], mj['f'], x_vals, yi, yj, disc_ij)
                        d_expr = mi['expr'] - mj['expr']
                        xs_ = mi['x_sym']
                        c_list = cands(d_expr, xs_, 'roots', X_LIM_L, X_LIM_R) if pts else []
                        for ix, iy in pts:
                            cx = exact_root(ix, d_expr, xs_, c_list, 'inter')
                            cy = exact_y_of(mi['expr'], xs_, cx) if cx is not None else None
                            if cx is not None and cy is None:
                                cy = exact_y_of(mj['expr'], mj['x_sym'], cx)
                            emit(ix, iy, pt_color, probe_f=mi['f'], cx=cx, cy=cy)

                    # ── функция × неявная ──
                    elif {ki, kj} == {'func', 'implicit'}:
                        fm = mi if ki == 'func' else mj
                        im_ = mi if ki == 'implicit' else mj
                        if im_.get('h') is None:
                            continue
                        pts = intersect_func_implicit(fm['f'], im_['h'],
                                                      X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T)
                        g_expr = im_['H'].subs(_Y_SYM, fm['expr'])
                        c_list = cands(g_expr, fm['x_sym'], 'roots', X_LIM_L, X_LIM_R) if pts else []
                        for ix, iy in pts:
                            cx = exact_root(ix, g_expr, fm['x_sym'], c_list, 'inter')
                            cy = exact_y_of(fm['expr'], fm['x_sym'], cx) if cx is not None else None
                            emit(ix, iy, pt_color, probe_f=fm['f'], cx=cx, cy=cy)

                    # ── неявная × неявная (в т.ч. две окружности) ──
                    elif ki == 'implicit' and kj == 'implicit':
                        if mi.get('h') is None or mj.get('h') is None:
                            continue
                        for ix, iy in intersect_implicit_implicit(mi['h'], mj['h'],
                                                                  X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T):
                            tag = ('ii', str(mi['H']), str(mj['H']))
                            cx = identify_value(ix, cache_key=tag + ('x',))
                            cy = identify_value(iy, cache_key=tag + ('y',))
                            if cx is None or cy is None or not pair_zero(mi['H'], mj['H'], cx, cy):
                                cx = cy = None
                            emit(ix, iy, pt_color, cx=cx, cy=cy)

                    # ── вертикаль × (функция или неявная) ──
                    elif 'vline' in (ki, kj):
                        vm = mi if ki == 'vline' else mj
                        om = mi if ki != 'vline' else mj
                        cx = vm.get('x')
                        if cx is None or not (X_LIM_L <= cx <= X_LIM_R):
                            continue
                        cx_e = vm.get('x_exact')
                        if om['kind'] == 'func':
                            yv = _safe_scalar(om['f'], cx)
                            if np.isfinite(yv):
                                cy = exact_y_of(om['expr'], om['x_sym'], cx_e) if cx_e is not None else None
                                emit(cx, yv, pt_color, probe_f=om['f'], cx=cx_e, cy=cy)
                        elif om['kind'] == 'implicit' and om.get('h') is not None:
                            # на вертикали x=cx ищем корни H(cx, y)=0
                            Hc = om['H'].subs(_X_SYM, cx_e if cx_e is not None else cx)
                            roots_y = _scan_roots_1d(lambda t: om['h'](cx, t), Y_LIM_B, Y_LIM_T)
                            c_list = (cands(Hc, _Y_SYM, 'roots', Y_LIM_B, Y_LIM_T)
                                      if (roots_y and cx_e is not None) else [])
                            for yr in roots_y:
                                cy = (exact_root(yr, Hc, _Y_SYM, c_list, 'inter')
                                      if cx_e is not None else None)
                                emit(cx, yr, pt_color, cx=cx_e, cy=cy)
                except SymbolicPending:
                    pass
                except Exception:
                    # Ошибка расчёта пересечений одной пары не должна ронять график
                    pass

    def lighten_color(hex_color, factor=0.45):
        hex_color = hex_color.lstrip('#')
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f'#{r:02x}{g:02x}{b:02x}'

    def draw_fill_lines(ax, xf, y1_fill, y2_fill, valid, fill_style, color,
                        x_lim_l, x_lim_r, y_lim_b, y_lim_t, density=0.01,
                        ax_px=(600.0, 600.0)):
        x_span = x_lim_r - x_lim_l
        y_span = y_lim_t - y_lim_b
        # Масштабы по осям могут отличаться (aspect 'auto'), поэтому штриховку
        # строим в ЭКРАННЫХ пикселях: линии под 45° на экране — это y = sign·k·x + c
        # в данных, где k = (px/ед. по X)/(px/ед. по Y). Шаг — тоже в пикселях,
        # чтобы плотность не зависела от окна просмотра.
        sx = ax_px[0] / x_span          # пикселей на единицу X
        sy = ax_px[1] / y_span          # пикселей на единицу Y
        k = sx / sy
        step_px = density * min(ax_px)
        step_c = step_px * np.sqrt(2) / sy      # шаг c в единицах Y
        lw   = 0.6

        # Стиль штриховки:
        #   0 — 45°,  сплошная диагональная линия
        #   1 — 135°, сплошная диагональная линия
        #   2 — "точки" — та же диагональная решётка (семейство параллельных
        #       линий + та же логика клиппинга по границе), что и у 45°,
        #       но вместо непрерывной линии на каждой диагонали ставятся
        #       отдельные точки на ФИКСИРОВАННОЙ глобальной сетке с шагом,
        #       зависящим от density. Сетка не зависит от формы границы —
        #       поэтому точки не "облипают" контур функции — и расстояние
        #       между любыми соседними точками (что вдоль диагонали, что
        #       между соседними диагоналями) всегда одинаковое.
        sign = -1 if fill_style == 1 else 1   # стили 0 и 2 — одно семейство 45°

        y_lo = np.minimum(y1_fill, y2_fill)
        y_hi = np.maximum(y1_fill, y2_fill)

        # Диапазон c должен покрыть все углы видимой области (линия y = sign·k·x + c)
        corners_c = [
            y_lim_b - sign * k * x_lim_l,
            y_lim_b - sign * k * x_lim_r,
            y_lim_t - sign * k * x_lim_l,
            y_lim_t - sign * k * x_lim_r,
        ]
        c_min = min(corners_c) - step_c
        c_max = max(corners_c) + step_c
        c_vals = np.arange(c_min, c_max, step_c)

        if fill_style == 2:
            # Шаг по x, который при движении вдоль линии под 45° на экране
            # даёт ровно евклидово (экранное) расстояние step_px между точками.
            dot_dx = max(step_px / np.sqrt(2) / sx, 1e-9)
            # Привязываем фазу решётки к глобальной системе координат
            # (x_lim_l), а не к границам конкретной заливки — так точки
            # не "прыгают" при изменении from/to.
            n0 = int(np.ceil((xf[0] - x_lim_l) / dot_dx))
            x_start = x_lim_l + n0 * dot_dx
            xs_lat = np.arange(x_start, xf[-1] + dot_dx * 0.5, dot_dx)
            # Оставляем только узлы решётки, строго попадающие в диапазон x
            # самой заливки. Без этого крайний узел мог оказаться чуть за
            # пределами [xf[0], xf[-1]] (особенно при малой плотности —
            # большом dot_dx), а np.clip ниже "притягивал" его индекс к
            # крайнему столбцу xf, и точка проходила проверку границ заливки,
            # хотя по x была уже вне окраса — отсюда точки за рамкой.
            xs_lat = xs_lat[(xs_lat >= xf[0]) & (xs_lat <= xf[-1])]
            if len(xs_lat) == 0:
                return
            idx = np.clip(np.searchsorted(xf, xs_lat), 0, len(xf) - 1)

            for c in c_vals:
                y_lat = sign * k * xs_lat + c
                ok = (valid[idx] &
                      (y_lat >= y_lo[idx]) & (y_lat <= y_hi[idx]))
                if np.any(ok):
                    ax.plot(xs_lat[ok], y_lat[ok], '.', color=color,
                            markersize=2.2, linewidth=0, zorder=4)
            return

        # 45° (style 0) и 135° (style 1) — сплошные параллельные диагональные линии
        for c in c_vals:
            # Линия: y = sign*k*x + c  (45° на экране)
            y_line = sign * k * xf + c

            # Клипируем линию по области заливки [y_lo, y_hi]
            y_clipped = np.where(
                valid & (y_line >= y_lo) & (y_line <= y_hi),
                y_line, np.nan
            )

            # Разбиваем на непрерывные куски и рисуем каждый
            seg_x, seg_y = [], []
            for xi, yi in zip(xf, y_clipped):
                if np.isfinite(yi):
                    seg_x.append(xi)
                    seg_y.append(yi)
                else:
                    if len(seg_x) > 1:
                        ax.plot(seg_x, seg_y, color=color,
                                linewidth=lw, zorder=4)
                    seg_x, seg_y = [], []
            if len(seg_x) > 1:
                ax.plot(seg_x, seg_y, color=color,
                        linewidth=lw, zorder=4)

    for fill_entry in FILL:
        try:
            # Формат: (f1, f2, x_от, x_до, стиль, границы, плотность)
            f1_idx, f2, fx_from, fx_to, fill_style = fill_entry[:5]
            show_borders = fill_entry[5] if len(fill_entry) > 5 else True
            density      = fill_entry[6] if len(fill_entry) > 6 else 0.01

            if f1_idx >= len(func_data):
                continue

            f1_func, y1_all, color1 = func_data[f1_idx]
            # Заливка относительно вертикальной линии не имеет смысла — пропускаем.
            if f1_func is None:
                continue

            fx_from = max(fx_from, X_LIM_L)
            fx_to   = min(fx_to,   X_LIM_R)
            if fx_from >= fx_to:
                continue

            mask = (x_vals >= fx_from) & (x_vals <= fx_to)
            xf   = x_vals[mask]
            if len(xf) < 2:
                continue

            y1_fill = _eval_array(f1_func, xf)
            y1_fill = np.clip(y1_fill, Y_LIM_B, Y_LIM_T)

            if isinstance(f2, str) or f2 is None:
                y2_fill = np.zeros_like(y1_fill)
                fill_color = lighten_color(color1)
            else:
                if f2 >= len(func_data):
                    continue
                f2_func = func_data[f2][0]
                if f2_func is None:
                    continue
                y2_fill = _eval_array(f2_func, xf)
                y2_fill = np.clip(y2_fill, Y_LIM_B, Y_LIM_T)
                fill_color = lighten_color(color1)

            valid = np.isfinite(y1_fill) & np.isfinite(y2_fill)

            # Рисуем штриховку реальными линиями
            draw_fill_lines(ax, xf, y1_fill, y2_fill, valid, fill_style,
                            fill_color, X_LIM_L, X_LIM_R, Y_LIM_B, Y_LIM_T,
                            density=density, ax_px=(_ax_w_px, _ax_h_px))

            # Вертикальные линии-границы по краям
            if show_borders:
                for xb in [fx_from, fx_to]:
                    y1b = _safe_val(f1_func, xb)
                    y1b = float(np.clip(y1b, Y_LIM_B, Y_LIM_T)) if np.isfinite(y1b) else None
                    if isinstance(f2, str) or f2 is None:
                        y2b = 0.0
                    else:
                        y2b = _safe_val(func_data[f2][0], xb)
                        y2b = float(np.clip(y2b, Y_LIM_B, Y_LIM_T)) if np.isfinite(y2b) else None
                    if y1b is not None and y2b is not None:
                        ax.plot([xb, xb], [y1b, y2b], color=fill_color,
                                linewidth=1.2, zorder=5)
        except Exception:
            # Некорректная запись заливки не должна ронять график
            continue

    _active_free_text_manager = FreeTextManager(fig, ax, font_size=_FS)

    _active_axis_label_manager = AxisLabelManager(
        fig, ax,
        artists={'x': _axis_label_x, 'y': _axis_label_y},
        home_offsets=_axis_home_offsets,
        default_fontsize=AXIS_FS)

    if standalone:
        plt.show()

    return {'ax': ax, 'errors': errors, 'pending': bool(_DRAW_STATE['pending'])}


class _EmptyFunction(Exception):
    """Пустая строка функции — тихо пропускается."""


if __name__ == "__main__":
    plot_function()
