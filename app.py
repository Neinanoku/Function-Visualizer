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
import queue
import threading
import traceback
import tkinter as tk
from tkinter import messagebox, filedialog, colorchooser

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Импортируем движок и редактор ────────────────────────────
# Добавляем путь к _MEIPASS чтобы PyInstaller находил модули-данные
if getattr(sys, 'frozen', False):
    _base = sys._MEIPASS
    if _base not in sys.path:
        sys.path.insert(0, _base)

import function_visualizer as fv
from math_editor import MathEditor, MathModel, IncompleteExpression


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

LEFT_PANEL_WIDTH = 500
REDRAW_DELAY_MS  = 250

UI_FONT = "Segoe UI"


def _resource_path(name):
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


# ═════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ВИДЖЕТЫ
# ═════════════════════════════════════════════════════════════

def styled_entry(parent, width=14, **kw):
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
        tk.Label(inner, text=title, bg=CARD_BG, fg=ACCENT,
                 font=(UI_FONT, 10, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(0, 6))
    inner._outer = outer   # храним ссылку чтобы pack вызывать снаружи
    return inner


def card_pack(widget, **kw):
    """pack карточки через её outer frame."""
    widget._outer.pack(**kw)


def small_button(parent, text, command, bg=BTN_DEL, fg="white", **kw):
    opts = dict(bg=bg, fg=fg, relief="flat", font=(UI_FONT, 9, "bold"),
                cursor="hand2", bd=0, padx=10, pady=4, activebackground=bg,
                activeforeground=fg, command=command)
    opts.update(kw)
    return tk.Button(parent, text=text, **opts)


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
        tk.Label(self.tip, text=self.text, bg="#fffbe6", fg=TEXT,
                 font=(UI_FONT, 9), relief="solid", bd=1, padx=6, pady=3).pack()

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
        [("x", "x", "переменная x"), ("y", "y", "переменная y (для уравнений, напр. x²+y²=9)"),
         ("a²", "sq", "квадрат"), ("aᵇ", "^", "степень")],
        [("(", "(", "открыть скобку"), (")", ")", "выйти из скобки"),
         ("√", "sqrt", "квадратный корень"), ("ⁿ√", "root", "корень n-й степени")],
        [("|a|", "abs", "модуль"), ("π", "pi", "число π"),
         ("e", "e", "число e"), ("a/b", "/", "дробь")],
        [("sin", "sin", None), ("cos", "cos", None), ("tan", "tan", None), ("cot", "cot", None)],
        [("ln", "ln", "натуральный логарифм"), ("log", "log", "десятичный логарифм"),
         ("logₐ", "logb", "логарифм по основанию a"), ("eˣ", "exp", "экспонента")],
        [("fn ▸", "__page__", "arcsin, arccos, sinh…"), ("↑", "up", "курсор вверх (числитель)"),
         ("↓", "down", "курсор вниз (знаменатель)"), ("", None, None)],
    ]
    LEFT_EXTRA = [
        [("arcsin", "asin", None), ("arccos", "acos", None), ("arctan", "atan", None), ("sec", "sec", None)],
        [("csc", "csc", None), ("sinh", "sinh", None), ("cosh", "cosh", None), ("tanh", "tanh", None)],
    ]
    RIGHT = [
        [("7", "7", None), ("8", "8", None), ("9", "9", None), ("÷", "/", "дробь")],
        [("4", "4", None), ("5", "5", None), ("6", "6", None), ("×", "*", "умножить")],
        [("1", "1", None), ("2", "2", None), ("3", "3", None), ("−", "-", "минус")],
        [("0", "0", None), (".", ".", "десятичная точка"), ("=", "=", "равно (уравнение / x = c)"),
         ("+", "+", "плюс")],
        [("←", "left", "курсор влево"), ("→", "right", "курсор вправо"),
         ("⌫", "backspace", "удалить"), ("C", "clear", "очистить поле")],
    ]

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
                    tk.Frame(parent, bg=CARD_BG, width=46, height=30).grid(row=r, column=c)
                    continue
                is_digit = numeric and (label.isdigit() or label == ".")
                bg = KEY_BG2 if is_digit else KEY_BG
                fnt = (UI_FONT, 10, "bold") if is_digit else (UI_FONT, 10)
                if token == "__page__":
                    label = "fn ▸" if self._page == 0 else "◂ back"
                    fnt = (UI_FONT, 9)
                b = tk.Button(parent, text=label, bg=bg, fg=KEY_FG, font=fnt,
                              relief="flat", bd=0, cursor="hand2", width=5, pady=3,
                              activebackground="#dde3ea", activeforeground=KEY_FG,
                              highlightthickness=1, highlightbackground=BORDER,
                              takefocus=0,
                              command=lambda t=token: self._press(t))
                b.grid(row=r, column=c, padx=1, pady=1, sticky="nsew")
                if tip:
                    Tooltip(b, tip)

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

        # Индекс функции
        self.idx_label = tk.Label(self.frame, text=f"f{idx}", width=2,
                                  bg=PANEL_BG, fg=SUBTEXT, font=(UI_FONT, 9, "bold"))
        self.idx_label.pack(side="left", padx=(6, 2))

        # Кликабельный цветной квадрат — выбор цвета
        self.dot = tk.Button(self.frame, bg=self.color, activebackground=self.color,
                             width=2, height=1, relief="flat", bd=0,
                             cursor="hand2", command=self._pick_color)
        self.dot.pack(side="left", padx=(2, 6))

        # Поле формулы — 2-D редактор без клавиатуры (ввод с экранной клавиатуры)
        self.editor = MathEditor(self.frame, font_size=15, on_change=lambda _e: on_change(),
                                 on_focus=lambda e: on_focus(self), width=260, height=40)
        self.editor.pack(side="left", padx=2, fill="x", expand=True)

        # Кнопка удалить
        tk.Button(self.frame, text="✕", bg=BTN_DEL, fg="white", activebackground=BTN_DEL,
                  relief="flat", font=(UI_FONT, 9, "bold"), cursor="hand2", bd=0, padx=6,
                  command=lambda: on_delete(self)).pack(side="right", padx=4)

        # ── Вторая строка: стиль линии + область определения ──
        self.frame2 = tk.Frame(parent, bg=PANEL_BG)
        self.frame2.pack(fill="x", padx=4, pady=(0, 3))

        tk.Label(self.frame2, text="", width=2, bg=PANEL_BG).pack(side="left", padx=(6, 2))

        # Ширина линии — кнопки ▼/▲
        tk.Label(self.frame2, text="w:", bg=PANEL_BG, fg=SUBTEXT,
                 font=(UI_FONT, 8)).pack(side="left", padx=(2, 1))
        self._lw_idx = 3   # default = 1.8
        self._lw_label = tk.Label(self.frame2, text=f"{self.get_linewidth()}", width=3,
                                  bg=ENTRY_BG, fg=TEXT, font=(UI_FONT, 9), relief="flat",
                                  highlightthickness=1, highlightbackground=BORDER)
        self._lw_label.pack(side="left")
        btn_f = dict(bg=PANEL_BG, fg=TEXT, relief="flat", bd=0, font=(UI_FONT, 8),
                     cursor="hand2", padx=2, activebackground=PANEL_BG)
        tk.Button(self.frame2, text="▼", command=lambda: self._lw_step(-1), **btn_f).pack(side="left")
        tk.Button(self.frame2, text="▲", command=lambda: self._lw_step(+1), **btn_f).pack(side="left", padx=(0, 4))

        # Тип линии
        self.linestyle_var = tk.StringVar(value=list(self.LINESTYLES.keys())[0])
        self.linestyle_var.trace_add("write", lambda *_: on_change())
        om = tk.OptionMenu(self.frame2, self.linestyle_var, *self.LINESTYLES.keys())
        om.config(bg=ENTRY_BG, fg=TEXT, activebackground=CARD_BG, activeforeground=TEXT,
                  relief="flat", font=(UI_FONT, 8), bd=0, highlightthickness=1,
                  highlightbackground=BORDER, width=7)
        om["menu"].config(bg=ENTRY_BG, fg=TEXT, activebackground=ACCENT, font=(UI_FONT, 9))
        om.pack(side="left", padx=(2, 6))

        # Область определения: функция строится только на [from, to]
        tk.Label(self.frame2, text="domain:", bg=PANEL_BG, fg=SUBTEXT,
                 font=(UI_FONT, 8)).pack(side="left", padx=(2, 4))
        self.dom_from = live_entry(self.frame2, 6, "-inf", on_change)
        self.dom_from.pack(side="left", padx=2)
        tk.Label(self.frame2, text="…", bg=PANEL_BG, fg=SUBTEXT,
                 font=(UI_FONT, 8)).pack(side="left")
        self.dom_to = live_entry(self.frame2, 6, "inf", on_change)
        self.dom_to.pack(side="left", padx=2)

    # ── ширина ───────────────────────────────────────────────
    def _lw_step(self, d):
        self._lw_idx = max(0, min(len(self.LW_VALUES) - 1, self._lw_idx + d))
        self._lw_label.config(text=f"{self.get_linewidth()}")
        self.on_change()

    def _pick_color(self):
        result = colorchooser.askcolor(color=self.color, title=f"Pick color for f{self.idx}")
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
        return self.LINESTYLES.get(self.linestyle_var.get(), "-")

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
            "style": self.linestyle_var.get(),
            "domain": [self.dom_from.get(), self.dom_to.get()],
        }

    def from_dict(self, d):
        try:
            self.editor.set_model(MathModel.from_json(d.get("model", {})))
        except Exception:
            try:
                self.editor.set_model(MathModel.from_text(d.get("text", "")))
            except Exception:
                self.editor.set_model(MathModel())
        self.set_color(d.get("color", self.color))
        self.set_linewidth(d.get("width", 1.8))
        if d.get("style") in self.LINESTYLES:
            self.linestyle_var.set(d["style"])
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

        def lbl(t, parent=None):
            tk.Label(parent or self.frame, text=t, bg=PANEL_BG, fg=SUBTEXT,
                     font=(UI_FONT, 9)).pack(side="left", padx=(4, 1))

        lbl("f1:")
        self.f1 = live_entry(self.frame, 3, "0", on_change); self.f1.pack(side="left")
        lbl("f2:")
        self.f2 = live_entry(self.frame, 4, "x", on_change); self.f2.pack(side="left")
        lbl("from:")
        self.x_from = live_entry(self.frame, 6, "-3", on_change); self.x_from.pack(side="left")
        lbl("to:")
        self.x_to = live_entry(self.frame, 6, "3", on_change); self.x_to.pack(side="left")

        lbl("style:")
        self.style_var = tk.StringVar(value=self.STYLES[0])
        self.style_var.trace_add("write", lambda *_: on_change())
        om = tk.OptionMenu(self.frame, self.style_var, *self.STYLES)
        om.config(bg=ENTRY_BG, fg=TEXT, activebackground=CARD_BG, activeforeground=TEXT,
                  relief="flat", font=(UI_FONT, 9), bd=0, highlightthickness=0, width=9)
        om["menu"].config(bg=ENTRY_BG, fg=TEXT, activebackground=ACCENT)
        om.pack(side="left", padx=4)

        tk.Button(self.frame, text="✕", bg=BTN_DEL, fg="white", activebackground=BTN_DEL,
                  relief="flat", font=(UI_FONT, 9, "bold"), cursor="hand2", bd=0, padx=6,
                  command=lambda: on_delete(self)).pack(side="right", padx=(10, 4))

        # вторая строка: границы + плотность
        self.frame2 = tk.Frame(parent, bg=PANEL_BG)
        self.frame2.pack(fill="x", padx=4, pady=(0, 3))

        self.borders_var = tk.IntVar(value=1)
        self.borders_var.trace_add("write", lambda *_: on_change())
        tk.Checkbutton(self.frame2, text="borders", variable=self.borders_var,
                       bg=PANEL_BG, fg=TEXT, selectcolor=ENTRY_BG, activebackground=PANEL_BG,
                       font=(UI_FONT, 8), bd=0, highlightthickness=0).pack(side="left", padx=4)

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
                 variable=self.density_var, length=110, resolution=5,
                 bg=CARD_BG, fg=TEXT, troughcolor=ACCENT, activebackground=BTN_DEL,
                 highlightthickness=0, bd=0, sliderrelief="flat", showvalue=False,
                 command=_on_density, font=(UI_FONT, 7)).pack(side="left")
        self._density_ind.pack(side="left", padx=(4, 0))

    def get(self):
        """Returns tuple (f1, f2, x_from, x_to, style, borders, density) or None on error."""
        try:
            f1 = int(self.f1.get().strip())
            f2_raw = self.f2.get().strip()
            f2 = f2_raw if f2_raw.lower() in ("x", "") else int(f2_raw)
            x_from  = fv.parse_number(self.x_from.get())
            x_to    = fv.parse_number(self.x_to.get())
            style   = self.STYLES.index(self.style_var.get())
            borders = bool(self.borders_var.get())
            # density: 0%→step=0.05 (редко),  100%→step=0.01 (густо)
            pct     = self.density_var.get() / 100.0
            density = 0.05 - pct * 0.04
            return (f1, f2, x_from, x_to, style, borders, density)
        except Exception:
            return None

    def to_dict(self):
        return {"f1": self.f1.get(), "f2": self.f2.get(), "from": self.x_from.get(),
                "to": self.x_to.get(), "style": self.style_var.get(),
                "borders": self.borders_var.get(), "density": self.density_var.get()}

    def from_dict(self, d):
        self.f1.var.set(str(d.get("f1", "0")))
        self.f2.var.set(str(d.get("f2", "x")))
        self.x_from.var.set(str(d.get("from", "-3")))
        self.x_to.var.set(str(d.get("to", "3")))
        if d.get("style") in self.STYLES:
            self.style_var.set(d["style"])
        self.borders_var.set(int(d.get("borders", 1)))
        self.density_var.set(int(d.get("density", 100)))
        self._density_ind.config(text=str(self.density_var.get()))

    def destroy(self):
        self.frame.destroy()
        self.frame2.destroy()


# ═════════════════════════════════════════════════════════════
#  ГЛАВНОЕ ОКНО
# ═════════════════════════════════════════════════════════════

class App(tk.Tk):
    PROJECT_EXT = ".fvproj"

    def __init__(self):
        super().__init__()
        self.title("Function Visualizer — Ariadna")
        self.configure(bg=APP_BG)
        self.resizable(True, True)
        self.minsize(980, 640)
        self.geometry("1320x820")

        self.func_rows = []
        self.fill_rows = []
        self.active_row = None
        self._logo_img = None
        self._redraw_job = None
        self._drawing = False
        self._redraw_wanted = False
        self._worker = None
        self._worker_queue = queue.Queue()
        self._loading = False
        self._pan = None
        self._tl_job = None

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

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(200, self._poll_worker)
        self.schedule_redraw()

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
        # ── Левая колонка (настройки) ───────────────────────
        left = tk.Frame(self, bg=APP_BG, width=LEFT_PANEL_WIDTH)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # ── Правая часть (график) ───────────────────────────
        right = tk.Frame(self, bg=APP_BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_header(left)
        self._build_settings(left)
        self._build_graph_pane(right)

    def _build_header(self, parent):
        hdr = tk.Frame(parent, bg=APP_BG)
        hdr.pack(fill="x")
        try:
            from PIL import Image, ImageTk
            img = Image.open(_resource_path("ariadna-logo1-trnsp.png")).convert("RGBA")
            h = 64
            w = int(img.width * h / img.height)
            img = img.resize((w, h), Image.LANCZOS)
            self._logo_img = ImageTk.PhotoImage(img)
            tk.Label(hdr, image=self._logo_img, bg=APP_BG).pack(side="left", padx=(16, 10), pady=6)
        except Exception:
            tk.Label(hdr, text="ariadna", bg=APP_BG, fg=ACCENT,
                     font=(UI_FONT, 18, "bold")).pack(side="left", padx=16, pady=6)

        title_frame = tk.Frame(hdr, bg=APP_BG)
        title_frame.pack(side="left", padx=(0, 10))
        tk.Label(title_frame, text="Function Visualizer", bg=APP_BG, fg=TEXT,
                 font=(UI_FONT, 16, "bold")).pack(anchor="w")
        tk.Label(title_frame, text="by Daniel", bg=APP_BG, fg=SUBTEXT,
                 font=(UI_FONT, 9)).pack(anchor="w")
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
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

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
                     bg=BTN_ADD).pack(anchor="w", padx=10, pady=(0, 6))

        # Экранная клавиатура — внутри карточки функций
        tk.Frame(self.func_card, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(0, 6))
        self.keypad = Keypad(self.func_card, self._on_key)
        self.keypad.pack(anchor="w", padx=10, pady=(0, 8))

        # ── View window ──────────────────────────────────────
        lim = card(p, "View Window")
        card_pack(lim, fill="x", padx=12, pady=4)
        g = tk.Frame(lim, bg=CARD_BG)
        g.pack(fill="x", padx=10, pady=(0, 10))

        def lim_row(label, defaults):
            r = tk.Frame(g, bg=CARD_BG)
            r.pack(fill="x", pady=2)
            tk.Label(r, text=label, bg=CARD_BG, fg=SUBTEXT,
                     font=(UI_FONT, 9), width=12, anchor="e").pack(side="left")
            entries = []
            for d in defaults:
                e = live_entry(r, 8, d, self.schedule_redraw)
                e.pack(side="left", padx=4)
                entries.append(e)
            return entries

        self.xlim_l, self.xlim_r = lim_row("X:  from / to", ["-5", "5"])
        self.ylim_b, self.ylim_t = lim_row("Y:  from / to", ["-5", "5"])
        tk.Label(g, text="Tip: mouse wheel over the graph zooms, drag pans.",
                 bg=CARD_BG, fg=SUBTEXT, font=(UI_FONT, 8)).pack(anchor="w", padx=4, pady=(4, 0))

        # ── Grid ─────────────────────────────────────────────
        grid_c = card(p, "Grid")
        card_pack(grid_c, fill="x", padx=12, pady=4)
        gr = tk.Frame(grid_c, bg=CARD_BG)
        gr.pack(fill="x", padx=10, pady=(0, 10))

        self.grid_var = self._live_int(1)
        styled_check(gr, "Show grid", self.grid_var).pack(side="left", padx=4)
        tk.Label(gr, text="  Step X:", bg=CARD_BG, fg=SUBTEXT, font=(UI_FONT, 9)).pack(side="left")
        self.xgrid_e = live_entry(gr, 5, "1", self.schedule_redraw)
        self.xgrid_e.pack(side="left", padx=4)
        tk.Label(gr, text="Step Y:", bg=CARD_BG, fg=SUBTEXT, font=(UI_FONT, 9)).pack(side="left")
        self.ygrid_e = live_entry(gr, 5, "1", self.schedule_redraw)
        self.ygrid_e.pack(side="left", padx=4)

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
        for i, (txt, var) in enumerate(checks):
            styled_check(grid_f, txt, var).grid(row=i // 3, column=i % 3, sticky="w", padx=6, pady=1)

        # Размер текста
        row3 = tk.Frame(dg, bg=CARD_BG); row3.pack(fill="x", pady=(6, 2))
        tk.Label(row3, text="Label size:", bg=CARD_BG, fg=SUBTEXT,
                 font=(UI_FONT, 9)).pack(side="left", padx=(6, 4))
        self.font_size_var = tk.IntVar(value=10)
        tk.Scale(row3, from_=6, to=20, orient="horizontal", variable=self.font_size_var,
                 length=160, bg=CARD_BG, fg=TEXT, troughcolor=ACCENT, activebackground=BTN_DEL,
                 highlightthickness=0, bd=0, sliderrelief="flat", font=(UI_FONT, 8),
                 command=lambda _v: self.schedule_redraw()).pack(side="left")
        tk.Label(row3, textvariable=self.font_size_var, bg=CARD_BG, fg=ACCENT,
                 font=(UI_FONT, 9, "bold"), width=3).pack(side="left", padx=2)

        # ── Fill ─────────────────────────────────────────────
        self.fill_card = card(p, "Area Fill")
        card_pack(self.fill_card, fill="x", padx=12, pady=4)
        tk.Label(self.fill_card,
                 text="f1 / f2 — function indices (0, 1, …) or 'x' for the X axis",
                 bg=CARD_BG, fg=SUBTEXT, font=(UI_FONT, 8)).pack(anchor="w", padx=10)
        self.fill_list = tk.Frame(self.fill_card, bg=PANEL_BG)
        self.fill_list.pack(fill="x", padx=8, pady=4)
        small_button(self.fill_card, "+ Add fill", self._add_fill,
                     bg=BTN_BLUE).pack(anchor="w", padx=10, pady=(0, 8))

        # ── Graph Labels ─────────────────────────────────────
        labels_card = card(p, "Graph Labels")
        card_pack(labels_card, fill="x", padx=12, pady=4)
        tk.Label(labels_card,
                 text="Double-click empty space on the graph to add a label. "
                      "Drag to move, scroll to rotate, right-click for options. "
                      "Point labels can be dragged too.",
                 bg=CARD_BG, fg=SUBTEXT, font=(UI_FONT, 8),
                 wraplength=LEFT_PANEL_WIDTH - 60, justify="left").pack(anchor="w", padx=10, pady=(0, 6))
        bl = tk.Frame(labels_card, bg=CARD_BG)
        bl.pack(anchor="w", padx=10, pady=(0, 8))
        small_button(bl, "Clear all labels", self._clear_labels).pack(side="left")
        small_button(bl, "Reset point labels", self._reset_point_labels).pack(side="left", padx=6)

        tk.Frame(p, bg=APP_BG, height=12).pack()

    def _live_int(self, value):
        var = tk.IntVar(value=value)
        var.trace_add("write", lambda *_: self.schedule_redraw())
        return var

    def _build_graph_pane(self, parent):
        bar = tk.Frame(parent, bg=APP_BG)
        bar.pack(fill="x", padx=10, pady=(10, 4))

        small_button(bar, "  Save image…", self._save_image, bg=BTN_SAVE,
                     font=(UI_FONT, 11, "bold"), padx=16, pady=7).pack(side="left")
        small_button(bar, "Reset view", self._reset_view, bg=BTN_DEL,
                     pady=7).pack(side="left", padx=(8, 0))
        small_button(bar, "Open project", self._open_project, bg=BTN_BLUE,
                     pady=7).pack(side="right")
        small_button(bar, "Save project", self._save_project, bg=BTN_BLUE,
                     pady=7).pack(side="right", padx=(0, 6))

        # Холст matplotlib
        frame = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        w = self.canvas.get_tk_widget()
        w.configure(bg="white", highlightthickness=0)
        w.pack(fill="both", expand=True)
        w.bind("<Configure>", self._on_canvas_resize)

        self.status = tk.Label(parent, text="", bg=APP_BG, fg=SUBTEXT,
                               font=(UI_FONT, 9), anchor="w")
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
        if not messagebox.askyesno("Clear all labels",
                                   f"Remove all {len(fv.FREE_TEXTS)} label(s) added on the graph?"):
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
            messagebox.showwarning("Cannot delete", "At least one function is required.")
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
    def schedule_redraw(self, *_):
        """Отложенная перерисовка: множество изменений подряд → одна отрисовка."""
        if self._loading:
            return
        if self._redraw_job is not None:
            try:
                self.after_cancel(self._redraw_job)
            except Exception:
                pass
        self._redraw_job = self.after(REDRAW_DELAY_MS, self._redraw)

    def _collect(self):
        """
        Собирает настройки из панели. Возвращает dict для движка или
        бросает ValueError с сообщением для строки состояния.
        """
        funcs, colors, widths, styles, domains = [], [], [], [], []
        for r in self.func_rows:
            if r.is_empty():
                r.editor.set_error(None)
                funcs.append("")       # пустая строка = нет кривой (индексы стабильны)
            else:
                try:
                    funcs.append(r.get())
                    r.editor.set_error(None)
                except IncompleteExpression as ex:
                    r.editor.set_error(str(ex))
                    funcs.append("")
            colors.append(r.get_color())
            widths.append(r.get_linewidth())
            styles.append(r.get_linestyle())
            domains.append(r.get_domain())

        def num(entry, what):
            try:
                return fv.parse_number(entry.get())
            except Exception:
                raise ValueError(f"{what}: cannot read '{entry.get()}'")

        xl = num(self.xlim_l, "X from"); xr = num(self.xlim_r, "X to")
        yb = num(self.ylim_b, "Y from"); yt = num(self.ylim_t, "Y to")
        if not all(map(lambda v: v == v and abs(v) != float("inf"), (xl, xr, yb, yt))):
            raise ValueError("View window must be finite")
        if xl >= xr or yb >= yt:
            raise ValueError("View window: left < right and bottom < top required")
        xg = num(self.xgrid_e, "Step X"); yg = num(self.ygrid_e, "Step Y")
        if xg <= 0 or yg <= 0:
            raise ValueError("Grid step must be positive")
        if (xr - xl) / xg > 2000 or (yt - yb) / yg > 2000:
            raise ValueError("Grid step is too small for this view window")

        fills = []
        for i, fr in enumerate(self.fill_rows):
            entry = fr.get()
            if entry is None:
                raise ValueError(f"Fill #{i + 1}: check parameters")
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
        if self._drawing:
            # уже рисуем (например, из вложенного события) — повторим позже
            self._redraw_wanted = True
            return
        self._drawing = True
        try:
            try:
                settings = self._collect()
            except ValueError as ex:
                self._set_status(str(ex), ERR_COLOR)
                return
            self._apply_to_engine(settings)

            fv.SYMBOLIC_MODE = 'cached'
            try:
                result = fv.plot_function(self.fig)
            except Exception:
                self._set_status("Plot error — see console", ERR_COLOR)
                traceback.print_exc()
                return

            errors = result.get('errors', {}) or {}
            for idx, r in enumerate(self.func_rows):
                if idx in errors and not r.is_empty():
                    r.editor.set_error(errors[idx])
            self.canvas.draw_idle()

            if result.get('pending'):
                self._start_worker(fv.take_pending_jobs())
                self._set_status("Refining labels (symbolic analysis)…")
            elif errors:
                self._set_status("Some functions have errors — hover the red field", ERR_COLOR)
            else:
                self._set_status("Ready")
        finally:
            self._drawing = False
            if self._redraw_wanted:
                self._redraw_wanted = False
                self.schedule_redraw()

    # ── фоновый поток для sympy ──────────────────────────────
    def _start_worker(self, jobs):
        if not jobs:
            return
        # Каждая пачка — свой daemon-поток: если sympy «завис» на одной
        # функции, остальные пачки всё равно посчитаются.
        def run():
            try:
                fv.run_jobs(jobs)
            except Exception:
                traceback.print_exc()
            finally:
                self._worker_queue.put("done")
        t = threading.Thread(target=run, daemon=True)
        t.start()
        self._worker = t

    def _poll_worker(self):
        got = False
        try:
            while True:
                self._worker_queue.get_nowait()
                got = True
        except queue.Empty:
            pass
        if got:
            self.schedule_redraw()
        self.after(150, self._poll_worker)

    def _on_canvas_resize(self, _event):
        # Пересчитать поля фигуры под новый размер (с задержкой)
        if self._tl_job is not None:
            try:
                self.after_cancel(self._tl_job)
            except Exception:
                pass
        self._tl_job = self.after(200, self._relayout)

    def _relayout(self):
        self._tl_job = None
        try:
            self.fig.tight_layout()
            self.canvas.draw_idle()
        except Exception:
            pass

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

    def _set_limits(self, xl, xr, yb, yt):
        def fmt(v):
            s = f"{v:.4g}"
            return s
        self._loading = True      # одна перерисовка вместо четырёх
        try:
            self.xlim_l.var.set(fmt(xl)); self.xlim_r.var.set(fmt(xr))
            self.ylim_b.var.set(fmt(yb)); self.ylim_t.var.set(fmt(yt))
        finally:
            self._loading = False
        self.schedule_redraw()

    def _on_graph_scroll(self, event):
        if event.inaxes is None or event.xdata is None:
            return
        if self._graph_hit_any(event):
            return            # колесо над подписью — поворот (движок)
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
        self._set_limits(nxl, nxr, nyb, nyt)

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
                     "lims": lims, "moved": False, "ax": event.inaxes}

    def _on_graph_motion(self, event):
        p = self._pan
        if p is None or event.x is None:
            return
        if not p["moved"] and abs(event.x - p["px"]) < 4 and abs(event.y - p["py"]) < 4:
            return
        p["moved"] = True
        ax = p["ax"]
        # переводим сдвиг в пикселях в единицы данных (по текущему масштабу осей)
        inv = ax.transData.inverted()
        x1, y1 = inv.transform((event.x, event.y))
        x0, y0 = inv.transform((p["px"], p["py"]))
        dx, dy = x1 - x0, y1 - y0
        xl, xr, yb, yt = p["lims"]
        ax.set_xlim(xl - dx, xr - dx)
        ax.set_ylim(yb - dy, yt - dy)
        p["cur"] = (xl - dx, xr - dx, yb - dy, yt - dy)
        self.canvas.draw_idle()

    def _on_graph_release(self, event):
        p = self._pan
        self._pan = None
        if p is None or not p["moved"]:
            return
        cur = p.get("cur")
        if cur is None:
            return
        self._set_limits(*cur)

    def _reset_view(self):
        self._set_limits(-5, 5, -5, 5)

    # ══════════════════════════════════════════════════════════
    #  СОХРАНЕНИЕ
    # ══════════════════════════════════════════════════════════
    def _save_image(self):
        path = filedialog.asksaveasfilename(
            title="Save graph image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("SVG vector", "*.svg"),
                       ("PDF", "*.pdf"), ("All files", "*.*")],
            initialfile="graph.png")
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            self._set_status(f"Saved: {path}", BTN_ADD)
        except Exception as ex:
            messagebox.showerror("Save error", str(ex))

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
            title="Save project", defaultextension=self.PROJECT_EXT,
            filetypes=[("Function Visualizer project", "*" + self.PROJECT_EXT), ("JSON", "*.json")],
            initialfile="graph" + self.PROJECT_EXT)
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._project_dict(), f, ensure_ascii=False, indent=2)
            self._set_status(f"Project saved: {path}", BTN_ADD)
        except Exception as ex:
            messagebox.showerror("Save error", str(ex))

    def _open_project(self):
        path = filedialog.askopenfilename(
            title="Open project",
            filetypes=[("Function Visualizer project", "*" + self.PROJECT_EXT),
                       ("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.load_project(data)
            self._set_status(f"Project loaded: {path}", BTN_ADD)
        except Exception as ex:
            messagebox.showerror("Open error", f"Cannot open project:\n{ex}")

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
            al = data.get("axis_labels", {})
            for key in ("x", "y"):
                if key in al and isinstance(al[key], dict):
                    fv.AXIS_LABELS[key].update(al[key])
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
        try:
            if self._redraw_job is not None:
                self.after_cancel(self._redraw_job)
        except Exception:
            pass
        self.destroy()


# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
