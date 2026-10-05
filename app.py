"""
Визуализатор функций — GUI
Зависимости: pip install ttkbootstrap matplotlib numpy sympy scipy
Сборка .exe: pyinstaller --onefile --windowed --name "FuncVisualizer" app.py
"""

import sys
import os
import ast
import traceback
import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ── Импортируем движок визуализатора ─────────────────────────
# Добавляем путь к _MEIPASS чтобы PyInstaller находил function_visualizer.py
if getattr(sys, 'frozen', False):
    _base = sys._MEIPASS
    if _base not in sys.path:
        sys.path.insert(0, _base)

import function_visualizer as fv


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
BTN_PLOT  = "#e94560"
BTN_ADD   = "#2ab850"
BTN_DEL   = "#84878c"

FUNC_COLORS = fv.CURVE_COLORS  # берём из движка


# ═════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ВИДЖЕТЫ
# ═════════════════════════════════════════════════════════════

def styled_label(parent, text, size=10, color=TEXT, bold=False, **kw):
    weight = "bold" if bold else "normal"
    return tk.Label(parent, text=text, bg=parent["bg"] if hasattr(parent, "__getitem__") else APP_BG,
                    fg=color, font=("Segoe UI", size, weight), **kw)


def styled_entry(parent, width=14, **kw):
    e = tk.Entry(parent, width=width, bg=ENTRY_BG, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 font=("Segoe UI", 10),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, **kw)
    return e


def styled_check(parent, text, var, **kw):
    return tk.Checkbutton(parent, text=text, variable=var,
                          bg=parent.cget("bg"), fg=TEXT,
                          selectcolor=ENTRY_BG, activebackground=CARD_BG,
                          activeforeground=TEXT, font=("Segoe UI", 10),
                          bd=0, highlightthickness=0, **kw)


def card(parent, title="", **kw):
    """Карточка-секция с заголовком. Возвращает inner frame для контента."""
    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    inner = tk.Frame(outer, bg=CARD_BG, bd=0, relief="flat")
    inner.pack(fill="both", expand=True)
    if title:
        tk.Label(inner, text=title, bg=CARD_BG, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(0, 6))
    inner._outer = outer   # храним ссылку чтобы pack вызывать снаружи
    return inner


def card_pack(widget, **kw):
    """pack карточки через её outer frame."""
    widget._outer.pack(**kw)


def row(parent, bg=None):
    bg = bg or parent.cget("bg")
    return tk.Frame(parent, bg=bg)


from tkinter import colorchooser

# ═════════════════════════════════════════════════════════════
#  СТРОКА ОДНОЙ ФУНКЦИИ
# ═════════════════════════════════════════════════════════════

class FuncRow:
    LINESTYLES = {"Solid": "-", "Dashed": "--", "Dotted": ":"}

    def __init__(self, parent, idx, on_delete):
        self.idx   = idx
        self.color = FUNC_COLORS[idx % len(FUNC_COLORS)]

        self.frame = tk.Frame(parent, bg=PANEL_BG)
        self.frame.pack(fill="x", padx=4, pady=3)

        # Индекс функции
        self.idx_label = tk.Label(
            self.frame, text=f"f{idx}", width=2,
            bg=PANEL_BG, fg=SUBTEXT,
            font=("Segoe UI", 9, "bold")
        )
        self.idx_label.pack(side="left", padx=(6, 2))

        # Кликабельный цветной квадрат — выбор цвета
        self.dot = tk.Button(
            self.frame,
            bg=self.color,
            width=2, height=1,
            relief="flat", bd=0,
            cursor="hand2",
            command=self._pick_color,
        )
        self.dot.pack(side="left", padx=(2, 6))

        # Поле ввода функции
        self.entry = styled_entry(self.frame, width=24)
        self.entry.pack(side="left", padx=2)
        self.entry.insert(0, "")

        # Ширина линии — кнопки +/- вместо Spinbox
        tk.Label(self.frame, text="w:", bg=PANEL_BG, fg=SUBTEXT,
                 font=("Segoe UI", 8)).pack(side="left", padx=(6, 1))
        self._lw = 1.8
        self._lw_vals = [0.5, 1.0, 1.5, 1.8, 2.5, 3.5, 5.0, 7.0]
        self._lw_idx  = 3   # default = 1.8

        self._lw_label = tk.Label(
            self.frame, text=f"{self._lw}", width=3,
            bg=ENTRY_BG, fg=TEXT, font=("Segoe UI", 9),
            relief="flat",
            highlightthickness=1, highlightbackground=BORDER,
        )
        self._lw_label.pack(side="left")

        def lw_dec():
            self._lw_idx = max(0, self._lw_idx - 1)
            self._lw = self._lw_vals[self._lw_idx]
            self._lw_label.config(text=f"{self._lw}")

        def lw_inc():
            self._lw_idx = min(len(self._lw_vals)-1, self._lw_idx + 1)
            self._lw = self._lw_vals[self._lw_idx]
            self._lw_label.config(text=f"{self._lw}")

        btn_f = dict(bg=PANEL_BG, fg=TEXT, relief="flat", bd=0,
                     font=("Segoe UI", 8), cursor="hand2", padx=2)
        tk.Button(self.frame, text="▼", command=lw_dec, **btn_f).pack(side="left")
        tk.Button(self.frame, text="▲", command=lw_inc, **btn_f).pack(side="left", padx=(0, 4))

        # Тип линии
        self.linestyle_var = tk.StringVar(value=list(self.LINESTYLES.keys())[0])
        om = tk.OptionMenu(self.frame, self.linestyle_var, *self.LINESTYLES.keys())
        om.config(bg=ENTRY_BG, fg=TEXT, activebackground=CARD_BG,
                  activeforeground=TEXT, relief="flat",
                  font=("Segoe UI", 8), bd=0, highlightthickness=1,
                  highlightbackground=BORDER, width=10)
        om["menu"].config(bg=ENTRY_BG, fg=TEXT, activebackground=ACCENT,
                          font=("Segoe UI", 9))
        om.pack(side="left", padx=(2, 4))

        # Кнопка удалить
        tk.Button(self.frame, text="✕", bg=BTN_DEL, fg="white",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  cursor="hand2", bd=0, padx=6,
                  command=lambda: on_delete(self)).pack(side="left", padx=4)

        # ── Вторая строка: область определения (domain) ───────
        # Функция строится только на [domain_from, domain_to], при этом
        # view window остаётся независимым. По умолчанию -inf..inf — вся
        # область определения функции.
        self.frame2 = tk.Frame(parent, bg=PANEL_BG)
        self.frame2.pack(fill="x", padx=4, pady=(0, 3))

        tk.Label(self.frame2, text="", width=2, bg=PANEL_BG).pack(side="left", padx=(6, 2))
        tk.Label(self.frame2, text="domain:", bg=PANEL_BG, fg=SUBTEXT,
                 font=("Segoe UI", 8)).pack(side="left", padx=(2, 4))

        tk.Label(self.frame2, text="from", bg=PANEL_BG, fg=SUBTEXT,
                 font=("Segoe UI", 8)).pack(side="left", padx=(0, 2))
        self.dom_from = styled_entry(self.frame2, width=7)
        self.dom_from.pack(side="left", padx=2)
        self.dom_from.insert(0, "-inf")

        tk.Label(self.frame2, text="to", bg=PANEL_BG, fg=SUBTEXT,
                 font=("Segoe UI", 8)).pack(side="left", padx=(6, 2))
        self.dom_to = styled_entry(self.frame2, width=7)
        self.dom_to.pack(side="left", padx=2)
        self.dom_to.insert(0, "inf")

    def _parse_domain_val(self, s, default):
        """Парсит границу домена: число, выражение (pi, 2pi, pi/2...), inf/-inf, пусто."""
        return fv.parse_number(s, default=default)

    def _pick_color(self):
        result = colorchooser.askcolor(
            color=self.color,
            title=f"Pick color for f{self.idx}",
        )
        if result and result[1]:
            self.color = result[1]
            self.dot.config(bg=self.color)

    def update_idx(self, new_idx):
        self.idx = new_idx
        self.idx_label.config(text=f"f{new_idx}")

    def get_color(self):
        return self.color

    def get_linewidth(self):
        return self._lw

    def get_linestyle(self):
        return self.LINESTYLES.get(self.linestyle_var.get(), "-")

    def get(self):
        return self.entry.get().strip()

    def get_domain(self):
        """Возвращает (from, to) области определения; по умолчанию (-inf, inf)."""
        try:
            d_from = self._parse_domain_val(self.dom_from.get(), float("-inf"))
        except ValueError:
            d_from = float("-inf")
        try:
            d_to = self._parse_domain_val(self.dom_to.get(), float("inf"))
        except ValueError:
            d_to = float("inf")
        if d_from > d_to:
            d_from, d_to = d_to, d_from
        return (d_from, d_to)

    def destroy(self):
        self.frame.destroy()
        self.frame2.destroy()


# ═════════════════════════════════════════════════════════════
#  СТРОКА ЗАЛИВКИ
# ═════════════════════════════════════════════════════════════

class FillRow:
    STYLES = ["45deg ////", "135deg \\\\\\\\", "Dots ...."]

    def __init__(self, parent, on_delete):
        self.frame = tk.Frame(parent, bg=PANEL_BG)
        self.frame.pack(fill="x", padx=4, pady=3)

        def lbl(t):
            tk.Label(self.frame, text=t, bg=PANEL_BG, fg=SUBTEXT,
                     font=("Segoe UI", 9)).pack(side="left", padx=(4, 1))

        lbl("f1:")
        self.f1 = styled_entry(self.frame, width=3); self.f1.pack(side="left")
        self.f1.insert(0, "0")

        lbl("f2:")
        self.f2 = styled_entry(self.frame, width=4); self.f2.pack(side="left")
        self.f2.insert(0, "x")

        lbl("from:")
        self.x_from = styled_entry(self.frame, width=6); self.x_from.pack(side="left")
        self.x_from.insert(0, "-3")

        lbl("to:")
        self.x_to = styled_entry(self.frame, width=6); self.x_to.pack(side="left")
        self.x_to.insert(0, "3")

        lbl("style:")

        self.style_var = tk.StringVar(value=self.STYLES[0])
        om = tk.OptionMenu(self.frame, self.style_var, *self.STYLES)
        om.config(bg=ENTRY_BG, fg=TEXT, activebackground=CARD_BG,
                  activeforeground=TEXT, relief="flat",
                  font=("Segoe UI", 9), bd=0, highlightthickness=0)
        om["menu"].config(bg=ENTRY_BG, fg=TEXT, activebackground=ACCENT)
        om.pack(side="left", padx=4)

        # Галочка — показывать вертикальные границы
        self.borders_var = tk.IntVar(value=1)
        tk.Checkbutton(self.frame, text="borders", variable=self.borders_var,
                       bg=PANEL_BG, fg=TEXT,
                       selectcolor=ENTRY_BG, activebackground=PANEL_BG,
                       font=("Segoe UI", 8), bd=0,
                       highlightthickness=0).pack(side="left", padx=4)

        # Слайдер плотности штриховки (0%=редко, 100%=густо)
        lbl("density:")
        self.density_var = tk.IntVar(value=100)
        self._density_ind = tk.Label(
            self.frame, text="100", width=3,
            bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 9, "bold"))

        def _on_density(val):
            try:
                self._density_ind.config(text=str(int(float(val))))
            except Exception:
                pass

        tk.Scale(self.frame, from_=0, to=100, orient="horizontal",
                 variable=self.density_var, length=80,
                 resolution=5,            # прыжки по 5, а не по 1
                 bg=CARD_BG, fg=TEXT,
                 troughcolor="#e94560",
                 activebackground=BTN_DEL,
                 highlightthickness=0, bd=0, sliderrelief="flat",
                 showvalue=False,
                 command=_on_density,
                 font=("Segoe UI", 7)).pack(side="left")

        # Индикатор значения справа от ползунка (как у Label size)
        self._density_ind.pack(side="left", padx=(4, 0))

        tk.Button(self.frame, text="✕", bg=BTN_DEL, fg="white",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  cursor="hand2", bd=0, padx=6,
                  command=lambda: on_delete(self)).pack(side="left", padx=(10, 4))

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
            density = 0.05 - pct * 0.04    # линейно от 0.05 до 0.01
            return (f1, f2, x_from, x_to, style, borders, density)
        except Exception:
            return None

    def destroy(self):
        self.frame.destroy()


# ═════════════════════════════════════════════════════════════
#  ГЛАВНОЕ ОКНО
# ═════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Function Visualizer — Ariadna")
        self.configure(bg=APP_BG)
        self.resizable(True, True)
        self.minsize(400, 400)

        self.func_rows = []
        self.fill_rows = []
        self._logo_img = None

        # Иконка окна и панели задач
        try:
            import os, sys
            base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            icon_path = os.path.join(base, "icon.ico")
            self.iconbitmap(icon_path)
        except Exception:
            pass

        self._build_ui()
        self._add_func()
        self._setup_clipboard_shortcuts()

        # Подбираем размер окна под контент после полной отрисовки
        self.update_idletasks()
        # Фиксируем ширину по содержимому, высота — авто
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        self.geometry(f"{w}x{h}")

    # ── Ctrl+C/X/V/A независимо от раскладки клавиатуры ─────
    def _setup_clipboard_shortcuts(self):
        """
        На Windows стандартные бинды Tk вида <Control-c> завязаны на keysym
        (символ), а не на физическую клавишу. Если раскладка не английская,
        физическая клавиша C/X/V/A при зажатом Ctrl присылает Tk совсем
        другой keysym (например, кириллический), и <Control-c> просто не
        срабатывает. event.keycode — это код физической клавиши (Windows
        virtual-key code), он одинаков независимо от текущей раскладки,
        поэтому ловим именно по нему и сами генерируем нужное virtual event.

        ВАЖНО: когда раскладка АНГЛИЙСКАЯ, встроенный бинд Tk уже
        срабатывает сам. Если мы тогда тоже сгенерируем <<Paste>>, действие
        выполнится ДВАЖДЫ (вставится "pipi" вместо "pi"). Поэтому наш
        обработчик вмешивается ТОЛЬКО когда встроенный бинд не сработал бы —
        то есть когда keysym НЕ совпадает с латинской буквой действия.
        """
        # keycode (физический) -> (virtual event, латинский keysym этого действия)
        KEYCODE_TO_ACTION = {
            67: ("<<Copy>>",  ("c", "C")),   # C
            88: ("<<Cut>>",   ("x", "X")),   # X
            86: ("<<Paste>>", ("v", "V")),   # V
        }
        entry_types = tuple(t for t in (tk.Entry, tk.Text, getattr(ttk, 'Entry', None))
                             if t is not None)

        def handler(event):
            # Игнорируем, если вместе с Ctrl зажат ещё и Shift —
            # это может быть другая комбинация (не наш случай в этой
            # программе, но на всякий случай не перехватываем).
            if event.state & 0x0001:
                return None
            widget = event.widget
            if not isinstance(widget, entry_types):
                return None

            action = KEYCODE_TO_ACTION.get(event.keycode)
            if action is not None:
                virtual_event, latin_keysyms = action
                # Если keysym — латинская буква этого действия, значит
                # раскладка английская и встроенный бинд Tk уже отработал.
                # Не дублируем.
                if event.keysym in latin_keysyms:
                    return None
                widget.event_generate(virtual_event)
                return "break"

            if event.keycode == 65:   # A — выделить всё в поле
                if event.keysym in ("a", "A"):
                    # английская раскладка — Tk сам обработает (если есть бинд);
                    # но select-all у Entry по умолчанию нет, поэтому делаем сами
                    # и здесь, и там это безопасно (идемпотентно).
                    pass
                try:
                    widget.select_range(0, 'end')
                    widget.icursor('end')
                except Exception:
                    pass
                return "break"
            return None

        self.bind_all("<Control-KeyPress>", handler)

    # ── UI ──────────────────────────────────────────────────
    def _build_ui(self):
        # ── Заголовок с логотипом ────────────────────────────
        hdr = tk.Frame(self, bg=APP_BG)
        hdr.pack(fill="x")

        # Логотип
        try:
            from PIL import Image, ImageTk
            import os, sys
            base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            logo_path = os.path.join(base, "ariadna-logo1-trnsp.png")
            img = Image.open(logo_path).convert("RGBA")
            h = 120
            w = int(img.width * h / img.height)
            img = img.resize((w, h), Image.LANCZOS)
            self._logo_img = ImageTk.PhotoImage(img)
            tk.Label(hdr, image=self._logo_img, bg=APP_BG).pack(
                side="left", padx=20, pady=10)
        except Exception:
            tk.Label(hdr, text="ariadna", bg=APP_BG, fg=ACCENT,
                     font=("Segoe UI", 22, "bold")).pack(side="left", padx=20, pady=10)

        # Заголовок и автор справа от логотипа
        title_frame = tk.Frame(hdr, bg=APP_BG)
        title_frame.pack(side="left", padx=(0, 20))

        tk.Label(title_frame, text="Function Visualizer",
                 bg=APP_BG, fg=TEXT,
                 font=("Segoe UI", 20, "bold")).pack(anchor="w")

        tk.Label(title_frame, text="by Daniel",
                 bg=APP_BG, fg=SUBTEXT,
                 font=("Segoe UI", 10)).pack(anchor="w")

        # Разделитель под заголовком
        tk.Frame(self, bg=ACCENT, height=3).pack(fill="x")

        # Прокручиваемая область
        canvas = tk.Canvas(self, bg=APP_BG, highlightthickness=0)
        scroll = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=APP_BG)
        self.scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        p = self.scroll_frame

        # ── Functions ────────────────────────────────────────
        self.func_card = card(p, "Functions")
        card_pack(self.func_card, fill="x", padx=12, pady=(12, 4))
        self.func_list = tk.Frame(self.func_card, bg=PANEL_BG)
        self.func_list.pack(fill="x", padx=8, pady=4)
        tk.Button(self.func_card, text="+ Add function",
                  bg=BTN_ADD, fg="white", relief="flat",
                  font=("Segoe UI", 9, "bold"), cursor="hand2", bd=0, padx=10, pady=4,
                  command=self._add_func).pack(anchor="w", padx=10, pady=(0, 8))

        # ── View window ──────────────────────────────────────
        lim = card(p, "View Window")
        card_pack(lim, fill="x", padx=12, pady=4)
        g = tk.Frame(lim, bg=CARD_BG)
        g.pack(fill="x", padx=10, pady=(0, 10))

        def lim_row(label, defaults):
            r = tk.Frame(g, bg=CARD_BG)
            r.pack(fill="x", pady=2)
            tk.Label(r, text=label, bg=CARD_BG, fg=SUBTEXT,
                     font=("Segoe UI", 9), width=12, anchor="e").pack(side="left")
            entries = []
            for d in defaults:
                e = styled_entry(r, width=8)
                e.insert(0, d)
                e.pack(side="left", padx=4)
                entries.append(e)
            return entries

        xl, xr = lim_row("X:  from / to", ["-5", "5"])
        yb, yt = lim_row("Y:  from / to", ["-5",  "5"])
        self.xlim_l, self.xlim_r = xl, xr
        self.ylim_b, self.ylim_t = yb, yt

        # ── Grid ─────────────────────────────────────────────
        grid_c = card(p, "Grid")
        card_pack(grid_c, fill="x", padx=12, pady=4)
        gr = tk.Frame(grid_c, bg=CARD_BG)
        gr.pack(fill="x", padx=10, pady=(0, 10))

        self.grid_var = tk.IntVar(value=1)
        styled_check(gr, "Show grid", self.grid_var).pack(side="left", padx=4)

        tk.Label(gr, text="  Step X:", bg=CARD_BG, fg=SUBTEXT,
                 font=("Segoe UI", 9)).pack(side="left")
        self.xgrid_e = styled_entry(gr, width=5)
        self.xgrid_e.insert(0, "1")
        self.xgrid_e.pack(side="left", padx=4)

        tk.Label(gr, text="Step Y:", bg=CARD_BG, fg=SUBTEXT,
                 font=("Segoe UI", 9)).pack(side="left")
        self.ygrid_e = styled_entry(gr, width=5)
        self.ygrid_e.insert(0, "1")
        self.ygrid_e.pack(side="left", padx=4)

        # ── Display options ───────────────────────────────────
        disp = card(p, "Display on Graph")
        card_pack(disp, fill="x", padx=12, pady=4)
        dg = tk.Frame(disp, bg=CARD_BG)
        dg.pack(fill="x", padx=10, pady=(0, 10))

        self.v_asimp = tk.IntVar(value=1)
        self.v_disc  = tk.IntVar(value=1)
        self.v_extr  = tk.IntVar(value=1)
        self.v_xtag  = tk.IntVar(value=1)
        self.v_ytag  = tk.IntVar(value=1)
        self.v_inter = tk.IntVar(value=1)
        self.v_show_values = tk.IntVar(value=1)
        self.v_xhide = tk.IntVar(value=0)
        self.v_yhide = tk.IntVar(value=0)

        checks = [
            ("Asymptotes",         self.v_asimp),
            ("Holes",              self.v_disc),
            ("Extrema",            self.v_extr),
            ("X-intercepts",       self.v_xtag),
            ("Y-intercepts",       self.v_ytag),
            ("Intersections",      self.v_inter),
            ("Show values",        self.v_show_values),
        ]
        hides = [
            ("Hide X labels",   self.v_xhide),
            ("Hide Y labels",   self.v_yhide),
        ]
        row1 = tk.Frame(dg, bg=CARD_BG); row1.pack(fill="x")
        row2 = tk.Frame(dg, bg=CARD_BG); row2.pack(fill="x")
        for txt, var in checks:
            styled_check(row1, txt, var).pack(side="left", padx=6, pady=2)
        for txt, var in hides:
            styled_check(row2, txt, var).pack(side="left", padx=6, pady=2)

        # Размер текста
        row3 = tk.Frame(dg, bg=CARD_BG); row3.pack(fill="x", pady=(4, 2))
        tk.Label(row3, text="Label size:", bg=CARD_BG, fg=SUBTEXT,
                 font=("Segoe UI", 9)).pack(side="left", padx=(6, 4))
        self.font_size_var = tk.IntVar(value=10)
        slider = tk.Scale(
            row3, from_=6, to=20, orient="horizontal",
            variable=self.font_size_var, length=160,
            bg=CARD_BG, fg=TEXT,
            troughcolor="#e94560",
            activebackground=BTN_DEL,
            highlightthickness=0, bd=0, sliderrelief="flat",
            font=("Segoe UI", 8),
        )
        slider.pack(side="left")
        self.font_size_label = tk.Label(
            row3, textvariable=self.font_size_var,
            bg=CARD_BG, fg=ACCENT, font=("Segoe UI", 9, "bold"), width=3
        )
        self.font_size_label.pack(side="left", padx=2)

        # ── Fill ─────────────────────────────────────────────
        self.fill_card = card(p, "Area Fill")
        card_pack(self.fill_card, fill="x", padx=12, pady=4)

        tk.Label(self.fill_card,
                 text="f1 / f2 — function indices (0, 1, …) or 'x' for the X axis",
                 bg=CARD_BG, fg=SUBTEXT, font=("Segoe UI", 8)).pack(anchor="w", padx=10)

        self.fill_list = tk.Frame(self.fill_card, bg=PANEL_BG)
        self.fill_list.pack(fill="x", padx=8, pady=4)
        tk.Button(self.fill_card, text="+ Add fill",
                  bg="#2a7ae0", fg="white", relief="flat",
                  font=("Segoe UI", 9, "bold"), cursor="hand2", bd=0, padx=10, pady=4,
                  command=self._add_fill).pack(anchor="w", padx=10, pady=(0, 8))

        # ── Graph Labels (свободные подписи прямо на графике) ──
        labels_card = card(p, "Graph Labels")
        card_pack(labels_card, fill="x", padx=12, pady=4)

        tk.Label(labels_card,
                 text="Double-click empty space on the graph to add a label. "
                      "Drag to move, scroll to rotate, right-click for options.",
                 bg=CARD_BG, fg=SUBTEXT, font=("Segoe UI", 8),
                 wraplength=260, justify="left").pack(anchor="w", padx=10, pady=(0, 6))

        tk.Button(labels_card, text="Clear all labels",
                  bg=BTN_DEL, fg="white", relief="flat",
                  font=("Segoe UI", 9, "bold"), cursor="hand2", bd=0, padx=10, pady=4,
                  command=self._clear_labels).pack(anchor="w", padx=10, pady=(0, 8))

        # ── Plot button ───────────────────────────────────────
        tk.Button(p, text="  Plot Graph",
                  bg=BTN_PLOT, fg="white",
                  font=("Segoe UI", 13, "bold"),
                  relief="flat", cursor="hand2",
                  bd=0, padx=20, pady=12,
                  command=self._plot).pack(fill="x", padx=12, pady=12)

        self.status = tk.Label(p, text="", bg=APP_BG, fg=SUBTEXT,
                               font=("Segoe UI", 9))
        self.status.pack(pady=(0, 10))

    # ── Свободные подписи ────────────────────────────────────
    def _clear_labels(self):
        if not fv.FREE_TEXTS:
            return
        if not messagebox.askyesno(
                "Clear all labels",
                f"Remove all {len(fv.FREE_TEXTS)} label(s) added on the graph?"):
            return
        fv.FREE_TEXTS.clear()
        mgr = fv._active_free_text_manager
        if mgr is not None:
            for record, artist in mgr.entries:
                try:
                    artist.remove()
                except Exception:
                    pass
            mgr.entries = []
            try:
                mgr.fig.canvas.draw_idle()
            except Exception:
                pass

    # ── Функции ─────────────────────────────────────────────
    def _add_func(self):
        idx = len(self.func_rows)
        row = FuncRow(self.func_list, idx, self._del_func)
        self.func_rows.append(row)

    def _del_func(self, row):
        if len(self.func_rows) <= 1:
            messagebox.showwarning("Cannot delete", "At least one function is required.")
            return
        row.destroy()
        self.func_rows.remove(row)
        for i, r in enumerate(self.func_rows):
            r.update_idx(i)

    # ── Заливка ─────────────────────────────────────────────
    def _add_fill(self):
        row = FillRow(self.fill_list, self._del_fill)
        self.fill_rows.append(row)

    def _del_fill(self, row):
        row.destroy()
        self.fill_rows.remove(row)

    # ── Построить ───────────────────────────────────────────
    def _plot(self):
        # Закрываем предыдущий график (если был) перед построением нового.
        plt.close('all')

        self.status.config(text="Computing...", fg=SUBTEXT)
        self.update_idletasks()

        try:
            funcs      = [r.get()            for r in self.func_rows if r.get()]
            colors     = [r.get_color()      for r in self.func_rows if r.get()]
            linewidths = [r.get_linewidth()  for r in self.func_rows if r.get()]
            linestyles = [r.get_linestyle()  for r in self.func_rows if r.get()]
            domains    = [r.get_domain()     for r in self.func_rows if r.get()]
            if not funcs:
                messagebox.showerror("Error", "Enter at least one function.")
                return

            xlim_l = fv.parse_number(self.xlim_l.get())
            xlim_r = fv.parse_number(self.xlim_r.get())
            ylim_b = fv.parse_number(self.ylim_b.get())
            ylim_t = fv.parse_number(self.ylim_t.get())

            if xlim_l >= xlim_r or ylim_b >= ylim_t:
                messagebox.showerror("Error", "Check limits: left < right, bottom < top.")
                return

            xgrid = fv.parse_number(self.xgrid_e.get())
            ygrid = fv.parse_number(self.ygrid_e.get())

            fills = []
            for fr in self.fill_rows:
                entry = fr.get()
                if entry is None:
                    messagebox.showerror("Error", "Check fill parameters.")
                    return
                fills.append(entry)

            # Пробрасываем в модуль визуализатора через глобальные переменные
            fv.FUNCS        = funcs
            fv.FUNC_DOMAINS = domains
            fv.CURVE_COLORS = colors
            fv.CURVE_WIDTHS = linewidths
            fv.CURVE_STYLES = linestyles
            fv.X_LIM_L = xlim_l
            fv.X_LIM_R = xlim_r
            fv.Y_LIM_B = ylim_b
            fv.Y_LIM_T = ylim_t
            fv.GRID    = self.grid_var.get()
            fv.X_GRID  = xgrid
            fv.Y_GRID  = ygrid
            fv.ASIMP   = self.v_asimp.get()
            fv.DISC    = self.v_disc.get()
            fv.EXTR    = self.v_extr.get()
            fv.X_TAG   = self.v_xtag.get()
            fv.Y_TAG   = self.v_ytag.get()
            fv.INTER        = self.v_inter.get()
            fv.SHOW_VALUES  = self.v_show_values.get()
            fv.X_HIDE       = self.v_xhide.get()
            fv.Y_HIDE  = self.v_yhide.get()
            fv.FONT_SIZE = self.font_size_var.get()
            fv.Y_HIDE  = self.v_yhide.get()
            fv.FILL    = fills

            fv.plot_function()
            self.status.config(text="Graph plotted successfully", fg=BTN_ADD)

        except Exception:
            err = traceback.format_exc()
            self.status.config(text="Error — check details", fg=BTN_DEL)
            messagebox.showerror("Plot Error", err)


# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
