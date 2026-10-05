"""
Function Visualizer 3.0 — ивритская версия (מדמה פונקציות).

Это тот же app.py с языком интерфейса LANG = "he": панель настроек справа,
график слева (оси по-прежнему направлены вправо и вверх), ивритский текст —
шрифтом David, цифры/латиница/формулы — прежним шрифтом (TeX Gyre Schola).
Сборка: build_he.bat → dist\\FuncVisualizer_he.exe
"""
import app

if __name__ == "__main__":
    app.main(lang="he")
