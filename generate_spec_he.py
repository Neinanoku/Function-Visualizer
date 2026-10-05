"""
Генерирует FuncVisualizer_he.spec (ивритская версия) — та же сборка, что и
generate_spec.py, но точка входа app_he.py и имя exe FuncVisualizer_he.
"""
from generate_spec import write_spec

if __name__ == "__main__":
    write_spec("app_he.py", "FuncVisualizer_he")
