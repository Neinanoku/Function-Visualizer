"""
Генерирует FuncVisualizer_ru.spec (русская версия) — та же сборка, что и
generate_spec.py, но точка входа app_ru.py и имя exe FuncVisualizer_ru.
"""
from generate_spec import write_spec

if __name__ == "__main__":
    write_spec("app_ru.py", "FuncVisualizer_ru")
