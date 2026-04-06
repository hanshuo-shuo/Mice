"""
Stub out optional dependencies that may not install cleanly.
Import this before anything that touches cellworld_game.
"""
import sys
import types


def _ensure_pulsekit():
    if "pulsekit" not in sys.modules:
        try:
            import pulsekit  # noqa: F401
        except ImportError:
            pk = types.ModuleType("pulsekit")

            class CodeBlock:
                def __init__(self, name):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    pass

            pk.CodeBlock = CodeBlock
            sys.modules["pulsekit"] = pk


_ensure_pulsekit()
