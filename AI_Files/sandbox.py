"""
Sandboxed Python code execution.
__builtins__ stripped to safelist. No network. Timeout + memory limit.
"""
import os
import sys
import subprocess
import tempfile
import shutil
import time as _time
from config import SANDBOX_DIR, SANDBOX_TIMEOUT_SECONDS, SANDBOX_MAX_OUTPUT_CHARS, SANDBOX_ENABLED
from logger import get_logger

log = get_logger("sandbox")

SAFE_BUILTINS = [
    "abs", "all", "any", "bin", "bool", "bytearray", "bytes", "callable",
    "chr", "complex", "dict", "dir", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "getattr", "hasattr", "hash", "hex",
    "id", "int", "isinstance", "issubclass", "iter", "len", "list",
    "map", "max", "min", "next", "object", "oct", "ord", "pow",
    "print", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "type", "vars", "zip",
    "True", "False", "None", "Exception", "ValueError", "TypeError",
    "KeyError", "IndexError", "RuntimeError", "StopIteration",
    "ZeroDivisionError", "AttributeError", "OverflowError",
    "ArithmeticError", "LookupError", "NameError",
]

SAFE_MODULES = {
    "math", "random", "string", "collections", "itertools", "functools",
    "operator", "decimal", "fractions", "statistics", "datetime", "time",
    "re", "json", "csv", "io", "copy", "enum", "dataclasses", "typing",
    "textwrap", "unicodedata", "hashlib", "hmac", "base64", "struct",
    "bisect", "heapq", "array", "pprint",
}


def execute_code(code, language="python"):
    if not SANDBOX_ENABLED:
        return {"success": False, "output": "", "error": "Code execution disabled", "duration": 0}
    if language != "python":
        return {"success": False, "output": "", "error": "Only Python supported", "duration": 0}

    work_dir = tempfile.mkdtemp(dir=SANDBOX_DIR, prefix="exec_")
    try:
        script_path = os.path.join(work_dir, "script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(_wrap_code(code))

        start = _time.time()
        result = subprocess.run(
            [sys.executable, "-u", script_path],
            capture_output=True, text=True, timeout=SANDBOX_TIMEOUT_SECONDS,
            cwd=work_dir,
            env={"PATH": "", "HOME": work_dir, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": ""}
        )
        duration = round(_time.time() - start, 3)
        stdout = (result.stdout or "")[:SANDBOX_MAX_OUTPUT_CHARS]
        stderr = (result.stderr or "")[:SANDBOX_MAX_OUTPUT_CHARS]

        if result.returncode == 0:
            return {"success": True, "output": stdout, "error": stderr, "duration": duration}
        return {"success": False, "output": stdout,
                "error": stderr or f"Exit code {result.returncode}", "duration": duration}
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "",
                "error": f"Timed out after {SANDBOX_TIMEOUT_SECONDS}s", "duration": SANDBOX_TIMEOUT_SECONDS}
    except Exception as e:
        log.error(f"Sandbox error: {e}")
        return {"success": False, "output": "", "error": str(e), "duration": 0}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _wrap_code(user_code):
    """Build sandbox wrapper. Uses repr() for user code — no f-string interpolation."""
    timeout_str = str(SANDBOX_TIMEOUT_SECONDS)
    safe_mods = repr(SAFE_MODULES)
    safe_bi = repr(SAFE_BUILTINS)

    preamble = (
        "import sys\nimport signal\n\n"
        "try:\n    import resource\n"
        "    resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))\n"
        "    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))\n"
        "except (ImportError, ValueError):\n    pass\n\n"
        "try:\n"
        "    signal.signal(signal.SIGALRM, lambda s,f: (_ for _ in ()).throw(TimeoutError()))\n"
        "    signal.alarm(" + timeout_str + ")\n"
        "except (AttributeError, OSError):\n    pass\n\n"
        "_orig = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__\n"
        "_safe = {}\n"
        "for _k in " + safe_bi + ":\n"
        "    if _k in _orig:\n        _safe[_k] = _orig[_k]\n\n"
        "_real_import = _orig.get('__import__', __import__)\n"
        "_allowed_mods = " + safe_mods + "\n"
        "def _restricted_import(name, *a, **kw):\n"
        "    if name.split('.')[0] not in _allowed_mods:\n"
        "        raise ImportError(f'Import of {name!r} blocked in sandbox')\n"
        "    return _real_import(name, *a, **kw)\n"
        "_safe['__import__'] = _restricted_import\n\n"
    )
    code_var = "\n_user_code = " + repr(user_code) + "\n\n"
    runner = (
        "_g = {'__builtins__': _safe}\n"
        "try:\n    exec(compile(_user_code, '<sandbox>', 'exec'), _g)\n"
        "except Exception as _e:\n"
        "    print(f'{type(_e).__name__}: {_e}', file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    return preamble + code_var + runner
