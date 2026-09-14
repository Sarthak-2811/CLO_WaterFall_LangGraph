"""
Sandbox execution environment for LLM-generated Monte Carlo scripts.

The original implementation blocked dangerous imports with a regex over the
raw source text (`_check_for_dangerous_imports`). That's trivially bypassed
(`__import__('os')`, string-built import names, `eval(...)`, etc.) because it
inspects the TEXT of the code, not what the code can actually reach at
runtime.

This version instead restricts what the executed code CAN reach:
  - a whitelist `__import__` that only allows numpy/pandas/json/math/random/
    itertools/statistics -- anything else raises ImportError, regardless of
    how the import is spelled or obfuscated.
  - a minimal `__builtins__` dict with no filesystem/process/network
    primitives (no open, no eval/exec/compile, no input).
  - a wall-clock timeout (SIGALRM) so a runaway loop can't hang the graph.

This is still Python's `exec()` under the hood and is NOT a substitute for a
real container/subprocess sandbox if you ever run this against untrusted
indentures rather than your own test PDFs -- but it closes the specific
bypasses the old blocklist was open to.
"""
import builtins
import contextlib
import io
import json
import signal
import traceback
from typing import Dict, Any, Tuple, Optional

ALLOWED_MODULES = {"numpy", "pandas", "json", "math", "random", "itertools", "statistics"}
EXECUTION_TIMEOUT_SECONDS = 30

_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
    "list", "map", "max", "min", "print", "range", "round", "set", "sorted",
    "str", "sum", "tuple", "zip", "isinstance", "issubclass", "type",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "RuntimeError", "StopIteration",
]


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout(f"Execution exceeded {EXECUTION_TIMEOUT_SECONDS}s wall-clock limit.")


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(
            f"SECURITY VIOLATION: import of '{name}' is blocked. "
            f"Allowed modules: {sorted(ALLOWED_MODULES)}."
        )
    return builtins.__import__(name, globals, locals, fromlist, level)


def _build_sandbox_globals() -> Dict[str, Any]:
    safe_builtins = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(builtins, n)}
    safe_builtins["__import__"] = _safe_import
    safe_builtins["True"] = True
    safe_builtins["False"] = False
    safe_builtins["None"] = None
    return {"__builtins__": safe_builtins}


def _extract_last_json(stdout: str) -> Optional[Dict[str, Any]]:
    lines = stdout.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def execute_simulation_code(code: str) -> Tuple[bool, Dict[str, Any], str]:
    """Executes generated code in a restricted namespace with a wall-clock timeout."""
    sandbox_globals = _build_sandbox_globals()
    stdout_buf = io.StringIO()

    timer_supported = hasattr(signal, "SIGALRM")
    old_handler = None
    try:
        if timer_supported:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(EXECUTION_TIMEOUT_SECONDS)

        with contextlib.redirect_stdout(stdout_buf):
            exec(code, sandbox_globals)

    except _Timeout as e:
        return False, {}, str(e)
    except ImportError as e:
        return False, {}, str(e)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n" + traceback.format_exc()
        return False, {}, error_msg
    finally:
        if timer_supported:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    stdout = stdout_buf.getvalue()
    result_dict = _extract_last_json(stdout)
    if result_dict is not None:
        return True, result_dict, ""
    return False, {}, f"Execution succeeded, but no valid JSON object found in stdout:\n{stdout}"
