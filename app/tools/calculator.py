"""The `calculator` tool — safe arithmetic WITHOUT eval().

Using eval() on user text is a classic security hole (it can run any code). Instead
we parse the expression into an Abstract Syntax Tree and walk it ourselves, allowing
ONLY numbers and basic math operators. Anything else returns a string starting with
"Error" (never raises), which is what the LLM and the contract test expect.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable

from langchain_core.tools import tool

# The only operations we permit. A node of any other type is rejected.
_BIN: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable[[float], float]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    # Names, function calls, attribute access, etc. are NOT allowed.
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (e.g. '2 + 3 * 4') and return the result.

    Supports + - * / // % ** and parentheses. Use this for any exact calculation.
    """
    try:
        result = _eval(ast.parse(expression, mode="eval").body)
    except Exception:
        return f"Error: cannot evaluate {expression!r} (only plain arithmetic is allowed)"
    # Show 14 rather than 14.0 when the result is a whole number.
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)
