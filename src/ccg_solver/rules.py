"""M1：组合规则。M0–M3 只开 > < >B <B，T 完全关掉。

每条规则是 ``(left, right, state) -> Category | None``：在 ``state`` 上做必要的合一并返回结果范畴；
不适用时返回 None。**规则不负责回滚**，调用方用 mark/rollback 包住。
规则引入的新变量用 ``fresh_var()``。应用后调用方检查所有词型变量仍 ``is_admissible``。
"""
from __future__ import annotations

from typing import Callable, Optional

from .category import Category, bwd, fresh_var, fwd
from .state import State

Rule = Callable[[Category, Category, State], Optional[Category]]


def forward_application(left: Category, right: Category, st: State) -> Category | None:
    """X/Y  Y  →  X"""
    x = fresh_var()
    return x if st.unify(left, fwd(x, right)) else None


def backward_application(left: Category, right: Category, st: State) -> Category | None:
    """Y  X\\Y  →  X"""
    x = fresh_var()
    return x if st.unify(right, bwd(x, left)) else None


def forward_composition(left: Category, right: Category, st: State) -> Category | None:
    """X/Y  Y/Z  →  X/Z"""
    x, y, z = fresh_var(), fresh_var(), fresh_var()
    if st.unify(left, fwd(x, y)) and st.unify(right, fwd(y, z)):
        return fwd(x, z)
    return None


def backward_composition(left: Category, right: Category, st: State) -> Category | None:
    """Y\\Z  X\\Y  →  X\\Z"""
    x, y, z = fresh_var(), fresh_var(), fresh_var()
    if st.unify(left, bwd(y, z)) and st.unify(right, bwd(x, y)):
        return bwd(x, z)
    return None


RULES: dict[str, Rule] = {
    ">": forward_application,
    "<": backward_application,
    ">B": forward_composition,
    "<B": backward_composition,
}
