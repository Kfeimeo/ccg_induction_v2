"""M1：组合规则（接口草案）。M0–M3 只开 > < >B <B，T 完全关掉。

每条规则是 ``(left, right, state) -> Category | None``：在 ``state`` 上做必要的合一并返回结果范畴；
不适用时返回 None。**规则不负责回滚**，调用方用 mark/rollback 包住。
规则引入的新变量用 ``fresh_var()``。应用后调用方检查所有词型变量仍 ``is_admissible``。
"""
from __future__ import annotations

from typing import Callable, Optional

from .category import Category
from .state import State

Rule = Callable[[Category, Category, State], Optional[Category]]


def forward_application(left: Category, right: Category, st: State) -> Category | None:
    """X/Y  Y  →  X"""
    raise NotImplementedError


def backward_application(left: Category, right: Category, st: State) -> Category | None:
    """Y  X\\Y  →  X"""
    raise NotImplementedError


def forward_composition(left: Category, right: Category, st: State) -> Category | None:
    """X/Y  Y/Z  →  X/Z"""
    raise NotImplementedError


def backward_composition(left: Category, right: Category, st: State) -> Category | None:
    """Y\\Z  X\\Y  →  X\\Z"""
    raise NotImplementedError


RULES: dict[str, Rule] = {
    ">": forward_application,
    "<": backward_application,
    ">B": forward_composition,
    "<B": backward_composition,
}
