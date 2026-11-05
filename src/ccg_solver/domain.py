"""M2：域。

域是**模式**（含变量的范畴）的有限集合，语义"变量是其中某个模式的实例"。ground 范畴是模式的特例。
初值 ``None`` = 复杂度界 L 下的全体候选（隐式，不枚举：见 docs/m2-addendum-3 的实现备注，
L=2 下按 论元数+深度 的全体候选超过 13 万个）。

``enumerate_categories`` 只作报告/调试工具，L ≤ 1 时可用。
"""
from __future__ import annotations

from typing import Iterable

from .category import Category, Functor, Var, atom, bwd, complexity, fresh_var, fwd, is_admissible, _functor

# 域枚举用的原子。不含裸 S：金标准里 S 总带特征，裸 S 只会让域翻倍。
DEFAULT_ATOMS: tuple[tuple[str, str | None], ...] = (
    ("S", "dcl"), ("S", "q"), ("N", None), ("NP", None), ("PP", None), ("conj", None),
)


def enumerate_categories(L: int, atoms=DEFAULT_ATOMS, max_slashes: int | None = None) -> list[Category]:
    """ground、``is_admissible``、``complexity ≤ L`` 且斜杠数 ≤ ``max_slashes`` 的范畴，按 (complexity, str) 排序。
    complexity（论元数+深度）≤ L 不给出斜杠数上界（论元内部的 arity 不计），所以 L ≥ 2 时结果按斜杠数截断：
    默认 max_slashes = L（L ≤ 1 时精确），只作报告用。"""
    max_slashes = L if max_slashes is None else max_slashes
    base = [atom(n, f) for n, f in atoms]
    levels = {0: list(base)}
    for n in range(1, max_slashes + 1):
        cur = []
        for i in range(n):
            for r in levels[i]:
                for a in levels[n - 1 - i]:
                    for mk in (fwd, bwd):
                        c = mk(r, a)
                        if is_admissible(c):
                            cur.append(c)
        levels[n] = cur
    out = [c for lv in levels.values() for c in lv if complexity(c) <= L]
    return sorted(set(out), key=lambda c: (complexity(c), str(c)))


# ---------------------------------------------------------------- 模式工具


def canonical(c: Category, names: dict[Var, str] | None = None) -> str:
    """变量按首次出现重命名的规范打印（alpha 等价 ⇔ 字符串相等）。"""
    names = {} if names is None else names
    if isinstance(c, Var):
        n = names.get(c)
        if n is None:
            n = names[c] = f"?_{len(names) + 1}"
        return n
    if isinstance(c, Functor):
        r, a = canonical(c.result, names), canonical(c.arg, names)
        if isinstance(c.result, Functor):
            r = f"({r})"
        if isinstance(c.arg, Functor):
            a = f"({a})"
        return f"{r}{c.slash}{a}"
    return str(c)


def freshen(c: Category, renames: dict[Var, Var] | None = None) -> Category:
    """所有变量换成新变量（同一变量换同一个）。"""
    renames = {} if renames is None else renames
    if isinstance(c, Var):
        v = renames.get(c)
        if v is None:
            v = renames[c] = fresh_var()
        return v
    if isinstance(c, Functor):
        return _functor(freshen(c.result, renames), c.slash, freshen(c.arg, renames))
    return c


class Domain:
    """不可变的模式集合，按规范形去重。"""

    __slots__ = ("_by_key", "_hash")

    def __init__(self, patterns: Iterable[Category]):
        d: dict[str, Category] = {}
        for p in patterns:
            d.setdefault(canonical(p), p)
        self._by_key = dict(sorted(d.items()))
        self._hash = hash(tuple(self._by_key))

    def __iter__(self):
        return iter(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)

    def __contains__(self, c: Category) -> bool:
        return canonical(c) in self._by_key

    def keys(self) -> tuple[str, ...]:
        return tuple(self._by_key)

    def items(self):
        return self._by_key.items()

    def __eq__(self, other) -> bool:
        if isinstance(other, Domain):
            return self._by_key.keys() == other._by_key.keys()
        if isinstance(other, (set, frozenset)):
            return set(self._by_key) == {canonical(c) for c in other}
        return NotImplemented

    def __hash__(self) -> int:
        return self._hash

    def __le__(self, other) -> bool:
        ok = other._by_key.keys() if isinstance(other, Domain) else {canonical(c) for c in other}
        return set(self._by_key) <= set(ok)

    def __repr__(self) -> str:
        return "Domain{" + ", ".join(self._by_key) + "}"
