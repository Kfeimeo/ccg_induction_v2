"""语法档案（实验版本机制）。接口草案。

``LEGACY`` 是 M0–M3 的现状：五个原子、无定义、只开 > <、depth ≤ 3。
``FIRST_ORDER_VP`` 是一阶实验：
- 新原子 ``VP``，定义 ``VP := S[dcl]\\NP``。定义原子在**合一**时按需展开（原子 vs 函子时展开再合一），
  但 ``resolve`` / 规范打印一律**折叠**成原子形式（遇到等于某定义右边的子项就换成该原子）。
- 规则集 ``>`` ``<`` ``>B`` ``<B`` ``>B²`` ``<B²``；无类型提升、无混合方向组合。
- ``max_depth = 0``：论元只能是原子（一阶）；结果位置可嵌套，arity 上限仍是 4。

使用：``State(profile=p)`` 决定定义展开/折叠与可容许性；``solve_sentence(..., profile=p)`` 决定规则集并把档案
传给状态。``p.activate()`` 把新原子名登记进 ``ATOMS``（幂等；对 LEGACY 无副作用）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .category import Category, Functor, MAX_ARITY, arity, depth, parse, register_atom, _functor, atom as _atom
from .rules import APPLICATION, COMPOSITION2, Rule


@dataclass(frozen=True)
class Profile:
    name: str
    extra_atoms: tuple[str, ...] = ()
    definitions: dict[str, str] = field(default_factory=dict)  # 原子名 → 范畴文本
    rules: dict[str, Rule] = field(default_factory=lambda: dict(APPLICATION))
    max_depth: int = 3
    max_arity: int = 4

    def activate(self) -> "Profile":
        for a in self.extra_atoms:
            register_atom(a)
        self._defs()  # 预解析，顺便校验定义文本
        return self

    def _defs(self) -> dict:
        cache = _DEF_CACHE.get(self.name)
        if cache is None:
            for a in self.extra_atoms:
                register_atom(a)
            parsed = {n: parse(t) for n, t in self.definitions.items()}
            cache = _DEF_CACHE[self.name] = (parsed, {v: _atom(n) for n, v in parsed.items()})
        return cache

    def definition(self, name: str) -> Category | None:
        return self._defs()[0].get(name)

    def fold(self, c: Category) -> Category:
        if not self.definitions:
            return c
        rev = self._defs()[1]
        if isinstance(c, Functor):
            c = _functor(self.fold(c.result), c.slash, self.fold(c.arg))
        return rev.get(c, c)

    def admissible(self, c: Category) -> bool:
        c = self.fold(c)
        from .category import subterms
        return all(arity(t) <= self.max_arity and depth(t) <= self.max_depth for t in subterms(c))


_DEF_CACHE: dict[str, tuple] = {}


LEGACY = Profile("legacy")

FIRST_ORDER_VP = Profile(
    "first_order_vp",
    extra_atoms=("VP",),
    definitions={"VP": "S[dcl]\\NP"},
    rules=COMPOSITION2,
    max_depth=0,
)
