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

from .category import Category
from .rules import APPLICATION, Rule


@dataclass(frozen=True)
class Profile:
    name: str
    extra_atoms: tuple[str, ...] = ()
    definitions: dict[str, str] = field(default_factory=dict)  # 原子名 → 范畴文本
    rules: dict[str, Rule] = field(default_factory=lambda: dict(APPLICATION))
    max_depth: int = 3
    max_arity: int = 4

    def activate(self) -> "Profile":
        return self  # 桩：实现时登记 extra_atoms

    def definition(self, name: str) -> Category | None:
        """已解析的定义（hash-cons），无定义返回 None。"""
        raise NotImplementedError

    def fold(self, c: Category) -> Category:
        """把等于某定义右边的子项换成该原子（自底向上，到不动点）。"""
        raise NotImplementedError

    def admissible(self, c: Category) -> bool:
        """折叠后 arity ≤ max_arity 且 depth ≤ max_depth（整棵树）。"""
        raise NotImplementedError


LEGACY = Profile("legacy")

FIRST_ORDER_VP = Profile(
    "first_order_vp",
    extra_atoms=("VP",),
    definitions={"VP": "S[dcl]\\NP"},
    rules={},  # 实现时填 COMPOSITION2
    max_depth=0,
)
