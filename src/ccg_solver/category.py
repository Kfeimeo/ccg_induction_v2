"""M0：范畴表示（接口草案，待测试用例确认后实现）。

设计约定（与 tests/test_m0_category.py 对应）：

- 三种节点：``Atom(name, feat)``、``Var(id)``、``Functor(result, slash, arg)``。
  全部 ``dataclass(frozen=True)``，经手写 hash-cons 表构造：结构相等即指针相等。
- 原子名固定为 ``ATOMS = ("S", "N", "NP", "PP", "conj")``；特征（如 ``S[dcl]``）是原子身份的一部分，
  ``S`` 与 ``S[dcl]`` 是不同原子。特征变量（feature polymorphism）不在 M0 范围内。
- 变量 id 全局唯一、单调递增，由 ``fresh_var()`` 分配；``var(id)`` 取回同一对象。
- 文本语法：``/`` ``\\`` 左结合，括号覆盖；单个大写字母 + 可选数字（``X``、``Y``、``T1``）且不是原子名的记号是变量，
  同一 ``env`` 内同名同变量；``?<id>`` 直接引用已有变量。打印时复合子项总加括号，变量打印为 ``?<id>``。
- 复杂度：``arity`` = 结果脊上的论元个数；``depth`` = 论元位置上的函子嵌套深度（原子论元不计）；
  ``complexity = arity + depth``。``MAX_ARITY = 4``，``is_admissible`` 对整棵树（含论元内部）检查论元数上限。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

ATOMS: tuple[str, ...] = ("S", "N", "NP", "PP", "conj")
MAX_ARITY: int = 4

FWD = "/"
BWD = "\\"


@dataclass(frozen=True)
class Atom:
    name: str
    feat: str | None = None


@dataclass(frozen=True)
class Var:
    id: int


@dataclass(frozen=True)
class Functor:
    result: "Category"
    slash: str
    arg: "Category"


Category = Atom | Var | Functor


def atom(name: str, feat: str | None = None) -> Atom:
    """hash-cons 的原子；``name`` 不在 ``ATOMS`` 内时抛 ``ValueError``。"""
    raise NotImplementedError


def fresh_var() -> Var:
    """分配一个全局唯一、id 单调递增的新变量。"""
    raise NotImplementedError


def var(id: int) -> Var:
    """按 id 取回（或登记）变量，hash-cons。"""
    raise NotImplementedError


def fwd(result: Category, arg: Category) -> Functor:
    """``result/arg``，hash-cons。"""
    raise NotImplementedError


def bwd(result: Category, arg: Category) -> Functor:
    """``result\\arg``，hash-cons。"""
    raise NotImplementedError


def parse(text: str, env: dict[str, Var] | None = None) -> Category:
    """解析范畴文本。``env`` 把变量名映射到变量；为 ``None`` 时本次调用内部使用临时 env（同名同变量，跨调用不共享）。"""
    raise NotImplementedError


def arity(c: Category) -> int:
    raise NotImplementedError


def depth(c: Category) -> int:
    raise NotImplementedError


def complexity(c: Category) -> int:
    raise NotImplementedError


def is_admissible(c: Category) -> bool:
    """整棵树内任一函子的论元数都 ≤ ``MAX_ARITY``。"""
    raise NotImplementedError


def free_vars(c: Category) -> set[Var]:
    raise NotImplementedError


def subterms(c: Category) -> Iterable[Category]:
    """前序遍历所有子项（含自身），供 dump / 因子图用。"""
    raise NotImplementedError
