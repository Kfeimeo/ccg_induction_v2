"""M0：范畴表示 —— hash-cons DAG、变量、解析/打印、复杂度度量。

设计约定：

- 三种节点：``Atom(name, feat)``、``Var(id)``、``Functor(result, slash, arg)``。
  全部 ``dataclass(frozen=True)``，经手写 hash-cons 表构造：结构相等即指针相等。
- 原子名固定为 ``ATOMS``；特征（如 ``S[dcl]``）是原子身份的一部分，``S`` 与 ``S[dcl]`` 是不同原子。
  特征变量（feature polymorphism）不在 M0 范围内。
- 变量 id 全局唯一、单调递增，由 ``fresh_var()`` 分配；``var(id)`` 取回同一对象。
- 文本语法：``/`` ``\\`` 左结合，括号覆盖。变量一律带 sigil：``?X`` 是命名变量（同一 ``env`` 内同名同变量，
  env 的键不含 sigil），``?<数字>`` 按 id 引用已有变量；裸大写一律是原子，不在 ``ATOMS`` 内即报错。
  打印时复合子项总加括号，变量打印为 ``?<id>``。
- 复杂度：``arity`` = 结果脊上的论元个数；``depth`` = 论元位置上的函子嵌套深度（原子论元不计）；
  ``complexity = arity + depth``。分别设限 ``MAX_ARITY = 4``、``MAX_DEPTH = 3``，
  ``is_admissible`` 对整棵树（含论元内部）检查。
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterator, Union

ATOMS: tuple[str, ...] = ("S", "N", "NP", "PP", "conj")
MAX_ARITY: int = 4
MAX_DEPTH: int = 3

FWD = "/"
BWD = "\\"


@dataclass(frozen=True, eq=False)
class Atom:
    name: str
    feat: str | None = None

    def __str__(self) -> str:
        return self.name if self.feat is None else f"{self.name}[{self.feat}]"


@dataclass(frozen=True, eq=False)
class Var:
    id: int

    def __str__(self) -> str:
        return f"?{self.id}"


@dataclass(frozen=True, eq=False)
class Functor:
    result: "Category"
    slash: str
    arg: "Category"

    def __str__(self) -> str:
        r = f"({self.result})" if isinstance(self.result, Functor) else str(self.result)
        a = f"({self.arg})" if isinstance(self.arg, Functor) else str(self.arg)
        return f"{r}{self.slash}{a}"


Category = Union[Atom, Var, Functor]

# eq=False 让 dataclass 保留 object 的 __eq__/__hash__（按身份）；hash-cons 保证结构相等 ⇔ 身份相等。

_atoms: dict[tuple[str, str | None], Atom] = {}
_vars: dict[int, Var] = {}
_functors: dict[tuple[int, str, int], Functor] = {}
_ids = itertools.count(1)


def atom(name: str, feat: str | None = None) -> Atom:
    if name not in ATOMS:
        raise ValueError(f"unknown atom {name!r}; atoms are {ATOMS}")
    key = (name, feat)
    a = _atoms.get(key)
    if a is None:
        a = _atoms[key] = Atom(name, feat)
    return a


def fresh_var() -> Var:
    return var(next(_ids))


def var(id: int) -> Var:
    v = _vars.get(id)
    if v is None:
        v = _vars[id] = Var(id)
    return v


def _functor(result: Category, slash: str, arg: Category) -> Functor:
    key = (id(result), slash, id(arg))
    f = _functors.get(key)
    if f is None:
        f = _functors[key] = Functor(result, slash, arg)
    return f


def fwd(result: Category, arg: Category) -> Functor:
    return _functor(result, FWD, arg)


def bwd(result: Category, arg: Category) -> Functor:
    return _functor(result, BWD, arg)


# ---------------------------------------------------------------- parse

_TOKEN = re.compile(r"\s*(?:(\?[A-Za-z0-9_]+)|([A-Za-z]+(?:\[[a-z]+\])?)|([/\\()]))")


def _tokenize(text: str) -> list[str]:
    pos, out = 0, []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            rest = text[pos:].strip()
            if not rest:
                break
            raise ValueError(f"bad token at {rest[:10]!r} in {text!r}")
        out.append(m.group(m.lastindex))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens: list[str], env: dict[str, Var]):
        self.toks, self.i, self.env = tokens, 0, env

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> str:
        t = self.peek()
        if t is None:
            raise ValueError("unexpected end of category")
        self.i += 1
        return t

    def category(self) -> Category:
        c = self.primary()
        while self.peek() in (FWD, BWD):  # 左结合
            slash = self.take()
            c = _functor(c, slash, self.primary())
        return c

    def primary(self) -> Category:
        t = self.take()
        if t == "(":
            c = self.category()
            if self.take() != ")":
                raise ValueError("expected ')'")
            return c
        if t.startswith("?"):
            name = t[1:]
            if name.isdigit():
                return var(int(name))
            if not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", name):
                raise ValueError(f"bad variable name {t!r}")
            v = self.env.get(name)
            if v is None:
                v = self.env[name] = fresh_var()
            return v
        if t in (FWD, BWD, ")"):
            raise ValueError(f"unexpected {t!r}")
        m = re.fullmatch(r"([A-Za-z]+)(?:\[([a-z]+)\])?", t)
        assert m is not None
        return atom(m.group(1), m.group(2))


def parse(text: str, env: dict[str, Var] | None = None) -> Category:
    p = _Parser(_tokenize(text), {} if env is None else env)
    c = p.category()
    if p.peek() is not None:
        raise ValueError(f"trailing tokens in {text!r}")
    return c


# ---------------------------------------------------------------- measures


def arity(c: Category) -> int:
    n = 0
    while isinstance(c, Functor):
        n, c = n + 1, c.result
    return n


def depth(c: Category) -> int:
    if not isinstance(c, Functor):
        return 0
    arg_d = 1 + depth(c.arg) if isinstance(c.arg, Functor) else 0
    return max(depth(c.result), arg_d)


def complexity(c: Category) -> int:
    return arity(c) + depth(c)


def is_admissible(c: Category) -> bool:
    return all(arity(t) <= MAX_ARITY and depth(t) <= MAX_DEPTH for t in subterms(c))


def free_vars(c: Category) -> set[Var]:
    return {t for t in subterms(c) if isinstance(t, Var)}


def subterms(c: Category) -> Iterator[Category]:
    yield c
    if isinstance(c, Functor):
        yield from subterms(c.result)
        yield from subterms(c.arg)
