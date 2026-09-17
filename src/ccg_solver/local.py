"""M1：局部求解器 —— 一句话 = 一行方程。

``solve_sentence(sent, lex, st)``：
- 在 ``st`` 之上深度优先枚举句子的所有推导（二叉括号化 × 规则），每步在状态上合一，
  只保留与当前状态一致的（存活）。
- 返回前把状态回滚到调用前：**调用不改变 st**。
- 按 σ 去重：σ = 推导蕴含的词型变量指派。规范打印时，句中词型变量本身和求解期间新建的局部变量
  按首次出现顺序重命名为 ``?_1 ?_2 …``；调用前已存在的其他全局变量保持原样。
  σ 相同的推导折叠为一条，``n_trees`` 记录折叠了多少棵树。
- 返回列表按 ``key`` 排序，因此输出确定。
- 剪枝：每次规则应用后，句中所有词型变量 resolve 后必须 ``is_admissible``；若给了 ``complexity_bound``（§4.6 的 L），
  还要求每个词型变量的 ``complexity`` ≤ L。
- ``rules`` 默认 ``APPLICATION``（> <）；传 ``COMPOSITION`` 开 >B <B。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Iterator

from .category import Category, Functor, Var, atom, complexity, fresh_var, is_admissible, _functor
from .corpus import Lexicon, Sentence
from .rules import APPLICATION, Rule
from .state import State


@dataclass(frozen=True)
class Tree:
    """推导树：叶子是词位置，内部节点是 (规则名, 左, 右)。"""
    rule: str | None
    span: tuple[int, int]
    children: tuple["Tree", ...] = ()

    def bracketed(self, words: tuple[str, ...]) -> str:
        if self.rule is None:
            return words[self.span[0]]
        l, r = self.children
        return f"[{self.rule} {l.bracketed(words)} {r.bracketed(words)}]"


@dataclass
class Derivation:
    key: tuple[str, ...]  # 句中每个 token 位置的 σ（规范打印），去重键
    sigma: dict[str, Category]  # 词型 → 范畴（含局部变量，未规范化）
    tree: Tree  # 代表树（折叠组里 bracketed 最小者）
    n_trees: int = 1
    _lex: Lexicon = field(default=None, repr=False)
    _floor: int = field(default=0, repr=False)

    def apply(self, st: State) -> bool:
        """把这条推导的词型指派并入全局状态（失败时 st 原子回滚）。局部变量先换成新变量。"""
        m = st.mark()
        renames: dict[Var, Var] = {}
        for word, cat in self.sigma.items():
            if not st.unify(self._lex.var_of(word), _freshen(cat, self._floor, renames)):
                st.rollback(m)
                return False
        return True


def _freshen(c: Category, floor: int, renames: dict[Var, Var]) -> Category:
    if isinstance(c, Var):
        if c.id < floor:
            return c
        v = renames.get(c)
        if v is None:
            v = renames[c] = fresh_var()
        return v
    if isinstance(c, Functor):
        return _functor(_freshen(c.result, floor, renames), c.slash, _freshen(c.arg, floor, renames))
    return c


def _canon(c: Category, local: set[Var], floor: int, names: dict[Var, str]) -> str:
    if isinstance(c, Var):
        if c in local or c.id >= floor:
            n = names.get(c)
            if n is None:
                n = names[c] = f"?_{len(names) + 1}"
            return n
        return str(c)
    if isinstance(c, Functor):
        r = _canon(c.result, local, floor, names)
        a = _canon(c.arg, local, floor, names)
        if isinstance(c.result, Functor):
            r = f"({r})"
        if isinstance(c.arg, Functor):
            a = f"({a})"
        return f"{r}{c.slash}{a}"
    return str(c)


def solve_sentence(
    sent: Sentence,
    lex: Lexicon,
    st: State,
    *,
    rules: dict[str, Rule] | None = None,
    complexity_bound: int | None = None,
) -> list[Derivation]:
    rules = APPLICATION if rules is None else rules
    word_vars = [lex.var_of(w) for w in sent.words]
    local = set(word_vars)
    floor = fresh_var().id  # 此后新建的变量都是局部的
    goal = atom("S", sent.feat)
    found: dict[tuple[str, ...], Derivation] = {}
    n = len(sent.words)

    def admissible() -> bool:
        # 注意：complexity_bound 是对每个词的硬界（§4.6 迭代加深的 L），不是"偏好更简单"的逐词贪心；
        # 解之间的取舍只能由 §4.5 的全局目标函数做。
        for v in local:
            c = st.resolve(v)
            if not is_admissible(c):
                return False
            if complexity_bound is not None and complexity(c) > complexity_bound:
                return False
        return True

    def span(i: int, j: int) -> Iterator[tuple[Tree, Category]]:
        if j == i + 1:
            yield Tree(None, (i, j)), word_vars[i]
            return
        for k in range(i + 1, j):
            for lt, lc in span(i, k):
                for rt, rc in span(k, j):
                    for name, rule in rules.items():
                        m = st.mark()
                        r = rule(lc, rc, st)
                        if r is not None and admissible():
                            yield Tree(name, (i, j), (lt, rt)), r
                        st.rollback(m)

    m0 = st.mark()
    for tree, root in span(0, n):
        m = st.mark()
        if st.unify(root, goal) and admissible():
            names: dict[Var, str] = {}
            key = tuple(_canon(st.resolve(v), local, floor, names) for v in word_vars)
            d = found.get(key)
            if d is None:
                sigma = {w: st.resolve(lex.var_of(w)) for w in dict.fromkeys(sent.words)}
                found[key] = Derivation(key, sigma, tree, 1, lex, floor)
            else:
                d.n_trees += 1
                if tree.bracketed(sent.words) < d.tree.bracketed(sent.words):
                    d.tree = tree
        st.rollback(m)
    st.rollback(m0)
    return [found[k] for k in sorted(found)]


def enumerate_trees(n: int, n_rules: int = len(APPLICATION)) -> int:
    """长度 n 的句子，规则集下的原始推导树数上界（Catalan(n-1) × n_rules^(n-1)）。"""
    c = comb(2 * (n - 1), n - 1) // n
    return c * n_rules ** (n - 1)
