"""M3/§8：评测 —— 原子范畴重命名匹配 + 词型级准确率。

解里的自由变量（``?_k``，词典级共享命名）要映射到金标准的原子名。做法：
1. 把解与金标准的范畴并行遍历，统计 (变量, 原子) 的结构共现次数作为权重；
2. 在变量与原子之间做**单射**的最大权匹配（变量 ≤ 7 时穷举所有单射，以"完全一致的词型数"为主序、
   共现权重为次序；否则按共现权重贪心）；
3. 按匹配重命名，逐词比对。原子名本身（S[dcl] 等）不重命名，因为求解器的原子集合就是金标准的原子集合。
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations, permutations

from .category import Atom, Category, Functor, Var, parse

_VAR = re.compile(r"\?_(\d+)")
_ATOM = re.compile(r"[A-Za-z]+(?:\[[a-z]+\])?")


def _parse_sol(text: str, env: dict) -> Category:
    return parse(_VAR.sub(r"?V\1", text), env)


def _cooccur(a: Category, b: Category, acc: Counter, names: dict[Var, str]) -> None:
    if isinstance(a, Var):
        if isinstance(b, Atom):
            acc[(names[a], str(b))] += 1
        return
    if isinstance(a, Functor) and isinstance(b, Functor) and a.slash == b.slash:
        _cooccur(a.result, b.result, acc, names)
        _cooccur(a.arg, b.arg, acc, names)


def _rename(text: str, mapping: dict[str, str]) -> str:
    return _VAR.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


def match_atoms(solution: dict[str, str], gold: dict[str, str]) -> dict[str, str]:
    """二分图最大匹配：解出的自由变量名 → 金标准原子名（单射）。只返回被匹配的变量。"""
    env: dict = {}
    sol_cats = {w: _parse_sol(c, env) for w, c in solution.items()}
    names = {v: "?_" + k[1:] for k, v in env.items()}
    vars_ = sorted(set(names.values()), key=lambda n: int(n[2:]))
    atoms = sorted({a for w in gold for a in _ATOM.findall(gold[w])})
    if not vars_ or not atoms:
        return {}
    acc: Counter = Counter()
    for w, c in sol_cats.items():
        if w in gold:
            _cooccur(c, parse(gold[w]), acc, names)

    def score(mapping: dict[str, str]) -> tuple[int, int]:
        exact = sum(1 for w in gold if w in solution and _rename(solution[w], mapping) == gold[w])
        return exact, sum(acc[(v, a)] for v, a in mapping.items())

    if len(vars_) <= 7:
        best, best_score = {}, (-1, -1)
        k = min(len(vars_), len(atoms))
        for vs in combinations(vars_, k):
            for perm in permutations(atoms, k):
                m = dict(zip(vs, perm))
                s = score(m)
                if s > best_score:
                    best, best_score = m, s
        return best
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for (v, a), _ in sorted(acc.items(), key=lambda kv: (-kv[1], kv[0])):
        if v not in mapping and a not in used:
            mapping[v] = a
            used.add(a)
    return mapping


def accuracy(solution: dict[str, str], gold: dict[str, str]) -> tuple[float, dict[str, str]]:
    """(词型级准确率, 重命名后的解)。"""
    mapping = match_atoms(solution, gold)
    renamed = {w: _rename(c, mapping) for w, c in solution.items()}
    hits = sum(1 for w in gold if renamed.get(w) == gold[w])
    return hits / max(1, len(gold)), renamed
