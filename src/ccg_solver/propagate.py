"""M2：跨句传播 —— backbone + 域过滤 + 优先队列消元，迭代到不动点。

``CorpusSolver(corpus, L, domain_filter=True, join_threshold=50)``
- 每个词型一个变量（``Lexicon``）；``State(complexity_bound=L)``，L 是所有范畴变量的全局上界；
  域初值 None（= L 下全体候选，隐式）。
- 优先队列（增补 4 §5）：非退化句（n-1 > L）优先，其中按 n 降序；退化句排在其后。同组内再按
  ``(未知变量数, 上次 |D|, 句子序号)`` 升序。
- ``degenerate(i)``：n-1 ≤ L（§5.5）。退化句上 backbone / 域过滤 / join 恒为零，只做单位子句检查。
- 句对 join（``join=True``）：对非退化且 ``|D| < join_threshold`` 的句子保留完整 σ 集；两句共享词型时做
  σ 元组级 mgu 关系交，互相剪掉没有相容伙伴的 σ；迭代到不动点。``Report.join_log_D`` 是 join 后的 Σ log|D_s|。
- 出队一句：``analyze_sentence`` 得 ``(D, backbone, projections)``。
  - ``|D| = 0``：冲突，抛 ``Conflict``（决策/回溯是 M3）。
  - ``|D| = 1``：单位子句，apply；若上次分析时 |D| > 1，记一次 ``unit_events``。
  - ``|D| > 1``：apply backbone；若 ``domain_filter``，每个词的域 ∩= 投影。
  - 句中任一词型变量的绑定或域发生变化 → 含该变量的其他句子重新入队（不动点迭代）。
- ``run()`` 结束后再扫一遍全部句子：若无任何变化则 ``fixed_point=True``。
- join（|D_s| < join_threshold 时保留完整 σ 集）：本轮只记录 ``results[i].derivations``，不参与传播。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .category import Category
from .corpus import Lexicon, Sentence
from .local import SentenceResult, analyze_sentence, solve_sentence
from .state import State


class Conflict(Exception):
    pass


def join_pair(Da, Db, shared: set[str], st: State):
    """句对 join：返回 (keep_a, keep_b)，各自保留至少有一个相容伙伴的 σ。不改变 st。"""
    raise NotImplementedError


@dataclass
class Report:
    order: list[int]
    unit_events: list[int]
    shrink_events: int
    rounds: int
    independent_log_D: float
    final_log_D: float
    fixed_point: bool
    join_log_D: float = float("nan")  # join 后的 Σ log|D_s|（join=False 时等于 final_log_D）

    @property
    def compression(self) -> float:
        if self.independent_log_D == 0:
            return 0.0
        return 1.0 - self.final_log_D / self.independent_log_D


class CorpusSolver:
    def __init__(
        self,
        corpus: list[Sentence],
        L: int,
        *,
        domain_filter: bool = True,
        join: bool = False,
        join_threshold: int = 200,
    ):
        self.corpus = list(corpus)
        self.L = L
        self.domain_filter = domain_filter
        self.join = join
        self.join_threshold = join_threshold
        self.lexicon = Lexicon()
        self.state = State(complexity_bound=L)
        for s in self.corpus:
            for w in s.words:
                self.lexicon.var_of(w)
        self._by_word: dict[str, set[int]] = {}
        for i, s in enumerate(self.corpus):
            for w in s.words:
                self._by_word.setdefault(w, set()).add(i)
        self.results: dict[int, SentenceResult] = {}
        self._last_D: dict[int, int] = {}
        self.independent: list[int] = []
        for s in self.corpus:
            n = len(solve_sentence(s, Lexicon(), State(complexity_bound=L)))
            if n == 0:
                raise Conflict(f"no derivation at L={L}: {s}")
            self.independent.append(n)

    # ------------------------------------------------------------ helpers
    def _snapshot(self, words) -> dict:
        st, lex = self.state, self.lexicon
        return {w: (st.resolve(lex.var_of(w)), (d.keys() if (d := st.domain(lex.var_of(w))) is not None else None)) for w in words}

    def _unknowns(self, s: Sentence) -> int:
        from .category import free_vars
        return sum(1 for w in dict.fromkeys(s.words) if free_vars(self.state.resolve(self.lexicon.var_of(w))))

    def degenerate(self, i: int) -> bool:
        return len(self.corpus[i]) - 1 <= self.L

    def _key(self, i: int):
        s = self.corpus[i]
        return (self.degenerate(i), -len(s), self._unknowns(s), self._last_D.get(i, self.independent[i]), i)

    def _process(self, i: int, rep_unit: list[int]) -> tuple[set[str], int]:
        """分析并传播第 i 句。返回 (状态发生变化的词, 域收缩次数)。"""
        s, st, lex = self.corpus[i], self.state, self.lexicon
        words = list(dict.fromkeys(s.words))
        before = self._snapshot(words)
        r = analyze_sentence(s, lex, st)
        n = len(r.derivations)
        prev = self._last_D.get(i, self.independent[i])
        self.results[i] = r
        self._last_D[i] = n
        if n == 0:
            raise Conflict(f"sentence {i} has no surviving derivation: {s}")
        if n == 1:
            if prev > 1:
                rep_unit.append(i)
            if not r.derivations[0].apply(st):
                raise Conflict(f"unit clause could not be applied: {s}")
        else:
            if not r.apply_backbone(st):
                raise Conflict(f"backbone could not be applied: {s}")
        shrinks = 0
        if self.domain_filter:
            for w, proj in r.projections.items():
                res = st.restrict(lex.var_of(w), proj)
                if res is None:
                    raise Conflict(f"domain of {w!r} emptied by {s}")
                shrinks += res
        after = self._snapshot(words)
        changed = {w for w in words if before[w] != after[w]}
        return changed, shrinks

    # ------------------------------------------------------------ main loop
    def run(self) -> Report:
        pending = set(range(len(self.corpus)))
        order: list[int] = []
        unit_events: list[int] = []
        shrink_events = 0
        while pending:
            i = min(pending, key=self._key)
            pending.remove(i)
            order.append(i)
            changed, shrinks = self._process(i, unit_events)
            shrink_events += shrinks
            for w in changed:
                pending |= self._by_word[w] - {i}
        # 不动点校验：再扫一遍，任何变化都算未收敛
        fixed = True
        for i in range(len(self.corpus)):
            changed, shrinks = self._process(i, [])
            if changed or shrinks:
                fixed = False
        final = sum(math.log(len(self.results[i].derivations)) for i in range(len(self.corpus)))
        indep = sum(math.log(n) for n in self.independent)
        return Report(order, unit_events, shrink_events, len(order), indep, final, fixed)

    def dump(self) -> dict:
        st, lex = self.state, self.lexicon
        dom_sizes = {}
        for w in lex.words():
            d = st.domain(lex.var_of(w))
            dom_sizes[w] = None if d is None else len(d)
        from .domain import canonical
        return {
            "lexicon": {w: canonical(st.resolve(lex.var_of(w))) for w in lex.words()},
            "domain_sizes": dom_sizes,
            "D_sizes": {i: len(r.derivations) for i, r in sorted(self.results.items())},
        }


def solve_corpus(corpus: list[Sentence], *, L_max: int = 4, **kw) -> tuple[int, CorpusSolver, Report]:
    """§4.6 迭代加深：L = 0,1,2,… 直到无冲突。返回最小可行 L。"""
    last = None
    for L in range(L_max + 1):
        try:
            solver = CorpusSolver(corpus, L, **kw)
            return L, solver, solver.run()
        except Conflict as e:
            last = e
    raise Conflict(f"no solution up to L={L_max}: {last}")
