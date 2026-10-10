"""M3：以目标函数为主的分支定界 + 回溯 + 冲突学习。

设计（增补 4 §4）：
- 外循环 ``solve_iterative(corpus, L_max, weights)``：L = 0,1,2,…，每层跑一次分支定界，按 §4.6 的 stall=2 停止；
  返回 ``(best_solution, min_feasible_L, per_L_costs)``。上一层的最优代价作为下一层的 incumbent 传入
  （解空间随 L 嵌套，所以在 L ≥ min_L 的层上"找不到更优"意味着该层最优 = 上一层最优，而非不可行）。
- ``BranchAndBound(corpus, L, weights, *, propagate=True, learn=True, node_limit=None, incumbent=None)``：
  - 语料先按句子去重（重复句不增加约束），``multiplicity`` 保留计数供报告。
  - 节点 = 全局 State 上的一段 trail；决策 = 给某句选一条 σ 并 apply。
  - 选句顺序：非退化句优先、按 n 降序（§4.2），同组内按当前 |D| 升序（最少分支先），再按句序号。
  - 分支顺序：该句的 σ 按 apply 后的下界升序（再按 σ-key）。
  - 剪枝：``lower_bound(...).total(w) >= best.total`` 即剪；任一句 |D| = 0 即冲突。
  - 传播（opportunistic）：决策后对**非退化**句做单位子句（|D| = 1 则直接 apply，记为蕴含而非决策，循环到不变）；
    退化句只检查 |D| > 0。域过滤不在搜索里做：M2 实测恒为零，且 |D| = 0 检查已覆盖冲突检测。
  - 冲突学习：某句 |D| = 0 时，从该句的词型出发，沿"共享词型"闭包收集相关的决策与蕴含；闭包里的**决策**
    构成 nogood（按 (句序号, σ-key)）。之后任何包含某个 nogood 的决策集合直接跳过。
  - 完整指派 = 每句都选了 σ（决策或蕴含）；此时按 ``corpus_cost`` 算真实代价（含 γ 项），更新 best。
  - |D| 计数按"句中词型的联合规范形"备忘，同一状态不重复求解。
  - 完备：``node_limit=None`` 时穷尽；返回的 best 是该 L 下代价最小的解（权重给定）。
- ``Solution``：``lexicon``（词型 → 规范打印的范畴，**词典级共享命名**，共享变量同名）、``cost``、``L``、
  ``choices``（句序号 → σ-key）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .category import Category, Functor, Var
from .corpus import Lexicon, Sentence
from .local import Derivation, solve_sentence
from .objective import Cost, Weights, corpus_cost, current_lexicon, lower_bound
from .state import State


@dataclass
class Solution:
    lexicon: dict[str, str]
    cost: Cost
    total: float
    L: int
    choices: dict[int, tuple[str, ...]]


@dataclass
class SearchStats:
    nodes: int = 0
    conflicts: int = 0
    pruned: int = 0
    nogoods: int = 0
    nogood_hits: int = 0
    solutions: int = 0  # 找到的完整指派数（含被更优解取代的）
    implied: int = 0  # 单位传播蕴含的指派次数


class _NodeLimit(Exception):
    pass


def joint_canonical(cats: dict[str, Category]) -> dict[str, str]:
    """词典级共享命名的规范打印：同一个变量在所有词里同名。"""
    names: dict[Var, str] = {}

    def p(c: Category) -> str:
        if isinstance(c, Var):
            n = names.get(c)
            if n is None:
                n = names[c] = f"?_{len(names) + 1}"
            return n
        if isinstance(c, Functor):
            r, a = p(c.result), p(c.arg)
            if isinstance(c.result, Functor):
                r = f"({r})"
            if isinstance(c.arg, Functor):
                a = f"({a})"
            return f"{r}{c.slash}{a}"
        return str(c)

    return {w: p(c) for w, c in cats.items()}


class BranchAndBound:
    def __init__(self, corpus: list[Sentence], L: int, weights: Weights = Weights(), *,
                 propagate: bool = True, learn: bool = True, node_limit: int | None = None,
                 incumbent: float | None = None):
        self.sents: list[Sentence] = list(dict.fromkeys(corpus))
        self.multiplicity = {i: corpus.count(s) for i, s in enumerate(self.sents)}
        self.L = L
        self.weights = weights
        self.propagate = propagate
        self.learn = learn
        self.node_limit = node_limit
        self.lexicon = Lexicon()
        self.state = State(complexity_bound=L)
        self.words_of: list[list[str]] = []
        for s in self.sents:
            ws = list(dict.fromkeys(s.words))
            self.words_of.append(ws)
            for w in ws:
                self.lexicon.var_of(w)
        self.by_word: dict[str, set[int]] = {}
        for i, ws in enumerate(self.words_of):
            for w in ws:
                self.by_word.setdefault(w, set()).add(i)
        self.stats = SearchStats()
        self.best: Solution | None = None
        self.best_total: float = float("inf") if incumbent is None else incumbent
        self.nogoods: list[frozenset[tuple[int, tuple[str, ...]]]] = []
        self._count_memo: dict[tuple[int, tuple[str, ...]], int] = {}
        # 搜索状态：decisions = [(句序号, σ-key)]（真正的分支点），implied = {句序号: σ-key}（单位传播）
        self._decisions: list[tuple[int, tuple[str, ...]]] = []
        self._implied: dict[int, tuple[str, ...]] = {}

    # ------------------------------------------------------------ helpers
    def degenerate(self, i: int) -> bool:
        return len(self.sents[i]) - 1 <= self.L

    def _sent_key(self, i: int) -> tuple:
        """|D_i| 的备忘键：句中词型的联合规范形，**加上**与它们共享自由变量的其他词型（全局复杂度界让
        这些词的形状也约束本句的推导：绑定共享变量可能让它们超过 L）。"""
        st, lex = self.state, self.lexicon
        from .category import free_vars
        cats = {w: st.resolve(lex.var_of(w)) for w in self.words_of[i]}
        fv = set().union(*(free_vars(c) for c in cats.values())) if cats else set()
        extra = {}
        if fv:
            for w in lex.words():
                if w not in cats:
                    c = st.resolve(lex.var_of(w))
                    if free_vars(c) & fv:
                        extra[w] = c
        jc = joint_canonical({**cats, **extra})
        return i, tuple(jc[w] for w in self.words_of[i]), tuple(sorted((w, jc[w]) for w in extra))

    def _D(self, i: int) -> list[Derivation]:
        return solve_sentence(self.sents[i], self.lexicon, self.state)

    def _count(self, i: int) -> int:
        key = self._sent_key(i)
        n = self._count_memo.get(key)
        if n is None:
            n = self._count_memo[key] = len(self._D(i))
        return n

    def _assigned(self) -> set[int]:
        return {i for i, _ in self._decisions} | set(self._implied)

    def _lb_total(self) -> float:
        return lower_bound(self.sents, self.lexicon, self.state).total(self.weights)

    def _learn_nogood(self, j: int) -> None:
        """冲突句 j：沿共享词型闭包收集相关的决策，记为 nogood。"""
        if not self.learn:
            return
        words = set(self.words_of[j])
        involved: set[int] = set()
        frontier = True
        assigned = self._assigned()
        while frontier:
            frontier = False
            for i in assigned - involved:
                if words & set(self.words_of[i]):
                    involved.add(i)
                    words |= set(self.words_of[i])
                    frontier = True
        ng = frozenset((i, k) for i, k in self._decisions if i in involved)
        if ng and ng not in self.nogoods:
            self.nogoods.append(ng)
            self.stats.nogoods += 1

    def _hits_nogood(self) -> bool:
        cur = set(self._decisions)
        return any(ng <= cur for ng in self.nogoods)

    def _unit_propagate(self, unassigned: set[int]) -> bool:
        """非退化句 |D| = 1 → 直接 apply（蕴含）。返回 False 表示冲突。循环到不变。"""
        changed = True
        while changed:
            changed = False
            for i in sorted(unassigned - set(self._implied)):
                if self.degenerate(i):
                    continue
                n = self._count(i)
                if n == 0:
                    return False
                if n == 1:
                    d = self._D(i)[0]
                    if not d.apply(self.state):
                        return False
                    self._implied[i] = d.key
                    self.stats.implied += 1
                    changed = True
        return True

    # ------------------------------------------------------------ search
    def run(self) -> Solution | None:
        try:
            self._node()
        except _NodeLimit:
            pass
        return self.best

    def _node(self) -> None:
        if self.node_limit is not None and self.stats.nodes >= self.node_limit:
            raise _NodeLimit
        self.stats.nodes += 1
        assigned = self._assigned()
        unassigned = set(range(len(self.sents))) - assigned
        # 冲突检查（所有未指派句）
        for i in sorted(unassigned):
            if self._count(i) == 0:
                self.stats.conflicts += 1
                self._learn_nogood(i)
                return
        if not unassigned:
            self._complete()
            return
        if self._lb_total() >= self.best_total:
            self.stats.pruned += 1
            return
        i = min(unassigned, key=lambda j: (self.degenerate(j), -len(self.sents[j]), self._count(j), j))
        ds = self._D(i)
        # 分支排序：apply 后的下界
        ranked = []
        for d in ds:
            m = self.state.mark()
            if d.apply(self.state):
                ranked.append((self._lb_total(), d.key, d))
            self.state.rollback(m)
        ranked.sort(key=lambda t: (t[0], t[1]))
        for lb, key, d in ranked:
            if lb >= self.best_total:
                self.stats.pruned += 1
                continue
            m = self.state.mark()
            implied_before = dict(self._implied)
            if not d.apply(self.state):
                self.state.rollback(m)
                continue
            self._decisions.append((i, key))
            if self.learn and self._hits_nogood():
                self.stats.nogood_hits += 1
            else:
                ok = True
                if self.propagate:
                    ok = self._unit_propagate(unassigned - {i})
                if ok:
                    self._node()
                else:
                    self.stats.conflicts += 1
                    conflict_j = next((j for j in sorted(unassigned - {i}) if self._count(j) == 0), i)
                    self._learn_nogood(conflict_j)
            self._decisions.pop()
            self._implied = implied_before
            self.state.rollback(m)

    def _complete(self) -> None:
        self.stats.solutions += 1
        cost = corpus_cost(self.sents, self.lexicon, self.state)
        total = cost.total(self.weights)
        if total < self.best_total:
            self.best_total = total
            choices = {i: k for i, k in self._decisions}
            choices.update(self._implied)
            self.best = Solution(
                joint_canonical(current_lexicon(self.sents, self.lexicon, self.state)),
                cost, total, self.L, dict(sorted(choices.items())),
            )

    def dump(self) -> dict:
        """notebook 用：当前最优、统计、每句 |D|（最优解下）。"""
        return {"best": self.best, "stats": self.stats, "nogoods": len(self.nogoods),
                "multiplicity": self.multiplicity}


def solve_iterative(corpus: list[Sentence], *, L_max: int = 4, weights: Weights = Weights(), stall: int = 2, **kw):
    """§4.6：返回 (best, min_feasible_L, {L: total 或 None})。
    L ≥ min_L 的层若在上一层 incumbent 下找不到更优解，记该层代价 = 当前最优（解空间嵌套）。"""
    best: Solution | None = None
    min_L: int | None = None
    per_L: dict[int, float | None] = {}
    stalled = 0
    for L in range(L_max + 1):
        bb = BranchAndBound(corpus, L, weights, incumbent=None if best is None else best.total, **kw)
        sol = bb.run()
        if sol is not None:
            per_L[L] = sol.total
            if min_L is None:
                min_L = L
            best, stalled = sol, 0
        elif min_L is None:
            per_L[L] = None
        else:
            per_L[L] = best.total
            stalled += 1
            if stalled >= stall:
                break
    return best, min_L, per_L
