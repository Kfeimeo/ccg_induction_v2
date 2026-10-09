"""M3：以目标函数为主的分支定界 + 回溯 + 冲突学习（接口草案）。

设计（增补 4 §4）：
- 外循环 ``solve_iterative(corpus, L_max, weights)``：L = 0,1,2,…，每层跑一次分支定界，按 §4.6 的 stall=2 停止；
  返回 ``(best_solution, min_feasible_L, per_L_costs)``。
- ``BranchAndBound(corpus, L, weights, *, propagate=True, learn=True, node_limit=None)``：
  - 节点 = 全局 State 上的一段 trail；决策 = 给某句选一条 σ 并 apply。
  - 选句顺序：非退化句优先、按 n 降序（§4.2），同组内按当前 |D| 升序（最少分支先）。
  - 分支顺序：该句的 σ 按 apply 后的下界升序。
  - 剪枝：``lower_bound(...).total(w) >= best.total`` 即剪；任一句 |D| = 0 即冲突。
  - 传播（opportunistic）：决策后对非退化句做单位子句与域过滤；退化句只检查 |D| > 0。
  - 冲突学习：冲突时记录 nogood = 导致该句 |D| = 0 的决策集合（只含与该句共享词型的决策，按 (句序号, σ-key)）；
    之后任何包含该 nogood 的决策前缀直接跳过。``stats.nogoods``、``stats.nogood_hits``。
  - 完整指派 = 每句都选了 σ；此时按 ``corpus_cost`` 算真实代价（含 γ 项），更新 best。
  - 完备：``node_limit=None`` 时穷尽；返回的 best 是该 L 下代价最小的解（权重给定）。
- ``Solution``：``lexicon``（词型 → 规范打印的范畴）、``cost``、``L``、``choices``（句序号 → σ-key）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Sentence
from .objective import Cost, Weights


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


class BranchAndBound:
    def __init__(self, corpus: list[Sentence], L: int, weights: Weights = Weights(), *,
                 propagate: bool = True, learn: bool = True, node_limit: int | None = None):
        raise NotImplementedError

    stats: SearchStats

    def run(self) -> Solution | None:
        raise NotImplementedError


def solve_iterative(corpus: list[Sentence], *, L_max: int = 4, weights: Weights = Weights(), stall: int = 2, **kw):
    """§4.6：返回 (best, min_feasible_L, {L: total 或 None})。"""
    raise NotImplementedError
