"""M2：跨句传播 —— backbone + 域过滤 + 优先队列消元，迭代到不动点（接口草案）。

``CorpusSolver(corpus, L, domain_filter=True, join_threshold=50)``
- 每个词型一个变量（``Lexicon``），每个变量的域初始化为 ``enumerate_categories(L)``。
- 优先队列键 ``(未知变量数, 存活 σ 数, 句长, 句子序号)``，升序；``未知变量数`` = 句中 resolve 后仍含变量的词型数。
- 出队一句：``analyze_sentence`` 得 ``(D, backbone, projections)``。
  - ``|D| = 0``：冲突，``run()`` 抛 ``Conflict``（M2 语料保证不会发生；决策/回溯是 M3）。
  - ``|D| = 1``：单位子句，apply；若这是因为之前的传播把它压到 1 的，记一次 ``unit_events``。
  - ``|D| > 1``：apply backbone；若 ``domain_filter``，域与 projections 取交集。
  - 任一共享变量的**域收缩**或合并 → 含该变量的其他句子重新入队（不动点迭代）。
- ``run()`` 返回 ``Report``；``results[i]`` 保留每句最后一次分析结果供 dump。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .category import Category
from .corpus import Lexicon, Sentence
from .local import SentenceResult
from .state import State


class Conflict(Exception):
    pass


@dataclass
class Report:
    order: list[int]  # 出队顺序（句子序号，可重复）
    unit_events: list[int]  # 被传播压成单位子句的句子序号
    shrink_events: int  # 域收缩次数
    rounds: int  # 出队总次数
    independent_log_D: float  # Σ log|D_s|，每句在空状态下独立求解
    final_log_D: float  # Σ log|D_s|，不动点后
    fixed_point: bool  # 结束时队列为空且再扫一遍无变化

    @property
    def compression(self) -> float:
        """1 - final/independent；independent 为 0 时定义为 0。"""
        raise NotImplementedError


class CorpusSolver:
    def __init__(self, corpus: list[Sentence], L: int, *, domain_filter: bool = True, join_threshold: int = 50):
        raise NotImplementedError

    lexicon: Lexicon
    state: State
    results: dict[int, SentenceResult]

    def run(self) -> Report:
        raise NotImplementedError

    def dump(self) -> dict:
        """notebook 用：每个词的 resolve 结果、域大小、每句 |D|。"""
        raise NotImplementedError


def solve_corpus(corpus: list[Sentence], *, L_max: int = 4, **kw) -> tuple[int, CorpusSolver, Report]:
    """§4.6 迭代加深：L = 0,1,2,… 直到无冲突。返回最小可行 L。"""
    raise NotImplementedError
