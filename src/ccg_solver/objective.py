"""M3：目标函数（§4.5，从 M4 提前）。接口草案。

cost = α₁·|词典条目数| + α₂·|不同范畴数| + β·Σ 范畴复杂度 + γ·Σ_句子 log(去重后 σ 数)

- 词典是 ``{词型: 范畴}``，范畴可含自由变量（原子未定），**按规范形计数**（alpha 等价的范畴算同一个）。
- **β 项是全局项，不是逐词规则。** 若实现成逐词贪心偏好，每个词都会想当原子范畴，结果没有任何句子可推导。
  代价只在完整指派上算，搜索里只用作下界（§4.6 / search.py）。
- γ 项需要用最终词典重新求解每句并按 σ 去重计数；部分指派下取 0 作为下界。
"""
from __future__ import annotations

from dataclasses import dataclass

from .category import Category
from .corpus import Lexicon, Sentence
from .state import State


@dataclass(frozen=True)
class Weights:
    alpha1: float = 0.0  # 刚性下条目数 = 词型数，常数；拆词（M4）后才有意义
    alpha2: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0


@dataclass(frozen=True)
class Cost:
    entries: int
    distinct: int
    complexity: int
    log_derivations: float

    def total(self, w: Weights) -> float:
        return w.alpha1 * self.entries + w.alpha2 * self.distinct + w.beta * self.complexity + w.gamma * self.log_derivations


def lexicon_cost(lexicon: dict[str, Category]) -> tuple[int, int, int]:
    """(条目数, 不同范畴数（按规范形）, Σ 复杂度)。不含 γ 项。"""
    raise NotImplementedError


def corpus_cost(corpus: list[Sentence], lex: Lexicon, st: State, **solve_kw) -> Cost:
    """完整代价：词典三项 + Σ log|D_s|（用当前状态重新求解每句）。某句 |D_s| = 0 时 log 项为 +inf。"""
    raise NotImplementedError


def lower_bound(corpus: list[Sentence], lex: Lexicon, st: State) -> Cost:
    """部分指派下的可容许下界：已 resolve 的词按现状计复杂度与不同范畴数，自由变量算 0，γ 项取 0。"""
    raise NotImplementedError
