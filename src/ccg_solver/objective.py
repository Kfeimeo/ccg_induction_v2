"""M3：目标函数（§4.5，从 M4 提前）。

cost = α₁·|词典条目数| + α₂·|不同范畴数| + β·Σ 范畴复杂度 + γ·Σ_句子 log(去重后 σ 数)

- 词典是 ``{词型: 范畴}``，范畴可含自由变量（原子未定），**不同范畴数按逐范畴规范形计数**
  （alpha 等价的范畴算同一个：``S[dcl]\\?X`` 与 ``S[dcl]\\?Y`` 是一个；``?X`` 与 ``?Y`` 也是一个）。
- **β 项是全局项，不是逐词规则。** 若实现成逐词贪心偏好，每个词都会想当原子范畴，结果没有任何句子可推导。
  代价只在完整指派上算；搜索里只用 ``lower_bound`` 作可容许下界（search.py），绝不按它逐词挑范畴。
- γ 项用当前状态重新求解每句并按 σ 去重计数；某句 |D| = 0 时为 +inf。
- 下界：自由变量复杂度 0（复杂度随绑定单调不减，所以 resolve 后的复杂度本身就是下界）；
  不同范畴数只计 **ground** 范畴（含变量的模式最终可能与别的范畴重合，不能计）；γ 项取 0。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .category import Category, complexity, free_vars
from .corpus import Lexicon, Sentence
from .domain import canonical
from .local import solve_sentence
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
    cats = list(lexicon.values())
    return len(cats), len({canonical(c) for c in cats}), sum(complexity(c) for c in cats)


def _corpus_words(corpus: list[Sentence]) -> list[str]:
    return list(dict.fromkeys(w for s in corpus for w in s.words))


def current_lexicon(corpus: list[Sentence], lex: Lexicon, st: State) -> dict[str, Category]:
    return {w: st.resolve(lex.var_of(w)) for w in _corpus_words(corpus)}


def corpus_cost(corpus: list[Sentence], lex: Lexicon, st: State, **solve_kw) -> Cost:
    """完整代价：词典三项 + Σ log|D_s|（用当前状态重新求解每句）。某句 |D_s| = 0 时 log 项为 +inf。"""
    entries, distinct, cx = lexicon_cost(current_lexicon(corpus, lex, st))
    log_d = 0.0
    for s in corpus:
        n = len(solve_sentence(s, lex, st, **solve_kw))
        if n == 0:
            log_d = math.inf
            break
        log_d += math.log(n)
    return Cost(entries, distinct, cx, log_d)


def lower_bound(corpus: list[Sentence], lex: Lexicon, st: State) -> Cost:
    """部分指派下的可容许下界：复杂度按 resolve 现状（自由变量 0），不同范畴数只计 ground 范畴，γ 项取 0。"""
    lexicon = current_lexicon(corpus, lex, st)
    ground = [c for c in lexicon.values() if not free_vars(c)]
    return Cost(len(lexicon), len({canonical(c) for c in ground}), sum(complexity(c) for c in lexicon.values()), 0.0)
