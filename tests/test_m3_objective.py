"""M3 测试（一）：目标函数（§4.5，提前自 M4）。

设计决定（请确认）：
1. 权重默认 α₁=0（刚性下条目数是常数）、α₂=β=γ=1。测试尽量权重无关：只断言分量和序关系。
2. 不同范畴数按规范形计（`?_1` 与 `?_2` 算同一个范畴，`S[dcl]\\?_1` 与 `S[dcl]\\?_2` 也是）。
3. 自由变量复杂度 0；γ 项在 |D_s| = 0 时为 +inf。
4. 下界：自由变量算 0、γ 项取 0；对任何完整指派的真实代价都 ≤，且在完整指派上等于真实代价减去 γ 项。
"""
import math

import pytest

from ccg_solver.category import complexity, parse
from ccg_solver.corpus import Lexicon, Sentence
from ccg_solver.objective import Cost, Weights, corpus_cost, lexicon_cost, lower_bound
from ccg_solver.state import State
from ccg_solver.toy import M2_CORPUS, M2_GOLD

SENTS = [Sentence.from_text(t) for t in M2_CORPUS]


def bound_gold(skip=()):
    lex, st = Lexicon(), State()
    for w in M2_GOLD:
        lex.var_of(w)
    for w, c in M2_GOLD.items():
        if w not in skip:
            assert lex.bind(st, w, parse(c))
    return lex, st


class TestLexiconCost:
    def test_gold_components(self):
        cats = {w: parse(c) for w, c in M2_GOLD.items()}
        entries, distinct, cx = lexicon_cost(cats)
        assert entries == 20
        assert distinct == len({M2_GOLD[w] for w in M2_GOLD}) == 7  # NP/N N N/N NP S\NP (S\NP)/NP (S\NP)\(S\NP)
        assert cx == sum(complexity(parse(c)) for c in M2_GOLD.values())

    def test_alpha_equivalent_categories_count_once(self):
        env = {}
        cats = {"a": parse("S[dcl]\\?X", env), "b": parse("S[dcl]\\?Y", env), "c": parse("?Z", env), "d": parse("?W", env)}
        assert lexicon_cost(cats)[1] == 2

    def test_free_var_complexity_zero(self):
        assert lexicon_cost({"a": parse("?X")})[2] == 0


class TestCorpusCost:
    def test_gold_cost(self):
        lex, st = bound_gold()
        c = corpus_cost(SENTS, lex, st)
        assert (c.entries, c.distinct) == (20, 7)
        assert c.log_derivations == 0.0  # 金标准下每句恰好 1 条 σ
        assert c.total(Weights()) == 7 + c.complexity

    def test_unsolvable_is_inf(self):
        lex, st = bound_gold()
        assert lex.bind(st, "quickly", parse("(S[dcl]\\NP)\\(S[dcl]\\NP)"))  # 已绑定，无变化
        lex2, st2 = bound_gold(skip=("sleeps",))
        assert lex2.bind(st2, "sleeps", parse("S[dcl]/NP"))  # 错的方向：含 sleeps 的句子无推导
        assert math.isinf(corpus_cost(SENTS, lex2, st2).log_derivations)

    def test_degenerate_lexicon_costs_more_than_gold(self):
        """§5.5：退化解 arity n-1 极其昂贵；再加 γ 项，任何正权重下都劣于金标准。"""
        gold_lex, gold_st = bound_gold()
        g = corpus_cost(SENTS, gold_lex, gold_st)
        # 退化词典：所有名词/限定词/副词做原子 ?, 动词吃掉整句 —— 对 "the cat sees the dog" 是 (((S\?)/?)/?)\?，
        # 在本语料下并非对每句都可推导，所以直接构造一个可推导但更宽的词典：全部词同一自由变量以外的最简形式
        # 这里用"句末词取 S、其余取 S/S"的 §4.5 极端语法：每句可推导但推导数巨大。
        lex, st = Lexicon(), State()
        words = {w for s in SENTS for w in s.words}
        last = {s.words[-1] for s in SENTS}
        for w in words:
            lex.var_of(w)
        feasible = True
        for w in words:
            cat = parse("S[dcl]") if w in last and w not in {s.words[i] for s in SENTS for i in range(len(s) - 1)} else parse("S[dcl]/S[dcl]")
            feasible &= lex.bind(st, w, cat)
        if not feasible:
            pytest.skip("极端语法在本语料不一致（有词既在句末又在句中）")
        wide = corpus_cost(SENTS, lex, st)
        for w in (Weights(), Weights(alpha2=0.1, beta=0.1, gamma=1), Weights(alpha2=5, beta=1, gamma=0.1)):
            assert g.total(w) < wide.total(w)

    def test_symmetry_is_cost_invariant(self):
        """§5.4：主宾翻转的自同构下 α₂、β 不变；只有 VP 修饰语句的 γ/可推导性才区分。"""
        a = {w: parse(c) for w, c in M2_GOLD.items()}
        b = dict(a)
        for v in ("sees", "chases", "likes"):
            b[v] = parse("(S[dcl]/NP)\\NP")
        assert lexicon_cost(a) == lexicon_cost(b)


class TestLowerBound:
    def test_bound_is_admissible_on_partial(self):
        lex, st = bound_gold(skip=("sees", "quickly", "big"))
        lb = lower_bound(SENTS, lex, st)
        full = corpus_cost(SENTS, *bound_gold())
        assert lb.log_derivations == 0.0
        assert lb.complexity <= full.complexity and lb.distinct <= full.distinct
        assert lb.total(Weights()) <= full.total(Weights())

    def test_bound_equals_cost_minus_gamma_on_complete(self):
        lex, st = bound_gold()
        lb, full = lower_bound(SENTS, lex, st), corpus_cost(SENTS, lex, st)
        assert (lb.entries, lb.distinct, lb.complexity) == (full.entries, full.distinct, full.complexity)

    def test_bound_monotone_in_bindings(self):
        lex, st = bound_gold(skip=("sees", "quickly", "big", "the"))
        lb0 = lower_bound(SENTS, lex, st).total(Weights())
        assert lex.bind(st, "the", parse("NP/N"))
        lb1 = lower_bound(SENTS, lex, st).total(Weights())
        assert lex.bind(st, "sees", parse("(S[dcl]\\NP)/NP"))
        lb2 = lower_bound(SENTS, lex, st).total(Weights())
        assert lb0 <= lb1 <= lb2
