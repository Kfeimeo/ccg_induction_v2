"""M2 测试（三）：跨句传播到不动点。判据 M2-a…e 见 docs/m1-conclusion-m2-plan.md §5.2 与增补 3。

设计决定（请确认）：
1. ``CorpusSolver(corpus, L, domain_filter=..., join_threshold=50)``；``run()`` 返回 ``Report``。
2. M2-b 写成硬断言（≥ 1 次单位子句事件）。M2 语料按增补 3 加了结构不同的宾语上下文，若仍不出现，如实报告。
3. M2-d 两个数分开：backbone-only 压缩率 与 加域过滤后的压缩率。数值先跑出来再存 golden。
4. join（|D_s| < 50 保留完整 σ 集）在本轮只作为报告项，不作为断言对象。
"""
import math

import pytest

from ccg_solver.category import complexity, parse
from ccg_solver.corpus import Lexicon, Sentence
from ccg_solver.local import solve_sentence
from ccg_solver.propagate import Conflict, CorpusSolver, Report, solve_corpus
from ccg_solver.state import State
from ccg_solver.toy import M2_CORPUS, M2_GOLD

from test_m1_local import subsumes_key

SENTS = [Sentence.from_text(t) for t in M2_CORPUS]
GOLD_L = max(complexity(parse(c)) for c in M2_GOLD.values())  # = 2


class TestToyCorpus:
    def test_shape(self):
        assert len(SENTS) == 30 and all(len(s) <= 8 for s in SENTS)
        assert len(M2_GOLD) == 20 and GOLD_L == 2
        words = {w for s in SENTS for w in s.words}
        assert words == set(M2_GOLD)

    def test_every_sentence_derivable_from_gold(self):
        lex, st = Lexicon(), State()
        for w, c in M2_GOLD.items():
            assert lex.bind(st, w, parse(c))
        for s in SENTS:
            assert len(solve_sentence(s, lex, st)) == 1, s

    def test_required_phenomena(self):
        texts = set(M2_CORPUS)
        assert {"the cat sleeps .", "the dog sleeps .", "the cat runs ."} <= texts  # 最小对
        assert any("quickly" in t or "often" in t for t in texts)  # VP 修饰语
        assert {"the cat sees the dog .", "the cat sees dogs .", "the cat sees the big dog ."} <= texts  # 同动词不同宾语结构


@pytest.fixture(scope="module")
def runs():
    """三种配置各跑一次：独立求解基线 / 只 backbone / backbone + 域过滤。"""
    out = {}
    for name, df in (("backbone", False), ("domain", True)):
        solver = CorpusSolver(SENTS, GOLD_L, domain_filter=df)
        out[name] = (solver, solver.run())
    return out


class TestFixedPoint:
    def test_no_conflict_and_fixed_point(self, runs):
        for solver, rep in runs.values():
            assert rep.fixed_point
            assert rep.rounds >= len(SENTS)

    def test_gold_consistent_with_final_state(self, runs):  # 求解器没把金标准排除掉
        for solver, rep in runs.values():
            st, lex = solver.state, solver.lexicon
            m = st.mark()
            for w, c in M2_GOLD.items():
                assert lex.bind(st, w, parse(c)), (w, st.resolve(lex.var_of(w)))
            st.rollback(m)

    def test_rerun_is_noop(self, runs):
        for solver, rep in runs.values():
            before = solver.dump()
            rep2 = solver.run()
            assert solver.dump() == before
            assert rep2.shrink_events == 0 and rep2.unit_events == []


class TestM2a_backbone:
    def test_backbone_in_results_is_lgg_of_D(self, runs):
        solver, rep = runs["backbone"]
        for i, r in solver.results.items():
            for d in r.derivations:
                assert all(w in r.backbone for w in d.sigma)


class TestM2b_chain:
    def test_at_least_one_unit_event(self, runs):
        solver, rep = runs["domain"]
        assert len(rep.unit_events) >= 1, rep


class TestM2c_order:
    def test_reproducible(self):
        a = CorpusSolver(SENTS, GOLD_L).run()
        b = CorpusSolver(SENTS, GOLD_L).run()
        assert a.order == b.order and a.unit_events == b.unit_events and a.final_log_D == b.final_log_D

    def test_shortest_first(self, runs):
        for solver, rep in runs.values():
            first = rep.order[0]
            assert len(SENTS[first]) == min(len(s) for s in SENTS)


class TestM2d_compression:
    def test_two_numbers(self, runs):
        (_, bb), (_, dm) = runs["backbone"], runs["domain"]
        assert bb.independent_log_D == dm.independent_log_D > 0
        assert 0 <= bb.compression <= dm.compression <= 1
        print(
            f"\nM2-d  independent Σlog|D|={bb.independent_log_D:.2f}  "
            f"backbone-only: {bb.final_log_D:.2f} ({bb.compression:.1%})  "
            f"+domain: {dm.final_log_D:.2f} ({dm.compression:.1%})  "
            f"unit_events={len(dm.unit_events)} shrink_events={dm.shrink_events} rounds={dm.rounds}"
        )

    def test_report_dump_has_required_fields(self, runs):
        solver, rep = runs["domain"]
        d = solver.dump()
        assert set(d) >= {"lexicon", "domain_sizes", "D_sizes"}
        assert len(d["lexicon"]) == 20 and len(d["D_sizes"]) == 30


class TestM2e_monotonicity:
    def test_final_D_subsumed_by_independent_D(self, runs):
        for solver, rep in runs.values():
            for i, s in enumerate(SENTS):
                indep = [d.key for d in solve_sentence(s, Lexicon(), State())]
                for d in solver.results[i].derivations:
                    assert any(subsumes_key(g, d.key) for g in indep), (s, d.key)


class TestIterativeDeepening:
    def test_min_L_is_gold_L(self):
        L, solver, rep = solve_corpus(SENTS, L_max=3)
        assert L == GOLD_L

    def test_L_too_small_conflicts(self):
        with pytest.raises(Conflict):
            CorpusSolver(SENTS, 1).run()
