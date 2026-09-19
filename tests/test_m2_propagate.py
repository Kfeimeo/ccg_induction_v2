"""M2 测试（三）：跨句传播到不动点。判据 M2-a…e 见 docs/m1-conclusion-m2-plan.md §5.2 与增补 3。

实测结论（2026-09-18，L=3，只开 > <）—— 记录为精确断言，不标 xfail：
- backbone 全部平凡：每个词的 lgg 都是裸变量。
- 域过滤（逐变量投影）全部平凡：每个词的极小化域都是 {?_1}，因为任何词在某条推导里都可以是纯论元。
- 因此 M2-b 单位子句事件 = 0，M2-d 两个压缩率都是 0.0%。信息全在变量间的相关性里，逐变量投影一个都传不出去。
- 迭代加深的最小可行 L = 2 < 金标准的 3（副词），且 L=1 时传播确实制造了冲突（域在 L=1 下够紧）。
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
GOLD_L = max(complexity(parse(c)) for c in M2_GOLD.values())  # = 3：副词 arity 2 + depth 1


class TestToyCorpus:
    def test_shape(self):
        assert len(SENTS) == 30 and all(len(s) <= 8 for s in SENTS)
        assert len(M2_GOLD) == 20 and GOLD_L == 3
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
    def test_no_unit_event_on_this_corpus(self, runs):
        """数据：逐变量传播在这部语料上一次单位子句都没制造出来。"""
        for solver, rep in runs.values():
            assert rep.unit_events == []

    def test_backbone_and_domains_are_trivial(self, runs):
        from ccg_solver.category import Var
        solver, rep = runs["domain"]
        st, lex = solver.state, solver.lexicon
        for w in lex.words():
            assert isinstance(st.resolve(lex.var_of(w)), Var)  # backbone 没绑定任何东西
            assert st.domain(lex.var_of(w)).keys() == ("?_1",)  # 极小化后的域 = 全体


class TestM2c_order:
    def test_reproducible(self):
        a = CorpusSolver(SENTS, GOLD_L).run()
        b = CorpusSolver(SENTS, GOLD_L).run()
        assert a.order == b.order and a.unit_events == b.unit_events and a.final_log_D == b.final_log_D

    def test_nondegenerate_longest_first(self, runs):
        """增补 4 §5：非退化句（n-1 > L）优先，按 n 降序；退化句排在其后。"""
        for solver, rep in runs.values():
            first = rep.order[0]
            assert len(SENTS[first]) == max(len(s) for s in SENTS) == 7
            first_pass = rep.order[: len(SENTS)]
            degen = [solver.degenerate(i) for i in first_pass]
            assert degen == sorted(degen)  # 所有非退化句先于所有退化句出队
            nd_lens = [len(SENTS[i]) for i, d in zip(first_pass, degen) if not d]
            assert nd_lens == sorted(nd_lens, reverse=True)

    def test_degenerate_split(self, runs):
        solver, _ = runs["domain"]
        nd = [i for i in range(len(SENTS)) if not solver.degenerate(i)]
        assert len(nd) == 12 and all(len(SENTS[i]) >= GOLD_L + 2 for i in nd)
        assert len(nd) * 3 >= len(SENTS)  # 增补 4 §7：非退化句至少占三分之一


class TestM2d_compression:
    def test_two_numbers(self, runs):
        (_, bb), (_, dm) = runs["backbone"], runs["domain"]
        assert bb.independent_log_D == dm.independent_log_D > 0
        assert bb.fixed_point and dm.fixed_point
        assert bb.compression == dm.compression == 0.0  # 数据：两个数都是 0
        assert dm.shrink_events > 0  # 域确实收缩过（从 None 到显式集合），只是收缩到的还是全体
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
                indep = [d.key for d in solve_sentence(s, Lexicon(), State(complexity_bound=GOLD_L))]
                final = [d.key for d in solver.results[i].derivations]
                # 逐条 subsumption 是 O(|D|²) 次合一（|D| 最大 3990，全量要数小时），只对前三句做；其余比较规范键集
                if i < 3:
                    for k in final:
                        assert any(subsumes_key(g, k) for g in indep), (s, k)
                assert set(final) <= set(indep), s


class TestIterativeDeepening:
    def test_min_L_is_a_lower_bound_below_gold(self):
        """数据：最小可行 L = 2，金标准需要 3（副词 arity 2 + depth 1）。L 是下界，不是金标准复杂度。"""
        L, solver, rep = solve_corpus(SENTS, L_max=3)
        assert L == 2 < GOLD_L
        assert rep.fixed_point

    def test_L1_conflicts_through_propagation(self):
        """L=1 时每句单独都有解（8 词句也有），是跨句传播制造了冲突：域在 L=1 下够紧。"""
        with pytest.raises(Conflict, match="no surviving derivation"):
            CorpusSolver(SENTS, 1).run()

    def test_L0_conflicts_immediately(self):
        with pytest.raises(Conflict, match="no derivation at L=0"):
            CorpusSolver(SENTS, 0)
