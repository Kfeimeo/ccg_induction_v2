"""M2 测试（四）：句对 join（增补 4 §3 有条件采纳）。

设计决定（请确认）：
1. ``join_pair(Da, Db, shared, st) -> (keep_a, keep_b)``：Da/Db 是两句的 Derivation 列表，shared 是共享词型；
   σa 与 σb 相容 ⇔ 把两条 σ 各自 freshen 后，在共享词上逐个合一全部成功（元组级：σ 内部共享的局部变量保持共享）。
   保留有至少一个相容伙伴的 σ。结果确定、对称。
2. ``CorpusSolver(..., join=True, join_threshold=200)``：只对非退化（n-1 > L）且 |D| < join_threshold 的句子做；
   增补 3 的阈值 50 在 L=3 下一句都覆盖不到（5 词句 |D|=127），所以阈值是参数，实测用 200。
3. join 剪掉的 σ 从 ``results[i].derivations`` 里移除；迭代到不动点；``Report.join_log_D`` 记录 join 后的 Σ log|D|，
   M2-d 变成三个数：backbone-only / +域过滤 / +join。
4. 退化句上 join 由推论 2 保证为零：测试断言退化句的 D 不变。
"""
import math

import pytest

from ccg_solver.category import parse
from ccg_solver.corpus import Lexicon, Sentence
from ccg_solver.local import analyze_sentence, solve_sentence
from ccg_solver.propagate import CorpusSolver, join_pair
from ccg_solver.state import State
from ccg_solver.toy import M2_CORPUS, M2_GOLD

from test_m1_local import gold_state
from test_m2_propagate import GOLD_L, SENTS


def D(text, lex, st, **kw):
    return solve_sentence(Sentence.from_text(text), lex, st, **kw)


class TestJoinPair:
    def test_degenerate_pair_keeps_everything(self):
        lex, st = Lexicon(), State(complexity_bound=GOLD_L)
        Da, Db = D("john sleeps .", lex, st), D("mary sleeps .", lex, st)
        ka, kb = join_pair(Da, Db, {"sleeps"}, st)
        assert [d.key for d in ka] == [d.key for d in Da]
        assert [d.key for d in kb] == [d.key for d in Db]

    def test_no_shared_words_is_identity(self):
        lex, st = Lexicon(), State(complexity_bound=GOLD_L)
        Da, Db = D("john sleeps .", lex, st), D("mary runs .", lex, st)
        ka, kb = join_pair(Da, Db, set(), st)
        assert len(ka) == len(Da) and len(kb) == len(Db)

    def test_incompatible_sigma_is_pruned(self):
        """b 句把 sees 固定成 (S\\NP)/NP 之后，a 句里 sees 的其他 13 个候选没有伙伴。"""
        lex, st = gold_state(skip=("sees",))
        Da = D("the cat sees the dog .", lex, st)
        assert len(Da) == 14
        lex2, st2 = gold_state()
        Db = D("the dog sees the cat .", lex2, st2)  # 固定词典：1 条 σ
        # 用 a 的 Lexicon 的词做共享；Db 的 sigma 是 ground，直接可比
        ka, kb = join_pair(Da, Db, {"the", "cat", "sees", "dog"}, st)
        assert [d.key[2] for d in ka] == ["(S[dcl]\\NP)/NP"]
        assert len(kb) == 1

    def test_symmetric_and_deterministic(self):
        lex, st = Lexicon(), State(complexity_bound=GOLD_L)
        Da, Db = D("the cat sees the dog .", lex, st), D("the dog chases the cat .", lex, st)
        shared = {"the", "cat", "dog"}
        ka, kb = join_pair(Da, Db, shared, st)
        kb2, ka2 = join_pair(Db, Da, shared, st)
        assert [d.key for d in ka] == [d.key for d in ka2]
        assert [d.key for d in kb] == [d.key for d in kb2]
        assert len(ka) <= len(Da) and len(kb) <= len(Db)

    def test_state_untouched(self):
        lex, st = Lexicon(), State(complexity_bound=GOLD_L)
        Da, Db = D("the cat sees the dog .", lex, st), D("the cat sleeps .", lex, st)
        m = st.mark()
        join_pair(Da, Db, {"the", "cat"}, st)
        assert st.mark() == m

    def test_gold_survives(self):
        lex, st = Lexicon(), State(complexity_bound=GOLD_L)
        a, b = "the cat sees the dog quickly .", "the big dog chases the small cat ."
        Da, Db = D(a, lex, st), D(b, lex, st)
        ka, kb = join_pair(Da, Db, {"the", "cat", "dog"}, st)
        from test_m1_local import subsumes_key
        for text, kept in ((a, ka), (b, kb)):
            lex_g, st_g = Lexicon(), State()
            for w, c in M2_GOLD.items():
                lex_g.bind(st_g, w, parse(c))
            gold_key = D(text, lex_g, st_g)[0].key
            assert any(subsumes_key(d.key, gold_key) for d in kept)


@pytest.fixture(scope="module")
def joined():
    solver = CorpusSolver(SENTS, GOLD_L, domain_filter=True, join=True, join_threshold=200)
    return solver, solver.run()


class TestJoinInCorpus:
    def test_fixed_point_and_gold(self, joined):
        solver, rep = joined
        assert rep.fixed_point
        st, lex = solver.state, solver.lexicon
        m = st.mark()
        for w, c in M2_GOLD.items():
            assert lex.bind(st, w, parse(c)), w
        st.rollback(m)

    def test_degenerate_sentences_untouched(self, joined):
        solver, rep = joined
        for i, s in enumerate(SENTS):
            if solver.degenerate(i):
                assert len(solver.results[i].derivations) == solver.independent[i], s

    def test_only_threshold_sentences_joined(self, joined):
        solver, rep = joined
        for i, s in enumerate(SENTS):
            if not solver.degenerate(i) and solver.independent[i] >= 200:
                assert len(solver.results[i].derivations) == solver.independent[i], s

    def test_three_numbers(self, joined):
        solver, rep = joined
        assert rep.join_log_D <= rep.final_log_D <= rep.independent_log_D
        # 数据（2026-09-19，L=3，阈值 200，只有 5 个 5 词句参与）：join 是非零的，但只作用于少数句子
        assert rep.final_log_D == rep.independent_log_D  # backbone / 域过滤仍是 0
        assert round(1 - rep.join_log_D / rep.independent_log_D, 3) == 0.015
        pruned = {i: len(solver.results[i].derivations) for i in (5, 6, 11, 15, 22)}
        assert pruned == {5: 61, 6: 61, 11: 112, 15: 125, 22: 115}
        assert rep.join_log_D < rep.final_log_D
        joined_sents = [i for i in range(len(SENTS)) if not solver.degenerate(i) and solver.independent[i] < 200]
        print(
            f"\nM2-d  independent={rep.independent_log_D:.2f}  backbone/domain={rep.final_log_D:.2f}  "
            f"+join={rep.join_log_D:.2f} ({1 - rep.join_log_D / rep.independent_log_D:.1%})  "
            f"joined sentences={len(joined_sents)}  "
            + " ".join(f"{i}:{solver.independent[i]}->{len(solver.results[i].derivations)}" for i in joined_sents)
        )

    def test_monotone(self, joined):
        solver, rep = joined
        for i, s in enumerate(SENTS):
            indep = {d.key for d in solve_sentence(s, Lexicon(), State(complexity_bound=GOLD_L))}
            assert {d.key for d in solver.results[i].derivations} <= indep


@pytest.mark.slow
def test_golden_join_threshold_800():
    """慢测试（约 1 分钟）：阈值 800 让 3 个 6 词句（|D|=720）参与；7 词句（3990）仍不参与。
    数据（2026-10-04，L=3）：join 把两个 6 词句从 720 剪到 346 / 253，剪它们的全是两个 5 词最小对，
    方向单向（6 词句没剪掉任何 5 词句的 σ）。整体 Σ log|D| 下降 3.1%，只算参与句子 6.5%。"""
    solver = CorpusSolver(SENTS, GOLD_L, domain_filter=True, join=True, join_threshold=800)
    rep = solver.run()
    assert rep.fixed_point
    assert rep.final_log_D == rep.independent_log_D
    assert round(1 - rep.join_log_D / rep.independent_log_D, 3) == 0.031
    joined = {i: len(solver.results[i].derivations) for i in range(len(SENTS))
              if not solver.degenerate(i) and solver.independent[i] < 800}
    assert joined == {5: 61, 6: 61, 8: 346, 11: 112, 14: 208, 15: 125, 16: 253, 22: 115, 23: 207, 27: 208}
    # 剪枝事件：(i, j, 剪掉的 σ 数 i, 剪掉的 σ 数 j)。6 词句（8, 16）只被剪，不剪别人
    assert solver.join_events == [
        (5, 6, 66, 66), (5, 8, 0, 346), (5, 11, 0, 15), (5, 15, 0, 2), (5, 16, 0, 435),
        (5, 22, 0, 10), (6, 8, 0, 28), (6, 16, 0, 32), (6, 22, 0, 2), (16, 23, 0, 1),
    ]
    # 只算参与句子：53.39 -> 49.90，6.5%
    sub_before = sum(math.log(solver.independent[i]) for i in joined)
    sub_after = sum(math.log(n) for n in joined.values())
    assert round(1 - sub_after / sub_before, 3) == 0.065
