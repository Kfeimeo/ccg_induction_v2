"""M3 测试（二）：分支定界 + 回溯 + 冲突学习 + 迭代加深。

设计决定（请确认）：
1. 节点 = State 上的 trail 段；决策 = 给某句 apply 一条 σ。选句：非退化优先、n 降序、同组 |D| 升序。
2. 分支按 apply 后的下界升序；下界 ≥ 当前最优即剪。
3. 完整指派时用 corpus_cost 算真实代价（含 γ）。穷尽搜索返回该 L 下代价最小的解。
4. 冲突学习：nogood = 让某句 |D|=0 的决策集合中与该句共享词型的那些；包含该 nogood 的前缀直接跳过。
5. 小语料上的"及格线"测试：解出的 the / sees / sleeps 经原子重命名后等于 NP/N、(S\\NP)/NP、S\\NP。
   这条是否能过**无法预判**，它是 M3 的核心实证；过不了就如实记录并报告。
6. CHILDES 测试需要 NLTK 语料，不在时 skip。
"""
import math

import pytest

from ccg_solver.category import parse
from ccg_solver.corpus import Sentence
from ccg_solver.evaluate import accuracy, match_atoms
from ccg_solver.objective import Cost, Weights, corpus_cost
from ccg_solver.search import BranchAndBound, Solution, solve_iterative
from ccg_solver.toy import M2_CORPUS, M2_GOLD

SENTS = [Sentence.from_text(t) for t in M2_CORPUS]
SMALL = [Sentence.from_text(t) for t in [
    "john sleeps .", "mary runs .", "the cat sleeps .", "the dog runs .",
    "the cat sees the dog .", "the dog sees the cat .", "the cat sees the dog quickly .",
]]
SMALL_GOLD = {w: M2_GOLD[w] for s in SMALL for w in s.words}


class TestEvaluate:
    def test_match_atoms_renames_free_vars(self):
        sol = {"the": "?_1/?_2", "cat": "?_2", "sleeps": "S[dcl]\\?_1"}
        gold = {"the": "NP/N", "cat": "N", "sleeps": "S[dcl]\\NP"}
        acc, renamed = accuracy(sol, gold)
        assert acc == 1.0
        assert renamed == gold
        assert match_atoms(sol, gold) == {"?_1": "NP", "?_2": "N"}

    def test_partial_accuracy(self):
        sol = {"the": "NP/N", "cat": "N", "sleeps": "S[dcl]/NP"}
        acc, _ = accuracy(sol, {"the": "NP/N", "cat": "N", "sleeps": "S[dcl]\\NP"})
        assert acc == pytest.approx(2 / 3)

    def test_matching_is_injective(self):
        sol = {"a": "?_1", "b": "?_2"}
        gold = {"a": "N", "b": "N"}
        r = match_atoms(sol, gold)
        assert len(set(r.values())) == len(r)  # 两个变量不能都映到 N


class TestBranchAndBound:
    def test_exhaustive_on_small_finds_feasible_solution(self):
        bb = BranchAndBound(SMALL, L=3)
        sol = bb.run()
        assert sol is not None
        assert math.isfinite(sol.total)
        assert set(sol.choices) == set(range(len(SMALL)))
        assert bb.stats.solutions >= 1 and bb.stats.nodes > 0

    def test_solution_is_consistent(self):
        """返回的词典必须让每句可推导（γ 项有限）。"""
        sol = BranchAndBound(SMALL, L=3).run()
        from ccg_solver.corpus import Lexicon
        from ccg_solver.state import State
        lex, st = Lexicon(), State(complexity_bound=3)
        env = {}
        for w, c in sol.lexicon.items():
            assert lex.bind(st, w, parse(c.replace("?_", "?V"), env)), w
        assert math.isfinite(corpus_cost(SMALL, lex, st).log_derivations)

    def test_optimal_is_no_worse_than_gold(self):
        """穷尽搜索的最优解代价 ≤ 金标准代价（金标准是可行解之一）。"""
        sol = BranchAndBound(SMALL, L=3).run()
        from ccg_solver.corpus import Lexicon
        from ccg_solver.state import State
        lex, st = Lexicon(), State(complexity_bound=3)
        for w, c in SMALL_GOLD.items():
            assert lex.bind(st, w, parse(c))
        gold_total = corpus_cost(SMALL, lex, st).total(Weights())
        assert sol.total <= gold_total + 1e-9

    def test_deterministic(self):
        a, b = BranchAndBound(SMALL, L=3).run(), BranchAndBound(SMALL, L=3).run()
        assert a.lexicon == b.lexicon and a.total == b.total

    def test_pruning_reduces_nodes_without_changing_optimum(self):
        full = BranchAndBound(SMALL, L=3, propagate=False, learn=False)
        s1 = full.run()
        pruned = BranchAndBound(SMALL, L=3, propagate=True, learn=True)
        s2 = pruned.run()
        assert s1.total == pytest.approx(s2.total)
        assert pruned.stats.nodes <= full.stats.nodes

    def test_nogood_learning_hits(self):
        bb = BranchAndBound(SMALL, L=3, learn=True)
        bb.run()
        assert bb.stats.conflicts > 0
        assert bb.stats.nogoods > 0
        assert bb.stats.nogood_hits >= 0  # 数据：是否真能命中，记录下来

    def test_node_limit_returns_best_so_far_or_none(self):
        bb = BranchAndBound(SMALL, L=3, node_limit=5)
        sol = bb.run()
        assert bb.stats.nodes <= 5
        assert sol is None or math.isfinite(sol.total)

    def test_infeasible_L_returns_none(self):
        assert BranchAndBound(SMALL, L=0).run() is None


WIDE = {  # 穷尽搜索在 SMALL 上的真实最优（2026-10-11，L=3，α₂=β=γ=1）：所有词落到 S 与 S 的修饰语
    "john": "S[dcl]", "mary": "S[dcl]", "cat": "S[dcl]", "dog": "S[dcl]",
    "the": "S[dcl]/S[dcl]", "sleeps": "S[dcl]\\S[dcl]", "runs": "S[dcl]\\S[dcl]",
    "quickly": "S[dcl]\\S[dcl]", "sees": "(S[dcl]/S[dcl])\\S[dcl]",
}


class TestPassLine:
    """§8 M3 及格线（先在玩具语料上）：恢复 NP/N、(S\\NP)/NP、S\\NP。

    **实测不通过，且不是权重问题。** 穷尽搜索的最优是 WIDE（§4.5 里点名的"句末词取 S、其余 S/S"那种极端语法），
    它在 α₂、β、γ 三个分量上**逐项 ≤** 金标准（4<6, 6<8, 0=0），任何正权重下都严格占优。γ 项没起作用：
    这部语法在语料每一句上都恰好 1 条推导，但接受 `john sleeps sleeps`、`the the cat`——它的过度泛化发生在
    语料外的字符串上，而 γ 只数语料内的推导（§5.2 的困境在目标函数层面再现）。不标 xfail，按实测记录。"""

    def test_measured_optimum_is_wide_grammar_not_gold(self):
        sol = BranchAndBound(SMALL, L=3).run()
        acc, renamed = accuracy(sol.lexicon, SMALL_GOLD)
        print(f"\nM3 small: acc={acc:.2f} total={sol.total:.2f} lexicon={renamed}")
        assert sol.lexicon == WIDE
        assert sol.total == 10.0 and sol.cost == Cost(9, 4, 6, 0.0)
        assert acc == 0.0
        assert renamed["the"] != "NP/N"  # 及格线三项全部不满足

    def test_wide_dominates_gold_componentwise(self):
        from ccg_solver.corpus import Lexicon
        from ccg_solver.state import State

        def cost_of(lexicon):
            lex, st = Lexicon(), State(complexity_bound=3)
            for w, c in lexicon.items():
                assert lex.bind(st, w, parse(c))
            return corpus_cost(SMALL, lex, st), lex, st

        wide, lex_w, st_w = cost_of(WIDE)
        gold, _, _ = cost_of(SMALL_GOLD)
        assert gold == Cost(9, 6, 8, 0.0)
        assert wide.distinct < gold.distinct and wide.complexity < gold.complexity
        assert wide.log_derivations == gold.log_derivations == 0.0
        for w in (Weights(), Weights(alpha2=10), Weights(beta=10), Weights(gamma=100)):
            assert wide.total(w) < gold.total(w)
        # 过度泛化在语料外：γ 看不见
        from ccg_solver.local import solve_sentence
        assert len(solve_sentence(Sentence.from_text("john sleeps sleeps ."), lex_w, st_w)) == 1
        assert len(solve_sentence(Sentence.from_text("the the cat ."), lex_w, st_w)) == 1


class TestIterativeDeepening:
    def test_stall_stop_and_reports(self):
        best, min_L, per_L = solve_iterative(SMALL, L_max=5)
        assert min_L == min(L for L, t in per_L.items() if t is not None)
        assert best is not None
        assert best.L >= min_L
        assert all(per_L[L] is None for L in range(min_L))
        # stall=2：找到更优后连续两层不改进就停，所以 per_L 的键数 ≤ best.L + 3
        assert max(per_L) <= best.L + 2

    def test_best_is_min_over_layers(self):
        best, _, per_L = solve_iterative(SMALL, L_max=5)
        assert best.total == min(t for t in per_L.values() if t is not None)


@pytest.mark.slow
def test_full_toy_corpus_anytime_optimum_is_wide_grammar():
    """30 句语料穷尽搜索不可行（7 词句根节点 3990 条 σ，25 分钟无果）。node_limit=1500 的 anytime 解
    （约 30 秒）仍是 WIDE 型语法：distinct 4、复杂度 18、γ 0，总代价 22；金标准 distinct 7、复杂度 18，总 25。"""
    bb = BranchAndBound(SENTS, L=3, node_limit=1500)
    sol = bb.run()
    acc, renamed = accuracy(sol.lexicon, M2_GOLD)
    print(f"\nM3 toy-30: acc={acc:.2f} total={sol.total:.2f} {bb.stats} lexicon={renamed}")
    assert sol.cost == Cost(20, 4, 18, 0.0) and sol.total == 22.0
    assert acc == 0.0
    assert {sol.lexicon[w] for w in ("john", "mary", "dogs", "cats", "birds", "a")} == {"S[dcl]"}
    from ccg_solver.corpus import Lexicon
    from ccg_solver.state import State
    lex, st = Lexicon(), State(complexity_bound=3)
    for w, c in M2_GOLD.items():
        assert lex.bind(st, w, parse(c))
    assert corpus_cost(SENTS, lex, st) == Cost(20, 7, 18, 0.0)


def _childes():
    try:
        from ccg_solver.data.childes import load_adult_utterances
        return load_adult_utterances("Eve", max_len=8, min_freq=20)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"CHILDES unavailable: {e}")


@pytest.mark.slow
def test_childes_feasible_solution():
    """M3 完成判据：CHILDES 长度 ≤ 8 子集能求出可行解；报告词典前 30、未解句数、平均 |D|。"""
    sents = _childes()
    best, min_L, per_L = solve_iterative(sents, L_max=3, node_limit=20000)
    assert best is not None
