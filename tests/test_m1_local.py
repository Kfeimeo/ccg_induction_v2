"""M1 测试（三）：单句求解 + σ 去重。判据 M1-a … M1-e 见 docs/m1-conclusion-m2-plan.md §1.1。

设计决定：
1. ``solve_sentence`` 不改变状态（返回前回滚）；要采纳某条推导用 ``Derivation.apply(st)``。
2. σ 的去重键 ``key``：按 token 位置列出词型范畴的规范打印，句中词型变量与局部新变量按首次出现重命名为
   ``?_1 ?_2 …``；调用前已存在的其他全局变量原样打印。
3. 返回按 ``key`` 排序；``n_trees`` = 折叠进这条 σ 的原始树数；``tree`` 是其中 bracketed 字符串最小的那棵。
4. 默认规则集只有 > <；>B <B 通过 ``rules=COMPOSITION`` 开启，golden 计数两套各记一份。
5. **不承诺唯一性**：``minimal(word)`` 断言精确集合，`sees` 的主宾对称是已知现象（CLAUDE.md §5.4）。
"""
import time

import pytest

from ccg_solver.category import Var, atom, complexity, free_vars, fresh_var, parse
from ccg_solver.corpus import Lexicon, Sentence
from ccg_solver.local import Derivation, Tree, enumerate_trees, solve_sentence
from ccg_solver.rules import APPLICATION, COMPOSITION
from ccg_solver.state import State

GOLD = {
    "the": "NP/N",
    "cat": "N",
    "dog": "N",
    "sleeps": "S[dcl]\\NP",
    "sees": "(S[dcl]\\NP)/NP",
}
CORPUS = [
    "the cat sleeps .",
    "the dog sleeps .",
    "the cat sees the dog .",
    "the dog sees the cat .",
]
# §2.3：VP 修饰语打破主宾对称
GOLD_VP = {**GOLD, "quickly": "(S[dcl]\\NP)\\(S[dcl]\\NP)"}
CORPUS_VP = CORPUS + ["the cat sees the dog quickly ."]


def gold_state(gold=GOLD, skip=()):
    lex, st = Lexicon(), State()
    for w in gold:
        lex.var_of(w)
    for w, c in gold.items():
        if w not in skip:
            assert lex.bind(st, w, parse(c))
    return lex, st


def _cat(key: str):
    return parse(key.replace("?_", "?V"))


def candidates(word, gold=GOLD, corpus=CORPUS, rules=None):
    """留出 word、其余固定；跨含它的句子取交集，得到 word 的候选范畴集合。"""
    lex, st = gold_state(gold, skip=(word,))
    sets = []
    for text in corpus:
        sent = Sentence.from_text(text)
        if word in sent.words:
            ds = solve_sentence(sent, lex, st, rules=rules)
            sets.append({d.key[sent.words.index(word)] for d in ds})
    return set.intersection(*sets)


def minimal(word, **kw):
    cands = candidates(word, **kw)
    lo = min(complexity(_cat(k)) for k in cands)
    return {k for k in cands if complexity(_cat(k)) == lo}


class TestFixedLexicon:  # M1-c
    def test_intransitive_one_sigma(self):
        lex, st = gold_state()
        ds = solve_sentence(Sentence.from_text("the cat sleeps ."), lex, st)
        assert len(ds) == 1
        d = ds[0]
        assert d.key == ("NP/N", "N", "S[dcl]\\NP")
        assert d.n_trees == 1
        assert d.tree.bracketed(("the", "cat", "sleeps")) == "[< [> the cat] sleeps]"

    def test_transitive_spurious_ambiguity_collapses(self):
        lex, st = gold_state()
        sent = Sentence.from_text("the cat sees the dog .")
        ds = solve_sentence(sent, lex, st)
        assert len(ds) == 1 and ds[0].n_trees == 1  # 只有应用规则时没有虚假歧义
        ds = solve_sentence(sent, lex, st, rules=COMPOSITION)
        assert len(ds) == 1
        assert ds[0].n_trees == 2  # [sees [the dog]] 与 [[sees the] dog]（>B）
        assert ds[0].key == ("NP/N", "N", "(S[dcl]\\NP)/NP", "NP/N", "N")

    def test_ungrammatical_has_no_derivation(self):
        lex, st = gold_state()
        assert solve_sentence(Sentence.from_text("cat the sleeps ."), lex, st) == []
        assert solve_sentence(Sentence.from_text("the cat sees ."), lex, st) == []

    def test_goal_feature_matters(self):
        lex, st = gold_state()
        assert solve_sentence(Sentence.from_text("the cat sleeps ?"), lex, st) == []

    def test_state_untouched(self):
        lex, st = gold_state()
        m = st.mark()
        solve_sentence(Sentence.from_text("the cat sees the dog ."), lex, st)
        assert st.mark() == m
        assert lex.categories(st)["sees"] is parse("(S[dcl]\\NP)/NP")

    def test_single_word_sentence(self):
        lex, st = Lexicon(), State()
        assert lex.bind(st, "go", parse("S[dcl]"))
        ds = solve_sentence(Sentence.from_text("go ."), lex, st)
        assert len(ds) == 1 and ds[0].key == ("S[dcl]",)
        assert ds[0].tree.bracketed(("go",)) == "go"


class TestFreeVariables:  # M1-a, M1-b
    def test_all_unknown_contains_gold_and_is_deterministic(self):
        lex, st = Lexicon(), State()
        sent = Sentence.from_text("the cat sleeps .")
        ds1 = solve_sentence(sent, lex, st)
        ds2 = solve_sentence(sent, lex, st)
        assert [d.key for d in ds1] == [d.key for d in ds2]
        assert [d.n_trees for d in ds1] == [d.n_trees for d in ds2]
        assert len(ds1) > 1  # 三个未知数一条方程，必然多解
        assert ("?_1/?_2", "?_2", "S[dcl]\\?_1") in {d.key for d in ds1}

    def test_gold_sigma_in_D_for_every_corpus_sentence(self):
        for gold, corpus in ((GOLD, CORPUS), (GOLD_VP, CORPUS_VP)):
            lex, st = Lexicon(), State()
            for text in corpus:
                sent = Sentence.from_text(text)
                keys = {d.key for d in solve_sentence(sent, lex, st)}
                # 金标准 σ 用局部变量表达后必须在解集中：把金标准原子换成局部变量的规范形式不可行，
                # 所以反过来：固定金标准，检查它是某条自由解的实例（subsumes）。
                lex_g, st_g = gold_state(gold)
                gold_key = solve_sentence(sent, lex_g, st_g)[0].key
                assert any(subsumes_key(k, gold_key) for k in keys), (text, gold_key)

    def test_keys_are_unique_and_sorted(self):
        lex, st = Lexicon(), State()
        ds = solve_sentence(Sentence.from_text("the cat sees the dog ."), lex, st)
        keys = [d.key for d in ds]
        assert len(keys) == len(set(keys))
        assert keys == sorted(keys)
        assert sum(d.n_trees for d in ds) <= enumerate_trees(5)

    def test_rigid_same_word_twice_shares_variable(self):
        lex, st = Lexicon(), State()
        ds = solve_sentence(Sentence.from_text("the cat sees the dog ."), lex, st)
        for d in ds:
            assert d.key[0] == d.key[3]  # 两个 the 同一变量

    def test_global_vars_keep_identity_in_key(self):
        lex, st = Lexicon(), State()
        x = fresh_var()
        assert lex.bind(st, "sleeps", parse(f"S[dcl]\\?{x.id}"))
        ds = solve_sentence(Sentence.from_text("cat sleeps ."), lex, st)
        assert (f"?{x.id}", f"S[dcl]\\?{x.id}") in {d.key for d in ds}


class TestBounds:  # M1-d + §4.6
    def test_six_unknowns_under_five_seconds(self):
        lex, st = Lexicon(), State()
        from ccg_solver.category import is_admissible
        t = time.perf_counter()
        ds = solve_sentence(Sentence.from_text("a b c d e f ."), lex, st)
        assert time.perf_counter() - t < 5.0
        assert ds and all(is_admissible(_cat(k)) for d in ds for k in d.key)

    def test_complexity_bound_is_iterative_deepening_order(self):
        lex, st = Lexicon(), State()
        sent = Sentence.from_text("the cat sees the dog .")
        by_L = [solve_sentence(sent, lex, st, complexity_bound=L) for L in range(0, 4)]
        assert by_L[0] == []  # 全原子推不出任何句子
        keys = [{d.key for d in ds} for ds in by_L]
        # 数据：L=1 已有 1 个解（the=S/?1, cat=?1/?2, sees=?2/S, dog=?1：全右嵌套），L=2 有 56，L=3 有 127
        assert [len(k) for k in keys] == [0, 1, 56, 127]
        assert keys[1] <= keys[2] <= keys[3]  # 单调：放宽 L 只增不减
        for ds, L in zip(by_L, range(4)):
            assert all(complexity(_cat(k)) <= L for d in ds for k in d.key)
        unbounded = {d.key for d in solve_sentence(sent, lex, st)}
        assert keys[3] <= unbounded

    def test_minimal_feasible_L_for_corpus(self):
        lex, st = gold_state()
        for text, L in (("the cat sleeps .", 1), ("the cat sees the dog .", 2)):
            sent = Sentence.from_text(text)
            assert solve_sentence(sent, lex, st, complexity_bound=L - 1) == []
            assert solve_sentence(sent, lex, st, complexity_bound=L)


def subsumes_key(general: tuple, specific: tuple) -> bool:
    """general 的一个实例是 specific：合一后 specific 的变量仍是变量且两两不合并（单向匹配）。"""
    st = State()
    envg, envs = {}, {}
    gs = [parse(k.replace("?_", "?V"), envg) for k in general]
    ss = [parse(k.replace("?_", "?W"), envs) for k in specific]
    if not all(st.unify(g, s) for g, s in zip(gs, ss)):
        return False
    svars = set().union(*(free_vars(c) for c in ss))
    reps = [st.resolve(v) for v in svars]
    return all(isinstance(r, Var) for r in reps) and len(set(reps)) == len(reps)


class TestMonotonicity:  # M1-e：D(C ∪ {s}) 限制到共享变量后 ⊆ D(C)
    @pytest.mark.parametrize("other", CORPUS[1:])
    def test_adding_a_sentence_only_shrinks_D(self, other):
        lex, st = Lexicon(), State()
        s1 = Sentence.from_text(CORPUS[0])
        before = [d.key for d in solve_sentence(s1, lex, st)]
        s2 = Sentence.from_text(other)
        for d2 in solve_sentence(s2, lex, st):
            m = st.mark()
            assert d2.apply(st)  # 取 s2 的一条推导作为"新增约束"
            after = [d.key for d in solve_sentence(s1, lex, st)]
            for k in after:
                assert any(subsumes_key(g, k) for g in before), (other, k)
            st.rollback(m)


class TestApply:
    def test_apply_binds_lexicon(self):
        lex, st = gold_state(skip=("sleeps",))
        ds = solve_sentence(Sentence.from_text("the cat sleeps ."), lex, st)
        d = next(d for d in ds if d.key[2] == "S[dcl]\\NP")
        assert d.apply(st)
        assert lex.categories(st)["sleeps"] is parse("S[dcl]\\NP")

    def test_apply_after_state_changed_can_fail_atomically(self):
        lex, st = gold_state(skip=("sleeps",))
        ds = solve_sentence(Sentence.from_text("the cat sleeps ."), lex, st)
        assert lex.bind(st, "sleeps", parse("S[dcl]/NP"))
        m = st.mark()
        assert all(not d.apply(st) for d in ds)
        assert st.mark() == m


class TestMinimalSigmas:
    """留出一个词，其余固定，跨句交集后取复杂度最小者。不是唯一性判据，是对解集形状的精确记录。"""

    def test_minimal_sigmas(self):
        assert minimal("the") == {"NP/N"}
        assert minimal("cat") == {"N"}
        assert minimal("dog") == {"N"}
        assert minimal("sleeps") == {"S[dcl]\\NP"}
        # 已知对称：主宾方向在无 VP 证据的语料下不可识别（CLAUDE.md §5.4）
        assert minimal("sees") == {"(S[dcl]\\NP)/NP", "(S[dcl]/NP)\\NP"}

    def test_vp_modifier_breaks_symmetry(self):
        assert minimal("sees", gold=GOLD_VP, corpus=CORPUS_VP) == {"(S[dcl]\\NP)/NP"}

    def test_adverb_itself_is_not_complexity_minimal(self):
        """数据：留出 quickly 时，复杂度最小的是句/NP/N 层面的修饰语（复杂度 1），金标准复杂度 2。
        又一个"标量复杂度选不出金标准"的实例；只有 α₂（范畴重用）能偏向 (S\\NP)\\(S\\NP)。"""
        cands = candidates("quickly", gold=GOLD_VP, corpus=CORPUS_VP)
        assert "(S[dcl]\\NP)\\(S[dcl]\\NP)" in cands
        assert minimal("quickly", gold=GOLD_VP, corpus=CORPUS_VP) == {"S[dcl]\\S[dcl]", "NP\\NP", "N\\N"}

    def test_symmetry_is_scalar_invariant(self):
        a, b = parse("(S[dcl]\\NP)/NP"), parse("(S[dcl]/NP)\\NP")
        assert complexity(a) == complexity(b)


# ---- golden values：留出词在单句下的 (|D|, 原始树数)。M4 开新规则前必须对比。
GOLDEN_APPLICATION = {
    ("the", "the cat sleeps ."): (2, 2),
    ("cat", "the cat sees the dog ."): (11, 11),
    ("sees", "the cat sees the dog ."): (14, 14),
    ("sleeps", "the cat sleeps ."): (2, 2),
    ("sees", "the cat sees the dog quickly ."): (15, 15),
    ("quickly", "the cat sees the dog quickly ."): (9, 9),
}
GOLDEN_COMPOSITION = {
    ("the", "the cat sleeps ."): (2, 2),
    ("cat", "the cat sees the dog ."): (14, 29),
    ("sees", "the cat sees the dog ."): (14, 19),
    ("sleeps", "the cat sleeps ."): (2, 2),
    ("sees", "the cat sees the dog quickly ."): (15, 25),
    ("quickly", "the cat sees the dog quickly ."): (11, 17),
}


@pytest.mark.parametrize("rules, golden", [(APPLICATION, GOLDEN_APPLICATION), (COMPOSITION, GOLDEN_COMPOSITION)])
def test_golden_counts(rules, golden):
    for (word, text), (n_sigma, n_trees) in golden.items():
        gold = GOLD_VP if "quickly" in text else GOLD
        lex, st = gold_state(gold, skip=(word,))
        ds = solve_sentence(Sentence.from_text(text), lex, st, rules=rules)
        assert (len(ds), sum(d.n_trees for d in ds)) == (n_sigma, n_trees), (word, text)


def test_enumerate_trees():
    assert enumerate_trees(1) == 1
    assert enumerate_trees(2) == 2
    assert enumerate_trees(3) == 2 * 4
    assert enumerate_trees(5, 4) == 14 * 4**4
