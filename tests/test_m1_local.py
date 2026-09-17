"""M1 测试（三）：单句求解 + σ 去重。完成判据"手工 5 词玩具语料能解出唯一词典"在本文件末尾。

设计决定（请确认）：
1. ``solve_sentence`` 不改变状态（返回前回滚）；要采纳某条推导用 ``Derivation.apply(st)``。
2. σ 的去重键 ``key``：按 token 位置列出词型范畴的规范打印，局部新变量按首次出现重命名为 ``?_1 ?_2 …``；
   全局已有变量（词型变量本身、调用前已存在的变量）原样打印。
3. 返回按 ``key`` 排序；``n_trees`` = 折叠进这条 σ 的原始树数；``tree`` 是其中 bracketed 字符串最小的那棵。
4. "唯一词典"的 M1 形式：5 词词典，逐个词留出，其余固定，含该词的句子必须把它解成唯一且正确的范畴。
   （跨句 backbone 是 M2 的事，M1 只保证单句方程本身是对的。）
"""
import pytest

from ccg_solver.category import atom, fresh_var, parse
from ccg_solver.corpus import Lexicon, Sentence
from ccg_solver.local import Derivation, Tree, enumerate_trees, solve_sentence
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


def gold_state(skip=()):
    lex, st = Lexicon(), State()
    for w, c in GOLD.items():
        lex.var_of(w)
    for w, c in GOLD.items():
        if w not in skip:
            assert lex.bind(st, w, parse(c))
    return lex, st


class TestFixedLexicon:
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
        ds = solve_sentence(Sentence.from_text("the cat sees the dog ."), lex, st)
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


class TestFreeVariables:
    def test_all_unknown_contains_gold_and_is_deterministic(self):
        lex, st = Lexicon(), State()
        sent = Sentence.from_text("the cat sleeps .")
        ds1 = solve_sentence(sent, lex, st)
        ds2 = solve_sentence(sent, lex, st)
        assert [d.key for d in ds1] == [d.key for d in ds2]
        assert len(ds1) > 1  # 三个未知数一条方程，必然多解
        keys = {d.key for d in ds1}
        # 金标准形状（用局部变量表达）在解集中
        assert ("?_1/?_2", "?_2", "S[dcl]\\?_1") in keys
        assert all(len(d.key) == 3 for d in ds1)

    def test_keys_are_unique(self):
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

    def test_admissibility_prunes(self):
        lex, st = Lexicon(), State()
        ds = solve_sentence(Sentence.from_text("a b c d e f ."), lex, st)
        for d in ds:
            for k in d.key:
                assert parse(k.replace("?_", "?V")).__class__  # 可解析
        from ccg_solver.category import is_admissible
        assert all(is_admissible(parse(k.replace("?_", "?V"))) for d in ds for k in d.key)

    def test_global_vars_keep_identity_in_key(self):
        lex, st = Lexicon(), State()
        x = fresh_var()
        assert lex.bind(st, "sleeps", parse(f"S[dcl]\\?{x.id}"))
        ds = solve_sentence(Sentence.from_text("cat sleeps ."), lex, st)
        assert (f"?{x.id}", f"S[dcl]\\?{x.id}") in {d.key for d in ds}


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
        assert not ds[0].apply(st)
        assert st.mark() == m


class TestUniqueLexicon:
    """M1 完成判据：5 词玩具语料，逐词留出，单句方程把它解成唯一且正确的范畴。"""

    @pytest.mark.xfail(
        strict=True,
        reason="假设被数据证伪：单句一个未知数不唯一（sees 有 14 个 σ），唯一性要靠跨句/目标函数；待决定 M1 判据",
    )
    @pytest.mark.parametrize("word", list(GOLD))
    def test_held_out_word_is_uniquely_recovered(self, word):
        lex, st = gold_state(skip=(word,))
        seen = False
        for text in CORPUS:
            sent = Sentence.from_text(text)
            if word not in sent.words:
                continue
            seen = True
            ds = solve_sentence(sent, lex, st)
            assert len(ds) == 1, f"{text}: {[d.key for d in ds]}"
            assert ds[0].key[sent.words.index(word)] == GOLD[word]
        assert seen

    def test_whole_corpus_consistent_with_gold(self):
        lex, st = gold_state()
        for text in CORPUS:
            assert len(solve_sentence(Sentence.from_text(text), lex, st)) == 1


def test_enumerate_trees():
    assert enumerate_trees(1) == 1
    assert enumerate_trees(2) == 4
    assert enumerate_trees(3) == 2 * 16
    assert enumerate_trees(5) == 14 * 4**4
