"""M1 测试（二）：Sentence 与 Lexicon。

设计决定（请确认）：
1. 输入句子最后一个 token 必须是 ``.`` 或 ``?``，决定目标 ``S[dcl]`` / ``S[q]``；标点本身不是词型、不设变量。
2. 小写化在 ``Sentence.from_text`` 里做；不做任何其他归一化（§7.4）。
3. Lexicon 变量按首次登记顺序分配 id（配合"代表元 = 较小 id"，先登记的词是合并后的代表元）。
"""
import pytest

from ccg_solver.category import Var, atom, parse
from ccg_solver.corpus import Lexicon, Sentence
from ccg_solver.state import State


class TestSentence:
    def test_from_text_dcl(self):
        s = Sentence.from_text("The cat sleeps .")
        assert s.words == ("the", "cat", "sleeps")
        assert s.feat == "dcl"
        assert len(s) == 3

    def test_from_text_q(self):
        s = Sentence.from_text("does the cat sleep ?")
        assert s.words == ("does", "the", "cat", "sleep") and s.feat == "q"

    @pytest.mark.parametrize("bad", ["the cat sleeps", ".", "", "the cat . sleeps", "the cat !"])
    def test_requires_final_punct(self, bad):
        with pytest.raises(ValueError):
            Sentence.from_text(bad)

    def test_hashable_and_equal(self):
        assert Sentence.from_text("a b .") == Sentence(("a", "b"), "dcl")
        assert {Sentence.from_text("a b .")} == {Sentence(("a", "b"), "dcl")}


class TestLexicon:
    def test_var_per_word_type(self):
        lex = Lexicon()
        assert lex.var_of("the") is lex.var_of("the")
        assert lex.var_of("the") is not lex.var_of("cat")
        assert isinstance(lex.var_of("the"), Var)
        assert "the" in lex and "dog" not in lex
        assert lex.words() == ["the", "cat"]

    def test_ids_follow_registration_order(self):
        lex = Lexicon()
        a, b = lex.var_of("a"), lex.var_of("b")
        assert a.id < b.id

    def test_categories_dump(self):
        lex, st = Lexicon(), State()
        assert lex.bind(st, "the", parse("NP/N"))
        lex.var_of("cat")
        cats = lex.categories(st)
        assert cats["the"] is parse("NP/N")
        assert cats["cat"] is lex.var_of("cat")  # 未解出的词显示为自己的变量

    def test_bind_conflict_is_atomic(self):
        lex, st = Lexicon(), State()
        assert lex.bind(st, "the", parse("NP/N"))
        m = st.mark()
        assert not lex.bind(st, "the", parse("S\\NP"))
        assert st.mark() == m
