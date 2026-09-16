"""M0 测试（一）：范畴表示 —— hash-cons DAG、变量、解析/打印、复杂度度量。

本文件里体现的设计决定（请确认）：
1. 特征是原子身份的一部分：``S`` 与 ``S[dcl]`` 是两个不同原子（特征变量不在 M0）。
2. 文本语法：斜线左结合；单个大写字母（可带数字）且非原子名 = 变量；``?<id>`` 引用已有变量。
3. 复杂度：arity = 结果脊上的论元数；depth = 论元位置上的函子嵌套深度（原子论元不计）。
4. ``MAX_ARITY = 4`` 对整棵树检查（论元内部超限同样不允许）。
"""
from dataclasses import FrozenInstanceError

import pytest

from ccg_solver.category import (
    ATOMS,
    MAX_ARITY,
    Atom,
    Functor,
    Var,
    arity,
    atom,
    bwd,
    complexity,
    depth,
    free_vars,
    fresh_var,
    fwd,
    is_admissible,
    parse,
    subterms,
    var,
)


class TestHashCons:
    def test_atom_identity(self):
        assert atom("S") is atom("S")
        assert atom("S", "dcl") is atom("S", "dcl")
        assert atom("S", "dcl") is not atom("S")
        assert atom("S", "dcl") is not atom("S", "q")

    def test_functor_identity_and_direction(self):
        assert fwd(atom("S"), atom("NP")) is fwd(atom("S"), atom("NP"))
        assert fwd(atom("S"), atom("NP")) is not bwd(atom("S"), atom("NP"))

    def test_structural_equality_is_pointer_equality(self):
        c1, c2 = parse("(S\\NP)/NP"), parse("(S\\NP)/NP")
        assert c1 == c2
        assert c1 is c2
        assert hash(c1) == hash(c2)
        assert parse("S/NP") != parse("S\\NP")
        assert parse("S/NP") is not parse("S\\NP")

    def test_subterm_sharing(self):
        tv = parse("(S\\NP)/NP")
        assert isinstance(tv, Functor)
        assert tv.result is parse("S\\NP")
        assert tv.arg is atom("NP")
        assert tv.result.result is atom("S")

    def test_usable_as_dict_key(self):
        d = {parse("S/NP"): 1}
        assert d[fwd(atom("S"), atom("NP"))] == 1

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            parse("S/NP").arg = atom("N")  # type: ignore[misc]

    def test_atoms_fixed_set(self):
        assert set(ATOMS) == {"S", "N", "NP", "PP", "conj"}
        with pytest.raises(ValueError):
            atom("Foo")
        assert isinstance(atom("conj"), Atom)


class TestVars:
    def test_fresh_ids_unique_and_increasing(self):
        a, b, c = fresh_var(), fresh_var(), fresh_var()
        assert a is not b and b is not c
        assert a.id < b.id < c.id

    def test_var_hash_consed_by_id(self):
        v = fresh_var()
        assert var(v.id) is v
        assert isinstance(v, Var)

    def test_parse_names_share_within_env(self):
        env: dict = {}
        c1 = parse("S/X", env)
        c2 = parse("X\\NP", env)
        assert isinstance(c1.arg, Var)
        assert c1.arg is c2.result
        assert env["X"] is c1.arg

    def test_parse_same_name_in_one_call_is_one_var(self):
        c = parse("X/X")
        assert c.result is c.arg

    def test_parse_without_env_is_fresh_across_calls(self):
        assert parse("S/X").arg is not parse("S/X").arg

    def test_parse_var_by_id(self):
        v = fresh_var()
        assert parse(f"S/?{v.id}").arg is v

    def test_free_vars(self):
        env: dict = {}
        c = parse("(X\\NP)/(Y/X)", env)
        assert free_vars(c) == {env["X"], env["Y"]}
        assert free_vars(parse("(S\\NP)/NP")) == set()
        v = fresh_var()
        assert free_vars(v) == {v}


class TestParsePrint:
    @pytest.mark.parametrize(
        "text, printed",
        [
            ("S", "S"),
            ("S[dcl]", "S[dcl]"),
            ("S[wq]", "S[wq]"),
            ("conj", "conj"),
            ("S/NP", "S/NP"),
            ("S\\NP", "S\\NP"),
            ("(S\\NP)/NP", "(S\\NP)/NP"),
            ("S\\NP/NP", "(S\\NP)/NP"),  # 左结合
            ("S/(S\\NP)", "S/(S\\NP)"),
            ("(S\\NP)/(S\\NP)", "(S\\NP)/(S\\NP)"),
            ("((S\\NP)/(S\\NP))/NP", "((S\\NP)/(S\\NP))/NP"),
            ("(S[dcl]\\NP)/NP", "(S[dcl]\\NP)/NP"),
            ("NP/N", "NP/N"),
            (" ( S \\ NP ) / NP ", "(S\\NP)/NP"),  # 容忍空白
        ],
    )
    def test_roundtrip(self, text, printed):
        c = parse(text)
        assert str(c) == printed
        assert parse(printed) is c

    def test_print_var(self):
        v = fresh_var()
        assert str(v) == f"?{v.id}"
        assert str(fwd(atom("S"), v)) == f"S/?{v.id}"
        assert str(bwd(v, fwd(atom("S"), atom("NP")))) == f"?{v.id}\\(S/NP)"

    def test_print_then_parse_preserves_vars(self):
        env: dict = {}
        c = parse("(X\\NP)/Y", env)
        assert parse(str(c)) is c  # 打印出的 ?<id> 能解析回同一变量

    @pytest.mark.parametrize(
        "bad",
        ["", "S/", "/NP", "(S\\NP", "S\\NP)", "S//NP", "Foo", "NPX", "NP[", "S[dcl", "S[]", "S NP", "?", "?x"],
    )
    def test_parse_errors(self, bad):
        with pytest.raises(ValueError):
            parse(bad)


class TestComplexity:
    @pytest.mark.parametrize(
        "text, ar, dp",
        [
            ("NP", 0, 0),
            ("S/NP", 1, 0),
            ("S\\NP", 1, 0),
            ("(S\\NP)/NP", 2, 0),
            ("((S\\NP)/PP)/NP", 3, 0),
            ("S/(S\\NP)", 1, 1),  # 类型提升
            ("(S\\NP)/(S\\NP)", 1, 1),  # 助动词
            ("((S\\NP)/(S\\NP))/NP", 2, 1),
            ("(S/(S\\NP))/N", 2, 1),  # 提升后的限定词
            ("S/(S/(S\\NP))", 1, 2),
            ("X/NP", 1, 0),  # 含变量按已知结构算（下界）
            ("X", 0, 0),
        ],
    )
    def test_arity_depth_complexity(self, text, ar, dp):
        c = parse(text)
        assert arity(c) == ar
        assert depth(c) == dp
        assert complexity(c) == ar + dp

    def test_max_arity(self):
        assert MAX_ARITY == 4
        assert is_admissible(parse("(((S/NP)/NP)/NP)/NP"))
        assert not is_admissible(parse("((((S/NP)/NP)/NP)/NP)/NP"))
        # 论元内部超限同样不允许
        assert not is_admissible(parse("S/(((((S/NP)/NP)/NP)/NP)/NP)"))
        assert is_admissible(atom("S"))
        assert is_admissible(fresh_var())


class TestSubterms:
    def test_preorder(self):
        c = parse("(S\\NP)/N")
        assert list(subterms(c)) == [c, parse("S\\NP"), atom("S"), atom("NP"), atom("N")]

    def test_atom_and_var(self):
        v = fresh_var()
        assert list(subterms(atom("S"))) == [atom("S")]
        assert list(subterms(v)) == [v]
