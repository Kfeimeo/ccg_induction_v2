"""M1 测试（一）：四条组合规则，T 关掉。

设计决定（请确认）：
1. 规则签名 ``rule(left, right, st) -> Category | None``，在 st 上合一，不回滚；调用方包 mark/rollback。
2. 规则对自由变量也适用：``?A ?B`` 经 ``>`` 得 ``?X``，并把 ``?A`` 绑成 ``?X/?B``（?X 是新变量）。
"""
import pytest

from ccg_solver.category import Var, atom, fresh_var, parse
from ccg_solver.rules import APPLICATION, COMPOSITION, RULES, backward_application, backward_composition, forward_application, forward_composition
from ccg_solver.state import State


def test_rule_sets():
    assert set(APPLICATION) == {">", "<"}
    assert set(COMPOSITION) == {">", "<", ">B", "<B"}
    assert RULES is APPLICATION  # M1–M3 默认；T 不存在


class TestApplication:
    def test_forward(self):
        st = State()
        r = forward_application(parse("NP/N"), atom("N"), st)
        assert st.resolve(r) is atom("NP")

    def test_backward(self):
        st = State()
        r = backward_application(atom("NP"), parse("S[dcl]\\NP"), st)
        assert st.resolve(r) is atom("S", "dcl")

    def test_forward_wrong_direction_or_arg(self):
        st = State()
        assert forward_application(parse("S\\NP"), atom("NP"), st) is None
        assert forward_application(parse("NP/N"), atom("NP"), st) is None
        assert forward_application(atom("N"), parse("NP/N"), st) is None

    def test_backward_wrong(self):
        st = State()
        assert backward_application(parse("S\\NP"), atom("NP"), st) is None
        assert backward_application(atom("N"), parse("S\\NP"), st) is None

    def test_forward_on_free_vars_binds_functor_shape(self):
        st, env = State(), {}
        a, b = parse("?A", env), parse("?B", env)
        r = forward_application(a, b, st)
        assert isinstance(st.resolve(r), Var)
        assert str(st.resolve(a)) == f"{st.resolve(r)}/{b}"  # ?A = ?X/?B
        assert st.resolve(b) is b  # 论元不动

    def test_backward_on_free_vars(self):
        st, env = State(), {}
        a, b = parse("?A", env), parse("?B", env)
        r = backward_application(a, b, st)
        assert str(st.resolve(b)) == f"{st.resolve(r)}\\{a}"

    def test_partial_functor_with_var(self):
        st, env = State(), {}
        r = forward_application(parse("S/?Y", env), parse("(S\\NP)", env), st)
        assert st.resolve(r) is atom("S")
        assert st.resolve(env["Y"]) is parse("S\\NP")

    def test_failure_may_leave_partial_bindings_caller_rolls_back(self):
        st, env = State(), {}
        m = st.mark()
        assert forward_application(parse("(?A/NP)/N", env), atom("NP"), st) is None
        st.rollback(m)
        assert st.resolve(env["A"]) is env["A"]


class TestComposition:
    def test_forward_B(self):
        st = State()
        r = forward_composition(parse("(S\\NP)/NP"), parse("NP/N"), st)
        assert st.resolve(r) is parse("(S\\NP)/N")

    def test_backward_B(self):
        st = State()
        r = backward_composition(parse("S\\NP"), parse("S\\S"), st)
        assert st.resolve(r) is parse("S\\NP")

    def test_B_needs_matching_middle(self):
        st = State()
        assert forward_composition(parse("S/NP"), parse("N/N"), st) is None
        assert forward_composition(parse("S/NP"), parse("NP\\N"), st) is None  # 混合方向不开
        assert forward_composition(parse("S/NP"), atom("NP"), st) is None
        assert backward_composition(parse("S\\NP"), parse("S/S"), st) is None

    def test_B_on_free_vars(self):
        st, env = State(), {}
        a, b = parse("?A", env), parse("?B", env)
        r = forward_composition(a, b, st)
        ra, rb, rr = st.resolve(a), st.resolve(b), st.resolve(r)
        # ?A = ?X/?Y, ?B = ?Y/?Z, 结果 ?X/?Z
        assert ra.slash == "/" and rb.slash == "/" and rr.slash == "/"
        assert ra.arg is rb.result
        assert rr.result is ra.result and rr.arg is rb.arg

    def test_B_with_feature(self):
        st = State()
        r = forward_composition(parse("(S[dcl]\\NP)/(S\\NP)"), parse("(S\\NP)/NP"), st)
        assert st.resolve(r) is parse("(S[dcl]\\NP)/NP")
        assert forward_composition(parse("S[dcl]/S[q]"), parse("S[dcl]/NP"), st) is None
