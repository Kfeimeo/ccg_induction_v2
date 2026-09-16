"""M0 测试（三）：trail —— 决策点标记、逐条撤销、dump 接口。

本文件里体现的设计决定（请确认）：
1. ``mark()`` 返回 trail 位置（int），``undo_to(mark)`` 撤销到该位置；向前撤销抛 ``ValueError``。
2. ``bindings()`` 只列出非代表元变量，值为其 ``resolve`` 结果（可能是另一个变量）。
"""
import pytest

from ccg_solver.category import atom, fresh_var, parse
from ccg_solver.state import State


class TestMarkUndo:
    def test_undo_restores_binding(self):
        st = State()
        x = fresh_var()
        m = st.mark()
        assert st.unify(x, atom("NP"))
        assert st.resolve(x) is atom("NP")
        st.undo_to(m)
        assert st.resolve(x) is x
        assert st.mark() == m

    def test_nested_marks(self):
        st = State()
        x, y = fresh_var(), fresh_var()
        m1 = st.mark()
        assert st.unify(x, atom("NP"))
        m2 = st.mark()
        assert st.unify(y, atom("S"))
        st.undo_to(m2)
        assert st.resolve(x) is atom("NP")
        assert st.resolve(y) is y
        st.undo_to(m1)
        assert st.resolve(x) is x
        assert st.resolve(y) is y

    def test_undo_var_var_merge(self):
        st = State()
        x, y = fresh_var(), fresh_var()
        m = st.mark()
        assert st.unify(x, y)
        assert st.same(x, y)
        st.undo_to(m)
        assert not st.same(x, y)
        assert st.resolve(x) is x and st.resolve(y) is y

    def test_undo_then_rebind_differently(self):
        st = State()
        x = fresh_var()
        m = st.mark()
        assert st.unify(x, atom("NP"))
        st.undo_to(m)
        assert st.unify(x, atom("S"))
        assert st.resolve(x) is atom("S")

    def test_undo_restores_structural_bindings(self):
        st, env = State(), {}
        pattern = parse("(X\\NP)/Y", env)
        m = st.mark()
        assert st.unify(pattern, parse("(S\\NP)/NP"))
        assert st.resolve(pattern) is parse("(S\\NP)/NP")
        st.undo_to(m)
        assert st.resolve(env["X"]) is env["X"]
        assert st.resolve(env["Y"]) is env["Y"]
        assert st.resolve(pattern) is pattern

    def test_undo_restores_chain_partially(self):
        st, env = State(), {}
        x, y = parse("X", env), parse("Y", env)
        assert st.unify(x, y)
        m = st.mark()
        assert st.unify(y, atom("NP"))
        st.undo_to(m)
        assert st.same(x, y)  # 决策点之前的合并保留
        assert st.resolve(x) is not atom("NP")

    def test_undo_to_current_mark_is_noop(self):
        st = State()
        x = fresh_var()
        assert st.unify(x, atom("NP"))
        m = st.mark()
        st.undo_to(m)
        assert st.resolve(x) is atom("NP")

    def test_undo_forward_is_error(self):
        st = State()
        with pytest.raises(ValueError):
            st.undo_to(st.mark() + 1)

    def test_undo_below_zero_is_error(self):
        st = State()
        with pytest.raises(ValueError):
            st.undo_to(-1)

    def test_failed_unify_inside_decision_then_backtrack(self):
        """模拟决策—冲突—回退：失败的 unify 不留痕，再回退到决策点。"""
        st, env = State(), {}
        assert st.unify(parse("X", env), parse("NP/N"))
        m = st.mark()
        assert st.unify(parse("Y", env), parse("S\\NP"))
        assert not st.unify(parse("Y", env), parse("X", env))  # 冲突
        st.undo_to(m)
        assert st.resolve(env["Y"]) is env["Y"]
        assert st.resolve(env["X"]) is parse("NP/N")


class TestDump:
    def test_bindings_lists_bound_vars_only(self):
        st, env = State(), {}
        assert st.unify(parse("X", env), atom("NP"))
        assert st.unify(parse("Y", env), parse("X", env))
        unbound = fresh_var()
        assert st.bindings() == {env["X"]: atom("NP"), env["Y"]: atom("NP")}
        assert unbound not in st.bindings()

    def test_bindings_var_var_shows_representative(self):
        st = State()
        x, y = fresh_var(), fresh_var()
        assert st.unify(x, y)
        b = st.bindings()
        assert len(b) == 1
        (k, v), = b.items()
        assert {k, v} == {x, y}

    def test_bindings_empty_after_undo(self):
        st = State()
        x = fresh_var()
        m = st.mark()
        assert st.unify(x, parse("(S\\NP)/NP"))
        assert st.bindings() == {x: parse("(S\\NP)/NP")}
        st.undo_to(m)
        assert st.bindings() == {}
