"""M0 测试（二）：一阶合一 + occurs check + 合同闭包。

本文件里体现的设计决定（请确认）：
1. ``State.unify`` 返回 bool；失败时**原子回滚**（不留下部分绑定），trail 长度不变。
2. ``S`` 与 ``S[dcl]`` 不合一（特征属于原子身份，见 test_m0_category）。
3. ``resolve`` 深度替换并返回 hash-cons 的范畴，因此可以用 ``is`` 比较。
"""
from ccg_solver.category import Var, atom, fresh_var, fwd, parse
from ccg_solver.state import State


class TestSpecCases:
    """里程碑表里明写的三个完成判据。"""

    def test_s_over_x_with_y_over_np(self):
        st, env = State(), {}
        assert st.unify(parse("S/X", env), parse("Y/NP", env))
        assert st.resolve(env["X"]) is atom("NP")
        assert st.resolve(env["Y"]) is atom("S")
        assert st.resolve(parse("S/X", env)) is parse("S/NP")
        assert st.resolve(parse("Y/NP", env)) is parse("S/NP")

    def test_direction_clash(self):
        st = State()
        assert not st.unify(parse("S/NP"), parse("S\\NP"))

    def test_occurs_check_direct(self):
        st, env = State(), {}
        x = parse("X", env)
        assert not st.unify(x, parse("S/X", env))
        assert not st.unify(parse("S/X", env), x)
        assert st.resolve(x) is x


class TestBasic:
    def test_identical_is_noop(self):
        st = State()
        m = st.mark()
        assert st.unify(parse("(S\\NP)/NP"), parse("(S\\NP)/NP"))
        v = fresh_var()
        assert st.unify(v, v)
        assert st.mark() == m

    def test_atom_mismatch(self):
        st = State()
        assert not st.unify(atom("S"), atom("NP"))
        assert not st.unify(atom("S", "dcl"), atom("S", "q"))
        assert not st.unify(atom("S", "dcl"), atom("S"))  # 特征属于原子身份
        assert not st.unify(atom("S"), parse("S/NP"))  # 原子 vs 函子

    def test_bind_var_to_atom_and_functor(self):
        st = State()
        x, y = fresh_var(), fresh_var()
        m = st.mark()
        assert st.unify(x, atom("NP"))
        assert st.unify(parse("(S\\NP)/NP"), y)
        assert st.resolve(x) is atom("NP")
        assert st.resolve(y) is parse("(S\\NP)/NP")
        assert st.mark() > m  # 成功的合并记入 trail

    def test_var_var_then_bind(self):
        st = State()
        x, y = fresh_var(), fresh_var()
        assert st.unify(x, y)
        assert st.same(x, y)
        assert st.resolve(x) in (x, y) and st.resolve(x) is st.resolve(y)
        assert st.unify(y, atom("NP"))
        assert st.resolve(x) is atom("NP")

    def test_bound_var_against_partial_structure(self):
        st, env = State(), {}
        assert st.unify(parse("X", env), parse("(S\\NP)/NP"))
        assert st.unify(parse("X", env), parse("(S\\NP)/Z", env))
        assert st.resolve(env["Z"]) is atom("NP")
        assert not st.unify(parse("X", env), parse("(S/NP)/Z", env))

    def test_compound_with_vars_both_sides(self):
        st, env = State(), {}
        assert st.unify(parse("(X\\NP)/Y", env), parse("(S[dcl]\\Z)/NP", env))
        assert st.resolve(env["X"]) is atom("S", "dcl")
        assert st.resolve(env["Y"]) is atom("NP")
        assert st.resolve(env["Z"]) is atom("NP")

    def test_resolve_follows_chains_deeply(self):
        st, env = State(), {}
        x, y, z = (parse(n, env) for n in "XYZ")
        assert st.unify(x, y)
        assert st.unify(y, z)
        assert st.unify(z, atom("NP"))
        assert st.resolve(parse("(X\\Y)/Z", env)) is parse("(NP\\NP)/NP")

    def test_resolve_keeps_unbound_vars(self):
        st = State()
        x = fresh_var()
        c = fwd(atom("S"), x)
        assert st.resolve(c) is c
        assert st.resolve(x) is x

    def test_most_general_unifier_binds_only_what_is_needed(self):
        st, env = State(), {}
        assert st.unify(parse("X/Y", env), parse("Z/NP", env))
        assert st.resolve(env["Y"]) is atom("NP")
        assert st.same(env["X"], env["Z"])
        assert isinstance(st.resolve(env["X"]), Var)  # X 仍是自由变量


class TestOccursCheck:
    def test_through_binding_chain(self):
        st, env = State(), {}
        assert st.unify(parse("X", env), parse("S/Y", env))
        assert not st.unify(parse("Y", env), parse("X", env))

    def test_var_var_then_cycle(self):
        st, env = State(), {}
        assert st.unify(parse("X", env), parse("Y", env))
        assert not st.unify(parse("X", env), parse("S/Y", env))

    def test_deep_cycle(self):
        st, env = State(), {}
        assert st.unify(parse("X", env), parse("S/Y", env))
        assert st.unify(parse("Y", env), parse("NP/Z", env))
        assert not st.unify(parse("Z", env), parse("X", env))
        assert not st.unify(parse("Z", env), parse("PP\\X", env))


class TestCongruence:
    def test_forward_closure(self):
        st, env = State(), {}
        assert st.unify(parse("X", env), parse("Y", env))
        assert st.same(parse("S/X", env), parse("S/Y", env))
        assert st.resolve(parse("S/X", env)) is st.resolve(parse("S/Y", env))

    def test_backward_closure_injectivity(self):
        st, env = State(), {}
        assert st.unify(parse("S/X", env), parse("S/Y", env))
        assert st.same(env["X"], env["Y"])

    def test_merging_vars_bound_to_compounds_merges_subterms(self):
        st, env = State(), {}
        assert st.unify(parse("A", env), parse("S/X", env))
        assert st.unify(parse("B", env), parse("S/Y", env))
        assert st.unify(parse("A", env), parse("B", env))
        assert st.same(env["X"], env["Y"])
        assert st.unify(env["X"], atom("NP"))
        assert st.resolve(env["B"]) is parse("S/NP")

    def test_merging_vars_bound_to_incompatible_compounds_fails(self):
        st, env = State(), {}
        assert st.unify(parse("A", env), parse("S/NP"))
        assert st.unify(parse("B", env), parse("S\\NP"))
        assert not st.unify(parse("A", env), parse("B", env))


class TestAtomicity:
    def test_failed_unify_leaves_no_partial_bindings(self):
        st, env = State(), {}
        m = st.mark()
        # 从左到右：X 先绑到 NP，随后 S vs NP 失败 —— X 必须回到未绑定
        assert not st.unify(parse("(X\\NP)/S", env), parse("(NP\\NP)/NP"))
        assert st.resolve(env["X"]) is env["X"]
        assert st.mark() == m

    def test_failed_unify_does_not_poison_state(self):
        st, env = State(), {}
        assert not st.unify(parse("(X\\NP)/S", env), parse("(NP\\NP)/NP"))
        assert st.unify(parse("X", env), atom("S"))
        assert st.resolve(env["X"]) is atom("S")

    def test_failed_occurs_check_leaves_no_partial_bindings(self):
        st, env = State(), {}
        m = st.mark()
        assert not st.unify(parse("(X\\Y)/Y", env), parse("(NP\\Z)/(S/Z)", env))
        assert st.resolve(env["X"]) is env["X"]
        assert st.resolve(env["Y"]) is env["Y"]
        assert st.mark() == m
