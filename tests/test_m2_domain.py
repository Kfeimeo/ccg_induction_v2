"""M2 测试（一）：域枚举 + State 的域表。

设计决定（请确认）：
1. 域是 ground 范畴的集合，用 ``DEFAULT_ATOMS``（S[dcl] S[q] N NP PP conj，不含裸 S）枚举，complexity ≤ L。
2. ``State.set_domain(v, cats)`` / ``State.domain(v)``（无域返回 None = 不受限）。域的改动记 trail，随 rollback 撤销。
3. 语义：var-var 合并 → 域取交集；绑定到 ground 范畴 → 必须在域内；绑定到含变量的模式 → 域过滤为该模式的实例。
   域为空 → unify 返回 False（原子回滚）。
"""
import pytest

from ccg_solver.category import atom, complexity, fresh_var, is_admissible, parse
from ccg_solver.domain import DEFAULT_ATOMS, enumerate_categories
from ccg_solver.state import State


class TestEnumerate:
    def test_L0_is_atoms(self):
        assert enumerate_categories(0) == [atom(n, f) for n, f in DEFAULT_ATOMS]

    def test_L1_count(self):
        # 6 原子 + 6·6·2 一元函子
        cats = enumerate_categories(1)
        assert len(cats) == 6 + 72
        assert parse("NP/N") in cats and parse("S[dcl]\\NP") in cats

    def test_L2_contains_gold_shapes(self):
        cats = set(enumerate_categories(2))
        for g in ("(S[dcl]\\NP)/NP", "(S[dcl]/NP)\\NP", "(S[dcl]\\NP)\\(S[dcl]\\NP)", "N/N"):
            assert parse(g) in cats
        assert parse("((S[dcl]\\NP)/NP)/NP") not in cats  # 复杂度 3

    @pytest.mark.parametrize("L", [0, 1, 2])
    def test_bounds_and_order(self, L):
        cats = enumerate_categories(L)
        assert all(complexity(c) <= L and is_admissible(c) for c in cats)
        assert len(set(cats)) == len(cats)
        assert cats == sorted(cats, key=lambda c: (complexity(c), str(c)))
        assert set(enumerate_categories(L)) <= set(enumerate_categories(L + 1))

    def test_deterministic(self):
        assert enumerate_categories(2) == enumerate_categories(2)


class TestStateDomains:
    def test_no_domain_means_unrestricted(self):
        st, v = State(), fresh_var()
        assert st.domain(v) is None
        assert st.unify(v, parse("((S[dcl]\\NP)/NP)/NP"))

    def test_ground_bind_must_be_in_domain(self):
        st, v = State(), fresh_var()
        st.set_domain(v, {parse("NP/N"), parse("N")})
        m = st.mark()
        assert not st.unify(v, parse("S[dcl]\\NP"))
        assert st.mark() == m
        assert st.unify(v, parse("NP/N"))

    def test_var_var_merge_intersects(self):
        st, a, b = State(), fresh_var(), fresh_var()
        st.set_domain(a, {parse("NP/N"), parse("N"), parse("N/N")})
        st.set_domain(b, {parse("N"), parse("N/N"), parse("NP")})
        assert st.unify(a, b)
        assert st.domain(a) == st.domain(b) == {parse("N"), parse("N/N")}

    def test_empty_intersection_is_conflict(self):
        st, a, b = State(), fresh_var(), fresh_var()
        st.set_domain(a, {parse("NP/N")})
        st.set_domain(b, {parse("N")})
        m = st.mark()
        assert not st.unify(a, b)
        assert st.mark() == m
        assert not st.same(a, b)

    def test_pattern_bind_filters_domain(self):
        st, v, env = State(), fresh_var(), {}
        st.set_domain(v, {parse("NP/N"), parse("N/N"), parse("S[dcl]\\NP"), parse("N")})
        assert st.unify(v, parse("?X/N", env))
        # v 现在是 ?X/N；域过滤到该模式的实例
        assert st.domain(v) == {parse("NP/N"), parse("N/N")}

    def test_pattern_bind_with_no_instances_is_conflict(self):
        st, v = State(), fresh_var()
        st.set_domain(v, {parse("N"), parse("NP")})
        assert not st.unify(v, parse("?X/?Y"))

    def test_singleton_domain_does_not_auto_bind(self):
        """域缩到 1 不自动绑定（保持"域"和"等式"两层分离，绑定由传播层决定）。"""
        st, v = State(), fresh_var()
        st.set_domain(v, {parse("N")})
        assert st.resolve(v) is v

    def test_rollback_restores_domain(self):
        st, a, b = State(), fresh_var(), fresh_var()
        st.set_domain(a, {parse("N"), parse("NP")})
        st.set_domain(b, {parse("N")})
        m = st.mark()
        assert st.unify(a, b)
        assert st.domain(a) == {parse("N")}
        st.rollback(m)
        assert st.domain(a) == {parse("N"), parse("NP")}
        assert st.domain(b) == {parse("N")}

    def test_restrict_records_shrink(self):
        st, v = State(), fresh_var()
        st.set_domain(v, {parse("N"), parse("NP"), parse("NP/N")})
        assert st.restrict(v, {parse("N"), parse("NP"), parse("S[dcl]")}) is True  # 收缩了
        assert st.domain(v) == {parse("N"), parse("NP")}
        assert st.restrict(v, {parse("N"), parse("NP")}) is False  # 没收缩
        assert st.restrict(v, {parse("S[dcl]")}) is None  # 变空：冲突，域不变
        assert st.domain(v) == {parse("N"), parse("NP")}

    def test_domain_of_bound_var_follows_representative(self):
        st, a, b = State(), fresh_var(), fresh_var()
        st.set_domain(b, {parse("N")})
        assert st.unify(a, b)
        assert st.domain(a) == {parse("N")}
        assert st.unify(a, parse("N"))
