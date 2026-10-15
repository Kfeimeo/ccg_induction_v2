"""M2 测试（一）：域枚举 + State 的域表。

设计决定：
1. 域是**模式**集合（ground 范畴是特例），按规范形去重；None = L 下全体候选（隐式，不枚举）。
   ``enumerate_categories`` 只是报告工具（L=2 下按 论元数+深度 有 13 万+，见 docs/m2-addendum-3）。
2. ``State.set_domain(v, cats)`` / ``State.domain(v)``。域的改动记 trail，随 rollback 撤销。
3. 语义：var-var 合并 → 域取交集（两两 mgu）；绑定到项 → 域过滤为与该项相容的模式；域为空 → unify False（原子回滚）。
4. 模式内变量不受词型域约束，但受 ``State(complexity_bound=L)`` 的**全局**复杂度界约束（所有范畴变量）。
"""
import pytest

from ccg_solver.category import atom, complexity, fresh_var, is_admissible, parse
from ccg_solver.domain import DEFAULT_ATOMS, enumerate_categories
from ccg_solver.state import State


class TestEnumerate:
    def test_L0_is_atoms(self):
        assert set(enumerate_categories(0)) == {atom(n, f) for n, f in DEFAULT_ATOMS}

    def test_L1_count(self):
        # 6 原子 + 6·6·2 一元函子
        cats = enumerate_categories(1)
        assert len(cats) == 6 + 72
        assert parse("NP/N") in cats and parse("S[dcl]\\NP") in cats

    def test_L2_is_too_big_to_enumerate(self):
        # 数据：论元数+深度 ≤ 2 的候选，2 斜杠内截断已有 1806 个，3 斜杠内 12174 个，4 斜杠内 136512 个
        # （论元内部 arity 不计入复杂度）。域因此走模式表示，不枚举。
        cats = enumerate_categories(2)
        assert len(cats) == 6 + 72 + 1728
        assert len(enumerate_categories(2, max_slashes=3)) == 12174
        for g in ("(S[dcl]\\NP)/NP", "(S[dcl]/NP)\\NP", "N/N"):
            assert parse(g) in set(cats)
        # 副词 (S\NP)\(S\NP)：arity 2 + depth 1 = 3
        assert parse("(S[dcl]\\NP)\\(S[dcl]\\NP)") in set(enumerate_categories(3, max_slashes=3))

    @pytest.mark.parametrize("L", [0, 1])
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

    def test_pattern_domains_intersect_by_mgu(self):
        st, a, b = State(), fresh_var(), fresh_var()
        st.set_domain(a, {parse("?X/N"), parse("S[dcl]\\?Y")})
        st.set_domain(b, {parse("NP/?Z"), parse("?W\\NP")})
        assert st.unify(a, b)
        assert st.domain(a) == {parse("NP/N"), parse("S[dcl]\\NP")}

    def test_global_complexity_bound_applies_to_pattern_vars(self):
        st, env = State(complexity_bound=1), {}
        x = parse("?X", env)
        assert st.unify(x, parse("S[dcl]/?Y", env))
        m = st.mark()
        assert not st.unify(env["Y"], parse("NP/N"))  # X 会变成 S[dcl]/(NP/N)，复杂度 2 > 1
        assert st.mark() == m
        assert st.unify(env["Y"], parse("NP"))

    def test_global_bound_prunes_domain_intersection(self):
        st, a, b = State(complexity_bound=1), fresh_var(), fresh_var()
        st.set_domain(a, {parse("?X/N")})
        st.set_domain(b, {parse("(S[dcl]\\NP)/?Z"), parse("NP/?Z")})
        assert st.unify(a, b)
        assert st.domain(a) == {parse("NP/N")}  # (S[dcl]\NP)/N 复杂度 2 被剪

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
