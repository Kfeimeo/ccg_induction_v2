"""M2 测试（二）：backbone（joint lgg）与域投影。

设计决定（请确认）：
1. backbone 是所有 σ 元组的最小公共泛化（anti-unification），跨词共享结构保留：
   若每条 σ 里 the 的论元都等于 cat 的范畴，backbone 里它们是同一个变量。
2. projections[w] = D 里 w 位置的模式集，与当前域取交（模式域，见 test_m2_domain）。无域时就是模式集本身。
"""
from ccg_solver.category import Var, atom, free_vars, parse
from ccg_solver.corpus import Lexicon, Sentence
from ccg_solver.domain import enumerate_categories
from ccg_solver.local import analyze_sentence, lgg, solve_sentence
from ccg_solver.state import State

from test_m1_local import GOLD, gold_state


def P(*texts):
    env = {}
    return tuple(parse(t, env) for t in texts)


class TestLgg:
    def test_identical(self):
        p = P("NP/N", "N")
        assert lgg([p, p]) == p

    def test_disagreement_becomes_var(self):
        g = lgg([P("NP/N", "N"), P("NP/N", "NP")])
        assert g[0] is parse("NP/N")
        assert isinstance(g[1], Var)

    def test_shared_structure_kept(self):
        # the=?a/?b, cat=?b  与  the=NP/N, cat=N  → the=?x/?y, cat=?y（同一个 y）
        g = lgg([P("?A/?B", "?B"), P("NP/N", "N")])
        assert g[0].arg is g[1]
        assert isinstance(g[0].result, Var)

    def test_same_pair_reuses_var(self):
        g = lgg([P("N/N", "N"), P("NP/NP", "NP")])
        # (N,NP) 出现三次，都映射到同一变量
        v = g[1]
        assert g[0].result is v and g[0].arg is v

    def test_functor_vs_atom(self):
        g = lgg([P("NP/N"), P("NP")])
        assert isinstance(g[0], Var)

    def test_slash_mismatch(self):
        g = lgg([P("S[dcl]\\NP"), P("S[dcl]/NP")])
        assert isinstance(g[0], Var)


class TestBackbone:
    def test_fixed_lexicon_backbone_is_exact(self):
        lex, st = gold_state()
        r = analyze_sentence(Sentence.from_text("the cat sees the dog ."), lex, st)
        assert len(r.derivations) == 1
        assert r.backbone == {w: parse(GOLD[w]) for w in ("the", "cat", "sees", "dog")}

    def test_all_unknown_backbone(self):
        lex, st = Lexicon(), State()
        r = analyze_sentence(Sentence.from_text("the cat sleeps ."), lex, st)
        assert len(r.derivations) > 1
        bb = r.backbone
        # 数据：三个词全自由时 backbone 完全平凡（sleeps 也可以是原子：the=S/?, cat=?/?2, sleeps=?2）。
        # 这正是只靠等式传播传不出信息、需要域过滤的原因（增补 3）。
        assert all(isinstance(bb[w], Var) for w in ("the", "cat", "sleeps"))
        assert len({bb[w] for w in bb}) == 3

    def test_backbone_is_implied_by_every_sigma(self):
        lex, st = Lexicon(), State()
        r = analyze_sentence(Sentence.from_text("the cat sees the dog ."), lex, st)
        # apply_backbone 之后每条 σ 仍可 apply：backbone 不排除任何存活推导
        m = st.mark()
        assert r.apply_backbone(st)
        for d in r.derivations:
            m2 = st.mark()
            assert d.apply(st), d.key
            st.rollback(m2)
        st.rollback(m)

    def test_apply_backbone_then_solve_unchanged(self):
        lex, st = Lexicon(), State()
        sent = Sentence.from_text("the cat sees the dog quickly .")
        r = analyze_sentence(sent, lex, st)
        before = [d.key for d in r.derivations]
        assert r.apply_backbone(st)
        after = [d.key for d in solve_sentence(sent, lex, st)]
        assert len(after) == len(before)  # backbone 是所有 σ 共有的，不丢解

    def test_partial_lexicon_backbone(self):
        lex, st = gold_state(skip=("sees",))
        r = analyze_sentence(Sentence.from_text("the cat sees the dog ."), lex, st)
        bb = r.backbone["sees"]
        assert isinstance(bb, Var)  # 14 个候选形状各异，没有公共结构
        lex, st = gold_state(skip=("sleeps",))
        r = analyze_sentence(Sentence.from_text("the cat sleeps ."), lex, st)
        # 两个 σ：S[dcl]\NP 与 (S[dcl]\(NP/N))\N（[the [cat sleeps]]），公共部分只有后向斜杠
        assert {d.key[2] for d in r.derivations} == {"S[dcl]\\NP", "(S[dcl]\\(NP/N))\\N"}
        assert r.backbone["sleeps"].slash == "\\"
        assert isinstance(r.backbone["sleeps"].result, Var) and isinstance(r.backbone["sleeps"].arg, Var)


class TestProjection:
    def test_no_domain_projection_is_pattern_set(self):
        lex, st = gold_state(skip=("sees",))
        r = analyze_sentence(Sentence.from_text("the cat sees the dog ."), lex, st)
        assert len(r.projections["sees"]) == 14
        assert r.projections["the"] == {parse("NP/N")}

    def test_projection_at_L2_via_global_bound(self):
        lex, st = gold_state(skip=("sees",))
        st.complexity_bound = 2
        r = analyze_sentence(Sentence.from_text("the cat sees the dog ."), lex, st)
        assert r.projections["sees"] == {parse("(S[dcl]\\NP)/NP"), parse("(S[dcl]/NP)\\NP")}

    def test_projection_is_subset_of_domain_and_covers_gold(self):
        lex, st = Lexicon(), State()
        dom = set(enumerate_categories(1))
        sent = Sentence.from_text("the cat sleeps .")
        for w in sent.words:
            st.set_domain(lex.var_of(w), dom)
        r = analyze_sentence(sent, lex, st)
        for w in ("the", "cat", "sleeps"):
            assert r.projections[w] <= dom
            assert parse(GOLD[w]) in r.projections[w]
        # 三个词全自由时 L=1 的域挡不住任何东西（每个 ground 候选都有某条 σ 相容）：数据
        assert len(r.projections["sleeps"]) == len(dom)
        # 固定 the/cat 后 sleeps 的投影只剩 S[dcl]\NP（另一候选复杂度 2，不在 L=1 域内）
        lex, st = gold_state(skip=("sleeps",))
        st.set_domain(lex.var_of("sleeps"), dom)
        r = analyze_sentence(sent, lex, st)
        assert r.projections["sleeps"] == {parse("S[dcl]\\NP")}

    def test_projection_respects_existing_domain(self):
        lex, st = gold_state(skip=("sees",))
        st.set_domain(lex.var_of("sees"), {parse("(S[dcl]\\NP)/NP"), parse("N")})
        st.set_domain(lex.var_of("sees"), {parse("(S[dcl]\\NP)/NP"), parse("N")})
        r = analyze_sentence(Sentence.from_text("the cat sees the dog ."), lex, st)
        assert r.projections["sees"] == {parse("(S[dcl]\\NP)/NP")}
