"""实验版本：一阶范畴 + 定义原子 VP + 二阶组合（语法档案 FIRST_ORDER_VP）。

设计决定（请确认）：
1. 档案是显式对象：``State(profile=)`` / ``solve_sentence(profile=)``。不传 = LEGACY，M0–M3 全部行为不变。
2. 定义原子：合一时展开（``VP`` 与 ``S[dcl]\\NP`` 合一成功），resolve/打印时折叠（``S[dcl]\\NP`` 永远显示为 ``VP``）。
   因此 σ 的去重键里不会同时出现 ``VP`` 和 ``S[dcl]\\NP`` 两种写法。
3. 一阶 = 论元全是原子（depth 0），结果位置可嵌套，arity ≤ 4 不变。``(S\\NP)\\(S\\NP)`` 非法，``VP\\VP`` 合法。
4. 规则集 > < >B <B >B² <B²；二阶组合定义为 ``X/Y (Y/Z)/W → (X/Z)/W`` 及镜像。无 T、无 B×。
5. 新原子只对 FIRST_ORDER_VP 生效的方式：``parse("VP")`` 在档案 activate 之后可用；LEGACY 下 ``VP`` 不出现在任何解里，
   因为 LEGACY 的 State 不展开定义，``VP`` 只是一个没人用的普通原子。
"""
import pytest

from ccg_solver.category import Var, atom, complexity, depth, parse
from ccg_solver.corpus import Lexicon, Sentence
from ccg_solver.local import solve_sentence
from ccg_solver.profile import FIRST_ORDER_VP, LEGACY, Profile
from ccg_solver.rules import APPLICATION, COMPOSITION2, backward_composition2, forward_composition2
from ccg_solver.state import State
from ccg_solver.toy import M2_CORPUS, M2_GOLD
from ccg_solver.toy_v2 import M2_GOLD_V2

P = FIRST_ORDER_VP.activate()
SENTS = [Sentence.from_text(t) for t in M2_CORPUS]


def gold_state_v2(skip=()):
    lex, st = Lexicon(), State(profile=P)
    for w in M2_GOLD_V2:
        lex.var_of(w)
    for w, c in M2_GOLD_V2.items():
        if w not in skip:
            assert lex.bind(st, w, parse(c)), w
    return lex, st


class TestProfileObject:
    def test_legacy_is_default(self):
        assert State().profile is LEGACY
        assert LEGACY.max_depth == 3 and LEGACY.definitions == {} and set(LEGACY.rules) == {">", "<"}

    def test_first_order_profile(self):
        assert set(P.rules) == {">", "<", ">B", "<B", ">B2", "<B2"}
        assert P.max_depth == 0 and P.max_arity == 4
        assert P.definition("VP") is parse("S[dcl]\\NP")
        assert P.definition("NP") is None

    def test_activate_registers_atom_idempotently(self):
        assert atom("VP") is atom("VP")
        assert FIRST_ORDER_VP.activate() is P
        assert parse("VP\\VP").arg is atom("VP")

    def test_no_type_raising_no_crossed(self):
        assert not {n for n in P.rules if "T" in n or "x" in n.lower()}


class TestFold:
    def test_fold_definition(self):
        assert P.fold(parse("S[dcl]\\NP")) is atom("VP")
        assert P.fold(parse("(S[dcl]\\NP)/NP")) is parse("VP/NP")
        assert P.fold(parse("(S[dcl]\\NP)\\(S[dcl]\\NP)")) is parse("VP\\VP")
        assert P.fold(parse("NP/N")) is parse("NP/N")
        assert LEGACY.fold(parse("S[dcl]\\NP")) is parse("S[dcl]\\NP")

    def test_admissible_first_order(self):
        assert P.admissible(parse("VP\\VP")) and P.admissible(parse("(VP/NP)/PP")) and P.admissible(atom("VP"))
        assert not P.admissible(parse("S[dcl]/(S[dcl]\\N)"))  # 论元是函子
        assert P.admissible(parse("(S[dcl]\\NP)\\(S[dcl]\\NP)"))  # 折叠后是 VP\\VP，合法
        assert not P.admissible(parse("(S[q]\\NP)\\(S[q]\\NP)"))  # 没有对应定义，折不掉
        assert not P.admissible(parse("((((VP/NP)/NP)/NP)/NP)/NP"))  # arity 5


class TestUnifyWithDefinitions:
    def test_atom_vs_expansion(self):
        st = State(profile=P)
        assert st.unify(atom("VP"), parse("S[dcl]\\NP"))
        assert st.unify(parse("S[dcl]\\NP"), atom("VP"))

    def test_atom_vs_functor_pattern(self):
        st, env = State(profile=P), {}
        assert st.unify(atom("VP"), parse("?X\\?Y", env))
        assert st.resolve(env["X"]) is atom("S", "dcl") and st.resolve(env["Y"]) is atom("NP")
        assert not st.unify(atom("VP"), parse("?X/?Y"))  # 方向不对
        assert not st.unify(atom("VP"), parse("S[q]\\NP"))

    def test_resolve_folds(self):
        st, env = State(profile=P), {}
        x = parse("?X", env)
        assert st.unify(x, parse("S[dcl]\\NP"))
        assert st.resolve(x) is atom("VP")
        y = parse("?Y", env)
        assert st.unify(y, parse("(S[dcl]\\NP)/NP"))
        assert st.resolve(y) is parse("VP/NP")

    def test_legacy_state_does_not_expand(self):
        st = State()  # LEGACY
        assert not st.unify(atom("VP"), parse("S[dcl]\\NP"))
        assert st.resolve(parse("S[dcl]\\NP")) is parse("S[dcl]\\NP")

    def test_first_order_bound_enforced_in_unify(self):
        st, env = State(profile=P), {}
        x = parse("?X", env)
        assert not st.unify(x, parse("S[dcl]/(S[q]\\NP)"))  # 一阶违规
        assert st.unify(x, parse("S[dcl]/(S[dcl]\\NP)"))  # 折叠后 S[dcl]/VP，合法
        assert st.resolve(x) is parse("S[dcl]/VP")


class TestRules:
    def test_np_vp_to_s(self):
        st = State(profile=P)
        r = APPLICATION["<"](atom("NP"), atom("VP"), st)
        assert r is not None and st.resolve(r) is atom("S", "dcl")

    def test_vp_modifier(self):
        st = State(profile=P)
        r = APPLICATION["<"](atom("VP"), parse("VP\\VP"), st)
        assert st.resolve(r) is atom("VP")

    def test_forward_composition2(self):
        st = State(profile=P)
        r = forward_composition2(parse("S[dcl]/VP"), parse("(VP/NP)/PP"), st)
        assert st.resolve(r) is parse("(S[dcl]/NP)/PP")
        assert forward_composition2(parse("S[dcl]/VP"), parse("VP/NP"), st) is None  # 一阶组合不归 B² 管

    def test_backward_composition2(self):
        st = State(profile=P)
        r = backward_composition2(parse("(VP\\NP)\\PP"), parse("S[dcl]\\VP"), st)
        assert st.resolve(r) is parse("VP\\PP")  # 折叠后

    def test_composition2_on_free_vars(self):
        st, env = State(profile=P), {}
        a, b = parse("?A", env), parse("?B", env)
        r = forward_composition2(a, b, st)
        ra, rb, rr = st.resolve(a), st.resolve(b), st.resolve(r)
        assert ra.slash == "/" and rb.slash == "/" and rb.result.slash == "/" and rr.result.slash == "/"
        assert ra.arg is rb.result.result


class TestToyCorpusV2:
    def test_gold_v2_is_first_order_and_informative(self):
        for w, c in M2_GOLD_V2.items():
            assert P.admissible(parse(c)), w
            assert P.fold(parse(M2_GOLD[w])) is parse(c), w  # 折叠旧金标准正好得到新金标准

    def test_every_sentence_one_sigma_under_gold(self):
        lex, st = gold_state_v2()
        for s in SENTS:
            ds = solve_sentence(s, lex, st, profile=P)
            assert len(ds) == 1, s
            assert all("S[dcl]\\NP" not in k for k in ds[0].key)  # 键里只出现折叠形式

    def test_legacy_suites_unaffected(self):
        lex, st = Lexicon(), State()
        ds = solve_sentence(Sentence.from_text("the cat sleeps ."), lex, st)
        assert len(ds) == 8  # M1 golden（LEGACY, L 无界）


class TestSymmetryUnderV2:
    """增补 2 的破对称证据在一阶 + VP 原子下是否仍然成立：实证，记录结果。"""

    def test_sees_candidates_with_vp_evidence(self):
        from test_m1_local import _cat
        lex, st = gold_state_v2(skip=("sees",))
        sets = []
        for text in ("the cat sees the dog .", "the cat sees the dog quickly ."):
            s = Sentence.from_text(text)
            ds = solve_sentence(s, lex, st, profile=P)
            sets.append({d.key[2] for d in ds})
        cands = set.intersection(*sets)
        lo = min(complexity(_cat(k)) for k in cands)
        minimal = {k for k in cands if complexity(_cat(k)) == lo}
        print(f"\nv2 sees candidates={sorted(cands)} minimal={minimal}")
        assert cands == {"VP/NP"}  # 数据：一阶 + VP 证据下 sees 唯一，主宾对称消失（LEGACY 下最小集有两个）


# 留出词在单句下的 (|D|, 树数)。LEGACY 对照：the (2,2)、sees (14,14)、quickly (9,9)
GOLDEN_V2 = {
    ("the", "the cat sleeps ."): (2, 2),
    ("sees", "the cat sees the dog ."): (2, 4),
    ("quickly", "the cat sees the dog quickly ."): (4, 9),
}


@pytest.mark.parametrize("word, text", [("the", "the cat sleeps ."), ("sees", "the cat sees the dog ."), ("quickly", "the cat sees the dog quickly .")])
def test_golden_counts_v2(word, text):
    lex, st = gold_state_v2(skip=(word,))
    ds = solve_sentence(Sentence.from_text(text), lex, st, profile=P)
    key = (word, text)
    print(f"\nv2 golden {key}: ({len(ds)}, {sum(d.n_trees for d in ds)})")
    if GOLDEN_V2:
        assert (len(ds), sum(d.n_trees for d in ds)) == GOLDEN_V2[key]
