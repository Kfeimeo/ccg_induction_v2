"""M0/M2：全局状态（"增广矩阵"）—— 带 trail 的并查集 + 一阶合一 + occurs check + 域表 + 全局复杂度界。

设计约定：

- ``unify(a, b) -> bool``：成功返回 True 并把必要的合并记入 trail；失败返回 False，且**原子性**：
  失败时回滚到调用前（trail 长度不变）。
- ``resolve(c)``：深度替换所有已绑定变量，返回 hash-cons 后的范畴；未绑定变量替换为其代表元。
- 合并方向确定：两个自由变量合并时代表元永远是 **id 较小者**，配路径压缩；golden 输出不随操作顺序漂移。
- ``mark()`` / ``rollback(mark)`` 是公开原语。trail 记录每一次 parent / binding / domain 改动。
- **域表**（M2）：``domain(v)`` 是 ``Domain``（模式集合）或 None（= L 下全体候选）。挂在代表元上。
  var-var 合并 → 两域交集（两两 mgu，超 L 剪掉）；绑定到项 → 域过滤为与该项相容的模式；域空 → 失败。
- **全局复杂度界**（M2）：``complexity_bound`` 不为 None 时，任何变量 resolve 后的复杂度都不得超过它
  （不只是词型变量；模式内变量同样受限）。
"""
from __future__ import annotations

from typing import Iterable

from .category import Atom, Category, Functor, Var, _functor, complexity, subterms
from .domain import Domain, canonical, freshen


class State:
    def __init__(self, complexity_bound: int | None = None) -> None:
        self._parent: dict[Var, Var] = {}  # 缺省：自己是代表元
        self._binding: dict[Var, Category] = {}  # 只在代表元上
        self._domain: dict[Var, Domain] = {}  # 只在代表元上；缺省 None
        # trail 条目：("p", var, old_parent | None) / ("b", var, None) / ("d", var, old_domain | None)
        self._trail: list[tuple] = []
        self.complexity_bound = complexity_bound
        # 域检查开关：CKY 搜索期间关掉（每次 bind 都做域交集太贵），在推导完成时统一检查（见 local.solve_sentence）。
        self.domain_checks = True
        # 两两 mgu 的备忘：(L, canonical(p), canonical(q)) -> canonical 结果 或 None。模式是 alpha 等价类，结果只依赖 L。
        self._mgu_memo: dict[tuple, Category | None] = {}

    # ------------------------------------------------------------ trail
    def mark(self) -> int:
        return len(self._trail)

    def rollback(self, mark: int) -> None:
        if mark < 0 or mark > len(self._trail):
            raise ValueError(f"bad mark {mark}; trail length is {len(self._trail)}")
        while len(self._trail) > mark:
            kind, v, old = self._trail.pop()
            if kind == "p":
                if old is None:
                    del self._parent[v]
                else:
                    self._parent[v] = old
            elif kind == "b":
                del self._binding[v]
            else:
                if old is None:
                    del self._domain[v]
                else:
                    self._domain[v] = old

    def _set_parent(self, v: Var, p: Var) -> None:
        self._trail.append(("p", v, self._parent.get(v)))
        self._parent[v] = p

    def _bind(self, rep: Var, c: Category) -> None:
        self._trail.append(("b", rep, None))
        self._binding[rep] = c

    def _set_domain_rep(self, rep: Var, d: Domain) -> None:
        self._trail.append(("d", rep, self._domain.get(rep)))
        self._domain[rep] = d

    # ------------------------------------------------------------ union-find
    def _find(self, v: Var) -> Var:
        root = v
        while root in self._parent:
            root = self._parent[root]
        while v is not root:  # 路径压缩，记入 trail 以便回滚
            nxt = self._parent[v]
            if nxt is not root:
                self._set_parent(v, root)
            v = nxt
        return root

    def _walk(self, c: Category) -> Category:
        while isinstance(c, Var):
            rep = self._find(c)
            b = self._binding.get(rep)
            if b is None:
                return rep
            c = b
        return c

    # ------------------------------------------------------------ queries
    def resolve(self, c: Category) -> Category:
        c = self._walk(c)
        if isinstance(c, Functor):
            r, a = self.resolve(c.result), self.resolve(c.arg)
            if r is c.result and a is c.arg:
                return c
            return _functor(r, c.slash, a)
        return c

    def same(self, a: Category, b: Category) -> bool:
        return self.resolve(a) is self.resolve(b)

    def bindings(self) -> dict[Var, Category]:
        out: dict[Var, Category] = {}
        for v in set(self._parent) | set(self._binding):
            r = self.resolve(v)
            if r is not v:
                out[v] = r
        return out

    # ------------------------------------------------------------ domains
    def domain(self, v: Var) -> Domain | None:
        return self._domain.get(self._find(v))

    def set_domain(self, v: Var, patterns: Iterable[Category]) -> None:
        self._set_domain_rep(self._find(v), patterns if isinstance(patterns, Domain) else Domain(patterns))

    def restrict(self, v: Var, patterns: Iterable[Category]) -> bool | None:
        """域 ∩= patterns。返回 True=收缩了，False=没变，None=变空（冲突，域不变）。"""
        rep = self._find(v)
        cur = self._domain.get(rep)
        new = self.minimize(Domain(patterns)) if cur is None else self.intersect(cur, patterns)
        if len(new) == 0:
            return None
        if cur is not None and new.keys() == cur.keys():
            return False
        self._set_domain_rep(rep, new)
        return True

    def mgu_pattern(self, p: Category, q: Category, kp: str, kq: str) -> Category | None:
        """两个模式（视为 alpha 等价类）的 mgu，超 L 剪掉；备忘。"""
        key = (self.complexity_bound, kp, kq)
        if key in self._mgu_memo:
            return self._mgu_memo[key]
        m = self.mark()
        saved, self.domain_checks = self.domain_checks, False
        pf, qf = freshen(p), freshen(q)
        r = None
        if self.unify(pf, qf):
            r = self.resolve(pf)
            if self.complexity_bound is not None and complexity(r) > self.complexity_bound:
                r = None
        self.domain_checks = saved
        self.rollback(m)
        self._mgu_memo[key] = r
        return r

    def subsumes_pattern(self, general: Category, specific: Category, kg: str, ks: str) -> bool:
        """general 的某个实例是 specific（备忘）。"""
        key = ("sub", kg, ks)
        if key in self._mgu_memo:
            return self._mgu_memo[key]
        m = self.mark()
        saved, self.domain_checks = self.domain_checks, False
        sf = freshen(specific)
        from .category import free_vars
        ok = self.unify(freshen(general), sf)
        if ok:
            reps = [self.resolve(v) for v in free_vars(sf)]
            ok = all(isinstance(r, Var) for r in reps) and len(set(reps)) == len(reps)
        self.domain_checks = saved
        self.rollback(m)
        self._mgu_memo[key] = ok
        return ok

    def minimize(self, dom: Domain) -> Domain:
        """去掉被同域其他模式包含的模式（"是其中某个的实例"语义下冗余）。"""
        items = list(dom.items())
        keep = []
        for i, (ki, p) in enumerate(items):
            redundant = False
            for j, (kj, q) in enumerate(items):
                if j == i:
                    continue
                if self.subsumes_pattern(q, p, kj, ki) and not (self.subsumes_pattern(p, q, ki, kj) and j > i):
                    redundant = True
                    break
            if not redundant:
                keep.append((ki, p))
        return Domain(p for _, p in keep)

    def intersect(self, a: Iterable[Category], b: Iterable[Category]) -> Domain:
        """两模式集合的交：两两 mgu（在本状态的 L 界下），去重并极小化。"""
        a = a if isinstance(a, Domain) else Domain(a)
        b = b if isinstance(b, Domain) else Domain(b)
        out = []
        for kp, p in a.items():
            for kq, q in b.items():
                r = self.mgu_pattern(p, q, kp, kq)
                if r is not None:
                    out.append(r)
        return self.minimize(Domain(out))

    def compatible_any(self, dom: Domain, term: Category) -> bool:
        """term 是否与域中某个模式相容（备忘）。"""
        t = self.resolve(term)
        kt = canonical(t)
        return any(self.mgu_pattern(p, t, kp, kt) is not None for kp, p in dom.items())

    def check_domains(self, vars: Iterable[Var]) -> bool:
        """对给定变量逐个检查当前 resolve 结果与域相容（domain_checks 关掉期间用）。"""
        for v in vars:
            dom = self.domain(v)
            if dom is not None and not self.compatible_any(dom, v):
                return False
        return True

    def compatible(self, dom: Domain, term: Category) -> Domain:
        """域中与 term 相容的模式，实例化到 term 的形状（term 的变量泛化为模式变量）。"""
        return self.intersect(dom, [self.resolve(term)])

    # ------------------------------------------------------------ unify
    def unify(self, a: Category, b: Category) -> bool:
        m = self.mark()
        if self._unify(a, b) and self._within_bound():
            return True
        self.rollback(m)
        return False

    def _within_bound(self) -> bool:
        L = self.complexity_bound
        if L is None:
            return True
        return all(complexity(self.resolve(v)) <= L for v in self._binding)

    def _occurs(self, v: Var, c: Category) -> bool:
        return any(t is v for t in subterms(self.resolve(c)))

    def _unify(self, a: Category, b: Category) -> bool:
        a, b = self._walk(a), self._walk(b)
        if a is b:
            return True
        if isinstance(a, Var) and isinstance(b, Var):
            lo, hi = (a, b) if a.id < b.id else (b, a)
            da, db = self._domain.get(lo), self._domain.get(hi)
            self._set_parent(hi, lo)
            if not self.domain_checks:
                if db is not None and da is None:
                    self._set_domain_rep(lo, db)
                elif db is not None:
                    self._set_domain_rep(lo, Domain([*da, *db]))  # 暂存并集，完成时再检查；交集留给传播层 restrict
                return True
            if db is not None:
                new = db if da is None else self.intersect(da, db)
                if len(new) == 0:
                    return False
                self._set_domain_rep(lo, new)
            return True
        if isinstance(a, Var):
            return self._bind_var(a, b)
        if isinstance(b, Var):
            return self._bind_var(b, a)
        if isinstance(a, Atom) or isinstance(b, Atom):
            return False
        assert isinstance(a, Functor) and isinstance(b, Functor)
        if a.slash != b.slash:
            return False
        return self._unify(a.result, b.result) and self._unify(a.arg, b.arg)

    def _bind_var(self, v: Var, c: Category) -> bool:
        if self._occurs(v, c):
            return False
        self._bind(v, c)
        dom = self._domain.get(v)
        if dom is not None and self.domain_checks:
            new = self.compatible(dom, c)
            if len(new) == 0:
                return False
            if new.keys() != dom.keys():
                self._set_domain_rep(v, new)
        return True
