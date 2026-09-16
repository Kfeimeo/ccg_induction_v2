"""M0：全局状态（"增广矩阵"）—— 带 trail 的并查集 + 一阶合一 + occurs check。

设计约定：

- ``unify(a, b) -> bool``：成功返回 True 并把必要的合并记入 trail；失败返回 False，且**原子性**：
  失败时回滚到调用前（trail 长度不变）。
- ``resolve(c)``：深度替换所有已绑定变量，返回 hash-cons 后的范畴；未绑定变量替换为其代表元。
- ``same(a, b)``：``resolve(a) is resolve(b)``。
- 合同闭包：前向（``X = Y`` ⇒ ``S/X = S/Y``）由 hash-cons + resolve 自动成立；
  反向（两复合项相等 ⇒ 子项对应合并）由 unify 递归完成。
- occurs check：变量不得（直接或经绑定链）出现在将要绑定给它的项中。
- 合并方向确定：两个自由变量合并时代表元永远是 **id 较小者**（不按秩），配路径压缩；
  golden 输出不随操作顺序漂移。
- ``mark()`` / ``rollback(mark)`` 是公开原语：决策层记录决策点、冲突时回退。trail 记录每一次
  parent / binding 改动（含路径压缩），逐条撤销。
- ``bindings()``：dump 接口，列出所有非代表元变量到其 ``resolve`` 结果的映射。
"""
from __future__ import annotations

from .category import Atom, Category, Functor, Var, _functor, subterms


class State:
    def __init__(self) -> None:
        self._parent: dict[Var, Var] = {}  # 缺省：自己是代表元
        self._binding: dict[Var, Category] = {}  # 只在代表元上
        # trail 条目：("p", var, old_parent | None) / ("b", var)
        self._trail: list[tuple] = []

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
            else:
                del self._binding[v]

    def _set_parent(self, v: Var, p: Var) -> None:
        self._trail.append(("p", v, self._parent.get(v)))
        self._parent[v] = p

    def _bind(self, rep: Var, c: Category) -> None:
        self._trail.append(("b", rep, None))
        self._binding[rep] = c

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
        """只解一层：变量 → 代表元或其绑定项。"""
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

    # ------------------------------------------------------------ unify
    def unify(self, a: Category, b: Category) -> bool:
        m = self.mark()
        if self._unify(a, b):
            return True
        self.rollback(m)
        return False

    def _occurs(self, v: Var, c: Category) -> bool:
        return any(t is v for t in subterms(self.resolve(c)))

    def _unify(self, a: Category, b: Category) -> bool:
        a, b = self._walk(a), self._walk(b)
        if a is b:
            return True
        if isinstance(a, Var) and isinstance(b, Var):
            lo, hi = (a, b) if a.id < b.id else (b, a)
            self._set_parent(hi, lo)
            return True
        if isinstance(a, Var):
            return self._bind_var(a, b)
        if isinstance(b, Var):
            return self._bind_var(b, a)
        if isinstance(a, Atom) or isinstance(b, Atom):
            return False  # 两个不同原子，或原子 vs 函子
        assert isinstance(a, Functor) and isinstance(b, Functor)
        if a.slash != b.slash:
            return False
        return self._unify(a.result, b.result) and self._unify(a.arg, b.arg)

    def _bind_var(self, v: Var, c: Category) -> bool:
        if self._occurs(v, c):
            return False
        self._bind(v, c)
        return True
