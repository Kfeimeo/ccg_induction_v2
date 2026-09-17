"""M1：句子与词典（刚性：每个词型一个变量）。"""
from __future__ import annotations

from dataclasses import dataclass

from .category import Category, Var, fresh_var
from .state import State

PUNCT_FEAT = {".": "dcl", "?": "q"}


@dataclass(frozen=True)
class Sentence:
    words: tuple[str, ...]
    feat: str  # 句末标点决定的 S 特征

    @classmethod
    def from_text(cls, text: str) -> "Sentence":
        """按空白切分，小写化；最后一个 token 必须是 ``.`` 或 ``?``，它决定 feat 且不进入 words。"""
        toks = text.lower().split()
        if len(toks) < 2 or toks[-1] not in PUNCT_FEAT:
            raise ValueError(f"sentence must end with one of {sorted(PUNCT_FEAT)}: {text!r}")
        words = tuple(toks[:-1])
        if any(w in PUNCT_FEAT for w in words):
            raise ValueError(f"punctuation inside sentence: {text!r}")
        return cls(words, PUNCT_FEAT[toks[-1]])

    def __len__(self) -> int:
        return len(self.words)


class Lexicon:
    """词型 → 类型变量。变量按首次登记顺序分配，同一 Lexicon 内同词同变量。"""

    def __init__(self) -> None:
        self._vars: dict[str, Var] = {}

    def var_of(self, word: str) -> Var:
        v = self._vars.get(word)
        if v is None:
            v = self._vars[word] = fresh_var()
        return v

    def __contains__(self, word: str) -> bool:
        return word in self._vars

    def words(self) -> list[str]:
        return list(self._vars)

    def categories(self, st: State) -> dict[str, Category]:
        return {w: st.resolve(v) for w, v in self._vars.items()}

    def bind(self, st: State, word: str, cat: Category) -> bool:
        return st.unify(self.var_of(word), cat)
