"""M1：句子与词典（刚性：每个词型一个变量）。接口草案。"""
from __future__ import annotations

from dataclasses import dataclass

from .category import Category, Var
from .state import State

PUNCT_FEAT = {".": "dcl", "?": "q"}


@dataclass(frozen=True)
class Sentence:
    words: tuple[str, ...]
    feat: str  # 句末标点决定的 S 特征

    @classmethod
    def from_text(cls, text: str) -> "Sentence":
        """按空白切分，小写化；最后一个 token 必须是 ``.`` 或 ``?``，它决定 feat 且不进入 words。"""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.words)


class Lexicon:
    """词型 → 类型变量。变量按首次登记顺序分配，同一 Lexicon 内同词同变量。"""

    def var_of(self, word: str) -> Var:
        raise NotImplementedError

    def __contains__(self, word: str) -> bool:
        raise NotImplementedError

    def words(self) -> list[str]:
        """按登记顺序。"""
        raise NotImplementedError

    def categories(self, st: State) -> dict[str, Category]:
        """dump：每个词在当前状态下 resolve 后的范畴。"""
        raise NotImplementedError

    def bind(self, st: State, word: str, cat: Category) -> bool:
        """便捷：``st.unify(var_of(word), cat)``。"""
        raise NotImplementedError
