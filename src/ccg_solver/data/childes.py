"""M3 语料：CHILDES Brown（Eve/Adam/Sarah）成人话语，CHAT 格式。

预处理（§7.4）：小写、不做词干还原、不用 subword。句末标点 ``.`` → dcl、``?`` → q，其他句末符号
（``!``、``+...`` 打断、``+/.`` 等）整句丢弃。CHAT 标记清理：``0v``/``0det`` 这类省略标记、``xxx``/``yyy``
不可辨、``&=laughs`` 事件、``[...]`` 注释、``(.)`` 停顿、``+<`` 重叠都删掉；``how_about`` 的下划线保留为一个 token。
频次截断（§7.3）：先在全部成人话语上数词频，只保留**所有词**出现次数 ≥ min_freq 的句子。
"""
from __future__ import annotations

import re
import zipfile
from collections import Counter
from pathlib import Path

from ..corpus import Sentence

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "childes"
CHILD_SPEAKER = "CHI"
_TOKEN_DROP = re.compile(r"^(0\w*|xxx|yyy|www|&.*|\+.*|\[.*\]|\(.*\)|<|>)$")
_STRIP = re.compile(r"[\[\]<>]")


def _raw_utterances(child: str) -> list[tuple[str, str]]:
    """(speaker, line) 列表，只含非 CHI 说话人；优先读解压目录，否则读 Brown.zip。"""
    out = []
    d = DATA_DIR / child
    if d.is_dir():
        files = sorted(d.glob("*.cha"))
        texts = [f.read_text(encoding="utf-8", errors="replace") for f in files]
    elif (DATA_DIR / "Brown.zip").exists():
        with zipfile.ZipFile(DATA_DIR / "Brown.zip") as z:
            names = sorted(n for n in z.namelist() if n.startswith(f"{child}/") and n.endswith(".cha"))
            texts = [z.read(n).decode("utf-8", errors="replace") for n in names]
    else:
        raise RuntimeError(f"CHILDES data not found under {DATA_DIR} (expected {child}/ or Brown.zip)")
    for text in texts:
        for line in text.splitlines():
            if line.startswith("*") and ":" in line:
                spk, body = line[1:].split(":", 1)
                if spk != CHILD_SPEAKER:
                    out.append((spk, body.strip()))
    return out


def _clean(body: str) -> tuple[list[str], str] | None:
    """返回 (words, feat) 或 None（无合格句末标点 / 清理后为空）。"""
    body = re.sub(r"\x15\d+_\d+\x15", "", body)  # 时间戳
    body = body.replace("‡", "").replace("„", "")
    toks = body.split()
    if not toks:
        return None
    end = toks[-1]
    if end == ".":
        feat = "dcl"
    elif end == "?":
        feat = "q"
    else:
        return None
    words = []
    for t in toks[:-1]:
        if _TOKEN_DROP.match(t):
            continue
        t = _STRIP.sub("", t)
        t = re.sub(r"@\w+$", "", t)  # @o 之类的后缀
        t = t.lower()
        if not t or not re.search(r"[a-z]", t):
            return None  # 含无法清理的符号，整句丢弃，宁缺毋滥
        words.append(t)
    return (words, feat) if words else None


def load_adult_utterances(child: str = "Eve", *, max_len: int = 8, min_freq: int = 20) -> list[Sentence]:
    cleaned = [c for _, body in _raw_utterances(child) if (c := _clean(body))]
    freq = Counter(w for words, _ in cleaned for w in words)
    out = []
    for words, feat in cleaned:
        if len(words) <= max_len and all(freq[w] >= min_freq for w in words):
            out.append(Sentence(tuple(words), feat))
    return out


def stats(sents: list[Sentence], L: int = 3) -> dict:
    types = {w for s in sents for w in s.words}
    n = len(sents)
    nondeg = sum(1 for s in sents if len(s) >= L + 2)
    return {
        "sentences": n,
        "types": len(types),
        "ratio": round(n / max(1, len(types)), 1),
        "mean_len": round(sum(len(s) for s in sents) / max(1, n), 2),
        "nondegenerate_frac": round(nondeg / max(1, n), 2),
        "questions_frac": round(sum(1 for s in sents if s.feat == "q") / max(1, n), 2),
    }
