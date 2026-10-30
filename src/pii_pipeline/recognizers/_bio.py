"""Token/char-span helpers for BIO token-classification models.

Ported from token_classification_ft/src/tokenize_bio.py and predict.py (the parts the v5.0 release
uses: plain softmax head, max merge across windows, first-sub-token labelling, visible line breaks,
script word boundaries, punctuation trim). Kept self-contained so the pipeline does not import the
sibling repo.
"""
from __future__ import annotations

IGNORE_ID = -100

OPEN, CLOSE = "([{【「『（［", ")]}】」』）］"
PAIRS = dict(zip(CLOSE, OPEN))
LEAD = "、，,：:;；·・"
# trailing full stop / comma is trimmed only after a digit for number-like labels
NUMERIC = {"ACCOUNT", "PHONE", "TIN", "DOB", "ADDRESS"}


def _trim_offset(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    return start, end


def _is_word_char(c: str) -> bool:
    return c.isalnum()


def script_of(c: str) -> str:
    if c.isdigit():
        return "digit"
    o = ord(c)
    if 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FA1F:
        return "han"
    if 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "kana"
    if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "hangul"
    if 0x0E00 <= o <= 0x0E7F:
        return "thai"
    return "other"


UNSPACED = {"han", "kana", "thai"}


def word_starts(text: str, offsets, script_boundaries: bool = False) -> list[bool]:
    """True where a (non-special) token begins a new word."""
    starts, prev_end = [], None
    for s, e in offsets:
        s, e = _trim_offset(text, s, e)
        if s >= e:
            starts.append(False)
            continue
        new = (prev_end is None or s > prev_end
               or not _is_word_char(text[s])
               or not _is_word_char(text[prev_end - 1]))
        if script_boundaries and not new:
            cur, prev = script_of(text[s]), script_of(text[prev_end - 1])
            new = cur in UNSPACED or cur != prev
        starts.append(new)
        prev_end = e
    return starts


# DeBERTa/XLM-R SentencePiece drops "\n" and "\t"; models trained with visible breaks see these markers
VISIBLE_BREAKS = {"\n": " ¶ ", "\t": " | ", "\r": " "}


def visible_breaks(text: str) -> tuple[str, list[int]]:
    out, back = [], []
    for i, c in enumerate(text):
        rep = VISIBLE_BREAKS.get(c, c)
        out.append(rep)
        back.extend([i] * len(rep))
    return "".join(out), back


def window_encode(tokenizer, text: str, max_length: int, stride: int,
                  show_breaks: bool = True, blank_lone_space: bool = False) -> list[dict]:
    """Tokenize once, then cut overlapping windows (incl. special tokens). Offsets refer to `text`."""
    shown, back = visible_breaks(text) if show_breaks else (text, list(range(len(text))))
    enc = tokenizer(shown, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = [(back[s], back[e - 1] + 1) if e > s else (back[s] if s < len(back) else len(text),) * 2
            for s, e in enc["offset_mapping"]]
    if blank_lone_space:
        for i, tok in enumerate(tokenizer.convert_ids_to_tokens(ids)):
            s, e = offs[i]
            if tok == "▁" and e > s and not text[s].isspace():
                offs[i] = (s - 1, s) if s > 0 and text[s - 1].isspace() else (s, s)
    first = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    last = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pre = [t for t in (first,) if t is not None]
    post = [t for t in (last,) if t is not None]
    body = max_length - len(pre) - len(post)
    step = max(1, body - stride)
    windows, start = [], 0
    while True:
        chunk = slice(start, start + body)
        windows.append({
            "input_ids": pre + ids[chunk] + post,
            "offset_mapping": [(0, 0)] * len(pre) + [tuple(o) for o in offs[chunk]] + [(0, 0)] * len(post),
            "special_tokens_mask": [1] * len(pre) + [0] * len(ids[chunk]) + [1] * len(post),
        })
        if start + body >= len(ids):
            break
        start += step
    return windows


def fill_continuations(names: list, starts: list[bool]) -> list:
    """Continuation sub-tokens take their word's label (B-X -> I-X)."""
    out, head = [], None
    for name, st in zip(names, starts):
        if st or head is None:
            head = name
            out.append(name)
        else:
            out.append(None if head is None else ("O" if head == "O" else "I-" + head.split("-", 1)[1]))
    return out


def labels_to_spans(offsets, label_names: list, text: str) -> list[dict]:
    spans, cur = [], None
    for (s, e), name in zip(offsets, label_names):
        if name is None or s == e:
            continue
        s, e = _trim_offset(text, s, e)
        if s >= e:
            continue
        if name == "O":
            if cur:
                spans.append(cur)
            cur = None
            continue
        tag, ent = name.split("-", 1)
        if tag == "I" and cur and cur["label"] == ent:
            cur["end"] = e
        else:
            if cur:
                spans.append(cur)
            cur = {"start": s, "end": e, "label": ent}
    if cur:
        spans.append(cur)
    return spans


def trim_span(text: str, sp: dict) -> dict | None:
    """Drop punctuation the tokenizer glued onto an entity. None if nothing is left."""
    a, b = sp["start"], sp["end"]
    while a < b and (text[a] in LEAD or text[a].isspace() or (text[a] in OPEN and not any(c in CLOSE for c in text[a:b]))):
        a += 1
    while a < b:
        c = text[b - 1]
        if c.isspace():
            b -= 1
        elif c in CLOSE and PAIRS[c] not in text[a:b - 1]:
            b -= 1
        elif c in ".。,，、;；:：" and sp["label"] in NUMERIC and b - 2 >= a and (text[b - 2].isdigit() or text[b - 2] in CLOSE):
            b -= 1
        else:
            break
    if b - a > 2 and text[a] in OPEN and PAIRS.get(text[b - 1]) == text[a] and not any(c in OPEN + CLOSE for c in text[a + 1:b - 1]):
        a, b = a + 1, b - 1
    if a >= b:
        return None
    return {**sp, "start": a, "end": b} if (a, b) != (sp["start"], sp["end"]) else sp
