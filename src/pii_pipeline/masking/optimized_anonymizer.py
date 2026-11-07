"""OptimizedAnonymizer: also mask short forms of detected names.

The model finds a name and its boundaries ("Tan Wei Ming"). Later the text often says "Wei Ming",
"Mr Tan" or "W. M. Tan". For each detected name (labels in `anonymizer.name_entities`, default PERSON)
this generates those forms and masks undetected copies with the full name's identity.

Guards against applying a partial form to a different person:
- a form shared by two detected names (both "Tan Wei Ming" and "Tan Ah Kow" give "Tan") is skipped
- a match inside a longer ASCII word is skipped ("Tan" in "Tango")
- a match whose neighbouring word is a capitalised word that is not part of this name is skipped
  ("Tan" in "Tan Ah Kow" when only "Tan Wei Ming" was detected); a title before it, or a capitalised
  word that starts a sentence, does not count
- forms are matched case-sensitively, longest first, and never over an existing span
"""
from __future__ import annotations

import re

from ..core.types import PIISpan
from ..nlp.context_enhancer import is_inside_ascii_word, neighbouring_words
from .anonymizer import PIIAnonymizer

TITLES = ("Mr", "Mrs", "Ms", "Mdm", "Miss", "Madam", "Dr", "Prof")
_TITLE_SET = {t.casefold() for t in TITLES} | {t.casefold() + "." for t in TITLES}
MIN_PART_CHARS = 2


def _is_title(word: str | None) -> bool:
    return word is not None and word.casefold().rstrip(".") in {t.casefold() for t in TITLES}


def name_variants(name: str) -> list[str]:
    """Short forms of a multi-word name, longest first. Names without spaces (e.g. Chinese script) have none."""
    words = [w.strip(",") for w in name.split()]
    core = [w for w in words if w and w.casefold() not in _TITLE_SET]
    if len(core) < 2:
        return []
    first, last = core[0], core[-1]
    out: set[str] = {" ".join(core[1:]), " ".join(core[:-1])}           # surname-first / surname-last
    out.update(w for w in core if len(w.strip(".")) >= MIN_PART_CHARS)    # single parts
    for given, surname in ((core[1:], first), (core[:-1], last)):
        initials = [w[0] + "." for w in given if w[:1].isalpha()]
        if initials:
            out.add(" ".join(initials) + " " + surname)                    # W. M. Tan
            out.add("".join(initials) + " " + surname)                     # W.M. Tan
    for title in ("Mr", "Ms", "Mrs", "Mdm", "Dr"):
        for part in {first, last}:
            out.add(f"{title} {part}")
            out.add(f"{title}. {part}")
    out.discard(name)
    out.discard(" ".join(core))
    return sorted((v for v in out if v.strip()), key=lambda v: (-len(v), v))


def _starts_sentence(text: str, word_start: int) -> bool:
    before = text[:word_start].rstrip(" \t")
    return not before or before[-1] in ".!?。！？\n:;"


class OptimizedAnonymizer(PIIAnonymizer):
    def __init__(self, *args, name_entities=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_entities = set(name_entities if name_entities is not None else self.config.name_entities)

    def _foreign_neighbour(self, text: str, start: int, end: int, parts: set[str]) -> bool:
        before, after = neighbouring_words(text, start, end)
        if after and after[0].isupper() and after.casefold() not in parts:
            return True
        if before and before[0].isupper() and before.casefold() not in parts and not _is_title(before):
            word_start = len(text[:start].rstrip(" \t")) - len(before)
            if not _starts_sentence(text, word_start):
                return True
        return False

    def _extra_spans(self, text: str, spans: list[PIISpan]) -> list[PIISpan]:
        names: dict[str, tuple[str, int, PIISpan]] = {}   # full name surface -> (entity, identity, span)
        for s in spans:
            if s.entity_type in self.name_entities:
                names.setdefault(s.text, (s.entity_type, s.metadata["identity"], s))

        # variant -> owners; a variant owned by two different identities is ambiguous
        owners: dict[str, set[tuple[str, int]]] = {}
        parts_of: dict[tuple[str, int], set[str]] = {}
        for surface, (entity, ident, _) in names.items():
            parts_of.setdefault((entity, ident), set()).update(w.casefold().strip(",.") for w in surface.split())
            for v in name_variants(surface):
                owners.setdefault(v, set()).add((entity, ident))

        occupied = [(s.start, s.end) for s in spans]
        added: list[PIISpan] = []
        for variant in sorted(owners, key=lambda v: (-len(v), v)):
            if len(owners[variant]) != 1:
                continue
            (entity, ident), = owners[variant]
            pattern = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, variant.split())) + r"(?!\w)")
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                if not self._free(occupied, start, end) or is_inside_ascii_word(text, start, end):
                    continue
                if self._foreign_neighbour(text, start, end, parts_of[(entity, ident)]):
                    continue
                occupied.append((start, end))
                added.append(PIISpan(start=start, end=end, text=text[start:end], entity_type=entity,
                                     score=0.5, recognizer_name="name_variant",
                                     metadata={"identity": ident}))
        return added
