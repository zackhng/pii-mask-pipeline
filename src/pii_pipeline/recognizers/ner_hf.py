"""Swappable NER recognizer: any Hugging Face token-classification model (BIO labels) as a Presidio
EntityRecognizer. Configure with `ner.class` + `ner.params` in the pipeline YAML."""
from __future__ import annotations

import bisect

import torch
from presidio_analyzer import EntityRecognizer, RecognizerResult
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ._bio import fill_continuations, labels_to_spans, trim_span, window_encode, word_starts


class HFTokenClassRecognizer(EntityRecognizer):
    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        max_length: int = 384,
        stride: int = 96,
        batch_windows: int = 16,
        trim_punct: bool | None = None,
        label_map: dict[str, str] | None = None,
        min_score: float = 0.0,
        supported_language: str = "en",
        name: str | None = None,
    ):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir, dtype=torch.float32)
        self.model = self.model.to(self.device).eval()
        cfg = self.model.config
        self.id2label = cfg.id2label
        self.show_breaks = bool(getattr(cfg, "visible_breaks", False))
        self.script_boundaries = bool(getattr(cfg, "script_boundaries", False))
        self.trim_punct = bool(getattr(cfg, "trim_punct", False)) if trim_punct is None else trim_punct
        self.max_length, self.stride, self.batch_windows = max_length, stride, batch_windows
        self.label_map = dict(label_map or {})
        self.min_score = min_score

        raw = {lab.split("-", 1)[1] for lab in self.id2label.values() if lab != "O"}
        supported = sorted({self.label_map.get(lab, lab) for lab in raw})
        super().__init__(supported_entities=supported, supported_language=supported_language,
                         name=name or type(self).__name__)
        self.is_loaded = True

    def load(self) -> None:  # model is loaded in __init__ so copies can share it
        self.is_loaded = True

    @torch.no_grad()
    def predict_spans(self, text: str) -> list[dict]:
        """Char spans {start, end, label, score} with raw model label names."""
        wins = window_encode(self.tok, text, self.max_length, self.stride, self.show_breaks,
                             blank_lone_space=self.script_boundaries)
        best: dict[tuple[int, int], tuple[float, int]] = {}
        pad = self.tok.pad_token_id
        for b in range(0, len(wins), self.batch_windows):
            chunk = wins[b:b + self.batch_windows]
            width = max(len(w["input_ids"]) for w in chunk)
            ids = torch.tensor([w["input_ids"] + [pad] * (width - len(w["input_ids"])) for w in chunk])
            mask = torch.tensor([[1] * len(w["input_ids"]) + [0] * (width - len(w["input_ids"])) for w in chunk])
            probs = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits.float().softmax(-1)
            top_p, top_lab = probs.max(-1)
            top_p, top_lab = top_p.tolist(), top_lab.tolist()
            for w, win in enumerate(chunk):
                for i, (s, e) in enumerate(win["offset_mapping"]):
                    if win["special_tokens_mask"][i] or s == e:
                        continue
                    p = top_p[w][i]
                    if (s, e) not in best or p > best[(s, e)][0]:
                        best[(s, e)] = (p, top_lab[w][i])

        offsets = sorted(best)
        if not offsets:
            return []
        starts = word_starts(text, offsets, self.script_boundaries)
        names = fill_continuations([self.id2label[best[o][1]] for o in offsets], starts)
        spans = labels_to_spans(offsets, names, text)

        ends = [o[1] for o in offsets]
        out = []
        for sp in spans:
            ps, k = [], bisect.bisect_left(ends, sp["start"])
            while k < len(offsets) and offsets[k][0] < sp["end"]:
                if starts[k]:
                    ps.append(best[offsets[k]][0])
                k += 1
            sp["score"] = round(sum(ps) / len(ps), 4) if ps else 0.0
            if self.trim_punct:
                sp = trim_span(text, sp)
                if sp is None:
                    continue
            out.append(sp)
        return out

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        for sp in self.predict_spans(text):
            entity = self.label_map.get(sp["label"], sp["label"])
            if entities and entity not in entities:
                continue
            if sp["score"] < self.min_score:
                continue
            results.append(RecognizerResult(
                entity_type=entity, start=sp["start"], end=sp["end"], score=sp["score"],
                recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name},
            ))
        return results
