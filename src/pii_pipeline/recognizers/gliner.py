"""GLiNER adapter: any GLiNER model as a Presidio EntityRecognizer.

GLiNER takes natural-language labels ("person", "phone number") at inference time. `labels` maps each
prompt label to the entity type the pipeline uses; a plain list uses the upper-cased label.

Long text is cut into overlapping character windows (`window_size`, `stride`), preferably at
whitespace, because GLiNER models have a fixed input length. Offsets are mapped back to the full text
and duplicates from overlapping windows keep their highest score.

Install with `pip install .[gliner]`. Configure under `model.gliner` in detector_config.yaml.
"""
from __future__ import annotations

from presidio_analyzer import EntityRecognizer, RecognizerResult

from ..core.errors import ModelNotAvailableError


def _load_gliner(model: str, device: str):
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise ModelNotAvailableError(
            "GLiNER models need the optional 'gliner' package: pip install .[gliner]") from exc
    try:
        loaded = GLiNER.from_pretrained(model)
    except Exception as exc:  # network / path / format errors from the library
        raise ModelNotAvailableError(f"could not load GLiNER model {model!r}: {type(exc).__name__}") from exc
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if hasattr(loaded, "to"):
        loaded = loaded.to(device)
    return loaded


def windows(text: str, size: int, stride: int) -> list[tuple[int, int]]:
    """Overlapping (start, end) character windows covering `text`. Each window after the first starts
    `stride` characters before the previous one ended; a cut moves back to whitespace when one is near."""
    if size <= 0:
        raise ValueError("window_size must be positive")
    stride = max(0, min(stride, size - 1))
    n = len(text)
    out, start = [], 0
    while start < n:
        end = min(n, start + size)
        if end < n:   # cut at the last whitespace in the second half of the window, if any
            cut = max(text.rfind(c, start + size // 2, end) for c in " \n\t")
            if cut > start:
                end = cut
        out.append((start, end))
        if end >= n:
            break
        nxt = end - stride
        if stride:    # start the overlap at a word boundary
            first = min((i for i in (text.find(c, nxt, end) for c in " \n\t") if i != -1), default=-1)
            if first != -1:
                nxt = first + 1
        start = max(nxt, start + 1)
    return out


class GLiNERRecognizer(EntityRecognizer):
    def __init__(
        self,
        model: str,
        labels: dict[str, str] | list[str],
        threshold: float = 0.5,
        window_size: int = 1500,
        stride: int = 200,
        device: str = "auto",
        flat_ner: bool = True,
        label_map: dict[str, str] | None = None,
        supported_language: str = "en",
        name: str | None = None,
    ):
        if isinstance(labels, dict):
            self.labels = {str(k): str(v) for k, v in labels.items()}
        else:
            self.labels = {str(lab): str(lab).upper().replace(" ", "_") for lab in labels}
        if not self.labels:
            raise ValueError("GLiNER needs at least one label")
        self.label_map = dict(label_map or {})
        self.threshold = float(threshold)
        self.window_size, self.stride = int(window_size), int(stride)
        self.flat_ner = flat_ner
        self.model_name = model
        self.model = _load_gliner(model, device)
        supported = sorted({self.label_map.get(e, e) for e in self.labels.values()})
        super().__init__(supported_entities=supported, supported_language=supported_language,
                         name=name or type(self).__name__)
        self.is_loaded = True

    def load(self) -> None:   # loaded in __init__ so per-language copies share the model
        self.is_loaded = True

    def predict_spans(self, text: str) -> list[dict]:
        """{start, end, label, score} with label already mapped to the entity type."""
        best: dict[tuple[int, int, str], float] = {}
        prompt_labels = list(self.labels)
        for w_start, w_end in windows(text, self.window_size, self.stride):
            chunk = text[w_start:w_end]
            if not chunk.strip():
                continue
            for ent in self.model.predict_entities(chunk, prompt_labels, threshold=self.threshold,
                                                   flat_ner=self.flat_ner):
                entity = self.labels.get(ent["label"], ent["label"])
                entity = self.label_map.get(entity, entity)
                key = (w_start + int(ent["start"]), w_start + int(ent["end"]), entity)
                best[key] = max(best.get(key, 0.0), float(ent.get("score", 0.0)))
        return [{"start": s, "end": e, "label": lab, "score": round(sc, 4)}
                for (s, e, lab), sc in sorted(best.items())]

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        for sp in self.predict_spans(text):
            if entities and sp["label"] not in entities:
                continue
            if sp["score"] < self.threshold:
                continue
            results.append(RecognizerResult(
                entity_type=sp["label"], start=sp["start"], end=sp["end"], score=sp["score"],
                recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name},
            ))
        return results
