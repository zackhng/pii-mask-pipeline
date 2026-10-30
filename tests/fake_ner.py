"""Stand-in NER recognizer for unit tests: finds fixed phrases, no model needed."""
import re

from presidio_analyzer import EntityRecognizer, RecognizerResult


class FakeNER(EntityRecognizer):
    def __init__(self, phrases: dict[str, str], score: float = 0.8, supported_language: str = "en", name=None):
        self.phrases = phrases
        self.fixed_score = score
        super().__init__(supported_entities=sorted(set(phrases.values())),
                         supported_language=supported_language, name=name or "FakeNER")

    def load(self) -> None:
        self.is_loaded = True

    def analyze(self, text, entities, nlp_artifacts=None):
        out = []
        for phrase, entity in self.phrases.items():
            for m in re.finditer(re.escape(phrase), text):
                if entities and entity not in entities:
                    continue
                out.append(RecognizerResult(
                    entity_type=entity, start=m.start(), end=m.end(), score=self.fixed_score,
                    recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name}))
        return out
