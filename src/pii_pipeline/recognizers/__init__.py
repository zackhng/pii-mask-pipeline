"""Presidio recognizers. The NER backends (ner_hf, gliner) are imported on demand so their heavy or
optional dependencies load only when configured."""
from .deterministic import deny_list_recognizers, pattern_recognizers, presidio_builtin_recognizers
from .table import StructuredLabelRecognizer, TableFieldRecognizer

__all__ = ["StructuredLabelRecognizer", "TableFieldRecognizer", "deny_list_recognizers",
           "pattern_recognizers", "presidio_builtin_recognizers"]
