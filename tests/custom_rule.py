"""Custom check rule for tests: keeps a span only if it contains the configured word."""


def contains_word(text, params, today):
    return params["word"] in text
