"""Regression tests for phrase-first title lexicon matching."""
import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "03_extract_title_features.py"
match_phrases_and_words = runpy.run_path(str(SCRIPT))["match_phrases_and_words"]


def test_phrase_not_double_counted_as_word():
    assert match_phrases_and_words("Worth It", ["worth it"], {"worth", "it"}) == ["worth it"]


def test_repeated_phrase_occurrences_preserved():
    assert match_phrases_and_words("Worth It and worth it", ["worth it"], {"worth", "it"}) == ["worth it", "worth it"]


def test_standalone_word_after_phrase_is_counted():
    assert match_phrases_and_words("Heart Attack attack", ["heart attack"], {"attack"}) == ["heart attack", "attack"]


def test_multiple_distinct_words_counted():
    assert match_phrases_and_words("best and biggest", [], {"best", "biggest"}) == ["best", "biggest"]


def test_no_match_returns_empty():
    assert match_phrases_and_words("ordinary title", [], {"best", "biggest"}) == []


if __name__ == "__main__":
    for test in [test_phrase_not_double_counted_as_word,
                 test_repeated_phrase_occurrences_preserved,
                 test_standalone_word_after_phrase_is_counted,
                 test_multiple_distinct_words_counted,
                 test_no_match_returns_empty]:
        test()
    print("5 title matching regression tests: PASS")
