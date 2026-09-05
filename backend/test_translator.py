import pytest
# Import your class from your actual file
from tagalog_to_baybayin import TagalogToBaybayin

@pytest.fixture
def engine():
    """Initializes the TagalogToBaybayin class for testing"""
    return TagalogToBaybayin()

# Baybayin bytes written as \uXXXX escapes.
#   ᜀ A   ᜁ I/E base   ᜂ U/O base
#    standalone I (I/E shape + line centred above)
#    standalone U (U/O shape + tick bottom-right)
#   ᜒ i-kudlit   ᜓ o-kudlit   ᜔ virama
#   ᜎ LA   ᜌ YA   ᜉ PA   ᜄ GA   ᜅ NGA   ᜆ TA
#   ᜈ NA  ᜊ BA   ᜋ MA   ᜐ SA
@pytest.mark.parametrize("input_text, expected, description", [
    ("", "", "Empty input guard clause"),
    ("mga", "ᜋᜄ", "Linguistic exception path"),
    ("a i u", "ᜀ  ", "Standalone vowels are dedicated glyphs"),
    ("Ilog", "ᜎᜓᜄ᜔", "Standalone I at word start"),
    ("Baya", "ᜊᜌ", "CV pair with inherent 'a'"),
    # Consonant+i / consonant+u keep the plain kudlits (U+1712 / U+1716);
    # only the *standalone* letters I and U get a dedicated glyph, so a typed
    # 'i' / 'u' does not round-trip through the BTL OCR as 'e' / 'o'.
    ("Ngiti", "ᜅᜒᜆᜒ", "Consonant+i uses the plain kudlit"),
    ("Opo", "ᜂᜉᜓ", "CV pair with o-kudlit (dot below)"),
    ("Iyo", "ᜌᜓ", "Word-initial standalone I keeps its glyph"),
    ("Salamat", "ᜐᜎᜋᜆ᜔", "Final consonant with virama"),
    ("Ang", "ᜀᜅ᜔", "Final digraph consonant logic"),
    ("A B", "ᜀ ᜊ᜔", "Space preservation between tokens"),
])
def test_white_box_coverage(engine, input_text, expected, description):
    """
    Test Case: Validates if the TTB engine correctly applies
    the Lopez Method and Regex tokenization.
    """
    # translate() returns (baybayin_text, confidence) — the test checks the text.
    result, _confidence = engine.translate(input_text)

    # This is the actual 'test'—checking if your code's output matches the 'Ground Truth'
    assert result == expected
