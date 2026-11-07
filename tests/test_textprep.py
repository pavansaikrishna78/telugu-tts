import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from textprep import Silence, Speech, TextPrep, number_to_telugu, transliterate  # noqa: E402


@pytest.fixture
def tp(tmp_path):
    return TextPrep(unknown_log=tmp_path / "unknown.txt")


# ------------------------------------------------------------ c. numbers
@pytest.mark.parametrize("n,expected", [
    (0, "సున్నా"),
    (7, "ఏడు"),
    (19, "పందొమ్మిది"),
    (26, "ఇరవై ఆరు"),
    (100, "వంద"),
    (150, "నూట యాభై"),
    (200, "రెండు వందలు"),
    (250, "రెండు వందల యాభై"),
    (1000, "వేయి"),
    (2000, "రెండు వేలు"),
    (2026, "రెండు వేల ఇరవై ఆరు"),
    (100000, "ఒక లక్ష"),
    (250000, "రెండు లక్షల యాభై వేలు"),
    (10000000, "ఒక కోటి"),
    (20000000, "రెండు కోట్లు"),
    (15000000, "ఒక కోటి యాభై లక్షలు"),
])
def test_integers(n, expected):
    assert number_to_telugu(n) == expected


@pytest.mark.parametrize("text,expected", [
    ("₹9,00,000", "తొమ్మిది లక్షల రూపాయలు"),
    ("9,00,000 రూపాయలు", "తొమ్మిది లక్షల రూపాయలు"),
    ("₹9,00,000 రూపాయలు", "తొమ్మిది లక్షల రూపాయలు"),
    ("Rs. 500", "ఐదు వందల రూపాయలు"),
    ("₹1", "ఒక రూపాయి"),
    ("₹100", "వంద రూపాయలు"),
    ("₹2,000", "రెండు వేల రూపాయలు"),
    ("₹1,00,000", "ఒక లక్ష రూపాయలు"),
    ("₹5 lakh", "ఐదు లక్షల రూపాయలు"),
    ("₹2 crore", "రెండు కోట్ల రూపాయలు"),
    ("₹500కి", "ఐదు వందల రూపాయలకి"),
    ("5,000 రూపాయలకు", "ఐదు వేల రూపాయలకు"),
    ("10%", "పది శాతం"),
    ("12.5%", "పన్నెండు పాయింట్ ఐదు శాతం"),
    ("1 శాతం", "ఒక శాతం"),
    ("2026-27", "రెండు వేల ఇరవై ఆరు – ఇరవై ఏడు"),
    ("2026లో", "రెండు వేల ఇరవై ఆరులో"),
    ("3.14", "మూడు పాయింట్ ఒకటి నాలుగు"),
    ("5 lakh", "ఐదు లక్షలు"),
    ("1 lakh", "ఒక లక్ష"),
    ("31-03-2026", "ముప్పై ఒకటి మార్చి రెండు వేల ఇరవై ఆరు"),
])
def test_number_expressions(tp, text, expected):
    assert tp.normalize(text) == expected


# ------------------------------------------------------------ d. legal references
@pytest.mark.parametrize("text,expected", [
    ("Section 2(77)", "సెక్షన్ టూ, సెవెంటీ సెవెన్"),
    ("2(11)(a)(i)", "టూ, ఎలెవన్, క్లాజ్ ఏ, సబ్ క్లాజ్ వన్"),
    ("Section 202(1)", "సెక్షన్ టూ హండ్రెడ్ టూ, వన్"),
    ("Section 2", "సెక్షన్ టూ"),
    ("సెక్షన్ 2(77)", "సెక్షన్ టూ, సెవెంటీ సెవెన్"),
    ("Section 80C", "సెక్షన్ ఎయిటీ సి"),
    ("Section 115BAC", "సెక్షన్ వన్ హండ్రెడ్ ఫిఫ్టీన్ బి ఏ సి"),
    ("Section 10(10D)", "సెక్షన్ టెన్, టెన్ డి"),
    ("Section 2(24)(iv)", "సెక్షన్ టూ, ట్వంటీ ఫోర్, సబ్ క్లాజ్ ఫోర్"),
    ("Section 2(22)(e)", "సెక్షన్ టూ, ట్వంటీ టూ, క్లాజ్ ఇ"),
    ("Section 2(1)(ia)", "సెక్షన్ టూ, వన్, సబ్ క్లాజ్ వన్ ఏ"),
    ("Section 2లో", "సెక్షన్ టూలో"),
])
def test_legal_references(tp, text, expected):
    assert tp.normalize(text) == expected


def test_three_digit_pair_style(tp):
    tp.rules["three_digit_style"] = "pair"
    assert tp.normalize("Section 202") == "సెక్షన్ టూ ఓ టూ"
    assert tp.normalize("Section 194") == "సెక్షన్ వన్ నైంటీ ఫోర్"


# ------------------------------------------------------------ e. English terms
@pytest.mark.parametrize("text,expected", [
    ("Assessee", "అసెస్సీ"),
    ("assessee", "అసెస్సీ"),
    ("Assessment", "అసెస్‌మెంట్"),
    ("HUF", "హెచ్ యు ఎఫ్"),
    ("LLP", "ఎల్ ఎల్ పి"),
    ("TDS", "టి డి ఎస్"),
    ("Tax Year", "టాక్స్ ఇయర్"),
    ("Company", "కంపెనీ"),
    ("Firm", "ఫర్మ్"),
    ("Rebate", "రిబేట్"),
    ("Surcharge", "సర్‌చార్జ్"),
    ("Assessees", "అసెస్సీలు"),
    ("TDSని", "టి డి ఎస్ని"),
])
def test_lexicon(tp, text, expected):
    assert tp.normalize(text) == expected


def test_lexicon_has_about_80_terms(tp):
    assert len(tp.lexicon) >= 80


def test_short_acronym_is_case_sensitive(tp):
    assert tp.normalize("IT") == "ఐ టి"
    assert "ఐ టి" not in tp.normalize("it")


def test_unknown_word_is_transliterated_and_logged(tp):
    result = tp.normalize("Bank")
    assert result and not any(c.isascii() and c.isalpha() for c in result)
    assert tp.unknown_log.read_text(encoding="utf-8").startswith("Bank\t")


def test_unknown_log_has_no_duplicates(tp):
    tp.normalize("Bank Bank")
    fresh = TextPrep(unknown_log=tp.unknown_log)
    fresh.normalize("Bank Zebra")
    words = [line.split("\t")[0] for line in tp.unknown_log.read_text(encoding="utf-8").splitlines()]
    assert words == ["Bank", "Zebra"]


def test_unknown_acronym_is_spelled(tp):
    assert tp.normalize("XYZ") == "ఎక్స్ వై జెడ్"


def test_known_word_is_not_logged(tp):
    tp.normalize("TDS")
    assert not tp.unknown_log.exists()


def test_transliterate_gives_telugu_only():
    assert all(not c.isascii() for c in transliterate("Bank"))


# ------------------------------------------------------------ a, b. markers and pauses
@pytest.mark.parametrize("marker", [
    "[ON SCREEN]: చూపించు",
    "[DURATION]: 45s",
    "[EMPHASIS]: గట్టిగా",
    "[VERIFY rate with ICAI]",
    "[CONFLICT with slide 4]",
    "[STRUCTURE MAP]: ఒకటి",
])
def test_markers_not_spoken(tp, marker):
    out = tp.prepare(f"నమస్తే.\n{marker}\n\nబాగున్నారా?")
    text = " ".join(s.text for s in out)
    assert text == "నమస్తే. బాగున్నారా?"


@pytest.mark.parametrize("label", ["[TRAP]:", "[EXAM POINT]:", "[QUICK CHECK]:", "[NEW EXAMPLE]:",
                                   "[REMEMBER]:", "[BRIDGE]:", "[ICAI EXTRA – no slide]"])
def test_kept_labels_keep_their_text(tp, label):
    out = tp.prepare(f"{label} ఇది ముఖ్యం.")
    assert [s.text for s in out] == ["ఇది ముఖ్యం."]


def test_markdown_symbols_removed(tp):
    out = tp.prepare("# శీర్షిక\n**ముఖ్యం** -> ఇది | అది → సరి\n---\n")
    text = " ".join(s.text for s in out)
    assert not set(text) & set("#*|>→-")


def test_pause_becomes_silence(tp):
    out = tp.prepare("మొదటి వాక్యం. [PAUSE 2s] రెండవ వాక్యం.")
    assert isinstance(out[0], Speech) and out[0].text == "మొదటి వాక్యం."
    assert out[1] == Silence(2.0)
    assert isinstance(out[2], Speech) and out[2].text == "రెండవ వాక్యం."


def test_pause_decimal_seconds(tp):
    out = tp.prepare("ఒకటి. [PAUSE 1.5s] రెండు.")
    assert out[1] == Silence(1.5)


# ------------------------------------------------------------ f. splitting and packing
def test_sentence_splitter(tp):
    assert tp.split_sentences("ఒకటి. రెండు? మూడు! నాలుగు। ఐదు\nఆరు") == \
        ["ఒకటి.", "రెండు?", "మూడు!", "నాలుగు।", "ఐదు", "ఆరు"]


def test_chunks_respect_limit_and_keep_sentences_whole(tp):
    sentence = " ".join(["పదం"] * 15) + "."  # about 8 seconds
    sentences = [sentence] * 10
    chunks = tp.pack_chunks(sentences)
    assert len(chunks) > 1
    assert all(tp.estimate_seconds(c) <= 25 for c in chunks)
    assert " ".join(chunks) == " ".join(sentences)
    assert all(c.endswith(".") for c in chunks)


def test_long_sentence_is_never_cut(tp):
    long_sentence = " ".join(["పదం"] * 80) + "."
    assert tp.pack_chunks(["చిన్నది.", long_sentence]) == ["చిన్నది.", long_sentence]


def test_short_tail_merges_when_it_fits(tp):
    a = " ".join(["పదం"] * 30) + "."
    b = " ".join(["పదం"] * 5) + "."
    assert tp.pack_chunks([a, b]) == [f"{a} {b}"]


# ------------------------------------------------------------ end to end
def test_full_sentence(tp):
    text = "Section 202(1) ప్రకారం, HUF కి ₹9,00,000 పై 10% Surcharge వర్తిస్తుంది."
    out = tp.prepare(text)
    assert [s.text for s in out] == [
        "సెక్షన్ టూ హండ్రెడ్ టూ, వన్ ప్రకారం, హెచ్ యు ఎఫ్ కి తొమ్మిది లక్షల రూపాయలు పై పది శాతం సర్‌చార్జ్ వర్తిస్తుంది."
    ]
