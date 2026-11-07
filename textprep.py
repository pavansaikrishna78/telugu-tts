"""Telugu text front-end: turns a lecture script into speakable chunks and pauses.

prepare(text) returns a list of Speech and Silence segments, in order.
Editable data lives in lexicon.tsv and legal_rules.yaml.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_LEXICON = ROOT / "lexicon.tsv"
DEFAULT_RULES = ROOT / "legal_rules.yaml"
DEFAULT_UNKNOWN_LOG = ROOT / "logs" / "unknown_terms.txt"

WORDS_PER_MINUTE = 110
MIN_CHUNK_SECONDS = 12
MAX_CHUNK_SECONDS = 25
DEFAULT_PAUSE_SECONDS = 1.0

TE = r"ఀ-౿"


@dataclass
class Speech:
    text: str
    est_seconds: float


@dataclass
class Silence:
    seconds: float


# ---------------------------------------------------------------- Telugu numbers

ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
        "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
        "పద్దెనిమిది", "పందొమ్మిది"]
TENS = {20: "ఇరవై", 30: "ముప్పై", 40: "నలభై", 50: "యాభై", 60: "అరవై", 70: "డెబ్బై", 80: "ఎనభై", 90: "తొంభై"}


def _under_100(n: int, before_noun: bool) -> str:
    if n < 20:
        return "ఒక" if (n == 1 and before_noun) else ONES[n]
    tens, unit = n - n % 10, n % 10
    if unit == 0:
        return TENS[tens]
    if unit == 1 and before_noun:
        return f"{TENS[tens]} ఒక్క"
    return f"{TENS[tens]} {ONES[unit]}"


def _count(n: int) -> str:
    """A count placed before వేల, లక్షల or కోట్ల."""
    return number_to_telugu(n, before_noun=True)


def number_to_telugu(n: int, before_noun: bool = False) -> str:
    """Indian-system words. before_noun gives the joined form: రెండు లక్షలు -> రెండు లక్షల రూపాయలు."""
    if n == 0:
        return ONES[0]
    parts: list[str] = []
    crore, n = divmod(n, 10**7)
    lakh, n = divmod(n, 10**5)
    thousand, n = divmod(n, 1000)
    hundred, rest = divmod(n, 100)

    def joined(is_last: bool) -> bool:
        return before_noun or not is_last

    if crore:
        last = not (lakh or thousand or hundred or rest)
        parts.append("ఒక కోటి" if crore == 1 else f"{_count(crore)} " + ("కోట్ల" if joined(last) else "కోట్లు"))
    if lakh:
        last = not (thousand or hundred or rest)
        parts.append("ఒక లక్ష" if lakh == 1 else f"{_count(lakh)} " + ("లక్షల" if joined(last) else "లక్షలు"))
    if thousand:
        last = not (hundred or rest)
        parts.append("వేయి" if thousand == 1 else f"{_count(thousand)} " + ("వేల" if joined(last) else "వేలు"))
    if hundred:
        if hundred == 1:
            parts.append("నూట" if rest else "వంద")
        else:
            parts.append(f"{ONES[hundred]} " + ("వందల" if joined(not rest) else "వందలు"))
    if rest:
        parts.append(_under_100(rest, before_noun))
    return " ".join(parts)


def _digits(s: str) -> str:
    return " ".join(ONES[int(d)] for d in s)


def _spoken_int(s: str, before_noun: bool = False) -> str:
    """Leading-zero strings and very long strings are read digit by digit."""
    s = s.replace(",", "")
    if (len(s) > 1 and s[0] == "0") or len(s) > 15:
        return _digits(s)
    return number_to_telugu(int(s), before_noun)


def _spoken_number(s: str, before_noun: bool = False) -> str:
    if "." in s:
        whole, frac = s.split(".", 1)
        return f"{_spoken_int(whole)} పాయింట్ {_digits(frac)}"
    return _spoken_int(s, before_noun)


# ---------------------------------------------------------------- patterns

# 9,00,000 (Indian) | 1,000,000 (Western) | 12.5 | 500
NUM = r"(?:\d{1,2}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
NOT_BEFORE = r"(?<![\d.,])"
SCALES = {
    "lakh": "lakh", "lakhs": "lakh", "lac": "lakh", "lacs": "lakh",
    "లక్ష": "lakh", "లక్షలు": "lakh", "లక్షల": "lakh",
    "crore": "crore", "crores": "crore", "కోటి": "crore", "కోట్లు": "crore", "కోట్ల": "crore",
}
SCALE_RE = "|".join(sorted(map(re.escape, SCALES), key=len, reverse=True))
SCALE_WORD = {"lakh": ("లక్ష", "లక్షల", "లక్షలు"), "crore": ("కోటి", "కోట్ల", "కోట్లు")}
CLASSIFIERS = "శాతం|లక్ష|కోటి|రూపాయి"
MONTHS = ["", "జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్", "జూలై", "ఆగస్టు", "సెప్టెంబర్",
          "అక్టోబర్", "నవంబర్", "డిసెంబర్"]
ROMAN_RE = re.compile(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

GROUP = r"\(\s*[0-9A-Za-z]{1,6}\s*\)"
LEGAL_NUM = r"\d+[A-Za-z]*"
SECTION_REF = re.compile(
    rf"(?:\bSections?\b|\bSec\b\.?|సెక్షన్లు?)\s*(?:[Nn]o\.?\s*)?({LEGAL_NUM})((?:{GROUP})*)", re.IGNORECASE)
BARE_REF = re.compile(rf"{NOT_BEFORE}(?<![A-Za-z])({LEGAL_NUM})((?:{GROUP})+)")

DROP_BRACKET = re.compile(r"\[\s*(?:VERIFY|CONFLICT)\b[^\]]*\]\s*:?", re.IGNORECASE)
DROP_LINE = re.compile(r"\[\s*(?:ON SCREEN|DURATION|EMPHASIS)\b[^\]]*\][^\n]*", re.IGNORECASE)
DROP_BLOCK = re.compile(r"\[\s*STRUCTURE MAP\b[^\]]*\].*?(?:\n\s*\n|\Z)", re.IGNORECASE | re.DOTALL)
KEEP_LABEL = re.compile(
    r"\[\s*(?:NARRATION|TRAP|EXAM POINT|QUICK CHECK|NEW EXAMPLE|REMEMBER|BRIDGE|ICAI EXTRA[^\]]*)\s*\]\s*:?",
    re.IGNORECASE)
PAUSE = re.compile(r"\[\s*PAUSE(?:\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?)?\s*\]\s*:?", re.IGNORECASE)
ANY_BRACKET_TAG = re.compile(r"\[[^\]\n]*\]\s*:?")
PAUSE_TOKEN = re.compile(r"\x00(\d+(?:\.\d+)?)\x00")

# Consonants carry a virama; the vowel table replaces it. Longest match first.
_CONS = [("sch", "ష్"), ("tch", "చ్"), ("ch", "చ్"), ("sh", "ష్"), ("th", "థ్"), ("ph", "ఫ్"), ("ck", "క్"),
         ("gh", "గ్"), ("kh", "ఖ్"), ("qu", "క్వ్"), ("b", "బ్"), ("c", "క్"), ("d", "డ్"), ("f", "ఫ్"), ("g", "గ్"),
         ("h", "హ్"), ("j", "జ్"), ("k", "క్"), ("l", "ల్"), ("m", "మ్"), ("n", "న్"), ("p", "ప్"), ("q", "క్"),
         ("r", "ర్"), ("s", "స్"), ("t", "ట్"), ("v", "వ్"), ("w", "వ్"), ("x", "క్స్"), ("y", "య్"), ("z", "జ్")]
_VOWELS = [("aa", "ా", "ఆ"), ("ai", "ై", "ఐ"), ("ay", "ే", "ఏ"), ("ee", "ీ", "ఈ"), ("ea", "ీ", "ఈ"),
           ("oo", "ూ", "ఊ"), ("ou", "ౌ", "ఔ"), ("ow", "ౌ", "ఔ"), ("oa", "ో", "ఓ"),
           ("a", "ా", "ఆ"), ("e", "ె", "ఎ"), ("i", "ి", "ఇ"), ("o", "ో", "ఓ"), ("u", "ు", "ఉ")]
VIRAMA = "్"


def transliterate(word: str) -> str:
    """Rough English-to-Telugu rules, only for words missing from the lexicon."""
    w = word.lower()
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiou":
        w = w[:-1]  # silent final e
    out, i = "", 0
    while i < len(w):
        for key, matra, independent in _VOWELS:
            if w.startswith(key, i):
                if out.endswith(VIRAMA):
                    out = out[:-1] + matra
                else:
                    out += independent
                i += len(key)
                break
        else:
            if w[i] == "y" and i == len(w) - 1 and out.endswith(VIRAMA):
                out = out[:-1] + "ీ"
                i += 1
                continue
            for key, cons in _CONS:
                if w.startswith(key, i):
                    if key == "c" and w[i + 1:i + 2] in ("e", "i", "y"):
                        cons = "స్"
                    out += cons
                    i += len(key)
                    break
            else:
                i += 1
    return out


class TextPrep:
    def __init__(self, lexicon_path=DEFAULT_LEXICON, rules_path=DEFAULT_RULES, unknown_log=DEFAULT_UNKNOWN_LOG):
        self.rules = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8"))
        self.en_numbers = {int(k): v for k, v in self.rules["english_numbers"].items()}
        self.letters = self.rules["letters"]
        self.unknown_log = Path(unknown_log) if unknown_log else None
        self.unknown_seen: set[str] = set()
        self.lexicon = self._load_lexicon(lexicon_path)
        self._lexicon_re = self._build_lexicon_regex()

    # ------------------------------------------------------------ lexicon
    @staticmethod
    def _load_lexicon(path) -> dict[str, str]:
        entries = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            key, _, value = line.partition("\t")
            if value.strip():
                entries[key.strip()] = value.strip()
        return entries

    @staticmethod
    def _case_sensitive(key: str) -> bool:
        return key.isupper() and len(key) <= 4

    def _build_lexicon_regex(self):
        alternatives = []
        for key in sorted(self.lexicon, key=len, reverse=True):
            body = r"\s+".join(map(re.escape, key.split()))
            alternatives.append(body if self._case_sensitive(key) else f"(?i:{body})")
        return re.compile(rf"(?<![A-Za-z])(?:{'|'.join(alternatives)})(?![A-Za-z])")

    def _lookup(self, matched: str) -> str | None:
        flat = " ".join(matched.split())
        if flat in self.lexicon:
            return self.lexicon[flat]
        for key, value in self.lexicon.items():
            if not self._case_sensitive(key) and key.lower() == flat.lower():
                return value
        return None

    def _latin_word(self, word: str) -> str:
        if word.endswith(("s", "S")) and len(word) > 3:
            stem = self._lookup(word[:-1])
            if stem:
                return stem + "లు"
        if word.isupper() and len(word) <= 6:
            spoken = self.spell(word)
        else:
            spoken = transliterate(word)
        self._log_unknown(word, spoken)
        return spoken

    def _log_unknown(self, word: str, guess: str) -> None:
        if word in self.unknown_seen:
            return
        self.unknown_seen.add(word)
        if not self.unknown_log:
            return
        self.unknown_log.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if self.unknown_log.exists():
            for line in self.unknown_log.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("\t")
                existing[key] = value
        existing.setdefault(word, guess)
        self.unknown_log.write_text(
            "".join(f"{k}\t{v}\n" for k, v in sorted(existing.items())), encoding="utf-8")

    def spell(self, letters: str) -> str:
        return " ".join(self.letters[c] for c in letters.lower() if c in self.letters)

    # ------------------------------------------------------------ legal references
    def english_number(self, n: int) -> str:
        table = self.en_numbers
        if n in table:
            return table[n]
        if n < 100:
            return f"{table[n - n % 10]} {table[n % 10]}"
        if n < 1000:
            hundreds, rest = divmod(n, 100)
            if self.rules.get("three_digit_style") == "pair":
                if rest == 0:
                    return f"{table[hundreds]} {self.rules['hundred']}"
                if rest < 10:
                    return f"{table[hundreds]} {self.rules['zero_in_pair']} {table[rest]}"
                return f"{table[hundreds]} {self.english_number(rest)}"
            head = f"{table[hundreds]} {self.rules['hundred']}"
            return f"{head} {self.english_number(rest)}" if rest else head
        thousands, rest = divmod(n, 1000)
        head = f"{self.english_number(thousands)} {self.rules['thousand']}"
        return f"{head} {self.english_number(rest)}" if rest else head

    def _alnum_ref(self, token: str) -> str:
        m = re.fullmatch(r"(\d+)([A-Za-z]*)", token)
        spoken = self.english_number(int(m.group(1)))
        return f"{spoken} {self.spell(m.group(2))}" if m.group(2) else spoken

    def _roman(self, s: str) -> str | None:
        """(iv) -> four, (ia) -> one + letter a. None if not a roman numeral."""
        for body, suffix in ((s, ""), (s[:-1], s[-1:])):
            if body and ROMAN_RE.match(body):
                values = [ROMAN_VALUES[c] for c in body]
                total = sum(-v if i + 1 < len(values) and v < values[i + 1] else v for i, v in enumerate(values))
                head = self.english_number(total)
                return f"{head} {self.spell(suffix)}" if suffix else head
        return None

    def _legal_groups(self, groups: str) -> list[str]:
        labels = self.rules["labels"]
        pieces, lettered_seen = [], 0
        for g in re.findall(r"\(\s*([0-9A-Za-z]+)\s*\)", groups):
            if g[0].isdigit():
                pieces.append(self._alnum_ref(g))
                continue
            if len(g) == 1 and g.isupper():
                pieces.append(f"{labels['item']} {self.spell(g)}")
                continue
            low, roman = g.lower(), None
            if lettered_seen > 0 or len(low) > 1 or low in self.rules["single_letters_read_as_roman"]:
                roman = self._roman(low)
            lettered_seen += 1
            pieces.append(f"{labels['sub_clause']} {roman}" if roman else f"{labels['clause']} {self.spell(low)}")
        return pieces

    def _legal(self, text: str) -> str:
        sep = self.rules["separator"]

        def section(m):
            return f"{self.rules['section_word']} " + sep.join(
                [self._alnum_ref(m.group(1))] + self._legal_groups(m.group(2)))

        def bare(m):
            return sep.join([self._alnum_ref(m.group(1))] + self._legal_groups(m.group(2)))

        return BARE_REF.sub(bare, SECTION_REF.sub(section, text))

    # ------------------------------------------------------------ numbers
    def _money(self, amount: str, scale: str | None, trailing: str | None) -> str:
        amount = amount.replace(",", "")
        paise = ""
        if scale:
            base = SCALES[scale]
            single, joined, _ = SCALE_WORD[base]
            head = f"ఒక {single}" if amount == "1" else f"{_spoken_number(amount, True)} {joined}"
            noun_ready = head
        else:
            whole, _, frac = amount.partition(".")
            if frac and int(frac):
                paise = f" {_spoken_int(frac.ljust(2, '0')[:2].lstrip('0') or '0')} పైసలు"
            noun_ready = None if whole == "1" else _spoken_int(whole, True)
        if trailing and trailing != "రూపాయలు":
            word = trailing  # keep an inflected form such as రూపాయలకు
        elif not scale and amount.split(".")[0] == "1":
            word = "రూపాయి"
        else:
            word = "రూపాయల" if paise else "రూపాయలు"
        lead = noun_ready if noun_ready is not None else "ఒక"
        return f"{lead} {word}{paise}"

    def _numbers(self, text: str) -> str:
        # dd-mm-yyyy
        def date(m):
            day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
            if not (1 <= day <= 31 and 1 <= month <= 12):
                return m.group(0)
            return f"{number_to_telugu(day)} {MONTHS[month]} {_spoken_int(year)}"
        text = re.sub(rf"{NOT_BEFORE}(\d{{1,2}})[-./](\d{{1,2}})[-./](\d{{4}})(?!\d)", date, text)

        # ₹9,00,000 | Rs. 5 lakh | ₹500కి
        def rupee_prefix(m):
            spoken = self._money(m.group(1), m.group(2), m.group(3))
            if m.group(4):  # attached suffix: రూపాయలు + కి -> రూపాయలకి
                spoken = re.sub(r"రూపాయలు$", "రూపాయల", spoken) + m.group(4)
            return spoken
        text = re.sub(
            rf"(?:₹|\bRs\.?|\bINR\b)\s*({NUM})(?:\s*({SCALE_RE}))?"
            rf"(?:\s+(రూపాయ[{TE}]*|[Rr]upees))?([{TE}]+)?",
            lambda m: rupee_prefix(m) if not (m.group(3) or "").lower() == "rupees"
            else self._money(m.group(1), m.group(2), None), text)

        # 9,00,000 రూపాయలు | 5 lakh rupees
        def rupee_suffix(m):
            word = None if m.group(3).lower() == "rupees" else m.group(3)
            return self._money(m.group(1), m.group(2), word)
        text = re.sub(rf"{NOT_BEFORE}({NUM})(?:\s*({SCALE_RE}))?\s*(రూపాయ[{TE}]*|[Rr]upees)", rupee_suffix, text)

        # percent
        text = re.sub(rf"{NOT_BEFORE}({NUM})\s*(?:%|[Pp]ercent\b)",
                      lambda m: f"{_spoken_number(m.group(1), True)} శాతం", text)

        # 2026-27
        text = re.sub(rf"{NOT_BEFORE}(\d{{4}})\s*[-–—]\s*(\d{{2}})(?!\d)",
                      lambda m: f"{_spoken_int(m.group(1))} – {_spoken_int(m.group(2))}", text)

        # 5 lakh without rupees
        def scale_plain(m):
            base = SCALES[m.group(2)]
            single, joined, plural = SCALE_WORD[base]
            amount = m.group(1).replace(",", "")
            if amount == "1":
                return f"ఒక {single}"
            form = plural if m.group(2) in ("lakh", "lakhs", "lac", "lacs", "crore", "crores") else m.group(2)
            return f"{_spoken_number(amount, True)} {form}"
        text = re.sub(rf"{NOT_BEFORE}({NUM})\s*({SCALE_RE})(?![A-Za-z])", scale_plain, text)

        # 31st -> థర్టీ ఫస్ట్
        text = re.sub(rf"{NOT_BEFORE}(\d+)(st|nd|rd|th)(?![A-Za-z])",
                      lambda m: self._english_ordinal(int(m.group(1))), text)

        # 80C, 26AS
        text = re.sub(rf"{NOT_BEFORE}(\d+)([A-Z]+)(?![A-Za-z])", lambda m: self._alnum_ref(m.group(0)), text)

        # 1 శాతం -> ఒక శాతం; other numbers
        def plain(m):
            return _spoken_number(m.group(1), bool(m.group(2)) and m.group(1) == "1") + (m.group(2) or "")
        return re.sub(rf"{NOT_BEFORE}({NUM})(?!\d)(\s*(?:{CLASSIFIERS}))?", plain, text)

    def _english_ordinal(self, n: int) -> str:
        words = {1: "ఫస్ట్", 2: "సెకండ్", 3: "థర్డ్", 4: "ఫోర్త్", 5: "ఫిఫ్త్", 6: "సిక్స్త్", 7: "సెవెన్త్", 8: "ఎయిత్",
                 9: "నైన్త్", 10: "టెన్త్", 11: "ఎలెవెన్త్", 12: "ట్వెల్త్", 13: "థర్టీన్త్", 14: "ఫోర్టీన్త్",
                 15: "ఫిఫ్టీన్త్", 16: "సిక్స్టీన్త్", 17: "సెవెంటీన్త్", 18: "ఎయిటీన్త్", 19: "నైన్టీన్త్",
                 20: "ట్వంటీయత్", 30: "థర్టీయత్"}
        if n in words:
            return words[n]
        if n < 100 and n % 10:
            return f"{self.english_number(n - n % 10)} {words[n % 10]}"
        return self.english_number(n)

    # ------------------------------------------------------------ pipeline steps
    def clean_markers(self, text: str) -> str:
        """Removes production markers. [PAUSE Ns] becomes an internal token."""
        text = DROP_BLOCK.sub("\n\n", text)
        text = DROP_BRACKET.sub(" ", text)
        text = DROP_LINE.sub(" ", text)
        text = KEEP_LABEL.sub(" ", text)
        text = PAUSE.sub(lambda m: f"\x00{m.group(1) or DEFAULT_PAUSE_SECONDS}\x00", text)
        text = ANY_BRACKET_TAG.sub(" ", text)
        text = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "\n", text)
        text = re.sub(r"(?m)^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|?\s*$", "\n", text)  # table rule
        text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
        text = re.sub(r"(?m)^\s*(?:>|[-*+]\s)\s*", "", text)
        return re.sub(r"->|→|=>|[*#|`_~]", " ", text)

    def normalize(self, text: str) -> str:
        """Numbers, legal references and English into Telugu script. Keeps line breaks."""
        text = self._legal(text)
        text = self._numbers(text)
        text = self._lexicon_re.sub(lambda m: self._lookup(m.group(0)) or m.group(0), text)
        text = re.sub(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",
                      lambda m: " ".join(self._latin_word(p) for p in re.split(r"[-']", m.group(0))), text)
        text = text.replace("&", " అండ్ ").replace("+", " ప్లస్ ").replace("=", " ఈక్వల్స్ ")
        text = re.sub(r"[()\[\]{}]", ", ", text)
        text = re.sub(r"[\"“”‘’/\\]", " ", text)
        text = re.sub(r"(?<=\s)-(?=\s)|(?<=[^\s–])-(?=[^\s–])", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" +([,.?!।])", r"\1", text)
        text = re.sub(r",(\s*,)+", ",", text)
        text = re.sub(r"(?m)^[\s,]+|[\s,]+$", "", text)
        return re.sub(r",([.?!।])", r"\1", text)

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.?!।॥])\s*|\n+", text)
        return [p.strip() for p in parts if p and p.strip(" ,.?!।॥\n")]

    @staticmethod
    def estimate_seconds(text: str) -> float:
        return len(text.split()) / WORDS_PER_MINUTE * 60

    @classmethod
    def pack_chunks(cls, sentences: list[str], max_seconds: float = MAX_CHUNK_SECONDS,
                    min_seconds: float = MIN_CHUNK_SECONDS) -> list[str]:
        """Whole sentences only. A single overlong sentence stays one chunk."""
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and cls.estimate_seconds(candidate) > max_seconds:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            if chunks and cls.estimate_seconds(current) < min_seconds \
                    and cls.estimate_seconds(f"{chunks[-1]} {current}") <= max_seconds:
                chunks[-1] = f"{chunks[-1]} {current}"
            else:
                chunks.append(current)
        return chunks

    def prepare(self, text: str) -> list[Speech | Silence]:
        cleaned = self.clean_markers(text)
        segments: list[Speech | Silence] = []
        pieces = PAUSE_TOKEN.split(cleaned)  # odd indexes hold pause lengths
        for i, piece in enumerate(pieces):
            if i % 2:
                segments.append(Silence(float(piece)))
                continue
            for chunk in self.pack_chunks(self.split_sentences(self.normalize(piece))):
                segments.append(Speech(chunk, self.estimate_seconds(chunk)))
        return segments
