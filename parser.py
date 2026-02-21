"""
parser.py — Triple-Stream PDF Parser

Reads PDFs from /QP, /MS, /ER directories — identifies year, season,
paper code, and document type (QP / MS / ER) by **reading the first
page** of each PDF.  Falls back to filename parsing if the cover page
doesn't yield enough information.

Splits each PDF into individual questions via regex, then merges the
three streams into unified question dictionaries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

from config import QP_DIR, MS_DIR, ER_DIR, RESOURCES_DIR, LOG_DIR

# ── Logging ──────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("parser")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(LOG_DIR / "parser.log", mode="a")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
logger.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
logger.addHandler(_ch)


# ── Boilerplate patterns to strip ────────────────────────────────────────
BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"©\s*\w.*", re.IGNORECASE),
    re.compile(r"turn\s+over", re.IGNORECASE),
    re.compile(r"page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),                          # lone page numbers
    re.compile(r"do\s+not\s+write\s+in\s+this", re.IGNORECASE),
    re.compile(r"question\s+paper\s+continues", re.IGNORECASE),
    re.compile(r"end\s+of\s+question\s+paper", re.IGNORECASE),
    re.compile(r"end\s+of\s+questions", re.IGNORECASE),
    re.compile(r"blank\s+page", re.IGNORECASE),
    re.compile(r"^\s*candidate\s+(?:number|surname)", re.IGNORECASE),
    re.compile(r"^\s*centre\s+number", re.IGNORECASE),
    re.compile(r"^\s*instructions?\s*$", re.IGNORECASE),
    re.compile(r"^\s*information\s*$", re.IGNORECASE),
    re.compile(r"^\s*advice\s*$", re.IGNORECASE),
    re.compile(r"^\s*total\s+mark\s+for\s+this\s+paper", re.IGNORECASE),
    # Pearson-specific boilerplate
    re.compile(r"^\s*\*P\w+\*\s*$"),                      # e.g. *P74633A0120*
    re.compile(r"^\s*P\d{5}[A-Z]\s*$"),                   # e.g. P74633A
    re.compile(r"^\s*DO\s*$"),                             # stray "DO" from vertical text
    re.compile(r"^\s*NOT\s*$"),                            # stray "NOT" from vertical text
    re.compile(r"^\s*WRITE\s*$"),                          # stray "WRITE"
    re.compile(r"^\s*IN\s*$"),                             # stray "IN"
    re.compile(r"^\s*THIS\s*$"),                           # stray "THIS"
    re.compile(r"^\s*AREA\s*$"),                           # stray "AREA"
    re.compile(r"^\s*AERA\s*$"),                           # reversed "AREA"
    re.compile(r"^\s*SIHT\s*$"),                           # reversed "THIS"
    re.compile(r"^\s*ETIRW\s*$"),                          # reversed "WRITE"
    re.compile(r"^\s*TON\s*$"),                            # reversed "NOT"
    re.compile(r"^\s*OD\s*$"),                             # reversed "DO"
    re.compile(r"^\s*NI\s*$"),                             # reversed "IN"
    re.compile(r"^\s*Z:\d+(/\d+)+\s*$"),                  # e.g. Z:1/1/1/1/1/
    re.compile(r"^\s*v\d+\s*$"),                           # e.g. v1
    re.compile(r"^\s*Other\s+names\s*$", re.IGNORECASE),
    re.compile(r"^\s*Please\s+check\s+the\s+examination", re.IGNORECASE),
    re.compile(r"^\s*Use\s+black\s+ink", re.IGNORECASE),
    re.compile(r"^\s*Fill\s+in\s+the\s+boxes", re.IGNORECASE),
    re.compile(r"^\s*Read\s+each\s+question\s+carefully", re.IGNORECASE),
    re.compile(r"^\s*Try\s+to\s+answer\s+every", re.IGNORECASE),
    re.compile(r"^\s*Check\s+your\s+answers", re.IGNORECASE),
    re.compile(r"^\s*You\s+do\s+not\s+need\s+any\s+other", re.IGNORECASE),
    re.compile(r"^\s*Total\s+Marks\s*$", re.IGNORECASE),
    re.compile(r"DD\d{5}"),                                # e.g. DD02652
    # Dotted answer lines (blank writing areas in QP)
    re.compile(r"^\s*\.{10,}\s*$"),                     # rows of dots
    # Stray single letters from vertical "DO NOT WRITE" fragments
    re.compile(r"^\s*[DO]\s*$"),                          # lone D or O
    # MS table-header remnants that bleed across sub-parts
    re.compile(r"^\s*Indicative\s+content\s*$", re.IGNORECASE),  # "Indicative content"
]

# Patterns that only apply to Mark Scheme text (not QP/ER)
# These are table-header remnants from the MS layout that bleed across sub-parts.
MS_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*Question\s+.*\bMark\s*$", re.IGNORECASE),  # "Question ... Mark"
    re.compile(r"^\s*Number\b", re.IGNORECASE),              # "Number ..." (MS table continuation)
    re.compile(r"^\s*Answer\s*$", re.IGNORECASE),             # "Answer"
    re.compile(r"^\s*Indicative\s+content\s*$", re.IGNORECASE),  # "Indicative content" (MS only)
]

# ── Question-boundary regex ─────────────────────────────────────────────
# Matches top-level question numbers.  We accept several formats:
#   "Question 1", "1 (a)", "1.", "1)", "1 As We Grow …"
# The last form (bare number + sentence) is common in Pearson papers.
# To avoid false positives on page numbers and dates like "25 June 2022"
# the bare-number + sentence pattern is restricted to numbers 1-9.
QUESTION_SPLIT_RE = re.compile(
    r"(?:^|\n)"                          # start of text / newline
    r"\s*"
    r"(?:"
        r"Question\s+(\d{1,2})"          # "Question 1" style  → group(1)
    r"|"
        r"(\d{1,2})"                     # any 1-2 digit num   → group(2)
        r"\s*"
        r"(?:"
            r"\(\s*[a-z]\s*\)"           #   followed by (a), (b), …
        r"|"
            r"(?:\.|\))\s"              #   or "1. " / "1) "
        r")"
    r"|"
        r"([1-9])"                       # single digit only    → group(3)
        r"\s+"
        r"(?![A-Z][a-z]+\s+\d{4})"      #   NOT followed by month + year (date)
        r"[A-Z][a-z]"                    #   start of a sentence "1 As We…"
    r")"
    ,
    re.MULTILINE,
)

# Mark-scheme: same logic.
MS_QUESTION_SPLIT_RE = re.compile(
    r"(?:^|\n)"
    r"\s*"
    r"(?:"
        r"Question\s+(\d{1,2})"          # → group(1)
    r"|"
        r"(\d{1,2})"                     # → group(2)
        r"\s*"
        r"(?:"
            r"\(\s*[a-z]\s*\)"
        r"|"
            r"(?:\.|\))\s"
        r")"
    r"|"
        r"([1-9])"                       # → group(3)
        r"\s+"
        r"(?![A-Z][a-z]+\s+\d{4})"
        r"[A-Z][a-z]"
    r")"
    ,
    re.MULTILINE,
)

# ── Marks extraction ────────────────────────────────────────────────────
# Matches: [5 marks], [5], (5 marks), (5), "Total: 20 marks", standalone "(6)"
MARKS_RE = re.compile(
    r"\[(\d{1,3})\s*(?:marks?|pts?)?\]"          # [5 marks] or [5]
    r"|"
    r"\((\d{1,3})\s*(?:marks?|pts?)?\)"          # (5 marks) or (5)
    r"|"
    r"(?:Total|total)[\s:]*(\d{1,3})\s*(?:marks?|pts?)?",
    re.IGNORECASE,
)


# ── Data structures ─────────────────────────────────────────────────────
@dataclass
class FileGroup:
    """Holds one synced set of QP / MS / ER paths."""
    year: int
    season: str
    paper_code: str
    qp_path: Optional[Path] = None
    ms_path: Optional[Path] = None
    er_path: Optional[Path] = None


@dataclass
class SubPart:
    """One sub-part of a question (e.g., part (a)(i), part (b))."""
    label: str           # "a_i", "a_ii", "b", "c", "stem", etc.
    marks: int = 0
    qp_text: str = ""
    ms_text: str = ""
    er_text: str = ""


@dataclass
class ParsedQuestion:
    """Single merged question record ready for enrichment."""
    year: int
    season: str
    paper_code: str
    question_num: int
    total_marks: int
    question_text: str
    mark_scheme_text: str
    examiner_report_text: str
    parts: list[SubPart] = field(default_factory=list)
    # filled later
    topic: str = ""
    subtopic: str = ""
    embedding: list[float] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────

def _clean_text(raw: str, extra_patterns: list[re.Pattern] | None = None) -> str:
    """Remove boilerplate lines from extracted text."""
    all_patterns = BOILERPLATE_PATTERNS
    if extra_patterns:
        all_patterns = BOILERPLATE_PATTERNS + extra_patterns
    lines = raw.splitlines()
    cleaned: list[str] = []
    for line in lines:
        if any(pat.search(line) for pat in all_patterns):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _extract_pdf_text(
    pdf_path: Path,
    extra_patterns: list[re.Pattern] | None = None,
) -> str:
    """Extract all text from a PDF, page by page, with boilerplate removed."""
    pages_text: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                raw = page.extract_text() or ""
                pages_text.append(_clean_text(raw, extra_patterns))
    except Exception as exc:
        logger.error("Failed to read %s: %s", pdf_path, exc)
        return ""
    return "\n".join(pages_text)


def _split_into_questions(
    full_text: str,
    split_re: re.Pattern = QUESTION_SPLIT_RE,
) -> dict[int, str]:
    """Split full document text into {question_number: text} dict."""
    # Find all question boundaries
    matches = list(split_re.finditer(full_text))
    if not matches:
        logger.warning("No question boundaries found; returning entire text as Q1.")
        return {1: full_text.strip()}

    questions: dict[int, str] = {}
    for i, m in enumerate(matches):
        # The question number can be in group(1), group(2), or group(3)
        # depending on which alternative matched.
        q_num_str = m.group(1) or m.group(2) or m.group(3)
        if not q_num_str:
            continue
        q_num = int(q_num_str)
        if q_num == 0:
            continue  # skip spurious zero matches
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        q_text = full_text[start:end].strip()
        # Merge: concatenate all chunks with the same question number.
        # This is essential for MS/ER where every sub-part (e.g. "1 (a)",
        # "1 (b)") matches the question-boundary regex independently.
        if q_num in questions:
            questions[q_num] += "\n" + q_text
        else:
            questions[q_num] = q_text

    return questions


def _extract_marks(text: str) -> int:
    """Extract total marks from a question text block.

    Strategy:
      - If a "Total: N marks" style line exists, use that.
      - Otherwise sum all the individual (N) mark indicators.
      - For very small sums (likely sub-parts weren't captured), fall
        back to the largest single value found.
    """
    all_marks = MARKS_RE.findall(text)
    if not all_marks:
        return 0

    bracket_vals: list[int] = []   # from [N] or [N marks]
    paren_vals: list[int] = []     # from (N) or (N marks)
    total_vals: list[int] = []     # from "Total: N"

    for groups in all_marks:
        if groups[0]:              # [N]
            bracket_vals.append(int(groups[0]))
        elif groups[1]:            # (N)
            paren_vals.append(int(groups[1]))
        elif groups[2]:            # Total: N
            total_vals.append(int(groups[2]))

    # Prefer explicit "Total" line
    if total_vals:
        return max(total_vals)
    # Bracket marks (common in many exam boards)
    if bracket_vals:
        return sum(bracket_vals)
    # Parenthesised marks (Pearson style): sum the sub-question marks
    if paren_vals:
        return sum(paren_vals)
    return 0


# ── Sub-part splitting ──────────────────────────────────────────────────

ROMAN_NUMERALS = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
ROMAN_TO_INT = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
    "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
}

# Matches a sub-part marker at the start of a line.
#   Groups:  (1) letter  (2) optional roman after letter  (3) standalone roman
# Handles QP "(a)(i)", MS "N (b)" styles.
SUBPART_MARKER_RE = re.compile(
    r"^\s*"
    r"(?:Question\s+\d{1,2}\s+|\d{1,2}\s+)?"   # optional prefix
    r"(?:"
        r"\(\s*([a-z])\s*\)"                     # group(1): letter (a)
        r"(?:\s*\(\s*([ivx]+)\s*\))?"            # group(2): optional roman (i)
    r"|"
        r"\(\s*([ivx]+)\s*\)"                    # group(3): standalone roman
    r")",
    re.MULTILINE,
)

# Stricter version for Examiner Reports — requires "Question N" or "N" prefix
# to avoid false positives from inline references like "(i) through to …"
SUBPART_MARKER_ER_RE = re.compile(
    r"^\s*"
    r"(?:Question\s+\d{1,2}\s+|\d{1,2}\s+)"    # REQUIRED prefix
    r"(?:"
        r"\(\s*([a-z])\s*\)"                     # group(1): letter
        r"(?:\s*\(\s*([ivx]+)\s*\))?"            # group(2): optional roman
    r"|"
        r"\(\s*([ivx]+)\s*\)"                    # group(3): standalone roman
    r")",
    re.MULTILINE,
)


def _subpart_sort_key(label: str) -> tuple:
    """Sort key for sub-part labels: stem < a < a_i < a_ii < … < b < …"""
    if label == "stem":
        return (0, 0, 0)
    if "_" in label:
        letter, roman = label.split("_", 1)
        return (1, ord(letter), ROMAN_TO_INT.get(roman, 99))
    return (1, ord(label), 0)


def _split_into_subparts(
    text: str,
    q_num: int,
    pattern: re.Pattern = SUBPART_MARKER_RE,
) -> dict[str, str]:
    """
    Split one question's text block into sub-parts.

    Works for all three streams (QP, MS, ER) by detecting patterns like:
      (a)(i), (ii), (b), Question N (c), N (d), etc.

    Returns dict mapping labels ("stem", "a_i", "b", …) to text blocks.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return {}

    current_letter: Optional[str] = None
    in_roman_seq: bool = False   # True after a (letter)(roman) combo
    markers: list[tuple[int, str]] = []     # (position, label)

    for m in matches:
        letter = m.group(1)
        roman_with_letter = m.group(2)
        standalone_roman = m.group(3)

        if letter:
            letter = letter.lower()
            if roman_with_letter:
                # Combined (a)(i) style
                roman = roman_with_letter.lower()
                if roman in ROMAN_NUMERALS:
                    current_letter = letter
                    in_roman_seq = True
                    label = f"{letter}_{roman}"
                else:
                    current_letter = letter
                    in_roman_seq = False
                    label = letter
            else:
                # Standalone (letter) — might be a roman numeral
                # continuing a sequence.  Single-char letters that are
                # valid roman digits (i, v, x) stay in the sequence
                # if one is active and the letter differs from the
                # current parent letter.
                if (in_roman_seq
                        and letter in ROMAN_NUMERALS
                        and current_letter
                        and letter != current_letter):
                    label = f"{current_letter}_{letter}"
                else:
                    current_letter = letter
                    in_roman_seq = False
                    label = letter
        elif standalone_roman:
            roman = standalone_roman.lower()
            if roman in ROMAN_NUMERALS and current_letter:
                label = f"{current_letter}_{roman}"
                in_roman_seq = True
            else:
                label = roman       # fallback
        else:
            continue

        markers.append((m.start(), label))

    if not markers:
        return {}

    parts: dict[str, str] = {}

    # Stem: introductory text before the first sub-part marker
    stem = text[: markers[0][0]].strip()
    if stem:
        parts["stem"] = stem

    for i, (pos, label) in enumerate(markers):
        end_pos = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        part_text = text[pos:end_pos].strip()

        # Strip trailing "Total for Question …" line
        part_text = re.sub(
            r"\n\s*\(Total\s+for\s+Question.*$", "",
            part_text, flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        if label in parts:
            parts[label] += "\n" + part_text
        else:
            parts[label] = part_text

    return parts


# ── File discovery ───────────────────────────────────────────────────────

# Standard naming: 2022_Summer_P1.pdf
FILENAME_RE = re.compile(
    r"^(\d{4})[_\-\s]+(Summer|Winter|Spring|Autumn|s|w|S|W)[_\-\s]+([\w]+)\.pdf$",
    re.IGNORECASE,
)

# Minimal naming: paper1.pdf, paper2.pdf
SIMPLE_FILENAME_RE = re.compile(
    r"^paper\s*[_\-]?\s*(\d{1,2})\.pdf$",
    re.IGNORECASE,
)

# Pearson naming: 4BS1-01-que-20231114.pdf
#   spec code  paper  type   date
PEARSON_FILENAME_RE = re.compile(
    r"^([\w]+)-(\d{2})-(que|rms|pef|ms|er|qp)-?(\d{8})?\.pdf$",
    re.IGNORECASE,
)
PEARSON_DOCTYPE_MAP = {
    "que": "QP", "qp": "QP",
    "rms": "MS", "ms": "MS",
    "pef": "ER", "er": "ER",
}

SEASON_MAP = {"s": "Summer", "w": "Winter"}


# ── First-page content detection ────────────────────────────────────────
# Regex patterns to extract metadata from the cover / first page of a PDF.

# Year: any 4-digit year between 2000–2099
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

# Season / session
_SEASON_RE = re.compile(
    r"\b(Summer|Winter|Spring|Autumn|June|January|May|October|November|"
    r"Jan|Jun|Oct|Nov)\b",
    re.IGNORECASE,
)
_SEASON_NORMALISE: dict[str, str] = {
    "summer": "Summer", "june": "Summer", "jun": "Summer", "may": "Summer",
    "winter": "Winter", "january": "Winter", "jan": "Winter",
    "november": "Winter", "nov": "Winter", "october": "Autumn",
    "oct": "Autumn", "autumn": "Autumn", "spring": "Spring",
}

# Paper code: e.g. "Paper 1", "Paper 2", "P1", "Component 01", "Unit C1"
# Also handles Pearson spec-style "Paper 4BS1/01" → extracts the 01 part
_PAPER_CODE_RE = re.compile(
    r"Paper\s+\w+/(\d{1,2})"             # "Paper 4BS1/01" → 01
    r"|"
    r"(?:Paper|Component|Unit)\s+(\d{1,2}|[A-Z]\d{0,2})"  # "Paper 1"
    r"|"
    r"\b([PCSFM]\d{1,2})\b",            # standalone P1, C3, etc.
    re.IGNORECASE,
)

# Document type detection
_DOCTYPE_PATTERNS: dict[str, list[re.Pattern]] = {
    "QP": [
        re.compile(r"question\s+paper", re.IGNORECASE),
        re.compile(r"answer\s+all\s+(?:the\s+)?questions", re.IGNORECASE),
        re.compile(r"write\s+your\s+answer", re.IGNORECASE),
        re.compile(r"instructions?\s+to\s+candidates", re.IGNORECASE),
        re.compile(r"time\s+allowed", re.IGNORECASE),
        re.compile(r"answer\s+book", re.IGNORECASE),
        re.compile(r"calculators?\s+(?:may|must)", re.IGNORECASE),
    ],
    "MS": [
        re.compile(r"mark\s+scheme", re.IGNORECASE),
        re.compile(r"marking\s+(?:scheme|guide|instructions)", re.IGNORECASE),
        re.compile(r"generic\s+marking\s+principles", re.IGNORECASE),
        re.compile(r"\bM1\b.*\bA1\b", re.IGNORECASE),
    ],
    "ER": [
        re.compile(r"examiner(?:s|'s|s')?\s+report", re.IGNORECASE),
        re.compile(r"principal\s+examiner", re.IGNORECASE),
        re.compile(r"examiners?\s+comment", re.IGNORECASE),
        re.compile(r"candidate(?:s)?\s+(?:found|struggled|performed)", re.IGNORECASE),
        re.compile(r"report\s+on\s+(?:the\s+)?examination", re.IGNORECASE),
    ],
}


@dataclass
class PDFMeta:
    """Metadata extracted from the first page of a PDF."""
    year: Optional[int] = None
    season: Optional[str] = None
    paper_code: Optional[str] = None
    doc_type: Optional[str] = None          # "QP", "MS", or "ER"
    confidence: float = 0.0                 # 0-1, how many fields were detected


def detect_pdf_metadata(pdf_path: Path) -> PDFMeta:
    """
    Read the first page (and optionally the second) of a PDF and extract:
      - year, season, paper_code, doc_type
    Returns a PDFMeta with whatever could be determined.
    """
    meta = PDFMeta()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Read up to the first 2 pages for cover-page info
            pages_to_read = min(2, len(pdf.pages))
            first_text = ""
            for i in range(pages_to_read):
                first_text += (pdf.pages[i].extract_text() or "") + "\n"
    except Exception as exc:
        logger.error("Could not read first page of %s: %s", pdf_path, exc)
        return meta

    if not first_text.strip():
        return meta

    fields_found = 0

    # ── Year ──────────────────────────────────────────────────────────
    year_matches = _YEAR_RE.findall(first_text)
    if year_matches:
        # Prefer years in a reasonable exam range
        candidates = [int(y) for y in year_matches if 2000 <= int(y) <= 2030]
        if candidates:
            meta.year = candidates[0]
            fields_found += 1

    # ── Season ────────────────────────────────────────────────────────
    season_match = _SEASON_RE.search(first_text)
    if season_match:
        raw = season_match.group(1).lower()
        meta.season = _SEASON_NORMALISE.get(raw, raw.capitalize())
        fields_found += 1

    # ── Paper code ────────────────────────────────────────────────────
    paper_match = _PAPER_CODE_RE.search(first_text)
    if paper_match:
        code = paper_match.group(1) or paper_match.group(2) or paper_match.group(3)
        # Normalise: "1" → "P1", "01" → "P1", "C1" stays "C1"
        code = code.strip().lstrip("0")
        if code.isdigit():
            code = f"P{code}"
        meta.paper_code = code.upper()
        fields_found += 1

    # ── Document type ─────────────────────────────────────────────────
    # Weight MS and ER signals higher — they're rarer and more
    # distinctive than generic QP phrases like "answer all questions".
    type_weights: dict[str, float] = {"QP": 1.0, "MS": 2.5, "ER": 2.5}
    type_scores: dict[str, float] = {"QP": 0.0, "MS": 0.0, "ER": 0.0}
    for dtype, patterns in _DOCTYPE_PATTERNS.items():
        for pat in patterns:
            if pat.search(first_text):
                type_scores[dtype] += type_weights[dtype]

    best_type = max(type_scores, key=type_scores.get)  # type: ignore[arg-type]
    if type_scores[best_type] > 0:
        meta.doc_type = best_type
        fields_found += 1

    meta.confidence = fields_found / 4.0

    logger.debug(
        "Detected metadata for %s: year=%s, season=%s, paper=%s, type=%s (conf=%.0f%%)",
        pdf_path.name, meta.year, meta.season, meta.paper_code,
        meta.doc_type, meta.confidence * 100,
    )
    return meta


@dataclass
class FilenameMeta:
    """Metadata extracted from a filename."""
    year: Optional[int] = None
    season: Optional[str] = None
    paper_code: Optional[str] = None
    doc_type: Optional[str] = None   # "QP", "MS", "ER" or None


def _parse_filename(path: Path) -> Optional[FilenameMeta]:
    """Extract whatever metadata we can from the filename.

    Supports:
      - Standard:  2022_Summer_P1.pdf
      - Simple:    paper1.pdf, paper2.pdf
      - Pearson:   4BS1-01-que-20231114.pdf
    """
    name = path.name

    # ── Standard format ───────────────────────────────────────────────
    m = FILENAME_RE.match(name)
    if m:
        year = int(m.group(1))
        season_raw = m.group(2)
        season = SEASON_MAP.get(season_raw.lower(), season_raw.capitalize())
        paper_code = m.group(3).upper()
        return FilenameMeta(year=year, season=season, paper_code=paper_code)

    # ── Pearson format ────────────────────────────────────────────────
    m = PEARSON_FILENAME_RE.match(name)
    if m:
        paper_num = m.group(2).lstrip("0") or "1"
        paper_code = f"P{paper_num}"
        doc_code = m.group(3).lower()
        doc_type = PEARSON_DOCTYPE_MAP.get(doc_code)
        # Try to extract year from the date portion (YYYYMMDD)
        date_str = m.group(4)
        year = int(date_str[:4]) if date_str and len(date_str) == 8 else None
        return FilenameMeta(year=year, paper_code=paper_code, doc_type=doc_type)

    # ── Simple format (paper1.pdf) ────────────────────────────────────
    m = SIMPLE_FILENAME_RE.match(name)
    if m:
        paper_code = f"P{m.group(1)}"
        return FilenameMeta(paper_code=paper_code)

    return None


def _resolve_metadata(
    pdf_path: Path,
    folder_hint: str,
) -> Optional[tuple[int, str, str, str]]:
    """
    Determine (year, season, paper_code, doc_type) for a PDF.

    Strategy:
      1. Read the first page and extract metadata via regex.
      2. Fall back to filename parsing for any missing fields.
      3. Use the folder name (QP / MS / ER) as doc_type hint if
         first-page detection is inconclusive.

    Returns None if year cannot be determined at all.
    """
    # Primary: first-page detection
    meta = detect_pdf_metadata(pdf_path)

    # Secondary: filename parsing
    fn = _parse_filename(pdf_path)

    year = meta.year or (fn.year if fn else None)
    season = meta.season or (fn.season if fn else None) or "Unknown"
    paper_code = meta.paper_code or (fn.paper_code if fn else None) or "P1"
    # Doc type: filename > first-page > folder hint
    # (filename-based type like Pearson "que"/"rms"/"pef" is very reliable)
    doc_type = (fn.doc_type if fn and fn.doc_type else None) or meta.doc_type or folder_hint

    if year is None:
        logger.warning(
            "Could not determine year for %s (neither first-page nor filename). Skipping.",
            pdf_path.name,
        )
        return None

    # Log which source provided each field
    sources = []
    if meta.year:
        sources.append("year:page")
    elif fn and fn.year:
        sources.append("year:filename")
    if meta.season:
        sources.append("season:page")
    elif fn and fn.season:
        sources.append("season:filename")
    if meta.paper_code:
        sources.append("paper:page")
    elif fn and fn.paper_code:
        sources.append("paper:filename")
    if fn and fn.doc_type:
        sources.append(f"type:filename({fn.doc_type})")
    elif meta.doc_type:
        sources.append(f"type:page({meta.doc_type})")
    else:
        sources.append(f"type:folder({folder_hint})")

    logger.info(
        "Resolved %s → %d %s %s [%s]  (sources: %s)",
        pdf_path.name, year, season, paper_code, doc_type,
        ", ".join(sources),
    )
    return year, season, paper_code, doc_type


def discover_file_groups(
    qp_dir: Path = QP_DIR,
    ms_dir: Path = MS_DIR,
    er_dir: Path = ER_DIR,
) -> list[FileGroup]:
    """
    Scan the three folders and group files by (year, season, paper_code).

    Each PDF's first page is read to detect year, season, paper code, and
    document type.  The folder it lives in serves as a hint for doc_type
    but can be overridden by what's actually on the cover page.
    """
    groups: dict[tuple[int, str, str], FileGroup] = {}

    def _register(directory: Path, folder_hint: str) -> None:
        if not directory.exists():
            logger.warning("Directory does not exist: %s", directory)
            return
        for pdf in sorted(directory.glob("**/*.pdf")):
            resolved = _resolve_metadata(pdf, folder_hint)
            if resolved is None:
                continue
            year, season, paper_code, doc_type = resolved

            key = (year, season, paper_code)
            if key not in groups:
                groups[key] = FileGroup(year=year, season=season, paper_code=paper_code)

            # Assign to the correct slot based on detected doc_type
            attr = {
                "QP": "qp_path",
                "MS": "ms_path",
                "ER": "er_path",
            }.get(doc_type, f"{folder_hint.lower()}_path")

            existing = getattr(groups[key], attr)
            if existing is not None and existing != pdf:
                logger.warning(
                    "Duplicate %s for %s: already have %s, now found %s. Keeping first.",
                    doc_type, key, existing.name, pdf.name,
                )
            else:
                setattr(groups[key], attr, pdf)

    _register(qp_dir, "QP")
    _register(ms_dir, "MS")
    _register(er_dir, "ER")

    ordered = sorted(groups.values(), key=lambda g: (g.year, g.season, g.paper_code))
    logger.info("Discovered %d file groups across QP/MS/ER.", len(ordered))
    return ordered


def discover_from_single_folder(
    resources_dir: Path = RESOURCES_DIR,
) -> list[FileGroup]:
    """
    Scan a single folder containing a mix of QP, MS, and ER PDFs.

    Each PDF's first page is read to detect year, season, paper code,
    and — crucially — document type.  Files are then auto-sorted into
    the correct slot (qp_path / ms_path / er_path) of each FileGroup.
    """
    if not resources_dir.exists():
        logger.warning("Resources directory does not exist: %s", resources_dir)
        return []

    groups: dict[tuple[int, str, str], FileGroup] = {}

    for pdf in sorted(resources_dir.glob("**/*.pdf")):
        resolved = _resolve_metadata(pdf, folder_hint="QP")  # hint is irrelevant; first-page wins
        if resolved is None:
            continue
        year, season, paper_code, doc_type = resolved

        key = (year, season, paper_code)
        if key not in groups:
            groups[key] = FileGroup(year=year, season=season, paper_code=paper_code)

        attr = {
            "QP": "qp_path",
            "MS": "ms_path",
            "ER": "er_path",
        }.get(doc_type, "qp_path")

        existing = getattr(groups[key], attr)
        if existing is not None and existing != pdf:
            logger.warning(
                "Duplicate %s for %s: already have %s, now found %s. Keeping first.",
                doc_type, key, existing.name, pdf.name,
            )
        else:
            setattr(groups[key], attr, pdf)

    ordered = sorted(groups.values(), key=lambda g: (g.year, g.season, g.paper_code))
    logger.info(
        "Discovered %d file groups from single resources folder (%d PDFs).",
        len(ordered), len(list(resources_dir.glob("**/*.pdf"))),
    )
    return ordered


# ── Main parse function ─────────────────────────────────────────────────

def parse_all(
    qp_dir: Path = QP_DIR,
    ms_dir: Path = MS_DIR,
    er_dir: Path = ER_DIR,
    resources_dir: Optional[Path] = None,
) -> list[ParsedQuestion]:
    """
    Full triple-stream parse:
      1. Discover & sync file groups.
      2. Extract + clean text.
      3. Split into questions.
      4. Merge QP / MS / ER by question number.

    If resources_dir is provided, all PDFs are read from that single
    folder and auto-sorted by document type detected on the first page.
    Returns a list of ParsedQuestion ready for enrichment.
    """
    if resources_dir is not None:
        file_groups = discover_from_single_folder(resources_dir)
    else:
        file_groups = discover_file_groups(qp_dir, ms_dir, er_dir)
    results: list[ParsedQuestion] = []

    for fg in file_groups:
        label = f"{fg.year}_{fg.season}_{fg.paper_code}"
        logger.info("Processing group: %s", label)

        # --- Extract text -------------------------------------------------
        qp_text = _extract_pdf_text(fg.qp_path) if fg.qp_path else ""
        ms_text = _extract_pdf_text(fg.ms_path, MS_BOILERPLATE_PATTERNS) if fg.ms_path else ""
        er_text = _extract_pdf_text(fg.er_path) if fg.er_path else ""

        if not qp_text:
            logger.warning("No QP text for %s — skipping group.", label)
            continue

        # --- Split into questions -----------------------------------------
        qp_questions = _split_into_questions(qp_text, QUESTION_SPLIT_RE)
        ms_questions = _split_into_questions(ms_text, MS_QUESTION_SPLIT_RE) if ms_text else {}
        er_questions = _split_into_questions(er_text, QUESTION_SPLIT_RE) if er_text else {}

        # --- Mismatch warning ---------------------------------------------
        if ms_questions and len(qp_questions) != len(ms_questions):
            logger.error(
                "MISMATCH [%s]: QP has %d questions but MS has %d. "
                "Proceeding with available data.",
                label, len(qp_questions), len(ms_questions),
            )

        if er_questions and len(qp_questions) != len(er_questions):
            logger.warning(
                "MISMATCH [%s]: QP has %d questions but ER has %d.",
                label, len(qp_questions), len(er_questions),
            )

        # --- Merge streams ------------------------------------------------
        for q_num, q_text in sorted(qp_questions.items()):
            marks = _extract_marks(q_text)
            # Fall back to MS marks if QP doesn't show them
            if marks == 0 and q_num in ms_questions:
                marks = _extract_marks(ms_questions[q_num])

            ms_q_text = ms_questions.get(q_num, "")
            er_q_text = er_questions.get(q_num, "")

            # --- Split into sub-parts ------------------------------------
            qp_subs = _split_into_subparts(q_text, q_num)
            ms_subs = _split_into_subparts(ms_q_text, q_num) if ms_q_text else {}
            er_subs = _split_into_subparts(er_q_text, q_num, pattern=SUBPART_MARKER_ER_RE) if er_q_text else {}

            # Collect all unique labels and sort naturally
            all_labels = sorted(
                set(list(qp_subs.keys()) + list(ms_subs.keys()) + list(er_subs.keys())),
                key=_subpart_sort_key,
            )

            merged_parts: list[SubPart] = []
            for lbl in all_labels:
                sp_qp = qp_subs.get(lbl, "")
                sp_ms = ms_subs.get(lbl, "")
                sp_er = er_subs.get(lbl, "")
                sp_marks = _extract_marks(sp_qp) if sp_qp else 0
                if sp_marks == 0 and sp_ms:
                    sp_marks = _extract_marks(sp_ms)
                merged_parts.append(SubPart(
                    label=lbl,
                    marks=sp_marks,
                    qp_text=sp_qp,
                    ms_text=sp_ms,
                    er_text=sp_er,
                ))

            pq = ParsedQuestion(
                year=fg.year,
                season=fg.season,
                paper_code=fg.paper_code,
                question_num=q_num,
                total_marks=marks,
                question_text=q_text,
                mark_scheme_text=ms_q_text,
                examiner_report_text=er_q_text,
                parts=merged_parts,
            )
            results.append(pq)
            logger.debug(
                "  Q%d — marks=%d, %d parts, QP=%d chars, MS=%d chars, ER=%d chars",
                q_num, marks, len(merged_parts),
                len(pq.question_text),
                len(pq.mark_scheme_text),
                len(pq.examiner_report_text),
            )

    logger.info("Parsed %d total questions from %d file groups.", len(results), len(file_groups))
    return results


# ── CLI quick-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = parse_all()
    for q in questions[:5]:
        print(
            f"[{q.year} {q.season} {q.paper_code}] Q{q.question_num} "
            f"({q.total_marks} marks) — "
            f"QP: {len(q.question_text)} chars, "
            f"MS: {len(q.mark_scheme_text)} chars, "
            f"ER: {len(q.examiner_report_text)} chars"
        )
