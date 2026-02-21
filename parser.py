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
    re.compile(r"do\s+not\s+write\s+in\s+this\s+margin", re.IGNORECASE),
    re.compile(r"question\s+paper\s+continues", re.IGNORECASE),
    re.compile(r"end\s+of\s+question\s+paper", re.IGNORECASE),
    re.compile(r"end\s+of\s+questions", re.IGNORECASE),
    re.compile(r"blank\s+page", re.IGNORECASE),
    re.compile(r"^\s*candidate\s+number", re.IGNORECASE),
    re.compile(r"^\s*centre\s+number", re.IGNORECASE),
    re.compile(r"^\s*instructions?\s*$", re.IGNORECASE),
    re.compile(r"^\s*information\s*$", re.IGNORECASE),
    re.compile(r"^\s*advice\s*$", re.IGNORECASE),
    re.compile(r"^\s*total\s+mark\s+for\s+this\s+paper", re.IGNORECASE),
]

# ── Question-boundary regex ─────────────────────────────────────────────
# Matches lines like "1", "1.", "1 ", "Question 1", "Q1", etc.
QUESTION_SPLIT_RE = re.compile(
    r"(?:^|\n)"                          # start of line
    r"\s*"
    r"(?:Question\s+|Q\.?\s*)*"          # optional "Question" / "Q" prefix
    r"(\d{1,2})"                         # the question number  (capture group 1)
    r"[\.\)\s]"                          # followed by dot / paren / space
    ,
    re.IGNORECASE,
)

# Stricter version for mark-scheme (often formatted differently)
MS_QUESTION_SPLIT_RE = re.compile(
    r"(?:^|\n)"
    r"\s*"
    r"(?:Question\s+|Q\.?\s*)*"
    r"(\d{1,2})"
    r"[\.\)\s\(]"
    ,
    re.IGNORECASE,
)

# ── Marks extraction ────────────────────────────────────────────────────
MARKS_RE = re.compile(
    r"\[(\d{1,3})\s*(?:marks?|pts?)?\]"
    r"|"
    r"\((\d{1,3})\s*(?:marks?|pts?)\)"
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
    # filled later
    topic: str = ""
    subtopic: str = ""
    embedding: list[float] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Remove boilerplate lines from extracted text."""
    lines = raw.splitlines()
    cleaned: list[str] = []
    for line in lines:
        if any(pat.search(line) for pat in BOILERPLATE_PATTERNS):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF, page by page, with boilerplate removed."""
    pages_text: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                raw = page.extract_text() or ""
                pages_text.append(_clean_text(raw))
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
        q_num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        q_text = full_text[start:end].strip()
        # Deduplicate: keep the longest version if we find multiple Q with same num
        if q_num in questions:
            if len(q_text) > len(questions[q_num]):
                questions[q_num] = q_text
        else:
            questions[q_num] = q_text

    return questions


def _extract_marks(text: str) -> int:
    """Try to extract total marks from a question text block."""
    all_marks = MARKS_RE.findall(text)
    if not all_marks:
        return 0
    # Each match is a tuple of groups; pick the non-empty one
    values: list[int] = []
    for groups in all_marks:
        for g in groups:
            if g:
                values.append(int(g))
    # Total marks is usually the last (or the sum of parts)
    return values[-1] if values else 0


# ── File discovery ───────────────────────────────────────────────────────

FILENAME_RE = re.compile(
    r"^(\d{4})[_\-\s]+(Summer|Winter|Spring|Autumn|s|w|S|W)[_\-\s]+([\w]+)\.pdf$",
    re.IGNORECASE,
)

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
_PAPER_CODE_RE = re.compile(
    r"(?:Paper|Component|Unit)\s*(\d{1,2}|[A-Z]\d{0,2})"
    r"|"
    r"\b([PCSFM]\d{1,2})\b",
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
        code = paper_match.group(1) or paper_match.group(2)
        # Normalise: "1" → "P1", "01" → "P1", "C1" stays "C1"
        code = code.strip().lstrip("0")
        if code.isdigit():
            code = f"P{code}"
        meta.paper_code = code.upper()
        fields_found += 1

    # ── Document type ─────────────────────────────────────────────────
    type_scores: dict[str, int] = {"QP": 0, "MS": 0, "ER": 0}
    for dtype, patterns in _DOCTYPE_PATTERNS.items():
        for pat in patterns:
            if pat.search(first_text):
                type_scores[dtype] += 1

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


def _parse_filename(path: Path) -> Optional[tuple[int, str, str]]:
    """Return (year, season, paper_code) from filename, or None."""
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    year = int(m.group(1))
    season_raw = m.group(2)
    season = SEASON_MAP.get(season_raw.lower(), season_raw.capitalize())
    paper_code = m.group(3).upper()
    return year, season, paper_code


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
    fn_parsed = _parse_filename(pdf_path)

    year = meta.year or (fn_parsed[0] if fn_parsed else None)
    season = meta.season or (fn_parsed[1] if fn_parsed else None) or "Unknown"
    paper_code = meta.paper_code or (fn_parsed[2] if fn_parsed else None) or "P1"
    doc_type = meta.doc_type or folder_hint

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
    elif fn_parsed:
        sources.append("year:filename")
    if meta.season:
        sources.append("season:page")
    elif fn_parsed and fn_parsed[1]:
        sources.append("season:filename")
    if meta.paper_code:
        sources.append("paper:page")
    elif fn_parsed and fn_parsed[2]:
        sources.append("paper:filename")
    if meta.doc_type:
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
        for pdf in sorted(directory.glob("*.pdf")):
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

    for pdf in sorted(resources_dir.glob("*.pdf")):
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
        len(ordered), len(list(resources_dir.glob("*.pdf"))),
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
        ms_text = _extract_pdf_text(fg.ms_path) if fg.ms_path else ""
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

            pq = ParsedQuestion(
                year=fg.year,
                season=fg.season,
                paper_code=fg.paper_code,
                question_num=q_num,
                total_marks=marks,
                question_text=q_text,
                mark_scheme_text=ms_questions.get(q_num, ""),
                examiner_report_text=er_questions.get(q_num, ""),
            )
            results.append(pq)
            logger.debug(
                "  Q%d — marks=%d, QP=%d chars, MS=%d chars, ER=%d chars",
                q_num, marks,
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
