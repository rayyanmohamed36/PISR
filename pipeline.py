"""
pipeline.py — Main orchestrator for the PISR Exam-Bank Pipeline.

Usage:
    python pipeline.py                   # full run (auto-detects resources/ folder)
    python pipeline.py --resources ./pdfs # use a custom single-folder of PDFs
    python pipeline.py --parse-only      # parse PDFs and print summary
    python pipeline.py --init-db         # create schema only
    python pipeline.py --search "query"  # hybrid retrieval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from config import LOG_DIR, QP_DIR, MS_DIR, ER_DIR, RESOURCES_DIR

# ── Top-level logger ────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("pipeline")


def run_full_pipeline(
    qp_dir: Path = QP_DIR,
    ms_dir: Path = MS_DIR,
    er_dir: Path = ER_DIR,
    resources_dir: Optional[Path] = None,
    skip_upload: bool = False,
) -> None:
    """Execute the complete: parse → enrich → upload pipeline."""
    from parser import parse_all
    from enrichment import enrich_questions
    from database import create_schema, upload_questions

    t0 = time.time()

    # ── 1. Parse ─────────────────────────────────────────────────────────
    logger.info("═══ STEP 1/3: Parsing PDFs ═══")
    questions = parse_all(qp_dir, ms_dir, er_dir, resources_dir=resources_dir)
    if not questions:
        logger.error("No questions parsed. Check your PDF folders and filenames.")
        sys.exit(1)
    logger.info("Parsed %d questions in %.1fs.", len(questions), time.time() - t0)

    # ── 2. Enrich ────────────────────────────────────────────────────────
    t1 = time.time()
    logger.info("═══ STEP 2/3: Enriching (classify + embed) ═══")
    questions = enrich_questions(questions)
    logger.info("Enriched %d questions in %.1fs.", len(questions), time.time() - t1)

    # ── 3. Upload ────────────────────────────────────────────────────────
    if skip_upload:
        logger.info("Skipping upload (--skip-upload flag).")
        # Dump to JSON as a fallback
        out_path = LOG_DIR / "parsed_questions.json"
        _dump_json(questions, out_path)
        logger.info("Saved parsed data to %s", out_path)
    else:
        t2 = time.time()
        logger.info("═══ STEP 3/3: Creating schema & uploading ═══")
        create_schema()
        count = upload_questions(questions)
        logger.info("Uploaded %d rows in %.1fs.", count, time.time() - t2)

    total_time = time.time() - t0
    logger.info("═══ Pipeline complete — %d questions in %.1fs ═══", len(questions), total_time)


def _dump_json(questions, path: Path) -> None:
    """Serialise ParsedQuestion list to JSON (minus embeddings for readability)."""
    data = []
    for q in questions:
        data.append({
            "year": q.year,
            "season": q.season,
            "paper_code": q.paper_code,
            "question_num": q.question_num,
            "total_marks": q.total_marks,
            "topic": q.topic,
            "subtopic": q.subtopic,
            "question_text": q.question_text[:500],
            "mark_scheme_text": q.mark_scheme_text[:500],
            "examiner_report_text": q.examiner_report_text[:300],
            "embedding_dim": len(q.embedding),
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _label_display(label: str) -> str:
    """Convert internal label like 'a_ii' to displayable '(a)(ii)'."""
    if label == "stem":
        return "stem"
    if "_" in label:
        letter, roman = label.split("_", 1)
        return f"({letter})({roman})"
    return f"({label})"


def _write_parsed_txt(questions, path: Path) -> None:
    """Write all parsed questions and sub-parts to a human-readable txt file."""
    sep = "=" * 80
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"PISR Parse Output — {len(questions)} questions\n")
        f.write(f"{'=' * 80}\n\n")

        for q in questions:
            f.write(f"{sep}\n")
            f.write(
                f"  {q.year} {q.season} {q.paper_code}  —  "
                f"Question {q.question_num}  ({q.total_marks} marks)"
                f"  [{len(q.parts)} parts]\n"
            )
            f.write(f"{sep}\n\n")

            if not q.parts:
                # No sub-parts detected — show full text
                f.write("  [QP]\n")
                for line in q.question_text.splitlines():
                    f.write(f"    {line}\n")
                f.write("\n  [MS]\n")
                for line in q.mark_scheme_text.splitlines():
                    f.write(f"    {line}\n")
                f.write("\n  [ER]\n")
                for line in q.examiner_report_text.splitlines():
                    f.write(f"    {line}\n")
                f.write("\n")
                continue

            for sp in q.parts:
                display = _label_display(sp.label)
                marks_str = f"  ({sp.marks} mark{'s' if sp.marks != 1 else ''})" if sp.marks else ""
                f.write(f"  ── Part {display}{marks_str} ──\n")

                if sp.qp_text:
                    f.write("  [QP]\n")
                    for line in sp.qp_text.splitlines():
                        f.write(f"    {line}\n")

                if sp.ms_text:
                    f.write("  [MS]\n")
                    for line in sp.ms_text.splitlines():
                        f.write(f"    {line}\n")

                if sp.er_text:
                    f.write("  [ER]\n")
                    for line in sp.er_text.splitlines():
                        f.write(f"    {line}\n")

                if not any([sp.qp_text, sp.ms_text, sp.er_text]):
                    f.write("    (no content)\n")

                f.write("\n")

            f.write("\n")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="PISR Exam-Bank Pipeline")
    ap.add_argument("--parse-only", action="store_true", help="Parse PDFs and print summary (no enrichment/upload)")
    ap.add_argument("--init-db", action="store_true", help="Create the database schema only")
    ap.add_argument("--search", type=str, default=None, help="Run a hybrid search query")
    ap.add_argument("--skip-upload", action="store_true", help="Run parse + enrich but skip DB upload")
    ap.add_argument("--top-k", type=int, default=10, help="Number of results for search")
    ap.add_argument("--resources", type=str, default=None,
                     help="Single folder with all PDFs (auto-detects QP/MS/ER from content)")
    ap.add_argument("--qp-dir", type=str, default=None, help="Override QP directory")
    ap.add_argument("--ms-dir", type=str, default=None, help="Override MS directory")
    ap.add_argument("--er-dir", type=str, default=None, help="Override ER directory")
    args = ap.parse_args()

    resources = Path(args.resources) if args.resources else None
    qp = Path(args.qp_dir) if args.qp_dir else QP_DIR
    ms = Path(args.ms_dir) if args.ms_dir else MS_DIR
    er = Path(args.er_dir) if args.er_dir else ER_DIR

    # Auto-detect: if resources/ has PDFs and no explicit --qp-dir etc., use it
    if resources is None and not any([args.qp_dir, args.ms_dir, args.er_dir]):
        if RESOURCES_DIR.exists() and list(RESOURCES_DIR.glob("*.pdf")):
            resources = RESOURCES_DIR
            logger.info("Auto-detected resources/ folder with PDFs — using single-folder mode.")

    if args.init_db:
        from database import create_schema
        create_schema()
        print("Database schema created successfully.")
        return

    if args.search:
        from database import find_question
        results = find_question(args.search, top_k=args.top_k)
        if not results:
            print("No results found.")
            return
        print(f"\n{'─' * 80}")
        print(f"  Found {len(results)} results for: \"{args.search}\"")
        print(f"{'─' * 80}\n")
        for i, r in enumerate(results, 1):
            print(
                f"  {i}. [{r['similarity']:.3f}]  "
                f"{r['year']} {r['season']} {r['paper_code']} Q{r['question_num']}  "
                f"({r['total_marks']} marks)"
            )
            print(f"     Topic: {r['topic']} > {r['subtopic']}")
            preview = r["question_text"][:120].replace("\n", " ")
            print(f"     {preview}…\n")
        return

    if args.parse_only:
        from parser import parse_all
        questions = parse_all(qp, ms, er, resources_dir=resources)
        total_parts = sum(len(q.parts) for q in questions)
        print(f"\nParsed {len(questions)} questions ({total_parts} sub-parts):\n")
        for q in questions:
            print(
                f"  {q.year} {q.season} {q.paper_code} Q{q.question_num} "
                f"({q.total_marks} marks, {len(q.parts)} parts) — "
                f"QP:{len(q.question_text)}ch  MS:{len(q.mark_scheme_text)}ch  "
                f"ER:{len(q.examiner_report_text)}ch"
            )
            for sp in q.parts:
                if sp.label == "stem":
                    print(f"      ├─ [stem]  ({sp.marks}m)")
                else:
                    print(
                        f"      ├─ ({_label_display(sp.label)})  "
                        f"({sp.marks}m)  "
                        f"QP:{len(sp.qp_text)}ch  MS:{len(sp.ms_text)}ch  ER:{len(sp.er_text)}ch"
                    )

        # Write detailed txt output
        out_path = LOG_DIR / "parsed_output.txt"
        _write_parsed_txt(questions, out_path)
        print(f"\nDetailed output saved to {out_path}")
        return

    # Default: full pipeline
    run_full_pipeline(qp, ms, er, resources_dir=resources, skip_upload=args.skip_upload)


if __name__ == "__main__":
    main()
