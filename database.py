"""
database.py — SQL DDL, batch uploader, and hybrid retrieval.

Handles:
  • Creating the `exam_bank` table with pgvector + HNSW index.
  • Batch-uploading enriched ParsedQuestion records via psycopg2.
  • Hybrid retrieval: metadata SQL filters + cosine similarity ranking.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import psycopg2
import psycopg2.extras

from config import BATCH_SIZE, DATABASE_URL, LOG_DIR
from parser import ParsedQuestion

# ── Logging ──────────────────────────────────────────────────────────────
logger = logging.getLogger("database")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(LOG_DIR / "database.log", mode="a")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
logger.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
logger.addHandler(_ch)


# ═════════════════════════════════════════════════════════════════════════
#  DDL — Table & Index Creation
# ═════════════════════════════════════════════════════════════════════════

DDL_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

DDL_TABLE = """
CREATE TABLE IF NOT EXISTS exam_bank (
    id              SERIAL PRIMARY KEY,

    -- metadata
    year            INTEGER       NOT NULL,
    season          VARCHAR(20)   NOT NULL,
    paper_code      VARCHAR(20)   NOT NULL,
    question_num    INTEGER       NOT NULL,
    total_marks     INTEGER       NOT NULL DEFAULT 0,

    -- classification
    topic           VARCHAR(100)  NOT NULL DEFAULT 'Unknown',
    subtopic        VARCHAR(100)  NOT NULL DEFAULT 'Unknown',

    -- content
    question_text        TEXT NOT NULL,
    mark_scheme_text     TEXT NOT NULL DEFAULT '',
    examiner_report_text TEXT NOT NULL DEFAULT '',

    -- embedding (OpenAI text-embedding-3-large → 2000 dimensions)
    embedding       vector(2000),

    -- prevent accidental duplicates
    UNIQUE (year, season, paper_code, question_num)
);
"""

DDL_HNSW_INDEX = """
CREATE INDEX IF NOT EXISTS idx_exam_bank_embedding
    ON exam_bank
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
"""

DDL_BTREE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_exam_bank_year      ON exam_bank (year);
CREATE INDEX IF NOT EXISTS idx_exam_bank_topic     ON exam_bank (topic);
CREATE INDEX IF NOT EXISTS idx_exam_bank_subtopic  ON exam_bank (subtopic);
CREATE INDEX IF NOT EXISTS idx_exam_bank_paper     ON exam_bank (paper_code);
"""


def get_connection():
    """Return a new psycopg2 connection."""
    return psycopg2.connect(DATABASE_URL)


def create_schema() -> None:
    """Create the pgvector extension, table, and indexes."""
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(DDL_EXTENSION)
                cur.execute(DDL_TABLE)
                cur.execute(DDL_HNSW_INDEX)
                cur.execute(DDL_BTREE_INDEXES)
        logger.info("Schema created / verified successfully.")
    finally:
        conn.close()


# ═════════════════════════════════════════════════════════════════════════
#  Batch Uploader
# ═════════════════════════════════════════════════════════════════════════

INSERT_SQL = """
INSERT INTO exam_bank (
    year, season, paper_code, question_num, total_marks,
    topic, subtopic,
    question_text, mark_scheme_text, examiner_report_text,
    embedding
) VALUES (
    %(year)s, %(season)s, %(paper_code)s, %(question_num)s, %(total_marks)s,
    %(topic)s, %(subtopic)s,
    %(question_text)s, %(mark_scheme_text)s, %(examiner_report_text)s,
    %(embedding)s
)
ON CONFLICT (year, season, paper_code, question_num)
DO UPDATE SET
    total_marks          = EXCLUDED.total_marks,
    topic                = EXCLUDED.topic,
    subtopic             = EXCLUDED.subtopic,
    question_text        = EXCLUDED.question_text,
    mark_scheme_text     = EXCLUDED.mark_scheme_text,
    examiner_report_text = EXCLUDED.examiner_report_text,
    embedding            = EXCLUDED.embedding;
"""


def _question_to_row(q: ParsedQuestion) -> dict[str, Any]:
    """Convert a ParsedQuestion into a dict suitable for psycopg2 parameterised query."""
    # pgvector expects the embedding as a string like '[0.1, 0.2, ...]'
    emb_str = (
        "[" + ",".join(f"{v:.8f}" for v in q.embedding) + "]"
        if q.embedding
        else None
    )
    return {
        "year": q.year,
        "season": q.season,
        "paper_code": q.paper_code,
        "question_num": q.question_num,
        "total_marks": q.total_marks,
        "topic": q.topic or "Unknown",
        "subtopic": q.subtopic or "Unknown",
        "question_text": q.question_text,
        "mark_scheme_text": q.mark_scheme_text,
        "examiner_report_text": q.examiner_report_text,
        "embedding": emb_str,
    }


def upload_questions(questions: list[ParsedQuestion]) -> int:
    """
    Batch-upsert enriched questions into exam_bank.
    Returns the number of rows affected.
    """
    conn = get_connection()
    total = 0
    try:
        with conn:
            with conn.cursor() as cur:
                for i in range(0, len(questions), BATCH_SIZE):
                    batch = questions[i : i + BATCH_SIZE]
                    rows = [_question_to_row(q) for q in batch]
                    psycopg2.extras.execute_batch(cur, INSERT_SQL, rows, page_size=BATCH_SIZE)
                    total += len(batch)
                    logger.info(
                        "Uploaded batch %d–%d (%d rows).",
                        i, i + len(batch) - 1, len(batch),
                    )
        logger.info("Upload complete: %d rows upserted.", total)
    except Exception as exc:
        logger.error("Upload failed: %s", exc)
        raise
    finally:
        conn.close()
    return total


# ═════════════════════════════════════════════════════════════════════════
#  Hybrid Retrieval
# ═════════════════════════════════════════════════════════════════════════

def _extract_filters_via_llm(user_query: str) -> dict[str, Any]:
    """
    Use an LLM to extract structured SQL filters + a semantic search
    string from a natural-language query.

    Returns dict with optional keys:
        year, season, paper_code, topic, subtopic, search_text
    """
    import openai as _openai
    from config import CLASSIFIER_MODEL, OPENAI_API_KEY

    client = _openai.OpenAI(api_key=OPENAI_API_KEY)

    system = (
        "You are a query parser for an exam question database. "
        "Given the user's natural-language query, extract structured filters "
        "and a semantic search string.\n"
        "Return ONLY a JSON object with these optional keys:\n"
        '  "year"        — integer (e.g. 2021)\n'
        '  "season"      — string (Summer, Winter, Spring, Autumn)\n'
        '  "paper_code"  — string (e.g. P1, P2)\n'
        '  "topic"       — string from the syllabus taxonomy\n'
        '  "subtopic"    — string from the syllabus taxonomy\n'
        '  "search_text" — the core semantic query to match against question content\n'
        "Omit any key for which you have no information. Output ONLY valid JSON."
    )

    resp = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_query},
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Could not parse filter JSON: %s", raw)
        return {"search_text": user_query}


def find_question(
    user_query: str,
    top_k: int = 10,
    similarity_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: metadata filtering + cosine-similarity ranking.

    1. Calls an LLM to parse the user query into SQL filters + search text.
    2. Embeds the search text.
    3. Executes a single SQL query combining WHERE filters with
       cosine_similarity ORDER BY.
    """
    from enrichment import embed_single

    # 1. Extract filters
    filters = _extract_filters_via_llm(user_query)
    search_text = filters.pop("search_text", user_query)
    logger.info("Extracted filters: %s | search_text: %s", filters, search_text)

    # 2. Embed the search text
    query_embedding = embed_single(search_text)
    emb_literal = "[" + ",".join(f"{v:.8f}" for v in query_embedding) + "]"

    # 3. Build dynamic SQL
    where_clauses: list[str] = []
    params: dict[str, Any] = {}

    for col in ("year", "season", "paper_code", "topic", "subtopic"):
        if col in filters and filters[col]:
            where_clauses.append(f"{col} = %({col})s")
            params[col] = filters[col]

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT
            id, year, season, paper_code, question_num, total_marks,
            topic, subtopic,
            question_text, mark_scheme_text, examiner_report_text,
            1 - (embedding <=> %(emb)s::vector) AS similarity
        FROM exam_bank
        {where_sql}
        ORDER BY embedding <=> %(emb)s::vector
        LIMIT %(top_k)s;
    """
    params["emb"] = emb_literal
    params["top_k"] = top_k

    logger.debug("Executing hybrid query:\n%s\nParams: %s", sql, {k: v for k, v in params.items() if k != "emb"})

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for row in rows:
        r = dict(row)
        sim = r.pop("similarity", 0)
        if sim >= similarity_threshold:
            r["similarity"] = round(float(sim), 4)
            # Don't send full embedding back to caller
            r.pop("embedding", None)
            results.append(r)

    logger.info("Hybrid search returned %d results (threshold=%.2f).", len(results), similarity_threshold)
    return results


# ── CLI quick-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        create_schema()
        print("Schema created.")
    elif len(sys.argv) > 1 and sys.argv[1] == "search":
        query = " ".join(sys.argv[2:]) or "quadratic equations 2022"
        results = find_question(query)
        for r in results:
            print(
                f"  [{r['similarity']:.3f}] {r['year']} {r['season']} "
                f"{r['paper_code']} Q{r['question_num']} — {r['topic']} > {r['subtopic']}"
            )
    else:
        print("Usage: python database.py init | search <query>")
