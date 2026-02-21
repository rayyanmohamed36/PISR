# PISR — Processing and Indexing System for Raisegrade

A fully automated pipeline that processes years of PDF exam papers (Question Papers, Mark Schemes, Examiner Reports) into a single-table **PostgreSQL + pgvector** database with LLM-powered classification and semantic search.

---

## Architecture

```
 ┌──────────┐   ┌──────────┐   ┌──────────┐
 │  /QP     │   │  /MS     │   │  /ER     │
 │  PDFs    │   │  PDFs    │   │  PDFs    │
 └────┬─────┘   └────┬─────┘   └────┬─────┘
      │              │              │
      └──────────┬───┘──────────────┘
                 │
        ┌────────▼────────┐
        │  Triple-Stream  │  pdfplumber + regex
        │     Parser      │  boilerplate removal
        └────────┬────────┘
                 │  ParsedQuestion dicts
        ┌────────▼────────┐
        │   Enrichment    │  GPT-4o-mini classification
        │     Layer       │  text-embedding-3-small (1536d)
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │   Batch Upload  │  psycopg2 upsert
        │   → PostgreSQL  │  pgvector HNSW index
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  Hybrid Search  │  SQL filters + cosine similarity
        │   find_question │
        └─────────────────┘
```

## Features

- **Triple-stream sync** — matches QP, MS, and ER files by filename, then merges Question N across all three documents into a single record.
- **Boilerplate removal** — strips headers, footers, page numbers, "Turn Over" notices, and other noise via 14 regex patterns.
- **Mismatch detection** — logs errors when the question count in a Mark Scheme doesn't match the Question Paper.
- **LLM classification** — sends each question to GPT-4o-mini to pick the best Topic > Subtopic from a configurable taxonomy.
- **Vector embeddings** — batches combined (Q + MS + ER) text through `text-embedding-3-small` for 1536-dim vectors.
- **Upsert logic** — `ON CONFLICT` prevents duplicates; re-running the pipeline safely updates existing rows.
- **Hybrid retrieval** — `find_question()` uses an LLM to parse natural-language queries into SQL filters + cosine-similarity ranking in a single query.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| PostgreSQL | 14+ with [pgvector](https://github.com/pgvector/pgvector) extension |
| OpenAI API key | any tier |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```
OPENAI_API_KEY=sk-your-key-here
DATABASE_URL=postgresql://user:password@localhost:5432/exam_bank
```

### 3. Place your PDFs

Drop files into the three input folders using the naming convention:

```
QP/2022_Summer_P1.pdf
MS/2022_Summer_P1.pdf
ER/2022_Summer_P1.pdf
```

**Filename format:** `<YEAR>_<Season>_<PaperCode>.pdf`

| Part | Examples |
|---|---|
| Year | `2020`, `2021`, `2022` |
| Season | `Summer`, `Winter`, `Spring`, `Autumn`, `s`, `w` |
| Paper Code | `P1`, `P2`, `C1`, `S1` |

### 4. Create the database

```bash
# Create the PostgreSQL database (if it doesn't exist)
createdb exam_bank

# Initialise the schema (extension + table + indexes)
python pipeline.py --init-db
```

Or apply the DDL manually:

```bash
psql exam_bank < schema.sql
```

### 5. Run the pipeline

```bash
# Full run: parse → classify → embed → upload
python pipeline.py

# Parse only (no API calls, no DB)
python pipeline.py --parse-only

# Parse + enrich, but skip the DB upload (saves to logs/parsed_questions.json)
python pipeline.py --skip-upload
```

### 6. Search

```bash
python pipeline.py --search "quadratic equations from 2021"
python pipeline.py --search "integration by parts Summer P2" --top-k 5
```

---

## Project Structure

```
PISR/
├── QP/                  # Question Paper PDFs
├── MS/                  # Mark Scheme PDFs
├── ER/                  # Examiner Report PDFs
├── logs/                # Runtime logs + JSON dumps
├── config.py            # Paths, API keys, topic taxonomy
├── parser.py            # Triple-stream PDF parser
├── enrichment.py        # LLM classification + embeddings
├── database.py          # DDL, batch uploader, hybrid retrieval
├── pipeline.py          # Main CLI orchestrator
├── schema.sql           # Standalone SQL for manual DB setup
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
└── README.md
```

---

## Data Model

Single table: **`exam_bank`**

| Column | Type | Description |
|---|---|---|
| `id` | `SERIAL PK` | Auto-incrementing primary key |
| `year` | `INTEGER` | Exam year (e.g. 2022) |
| `season` | `VARCHAR(20)` | Summer, Winter, etc. |
| `paper_code` | `VARCHAR(20)` | P1, P2, C1, etc. |
| `question_num` | `INTEGER` | Question number within the paper |
| `total_marks` | `INTEGER` | Marks for the question |
| `topic` | `VARCHAR(100)` | LLM-assigned topic |
| `subtopic` | `VARCHAR(100)` | LLM-assigned subtopic |
| `question_text` | `TEXT` | Full question text |
| `mark_scheme_text` | `TEXT` | Corresponding mark scheme |
| `examiner_report_text` | `TEXT` | Corresponding examiner report |
| `embedding` | `vector(1536)` | Combined-text embedding |

**Indexes:** HNSW on `embedding` (cosine), B-tree on `year`, `topic`, `subtopic`, `paper_code`.

---

## Topic Taxonomy

The default taxonomy in `config.py` covers:

- **Algebra** — Linear/Quadratic/Simultaneous Equations, Sequences, Polynomials, Binomial Expansion, …
- **Calculus** — Differentiation, Integration, Differential Equations, Numerical Methods, …
- **Geometry** — Coordinate Geometry, Vectors, Trigonometry, Parametric Equations, …
- **Statistics** — Probability, Distributions (Normal/Binomial/Poisson), Hypothesis Testing, …
- **Mechanics** — Kinematics, Forces, Moments, Energy, Projectiles, …
- **Pure Mathematics** — Proof, Complex Numbers, Matrices, Polar Coordinates, …
- **Number** — Surds, Ratio, Number Theory

Edit `TOPIC_TAXONOMY` in `config.py` to match your syllabus.

---

## Configuration Reference

All settings live in `config.py` and can be overridden via environment variables:

| Setting | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/exam_bank` | PostgreSQL connection string |
| `OPENAI_API_KEY` | — | Your OpenAI key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for 1536-dim embeddings |
| `CLASSIFIER_MODEL` | `gpt-4o-mini` | Model for topic classification |
| `BATCH_SIZE` | `50` | Rows per DB insert batch |
| `EMBEDDING_BATCH` | `20` | Texts per embedding API call |
| `MAX_RETRIES` | `3` | Retries on API failure |
| `RETRY_DELAY` | `2.0` | Base delay (seconds) between retries |

---

## Error Handling & Logging

- All modules write to `logs/` (`parser.log`, `enrichment.log`, `database.log`, `pipeline.log`).
- **Question-count mismatches** between QP and MS are logged as `ERROR` level.
- API rate limits trigger automatic exponential backoff.
- Failed embeddings fall back to zero vectors so the pipeline never crashes mid-run.
- Unexpected filenames are logged as warnings and skipped.

---

## License

Private — Raisegrade.
