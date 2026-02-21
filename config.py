"""
config.py — Central configuration for the PISR exam-bank pipeline.

Environment variables (set in .env or export before running):
    OPENAI_API_KEY      – OpenAI API key (used for embeddings + classification)
    DATABASE_URL        – PostgreSQL connection string
                          e.g. postgresql://user:pass@localhost:5432/exam_bank
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
QP_DIR = BASE_DIR / "QP"
MS_DIR = BASE_DIR / "MS"
ER_DIR = BASE_DIR / "ER"
RESOURCES_DIR = BASE_DIR / "resources"   # dump all PDFs here; auto-sorted
LOG_DIR = BASE_DIR / "logs"

# ── Database ─────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/exam_bank",
)

# ── OpenAI ───────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "text-embedding-3-small"   # 1536-dim
CLASSIFIER_MODEL: str = "gpt-4o-mini"             # cheap & fast for classification

# ── Pipeline tuning ─────────────────────────────────────────────────────
BATCH_SIZE: int = 50          # rows per INSERT batch
EMBEDDING_BATCH: int = 20    # texts per embedding API call
MAX_RETRIES: int = 3
RETRY_DELAY: float = 2.0     # seconds

# ── Topic taxonomy ──────────────────────────────────────────────────────
# Extend / replace this dict to match your syllabus.
TOPIC_TAXONOMY: dict[str, list[str]] = {
    "Algebra": [
        "Linear Equations",
        "Quadratic Equations",
        "Simultaneous Equations",
        "Inequalities",
        "Sequences and Series",
        "Polynomials",
        "Partial Fractions",
        "Binomial Expansion",
        "Functions and Graphs",
        "Logarithms and Exponentials",
    ],
    "Calculus": [
        "Differentiation",
        "Integration",
        "Differential Equations",
        "Applications of Differentiation",
        "Applications of Integration",
        "Numerical Methods",
    ],
    "Geometry": [
        "Coordinate Geometry",
        "Circle Geometry",
        "Vectors",
        "Trigonometry",
        "Trigonometric Identities",
        "Parametric Equations",
    ],
    "Statistics": [
        "Data Representation",
        "Probability",
        "Discrete Random Variables",
        "Continuous Random Variables",
        "Normal Distribution",
        "Binomial Distribution",
        "Poisson Distribution",
        "Hypothesis Testing",
        "Correlation and Regression",
    ],
    "Mechanics": [
        "Kinematics",
        "Forces and Newton's Laws",
        "Moments",
        "Work, Energy and Power",
        "Momentum and Impulse",
        "Projectiles",
        "Circular Motion",
    ],
    "Pure Mathematics": [
        "Proof",
        "Complex Numbers",
        "Matrices",
        "Further Calculus",
        "Hyperbolic Functions",
        "Polar Coordinates",
        "Maclaurin and Taylor Series",
    ],
    "Number": [
        "Surds and Indices",
        "Ratio and Proportion",
        "Number Theory",
    ],
}

# Flat list for prompt injection
ALL_TOPICS: list[str] = []
for topic, subs in TOPIC_TAXONOMY.items():
    for sub in subs:
        ALL_TOPICS.append(f"{topic} > {sub}")
