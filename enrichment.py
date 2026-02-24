"""
enrichment.py — Automated Enrichment Layer

Uses OpenAI to:
  1. Classify each question into Topic > Subtopic from the taxonomy.
  2. Generate a 2000-dim embedding of the combined (Q + MS + ER) text.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import openai

from config import (
    ALL_TOPICS,
    CLASSIFIER_MODEL,
    EMBEDDING_BATCH,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    LOG_DIR,
    MAX_RETRIES,
    OPENAI_API_KEY,
    RETRY_DELAY,
    TOPIC_TAXONOMY,
)
from parser import ParsedQuestion

# ── Logging ──────────────────────────────────────────────────────────────
logger = logging.getLogger("enrichment")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(LOG_DIR / "enrichment.log", mode="a")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
logger.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
logger.addHandler(_ch)

# ── OpenAI client ────────────────────────────────────────────────────────
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# ── Classification ───────────────────────────────────────────────────────

CLASSIFICATION_SYSTEM_PROMPT = (
    "You are an expert exam question classifier. "
    "Given a maths / science exam question, you MUST output valid JSON with exactly "
    'two keys: "topic" and "subtopic". '
    "Choose the BEST match from the provided taxonomy list. "
    "Do NOT invent new topics. Output ONLY the JSON object, no markdown."
)


def _build_classification_user_prompt(question_text: str) -> str:
    taxonomy_str = "\n".join(f"  - {t}" for t in ALL_TOPICS)
    return (
        f"Taxonomy (Topic > Subtopic):\n{taxonomy_str}\n\n"
        f"Question:\n{question_text}\n\n"
        'Respond with JSON: {{"topic": "...", "subtopic": "..."}}'
    )


def classify_question(question_text: str) -> tuple[str, str]:
    """Return (topic, subtopic) for the given question text."""
    user_prompt = _build_classification_user_prompt(question_text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=CLASSIFIER_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            topic = data.get("topic", "Unknown")
            subtopic = data.get("subtopic", "Unknown")

            # Validate against taxonomy
            if topic in TOPIC_TAXONOMY:
                if subtopic not in TOPIC_TAXONOMY[topic]:
                    logger.warning(
                        "Subtopic '%s' not in taxonomy for '%s'. Keeping anyway.",
                        subtopic, topic,
                    )
            else:
                logger.warning("Topic '%s' not in taxonomy. Keeping anyway.", topic)

            return topic, subtopic

        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Classification JSON parse error (attempt %d): %s", attempt, exc)
        except openai.RateLimitError:
            wait = RETRY_DELAY * attempt
            logger.warning("Rate limited; sleeping %.1fs (attempt %d).", wait, attempt)
            time.sleep(wait)
        except openai.APIError as exc:
            logger.error("OpenAI API error (attempt %d): %s", attempt, exc)
            time.sleep(RETRY_DELAY)

    logger.error("Classification failed after %d retries. Returning Unknown.", MAX_RETRIES)
    return "Unknown", "Unknown"


# ── Embeddings ───────────────────────────────────────────────────────────

def _combine_text(q: ParsedQuestion) -> str:
    """Combine the three content fields into one string for embedding."""
    parts = [
        f"Question: {q.question_text}",
        f"Mark Scheme: {q.mark_scheme_text}" if q.mark_scheme_text else "",
        f"Examiner Report: {q.examiner_report_text}" if q.examiner_report_text else "",
    ]
    return "\n\n".join(p for p in parts if p)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts in batches.
    Returns a list of 2000-dim float vectors.
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH):
        batch = texts[i : i + EMBEDDING_BATCH]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    dimensions=EMBEDDING_DIMENSIONS,
                )
                batch_embeddings = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_embeddings)
                logger.debug(
                    "Embedded batch %d–%d (%d texts).",
                    i, i + len(batch) - 1, len(batch),
                )
                break
            except openai.RateLimitError:
                wait = RETRY_DELAY * attempt
                logger.warning("Embedding rate-limited; sleeping %.1fs.", wait)
                time.sleep(wait)
            except openai.APIError as exc:
                logger.error("Embedding API error (attempt %d): %s", attempt, exc)
                time.sleep(RETRY_DELAY)
        else:
            # All retries exhausted – fill with zero vectors
            logger.error(
                "Embedding failed for batch %d–%d. Filling with zeros.",
                i, i + len(batch) - 1,
            )
            all_embeddings.extend([[0.0] * 2000] * len(batch))

    return all_embeddings


def embed_single(text: str) -> list[float]:
    """Convenience: embed a single string."""
    result = embed_texts([text])
    return result[0] if result else [0.0] * 2000


# ── Orchestration: enrich a batch of ParsedQuestions ─────────────────────

def enrich_questions(questions: list[ParsedQuestion]) -> list[ParsedQuestion]:
    """
    For each ParsedQuestion:
      1. Classify → topic, subtopic
      2. Embed combined text → 2000-dim vector
    Mutates in place and returns the same list.
    """
    logger.info("Enriching %d questions …", len(questions))

    # Step 1: Classification (sequential – each call is tiny)
    for idx, q in enumerate(questions):
        logger.info(
            "  [%d/%d] Classifying %s_%s_%s Q%d …",
            idx + 1, len(questions),
            q.year, q.season, q.paper_code, q.question_num,
        )
        q.topic, q.subtopic = classify_question(q.question_text)
        logger.info("    → %s > %s", q.topic, q.subtopic)

    # Step 2: Embeddings (batched)
    combined_texts = [_combine_text(q) for q in questions]
    embeddings = embed_texts(combined_texts)
    for q, emb in zip(questions, embeddings):
        q.embedding = emb

    logger.info("Enrichment complete for %d questions.", len(questions))
    return questions


# ── CLI quick-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Minimal smoke test
    dummy = ParsedQuestion(
        year=2023,
        season="Summer",
        paper_code="P1",
        question_num=1,
        total_marks=5,
        question_text="Solve x^2 - 5x + 6 = 0.",
        mark_scheme_text="x = 2 or x = 3. M1 A1.",
        examiner_report_text="Well answered by most candidates.",
    )
    enriched = enrich_questions([dummy])
    q = enriched[0]
    print(f"Topic: {q.topic}, Subtopic: {q.subtopic}")
    print(f"Embedding dim: {len(q.embedding)}, first 5: {q.embedding[:5]}")
