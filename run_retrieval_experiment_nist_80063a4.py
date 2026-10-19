"""Run retrieval and re-ranking over NIST SP 800-63A-4.

This script is the executable development version of the assignment notebook.
It deliberately keeps answer generation separate from retrieval evidence so a
rank shift is never confused with an LLM-quality claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer


ROOT = Path(__file__).resolve().parent
# Set PDF_PATH in the environment if your PDF is stored elsewhere.
# The script checks the script folder, ./data, and Kaggle input folders.
def locate_pdf() -> Path:
    configured = os.getenv("PDF_PATH")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"PDF_PATH does not exist: {candidate}")

    filename = "NIST.SP.800-63a-4.pdf"
    candidates = [ROOT / filename, ROOT / "data" / filename,
                  Path("/kaggle/input") / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = list(kaggle_input.rglob(filename))
        if matches:
            return matches[0].resolve()

    raise FileNotFoundError(
        "Could not find NIST.SP.800-63a-4.pdf. Place it beside this script, "
        "inside ./data, or in /kaggle/input; alternatively set PDF_PATH."
    )

PDF_PATH = locate_pdf()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT / "outputs_nist_80063a4")))
# This is a new corpus, so do not compare it to the old 63B-4 checksum.
EXPECTED_SHA256 = None
BI_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_WORDS = 180
CHUNK_OVERLAP = 40
SEED = 42


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page_start: int
    page_end: int
    text: str


CANDIDATE_QUERIES = [
    "What does identity evidence validation establish during identity proofing?",
    "What is the purpose of identity evidence verification?",
    "What requirements apply to identity proofing at IAL3?",
    "What role does a trained proofing agent or CSP representative have at IAL3?",
    "What biometric requirements apply to IAL3 identity proofing?",
    "What is fraud mitigation intended to detect, prevent, or respond to?",
    "What should a CSP do when an applicant cannot complete the standard identity-proofing process?",
    "What is the difference between identity evidence validation and verification?",
    "What requirements apply to collecting and processing applicant information during identity proofing?",
    "What steps help establish that identity evidence belongs to the applicant?",
    "What should happen when identity evidence appears fraudulent or altered?",
    "What requirements apply to remote identity proofing?",
    "What are the requirements for in-person identity proofing?",
    "How should a CSP handle exceptions or alternative proofing paths?",
    "What is the purpose of enrollment after identity proofing?",
    "What safeguards help protect applicant information during proofing and enrollment?",
    "What is required when biometric information is collected during identity proofing?",
    "How should a CSP address risks of impersonation and synthetic identity fraud?",
    "What evidence must be validated before it can support an identity claim?",
    "What are the identity assurance levels and how do they differ?",
    "What does the guideline say about proofing agents and applicant interaction?",
    "What should the CSP document or communicate when a proofing attempt fails?",
    "How does the guideline treat exception handling for applicants who cannot follow the usual process?",
    "What is the relationship between identity resolution, evidence validation, and identity verification?",
]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_page_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pages(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(path)
    return [
        (page_number, clean_page_text(page.extract_text() or ""))
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def chunk_pages(
    pages: list[tuple[int, str]], chunk_words: int, overlap: int
) -> list[Chunk]:
    if not 0 <= overlap < chunk_words:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_words")

    words: list[tuple[str, int]] = []
    for page_number, text in pages:
        words.extend((word, page_number) for word in text.split())

    chunks: list[Chunk] = []
    step = chunk_words - overlap
    for index, start in enumerate(range(0, len(words), step)):
        window = words[start : start + chunk_words]
        if len(window) < 40:
            break
        page_numbers = [page for _, page in window]
        chunks.append(
            Chunk(
                chunk_id=f"C{index:04d}",
                page_start=min(page_numbers),
                page_end=max(page_numbers),
                text=" ".join(word for word, _ in window),
            )
        )
    return chunks


def dense_top_k(
    query: str,
    model: SentenceTransformer,
    corpus_embeddings: np.ndarray,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    query_embedding = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    # A small float64 reduction avoids spurious Accelerate/BLAS overflow
    # warnings seen with float32 matmul on some macOS Python builds.
    scores = np.sum(
        np.asarray(corpus_embeddings, dtype=np.float64)
        * np.asarray(query_embedding, dtype=np.float64),
        axis=1,
    )
    if not np.isfinite(scores).all():
        raise RuntimeError("Dense retrieval produced non-finite similarity scores")
    indices = np.argsort(-scores)[:top_k]
    elapsed_ms = (time.perf_counter() - started) * 1000
    return indices, scores[indices], elapsed_ms


def rerank(
    query: str,
    candidate_indices: np.ndarray,
    chunks: list[Chunk],
    model: CrossEncoder,
) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    pairs = [(query, chunks[index].text) for index in candidate_indices]
    scores = np.asarray(model.predict(pairs, show_progress_bar=False))
    order = np.argsort(-scores)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return candidate_indices[order], scores[order], elapsed_ms


def main() -> None:
    np.random.seed(SEED)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")
    actual_hash = sha256(PDF_PATH)
    print(f"Using PDF: {PDF_PATH}")
    print(f"PDF SHA-256: {actual_hash}")

    pages = extract_pages(PDF_PATH)
    chunks = chunk_pages(pages, CHUNK_WORDS, CHUNK_OVERLAP)
    print(f"Extracted {len(pages)} pages into {len(chunks)} chunks")

    bi_encoder = SentenceTransformer(BI_ENCODER_NAME, device="cpu")
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device="cpu")

    embedding_started = time.perf_counter()
    corpus_embeddings = bi_encoder.encode(
        [chunk.text for chunk in chunks],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embedding_seconds = time.perf_counter() - embedding_started
    print(f"Embedded corpus in {embedding_seconds:.2f} seconds")
    np.save(OUTPUT_DIR / "corpus_embeddings.npy", corpus_embeddings)
    (OUTPUT_DIR / "chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2), encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    top_chunks: dict[str, dict[str, list[dict[str, object]]]] = {}
    for query_id, query in enumerate(CANDIDATE_QUERIES, start=1):
        dense_indices, dense_scores, dense_ms = dense_top_k(
            query, bi_encoder, corpus_embeddings
        )
        reranked_indices, cross_scores, rerank_ms = rerank(
            query, dense_indices, chunks, cross_encoder
        )
        dense_rank = {int(index): rank for rank, index in enumerate(dense_indices, 1)}
        cross_rank = {
            int(index): rank for rank, index in enumerate(reranked_indices, 1)
        }

        for index in dense_indices:
            index = int(index)
            chunk = chunks[index]
            rows.append(
                {
                    "query_id": f"Q{query_id:02d}",
                    "query": query,
                    "chunk_id": chunk.chunk_id,
                    "pages": f"{chunk.page_start}-{chunk.page_end}",
                    "dense_rank": dense_rank[index],
                    "reranked_rank": cross_rank[index],
                    "dense_score": float(dense_scores[dense_rank[index] - 1]),
                    "cross_score": float(cross_scores[cross_rank[index] - 1]),
                    "text": chunk.text,
                }
            )

        def serialize(indices: np.ndarray, scores: np.ndarray) -> list[dict[str, object]]:
            return [
                {
                    **asdict(chunks[int(index)]),
                    "score": float(score),
                }
                for index, score in zip(indices[:3], scores[:3])
            ]

        top_chunks[f"Q{query_id:02d}"] = {
            "query": query,
            "baseline_top3": serialize(dense_indices, dense_scores),
            "reranked_top3": serialize(reranked_indices, cross_scores),
            "baseline_retrieval_ms": dense_ms,
            "reranking_ms": rerank_ms,
            "advanced_total_ms": dense_ms + rerank_ms,
        }
        print(
            f"Q{query_id:02d}: dense={dense_ms:.1f} ms, "
            f"rerank={rerank_ms:.1f} ms"
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "candidate_rank_shifts.csv", index=False)
    (OUTPUT_DIR / "top_chunks.json").write_text(
        json.dumps(top_chunks, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "pdf_path": str(PDF_PATH),
                "document": "NIST SP 800-63A-4",
                "pdf_sha256": actual_hash,
                "page_count": len(pages),
                "chunk_count": len(chunks),
                "chunk_words": CHUNK_WORDS,
                "chunk_overlap": CHUNK_OVERLAP,
                "bi_encoder": BI_ENCODER_NAME,
                "cross_encoder": CROSS_ENCODER_NAME,
                "embedding_seconds": embedding_seconds,
                "seed": SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    shifts = frame[
        (frame["dense_rank"] > 3) & (frame["reranked_rank"] <= 3)
    ].sort_values(["query_id", "reranked_rank"])
    print("\nChunks promoted from dense ranks 4-10 into reranked Top 3:")
    print(
        shifts[
            [
                "query_id",
                "chunk_id",
                "pages",
                "dense_rank",
                "reranked_rank",
                "text",
            ]
        ].to_string(index=False, max_colwidth=100)
    )


if __name__ == "__main__":
    main()
