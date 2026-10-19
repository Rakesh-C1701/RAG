# Evidence Worksheet — NIST SP 800-63A-4 RAG

Use this worksheet alongside `rocky_rag_reranking.ipynb`. Values below are taken from the notebook's saved outputs. Re-run the notebook if you change the PDF, models, queries, chunking settings, hardware, or generator configuration.

## 1. Reproducibility facts

- **Document:** NIST SP 800-63A-4, *Digital Identity Guidelines: Identity Proofing and Enrollment* (July 2025).
- **PDF filename:** `NIST.SP.800-63a-4.pdf`.
- **PDF SHA-256:** `67b528e1c6ea76f816d5f29cc379a2530aa3e9114a35c6e98780e03707c7ec2b`.
- **Pages extracted:** 114.
- **Chunks:** 317.
- **Chunking:** sentence-aware packing, target 160 words, overlap of 2 sentences.
- **Observed chunk lengths:** minimum 18 words, median 166 words, maximum 335 words.
- **Bi-encoder:** `BAAI/bge-small-en-v1.5`.
- **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Device:** CPU.
- **Embedding matrix:** 317 × 384.
- **One-time embedding/indexing time:** 33.1 seconds.
- **Retrieval candidate depth:** Top 10 dense candidates.
- **Context depth:** Top 3 chunks supplied to the answer generator.
- **Configured generator:** `gemini:gemini-2.0-flash` (fresh generation requires `GEMINI_API_KEY`).

## 2. Saved rank-shift evidence

| Test | Category | Gold chunk | Page | Dense rank | Re-ranked rank | Shift | Saved verdict |
|---|---|---:|---:|---:|---:|---|---|
| T1 | Definition / comparison | 42 | 14 | 1 | 2 | #1 → #2 | Fine in both |
| T2 | Evidence validation | Not identified | — | — | — | — | Gold not in Top 10 or gold-pattern mismatch |
| T3 | Fraud management | 39 | 13 | 1 | 1 | #1 → #1 | Fine in both |
| T4 | Exception handling | 49 | 17 | 3 | 1 | #3 → #1 | Fine in both |

### Interpretation

- **T1 — IAL3 vs. IAL2:** Chunk C0042 on page 14 is the gold chunk. Dense retrieval placed it first, but re-ranking moved it to second. The relevant chunk remains within the Top 3 in both pipelines; re-ranking did not improve this gold rank.
- **T2 — Evidence validation:** No chunk in the dense Top 10 matched all configured gold patterns (`Evidence validation`, `genuine`, `authentic`). The retrieved candidate list includes a passage on page 25 stating that the CSP shall validate the authenticity, accuracy, and validity of presented evidence, so the pattern set may be too restrictive or inconsistent with the document's wording. This is not evidence that the relevant passage is absent from the full PDF. Review the gold patterns and rerun if possible.
- **T3 — Fraud mitigation:** Chunk C0039 on page 13 remained at rank one. The baseline already retrieved the gold chunk first.
- **T4 — Exception handling:** Chunk C0049 on page 17 improved from dense rank 3 to re-ranked rank 1. This is the clearest positive rank shift in this run.

The saved verdict logic labels T1, T3, and T4 “fine in both” because the gold chunk remains in the Top 3, but the rank shift itself should still be reported accurately. This run does not show that re-ranking improves every query.

## 3. Answer-generation evidence

The notebook is configured with `ONLY_RESCUED = True`, so it sends a query to the generator only when the gold chunk moves from dense rank 4–10 into the re-ranked Top 3. None of the four saved tests received the `RESCUED by re-ranker` verdict. As a result, the saved `answers` dictionary is empty and the notebook does **not** contain paired baseline and re-ranked answers for these tests.

| Test | Baseline answer saved? | Re-ranked answer saved? | Evidence status |
|---|---|---|---|
| T1 | No | No | Not generated under the current `ONLY_RESCUED` setting |
| T2 | No | No | Gold passage not identified by the configured pattern check |
| T3 | No | No | Not generated under the current `ONLY_RESCUED` setting |
| T4 | No | No | Not generated under the current `ONLY_RESCUED` setting |

If the assignment requires paired answers, set `ONLY_RESCUED = False`, configure a valid generator/API key, rerun the answer cell, and copy the actual outputs into this worksheet. Do not invent or copy answers from another document's worksheet.

## 4. Latency evidence

The notebook warmed up the re-ranking pipeline once and then measured 10 repetitions per query. The reported values are medians. Model loading, PDF extraction, indexing, and answer generation are excluded.

| Test | Baseline retrieval (ms) | Re-ranking stage (ms) | Advanced total (ms) |
|---|---:|---:|---:|
| T1 | 25.7 | 448.8 | 474.5 |
| T2 | 25.8 | 550.4 | 576.2 |
| T3 | 25.5 | 425.4 | 450.9 |
| T4 | 25.3 | 439.7 | 465.0 |
| **Mean** | **25.6** | **466.1** | **491.6** |

On this CPU run, the advanced retrieval total is about **19.2×** the baseline mean (491.6 / 25.6). The cross-encoder stage dominates latency. Exact timings vary with hardware and system load.

## 5. Scaling evidence

The saved notebook measured approximately **56.7 ms per query–chunk pair** for the cross-encoder on CPU and extrapolated the cost of scoring every chunk:

| Number of chunks scored | Estimated cross-encoder time per query |
|---:|---:|
| 10 | 0.89 seconds |
| 317 | 17.98 seconds |
| 10,000 | 567.12 seconds |
| 1,000,000 | 56,711.77 seconds |

The 10,000- and 1,000,000-chunk figures are linear extrapolations, not direct measurements at those corpus sizes. They illustrate why the cross-encoder is applied only to a small candidate set after the bi-encoder searches the full collection.

## 6. Short answers for the report sections

### Architecture comparison

The bi-encoder embeds each chunk independently and the chunk vectors can be precomputed once. At query time, the query is embedded and compared against the stored vectors. The cross-encoder jointly processes each query–chunk pair, allowing it to score their relevance using token-level interactions. This additional interaction costs an inference per candidate, so the notebook re-ranks only the Top 10 dense candidates.

### Qualitative and rank-shift analysis

T1 changed from rank 1 to rank 2, so the re-ranker slightly lowered the gold passage but it remained in the Top 3. T3 stayed at rank 1. T4 improved from rank 3 to rank 1, moving the gold passage to the first position. T2 did not yield a gold chunk under the current regular-expression match. The retrieved page-25 passage uses “authenticity, accuracy, and validity,” while the configured patterns require “genuine” and “authentic”; check the pattern set before interpreting this as a retrieval failure.

### Latency profiling

The timings were collected on CPU and cover query embedding/dense ranking, re-ranking of ten candidates, and the combined retrieval path. Mean baseline latency was 25.6 ms; mean re-ranking latency was 466.1 ms; mean advanced total was 491.6 ms. Re-ranking increases latency substantially in this run.

### Scaling

A cross-encoder scores each query–chunk pair separately, so scoring all chunks makes the per-query cost grow roughly linearly with corpus size. Dense retrieval narrows the full collection to a small Top 10 first, limiting the number of expensive cross-encoder evaluations.

## 7. Files and final checks

The notebook saves `rank_shifts.csv`, `candidates.csv`, `latency.csv`, `scaling.csv`, `answers.json`, and `run_info.json` in `outputs/`.

Before submission:

- Confirm the submitted notebook and worksheet both refer to **NIST SP 800-63A-4**, not SP 800-63B-4.
- Keep this PDF checksum only if the submitted PDF is byte-for-byte the same file.
- Review T2's gold regex and rerun the evaluation if possible.
- If paired generated answers are required, rerun answer generation with `ONLY_RESCUED = False` and retain the actual outputs.
- Submit the notebook and the required output artifacts from the same run.
