# Analysis Report: NIST SP 800-63A-4 RAG — Baseline vs. Cross-Encoder Re-Ranking

## 1. Objective

This experiment compares two retrieval pipelines over the NIST SP 800-63A-4 document, *Digital Identity Guidelines: Identity Proofing and Enrollment*. The baseline uses a bi-encoder to retrieve the ten most similar chunks and supplies its top three chunks to the answer generator. The second pipeline applies a cross-encoder to the same ten candidates, reorders them, and supplies the re-ranked top three chunks. The experiment evaluates retrieval ordering, the effects of re-ranking on selected test queries, retrieval latency, and the estimated cost of scoring larger candidate collections.

## 2. Dataset and setup

The notebook's saved run reports the following configuration:

- **Source PDF:** `NIST.SP.800-63a-4.pdf` (NIST SP 800-63A-4, Identity Proofing and Enrollment).
- **Pages extracted:** 114.
- **Chunks:** 317.
- **Chunking method:** sentence-aware packing with a target of 160 words and an overlap of two sentences between neighboring chunks.
- **Chunk size:** minimum 18 words, median 166 words, maximum 335 words.
- **Bi-encoder:** `BAAI/bge-small-en-v1.5`.
- **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Device:** CPU.
- **Embedding matrix:** 317 chunk vectors with 384 dimensions.
- **One-time embedding/indexing time:** 33.1 seconds.
- **Retrieval settings:** dense Top 10 candidates and Top 3 context chunks.
- **Answer-generation configuration:** Gemini backend with model name `gemini-2.0-flash` configured in the notebook. The answer-generation cell is set to process only queries marked as rescued by the re-ranker.

The chunks retain page numbers, allowing retrieved evidence to be traced back to the source document. The bi-encoder embeddings are normalized so cosine similarity can be computed using a dot product.

## 3. Retrieval rank-shift results

The saved rank-shift table contains four test queries:

| Test | Category | Gold chunk | Page | Dense rank | Re-ranked rank | Shift | Saved verdict |
|---|---|---:|---:|---:|---:|---|---|
| T1 | Definition / comparison | 42 | 14 | 1 | 2 | #1 → #2 | Fine in both |
| T2 | Evidence validation | Not identified | — | — | — | — | Gold not in Top 10 (recall miss) or pattern matched nothing |
| T3 | Fraud management | 39 | 13 | 1 | 1 | #1 → #1 | Fine in both |
| T4 | Exception handling | 49 | 17 | 3 | 1 | #3 → #1 | Fine in both |

### Interpretation by test

- **T1 — Definition / comparison:** The gold chunk was already ranked first by dense retrieval, then moved to rank two after re-ranking. The gold passage remained within the Top 3 context, but its rank worsened. This is a small ranking regression for the target passage, although the notebook classifies the answer context as adequate in both pipelines.
- **T2 — Evidence validation:** No chunk satisfying all configured gold regex patterns was found in the dense Top 10. This can mean the relevant chunk was not retrieved, or that the gold patterns were too restrictive or did not match the extracted PDF text. Since the gold passage was not identified in the candidate set, the cross-encoder could not promote it. This case should be treated as an unresolved retrieval/evaluation issue, not as a successful reranking result.
- **T3 — Fraud management:** The gold chunk remained at rank one in both pipelines. Dense retrieval already retrieved the relevant passage at the top, so re-ranking did not improve its position.
- **T4 — Exception handling:** The gold chunk improved from rank three to rank one. This is the clearest positive result in the saved rank-shift table: re-ranking promoted the target passage to the highest position.

Across these four tests, the gold passage improved for one query (T4), stayed at rank one for one query (T3), moved down one position for one query (T1), and was not identified in the Top 10 for one query (T2). These results show that re-ranking is not guaranteed to improve every query. Its potential benefit depends on whether dense retrieval includes the relevant passage among its candidates.

## 4. Architecture comparison

### Baseline: bi-encoder dense retrieval

The bi-encoder embeds each chunk independently during the one-time indexing stage. At query time, it embeds the query, compares the query vector with all stored chunk vectors, and returns the ten highest-scoring candidates. The first three candidates become the context passed to the answer generator. Precomputing the chunk vectors makes the query-time search relatively efficient.

### Advanced: dense retrieval followed by cross-encoder re-ranking

The advanced pipeline first retrieves the same ten candidates using the bi-encoder. The cross-encoder then scores each query–chunk pair jointly and sorts the candidates by the resulting relevance scores. The top three re-ranked chunks are used as context. Joint scoring can better assess interactions between the query and passage, but it requires additional model inference for each candidate. Because the cross-encoder only sees the dense Top 10, it cannot recover a relevant chunk that was excluded from that initial candidate pool.

## 5. Retrieval latency

The notebook reports median retrieval timings from ten repetitions per test after a warm-up. PDF parsing, model loading, one-time corpus embedding, and answer generation are excluded.

| Test | Baseline retrieval (ms) | Re-ranking 10 candidates (ms) | Advanced pipeline total (ms) |
|---|---:|---:|---:|
| T1 | 25.7 | 448.8 | 474.5 |
| T2 | 25.8 | 550.4 | 576.2 |
| T3 | 25.5 | 425.4 | 450.9 |
| T4 | 25.3 | 439.7 | 465.0 |
| **Mean** | **25.6** | **466.1** | **491.6** |

Baseline retrieval takes about 25.6 ms on average, while the re-ranking stage takes about 466.1 ms. The advanced pipeline's mean total is approximately 19.2 times the baseline latency (491.6 / 25.6). In this CPU run, the cross-encoder stage dominates query-time retrieval cost. These timings are measurements from this environment and may vary with hardware, runtime, and system load.

## 6. Scaling experiment

The notebook estimates cross-encoder cost from a measured rate of approximately 56.7 ms per query–chunk pair on CPU. It reports the following values for scoring every chunk in a collection of the stated size:

| Number of chunks scored | Estimated cross-encoder time per query |
|---:|---:|
| 10 | 0.89 seconds |
| 317 | 17.98 seconds |
| 10,000 | 567.12 seconds |
| 1,000,000 | 56,711.77 seconds |

The values for 10,000 and 1,000,000 chunks are linear extrapolations, not direct measurements on collections of those sizes. They illustrate why cross-encoding every chunk is expensive. A two-stage architecture is more practical: the bi-encoder searches the full corpus, and the cross-encoder re-scores only a small candidate set.

## 7. Answer-generation evidence and limitations

The answer-generation cell has `ONLY_RESCUED = True`, so it sends only tests whose gold chunk moves from dense rank 4–10 into the re-ranked Top 3 to the LLM. In the saved rank-shift table, none of the four tests is labeled `RESCUED by re-ranker`: T1, T3, and T4 are labeled `fine in both`, and T2 has no identified gold chunk. Consequently, the saved notebook does not provide paired generated answers for these four tests in its current configuration. This report does not infer or invent those answers.

For a fuller answer-quality comparison, the notebook could be rerun with `ONLY_RESCUED = False` after confirming that the configured API key and model are available. The T2 gold patterns should also be checked against the actual extracted text before rerunning evaluation.

## 8. Conclusion

The saved experiment demonstrates the trade-off between efficient first-stage retrieval and more expensive second-stage relevance scoring. The strongest positive rank shift is T4, where the exception-handling gold chunk moved from rank three to rank one. T3 remained at rank one, T1 moved from rank one to rank two, and T2's gold passage was not identified in the dense Top 10 under the current matching rule.

Latency results show that re-ranking significantly increases retrieval time on CPU: the mean baseline was 25.6 ms, compared with 491.6 ms for the advanced pipeline including re-ranking. This supports applying the cross-encoder to a limited candidate set rather than the entire corpus. The main limitations are the small four-query test set, the unresolved T2 gold-pattern/retrieval issue, and the absence of paired generated answers in the saved outputs. Results should therefore be interpreted as a small, document-specific experiment rather than a general guarantee that re-ranking improves every query.

## 9. Reproducibility

The notebook exports its results to an `outputs/` directory:

- `rank_shifts.csv`
- `candidates.csv`
- `latency.csv`
- `scaling.csv`
- `answers.json`
- `run_info.json` (PDF filename, chunk count, model names, device, indexing time, and PDF SHA-256 checksum)

For submission, keep the report consistent with the same notebook run and PDF. Include the executed notebook and the requested output files if required by the assignment. Do not report paired answer comparisons unless those answers were actually generated and saved.
