# RAG Assignment: Baseline Retrieval vs. Cross-Encoder Re-Ranking

This project compares dense retrieval with a two-stage retrieval pipeline using **NIST SP 800-63A-4, Digital Identity Guidelines: Identity Proofing and Enrollment**.

The primary deliverable is the Jupyter notebook `NIST_SP_800-63A-4_RAG_results.ipynb`. It extracts text from the PDF, builds a shared chunk corpus, compares bi-encoder retrieval against cross-encoder re-ranking, profiles retrieval latency, and exports evidence for the report.

## Experiment overview

1. Extract text and page numbers from the NIST SP 800-63A-4 PDF.
2. Build sentence-aware chunks with overlapping sentences.
3. Embed each chunk once with the bi-encoder.
4. **Baseline:** retrieve the Top 10 chunks using dense similarity and pass the Top 3 to the answer generator.
5. **Advanced retrieval:** re-rank the same Top 10 candidates with a cross-encoder and pass the re-ranked Top 3 to the answer generator.
6. Record rank shifts, candidate rankings, latency, scaling estimates, generated answers, and run metadata.

Because both workflows use the same dense Top 10 candidate pool, re-ranking can improve the order of those candidates but cannot recover a relevant chunk that was not retrieved in the first stage.

## Source document

- **Document:** NIST SP 800-63A-4, *Digital Identity Guidelines: Identity Proofing and Enrollment*.
- **Official PDF:** <https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63A-4.pdf>
- **Kaggle setup:** Attach the PDF as a notebook input dataset. The notebook's PDF-discovery cell searches `/kaggle/input` and assigns the discovered file to `PDF_PATH`.

The extracted page count, chunk count, and SHA-256 checksum are run-specific metadata. Use the values printed or exported by your final execution rather than assuming a fixed count.

## Environment and setup

The notebook is designed to run in Kaggle or a local Python/Jupyter environment.

### Kaggle

1. Open the notebook in Kaggle.
2. Use **Add Input** to attach the NIST SP 800-63A-4 PDF dataset.
3. Run the PDF-discovery cell and confirm that it prints the PDF path.
4. Run the dependency-installation cell if required.
5. Run the notebook cells from top to bottom.
6. Download or commit the executed notebook and the `outputs/` files required for submission.

### Local Python environment

Create and activate a virtual environment, then install the dependencies used by the notebook:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pypdf sentence-transformers torch numpy pandas matplotlib google-genai openai requests
```

Place the PDF in a known location and set `PDF_PATH` to that exact path, or adapt the notebook's PDF-discovery cell to search your local data folder.

## Models and configuration

The notebook currently configures:

- **Bi-encoder:** `BAAI/bge-small-en-v1.5`
- **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Dense candidate depth:** Top 10
- **Context depth:** Top 3
- **Chunking target:** approximately 160 words, with overlap of two sentences
- **Device:** CUDA when available; otherwise CPU
- **Answer generator default:** Gemini, model `gemini-2.0-flash`

The notebook supports `gemini`, `openai`, and `ollama` generation backends. Select a backend that is available in your environment. For Gemini, set `GEMINI_API_KEY`; for OpenAI, set `OPENAI_API_KEY` and select the `openai` backend; for Ollama, make sure the local Ollama service and selected model are running.

Keep API keys in environment variables or a private local `.env` file. **Never commit API keys to the notebook or GitHub.**

## Evaluation queries and interpretation

The notebook contains four document-specific test queries covering:

- IAL3 versus IAL2 requirements
- Evidence validation
- Fraud mitigation
- Exception handling

For each query, the notebook searches for a gold chunk using the provided regular-expression patterns, then compares its rank in dense retrieval and after re-ranking. A query counts as **rescued by the re-ranker** only when its gold chunk is ranked below the Top 3 by the baseline but enters the re-ranked Top 3.

A missing gold chunk can mean that dense retrieval did not return the passage in its Top 10, or that the gold patterns did not match the extracted text. Inspect the source passage and matching patterns before interpreting it as a definitive retrieval failure. Report negative or unchanged rank shifts honestly.

## Generated outputs

When the notebook completes, it writes the following files under `outputs/`:

- `rank_shifts.csv` — gold-chunk rank comparison and verdicts for each query.
- `candidates.csv` — candidate chunk IDs, dense ranks, cross-encoder ranks, scores, and text previews.
- `latency.csv` — baseline, re-ranking, and advanced retrieval timings.
- `scaling.csv` — estimated cost of applying the cross-encoder to different collection sizes.
- `answers.json` — generated baseline and re-ranked answers for queries selected by the notebook's answer-generation settings.
- `run_info.json` — PDF filename and checksum, chunk count, model names, device, and indexing time.

These outputs should be generated by the final run and submitted if required. Latency depends on hardware, runtime conditions, and model loading state; use the values from your own run.

## Reproducibility and limitations

- Run all cells from top to bottom after changing the PDF, queries, chunking, model, backend, or hardware configuration.
- Check that `PDF_PATH` points to the intended SP 800-63A-4 PDF and confirm the checksum in `outputs/run_info.json`.
- The cross-encoder re-ranks only the dense Top 10 candidates; it does not search the full corpus independently.
- If `ONLY_RESCUED = True`, the notebook generates paired answers only for queries classified as rescued. If no query is rescued, `answers.json` may be empty. This is a configuration choice, not evidence that answer generation succeeded for every query.
- Do not claim an end-to-end answer improvement unless the saved baseline and re-ranked answers support that conclusion.
- Re-run the notebook after any configuration change and keep the report, worksheet, notebook outputs, and checksum consistent with the same final run.

## Repository contents

A suggested repository structure is:

```text
.
├── notebooks/
│   └── NIST_SP_800-63A-4_RAG_results.ipynb
├── report/
│   ├── analysis_report.md
│   └── evidence_worksheet.md
├── README.md
└── requirements.txt  # optional, if used by the project
```

Only include files that are part of your own project and required for submission. Do not commit downloaded model weights, large temporary files, credentials, or unrelated personal data.
