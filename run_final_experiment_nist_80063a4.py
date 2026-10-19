"""Run the NIST SP 800-63A-4 final retrieval/reranking experiment."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_retrieval_experiment import (
    BI_ENCODER_NAME,
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    CROSS_ENCODER_NAME,
    OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
    chunk_pages,
    dense_top_k,
    extract_pages,
    rerank,
    sha256,
)

# Assignment-specific paths: keep this script beside the PDF and helper module.
ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "NIST.SP.800-63a-4.pdf"
OUTPUT_DIR = ROOT / "outputs_nist_80063a4"


load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

LOCAL_LLM_NAME = os.getenv("LOCAL_LLM_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
GENERATOR_BACKEND = os.getenv("GENERATOR_BACKEND", "local").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
LATENCY_REPEATS = 5

FINAL_CASES = [
    {
        "case_id": "F1",
        "failure_mode": "Specific IAL requirement",
        "query": "What additional identity-proofing elements are required at IAL3?",
        "gold_phrase": "trained CSP representative",
        "expected": "IAL3 requires direct interaction with a trained CSP representative during an attended on-site session and collection of at least one biometric characteristic.",
    },
    {
        "case_id": "F2",
        "failure_mode": "Evidence validation definition",
        "query": "What does evidence validation confirm?",
        "gold_phrase": "supplied identity evidence is genuine",
        "expected": "Evidence validation confirms that supplied identity evidence is genuine, authentic, and accurate.",
    },
    {
        "case_id": "F3",
        "failure_mode": "Fraud mitigation definition",
        "query": "What is the purpose of fraud mitigation in identity proofing?",
        "gold_phrase": "Detect, respond to, and prevent access",
        "expected": "Fraud mitigation detects, responds to, and prevents access to benefits, services, data, or assets using a fraudulent identity.",
    },
    {
        "case_id": "F4",
        "failure_mode": "Exception handling / process design",
        "query": "What does the guideline say about exception handling processes?",
        "gold_phrase": "Offer Robust and Responsive Exception Handling Processes",
        "expected": "CSPs should offer robust and responsive exception-handling processes.",
    },
]


def make_prompt(query: str, selected_chunks: list) -> str:
    context = "\n\n".join(
        f"[{chunk.chunk_id}, pages {chunk.page_start}-{chunk.page_end}]\n{chunk.text}"
        for chunk in selected_chunks
    )
    return f"""Answer the question using only the supplied NIST SP 800-63A-4 context.
Preserve important requirements, qualifiers, and definitions. Do not import rules from NIST SP 800-63B or other documents. If the supplied passages do not directly support an answer, say exactly Insufficient context.
Answer in one short sentence.

Context:
{context}

Question: {query}
Answer:"""


def generate_answer(prompt: str, tokenizer, model) -> tuple[str, float]:
    messages = [
        {
            "role": "system",
            "content": "You answer questions only from supplied documentary evidence.",
        },
        {"role": "user", "content": prompt},
    ]
    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered = prompt
    inputs = tokenizer(rendered, return_tensors="pt")
    started = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=22,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - started
    answer_tokens = outputs[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip(), elapsed


def generate_gemini_answer(prompt: str, client, model_name: str) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.models.generate_content(model=model_name, contents=prompt)
    elapsed = time.perf_counter() - started
    return (response.text or "").strip(), elapsed


def generate_openai_answer(prompt: str, client, model_name: str) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.responses.create(model=model_name, input=prompt)
    elapsed = time.perf_counter() - started
    return (response.output_text or "").strip(), elapsed


def median_latencies(query, bi_encoder, cross_encoder, embeddings, chunks):
    baseline_samples = []
    rerank_samples = []
    advanced_samples = []
    for _ in range(LATENCY_REPEATS):
        dense_indices, _, dense_ms = dense_top_k(query, bi_encoder, embeddings)
        _, _, rerank_ms = rerank(query, dense_indices, chunks, cross_encoder)
        baseline_samples.append(dense_ms)
        rerank_samples.append(rerank_ms)
        advanced_samples.append(dense_ms + rerank_ms)
    return {
        "baseline_ms": statistics.median(baseline_samples),
        "reranking_only_ms": statistics.median(rerank_samples),
        "advanced_total_ms": statistics.median(advanced_samples),
    }


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if GENERATOR_BACKEND == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before selecting the OpenAI backend")
    if GENERATOR_BACKEND == "gemini" and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY before selecting the Gemini backend")
    if GENERATOR_BACKEND not in {"local", "gemini", "openai"}:
        raise ValueError("GENERATOR_BACKEND must be 'local', 'gemini', or 'openai'")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}. Put NIST.SP.800-63a-4.pdf beside this script or edit PDF_PATH."
        )
    print(f"Using PDF: {PDF_PATH}")
    print(f"PDF SHA-256: {sha256(PDF_PATH)}")

    pages = extract_pages(PDF_PATH)
    chunks = chunk_pages(pages, CHUNK_WORDS, CHUNK_OVERLAP)
    bi_encoder = SentenceTransformer(BI_ENCODER_NAME, device="cpu")
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device="cpu")
    embeddings = bi_encoder.encode(
        [chunk.text for chunk in chunks],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    # Warm both models before timed trials.
    warm_indices, _, _ = dense_top_k(FINAL_CASES[0]["query"], bi_encoder, embeddings)
    rerank(FINAL_CASES[0]["query"], warm_indices, chunks, cross_encoder)

    tokenizer = llm = gemini_client = openai_client = None
    if GENERATOR_BACKEND == "local":
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_NAME)
        llm = AutoModelForCausalLM.from_pretrained(LOCAL_LLM_NAME)
    elif GENERATOR_BACKEND == "gemini":
        from google import genai

        gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    elif GENERATOR_BACKEND == "openai":
        from openai import OpenAI

        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    results = []
    rank_rows = []
    for case in FINAL_CASES:
        query = case["query"]
        dense_indices, dense_scores, _ = dense_top_k(query, bi_encoder, embeddings)
        reranked_indices, cross_scores, _ = rerank(
            query, dense_indices, chunks, cross_encoder
        )
        dense_ranks = {int(index): rank for rank, index in enumerate(dense_indices, 1)}
        reranked_ranks = {
            int(index): rank for rank, index in enumerate(reranked_indices, 1)
        }
        gold_indices = [
            index
            for index, chunk in enumerate(chunks)
            if case["gold_phrase"].lower() in chunk.text.lower()
        ]
        eligible_gold = [index for index in gold_indices if index in dense_ranks]
        if not eligible_gold:
            print(
                f"WARNING: gold phrase for {case['case_id']} was not found in dense Top 10; "
                "recording retrieval outputs without a gold-rank claim."
            )
            gold_index = None
        else:
            gold_index = min(eligible_gold, key=lambda index: reranked_ranks[index])

        baseline_chunks = [chunks[int(index)] for index in dense_indices[:3]]
        advanced_chunks = [chunks[int(index)] for index in reranked_indices[:3]]
        if GENERATOR_BACKEND == "gemini":
            answer1, answer1_seconds = generate_gemini_answer(
                make_prompt(query, baseline_chunks), gemini_client, GEMINI_MODEL
            )
            answer2, answer2_seconds = generate_gemini_answer(
                make_prompt(query, advanced_chunks), gemini_client, GEMINI_MODEL
            )
        elif GENERATOR_BACKEND == "openai":
            answer1, answer1_seconds = generate_openai_answer(
                make_prompt(query, baseline_chunks), openai_client, OPENAI_MODEL
            )
            answer2, answer2_seconds = generate_openai_answer(
                make_prompt(query, advanced_chunks), openai_client, OPENAI_MODEL
            )
        else:
            answer1, answer1_seconds = generate_answer(
                make_prompt(query, baseline_chunks), tokenizer, llm
            )
            answer2, answer2_seconds = generate_answer(
                make_prompt(query, advanced_chunks), tokenizer, llm
            )
        latencies = median_latencies(
            query, bi_encoder, cross_encoder, embeddings, chunks
        )

        for candidate_index in dense_indices:
            candidate_index = int(candidate_index)
            chunk = chunks[candidate_index]
            rank_rows.append(
                {
                    "case_id": case["case_id"],
                    "query": query,
                    "chunk_id": chunk.chunk_id,
                    "pages": f"{chunk.page_start}-{chunk.page_end}",
                    "dense_rank": dense_ranks[candidate_index],
                    "reranked_rank": reranked_ranks[candidate_index],
                    "dense_score": float(
                        dense_scores[dense_ranks[candidate_index] - 1]
                    ),
                    "cross_score": float(
                        cross_scores[reranked_ranks[candidate_index] - 1]
                    ),
                    "is_gold": candidate_index == gold_index,
                    "text": chunk.text,
                }
            )

        results.append(
            {
                **case,
                "gold_chunk_id": chunks[gold_index].chunk_id if gold_index is not None else None,
                "gold_pages": f"{chunks[gold_index].page_start}-{chunks[gold_index].page_end}" if gold_index is not None else None,
                "dense_rank": dense_ranks[gold_index] if gold_index is not None else None,
                "reranked_rank": reranked_ranks[gold_index] if gold_index is not None else None,
                "baseline_top3": [chunk.chunk_id for chunk in baseline_chunks],
                "reranked_top3": [chunk.chunk_id for chunk in advanced_chunks],
                "answer1_baseline": answer1,
                "answer2_reranked": answer2,
                "answer1_generation_seconds": answer1_seconds,
                "answer2_generation_seconds": answer2_seconds,
                "generator_backend": GENERATOR_BACKEND,
                "generator_model": {
                    "local": LOCAL_LLM_NAME,
                    "gemini": GEMINI_MODEL,
                    "openai": OPENAI_MODEL,
                }[GENERATOR_BACKEND],
                **latencies,
            }
        )
        if gold_index is not None:
            print(
                f"{case['case_id']}: {chunks[gold_index].chunk_id} "
                f"rank {dense_ranks[gold_index]} -> {reranked_ranks[gold_index]}"
            )
        print("Answer 1:", answer1)
        print("Answer 2:", answer2)

    pd.DataFrame(rank_rows).to_csv(OUTPUT_DIR / "final_rank_details_nist_80063a4.csv", index=False)
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "final_results_nist_80063a4.csv", index=False)
    (OUTPUT_DIR / "final_results_nist_80063a4.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
