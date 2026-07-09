"""
Evaluation harness for the RAG pipeline.

This file grades answers TWO different ways, on purpose:

1. Keyword matching (the original, cheap check) -- does the generated
   answer contain an expected word/phrase? Fast, free, deterministic, but
   blunt: an answer could contain the right keyword purely by coincidence,
   next to otherwise wrong or unsupported reasoning.

2. LLM-as-judge (the more advanced check, added here) -- ask a separate
   Claude call to actually READ the question, the generated answer, and a
   short description of what a correct answer needs to say, then judge
   whether it's actually correct. This catches cases keyword matching
   can't: an answer that's correct but phrases things differently than we
   expected, or an answer that's wrong despite mentioning the right words.

We keep BOTH metrics rather than replacing one with the other, because
comparing them is itself diagnostic: if keyword accuracy and LLM-judged
accuracy disagree a lot on the same answers, that's a signal your keyword
list is either too strict (missing valid phrasings) or too loose (letting
wrong answers slip through).
"""

import os
import json
from typing import List, Dict, Any
from chunking import chunk_documents
from retrieval import HybridIndex
from generate import generate_answer

# The labeled test set. Each entry pairs a question with what a CORRECT
# answer must satisfy:
#   - expected_doc_ids: which source document(s) actually contain the answer
#     (used to grade retrieval quality)
#   - expected_keywords: words/phrases a correct answer should contain
#     (used for the cheap keyword check)
#   - expected_facts: a short plain-language description of what a correct
#     answer needs to say (used as the grading rubric for the LLM judge --
#     this is richer than a keyword list because it can express "the
#     discount was ~10% AND traded for a 2-year term," not just "contains
#     the word 10%.")
TEST_SET = [
    {
        "question": "Why did Halberg Retail Group churn?",
        "expected_doc_ids": ["call_003"],
        "expected_keywords": ["cost", "onboarding", "price"],
        "expected_facts": "Halberg churned mainly due to cost/price relative to a cheaper competitor, and because onboarding took too long (three months instead of the promised one month).",
    },
    {
        "question": "What discount did Birkenstadt Manufacturing negotiate and what did they trade for it?",
        "expected_doc_ids": ["call_006"],
        "expected_keywords": ["10%", "2-year"],
        "expected_facts": "Birkenstadt got a 10% discount in exchange for agreeing to a 2-year contract term instead of the 3-year term they originally requested.",
    },
    {
        "question": "What feature convinced Solaris Energy Partners to consider switching providers?",
        "expected_doc_ids": ["call_007"],
        "expected_keywords": ["hybrid search", "semantic"],
        "expected_facts": "Solaris was interested in switching because of hybrid/semantic search capability, since their current tool's keyword-only search missed relevant results.",
    },
    {
        "question": "Why did Ferra Steel Works not move forward with the deal?",
        "expected_doc_ids": ["call_009"],
        "expected_keywords": ["budget", "restructuring"],
        "expected_facts": "Ferra paused the deal due to a company-wide budget freeze caused by a restructuring, not due to dissatisfaction with the product.",
    },
    {
        "question": "What data residency requirement did Solaris Energy Partners raise?",
        "expected_doc_ids": ["call_008"],
        "expected_keywords": ["EU"],
        "expected_facts": "Solaris required that their customer data stay hosted within the EU.",
    },
    {
        "question": "What compliance document did Birkenstadt's legal team require beyond GDPR?",
        "expected_doc_ids": ["call_006"],
        "expected_keywords": ["BDSG", "data protection"],
        "expected_facts": "Birkenstadt's legal team required a data processing addendum compliant with Germany's BDSG data protection law, in addition to standard GDPR terms.",
    },
    {
        "question": "What specific retrieval feature did Vindra Cosmetics value and why?",
        "expected_doc_ids": ["call_010"],
        "expected_keywords": ["structured retrieval", "region", "regulatory"],
        "expected_facts": "Vindra valued structured retrieval that let them filter product documentation by region and regulatory status, since cosmetics regulations differ across the EU, US, and UK.",
    },
    {
        "question": "What technical integration problem did Nordwind Logistics run into during implementation?",
        "expected_doc_ids": ["call_002"],
        "expected_keywords": ["date format", "ERP"],
        "expected_facts": "Nordwind's legacy ERP system used a non-standard date format, which slowed down the initial API integration.",
    },
]


def retrieval_hit(retrieved_chunks: List[Dict[str, Any]], expected_doc_ids: List[str]) -> bool:
    retrieved_doc_ids = {c["doc_id"] for c in retrieved_chunks}
    return any(doc_id in retrieved_doc_ids for doc_id in expected_doc_ids)


def keyword_answer_correct(answer: str, expected_keywords: List[str]) -> bool:
    """The original, cheap check: does the answer contain an expected keyword."""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


def llm_judge_answer(question: str, answer: str, expected_facts: str) -> Dict[str, Any]:
    """
    Ask Claude to grade whether `answer` correctly addresses `expected_facts`,
    given the original `question`. This is a SEPARATE API call from the one
    that generated the answer -- the judge never sees the retrieved context,
    only the question, the answer, and what a correct answer needed to say.
    That separation matters: we're grading the FINAL answer's correctness
    against ground truth, not re-checking whether it matches its own context.

    Returns {"correct": bool, "reasoning": str}. If no API key is available,
    returns a clearly-labeled placeholder instead of silently guessing.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"correct": None, "reasoning": "[skipped -- no ANTHROPIC_API_KEY set]"}

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # The judge prompt is deliberately structured: give it the question, the
    # answer to grade, and the rubric (what facts a correct answer needs),
    # then ask for a strict, parseable verdict format (a single CORRECT/
    # INCORRECT line first, reasoning after) so we can reliably parse the
    # verdict out of the response without fragile string-matching on prose.
    judge_prompt = (
        f"Question: {question}\n\n"
        f"Answer to grade: {answer}\n\n"
        f"Facts a correct answer must include: {expected_facts}\n\n"
        "Does the answer correctly convey those facts (paraphrasing is fine, "
        "exact wording is not required)? Respond with exactly one word on the "
        "first line -- CORRECT or INCORRECT -- followed by one sentence of "
        "reasoning on the next line."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()

    # Parse the verdict off the first line. We check startswith rather than
    # exact equality in case the model adds punctuation, and check
    # "INCORRECT" isn't accidentally matched by a naive "in" check on
    # "CORRECT" (since the substring "CORRECT" appears inside "INCORRECT").
    first_line = text.splitlines()[0].strip().upper() if text else ""
    verdict = first_line.startswith("CORRECT")  # False for both "INCORRECT" and anything else

    return {"correct": verdict, "reasoning": text}


def run_eval(index: HybridIndex, top_k: int = 3, verbose: bool = True) -> Dict[str, float]:
    retrieval_hits = 0
    keyword_hits = 0
    llm_judge_hits = 0
    llm_judge_available = os.environ.get("ANTHROPIC_API_KEY") is not None
    results = []

    for case in TEST_SET:
        retrieved = index.hybrid_search(case["question"], top_k=top_k)
        r_hit = retrieval_hit(retrieved, case["expected_doc_ids"])

        answer = generate_answer(case["question"], retrieved)

        k_hit = keyword_answer_correct(answer, case["expected_keywords"])

        # NEW: also grade this same answer with the LLM judge, using the
        # richer "expected_facts" rubric instead of a keyword list.
        judge_result = llm_judge_answer(case["question"], answer, case["expected_facts"])

        retrieval_hits += int(r_hit)
        keyword_hits += int(k_hit)
        if judge_result["correct"] is not None:
            llm_judge_hits += int(judge_result["correct"])

        results.append({
            "question": case["question"],
            "retrieval_hit": r_hit,
            "keyword_correct": k_hit,
            "llm_judge_correct": judge_result["correct"],
            "llm_judge_reasoning": judge_result["reasoning"],
            "answer": answer,
        })

        if verbose:
            status_r = "PASS" if r_hit else "FAIL"
            status_k = "PASS" if k_hit else "FAIL"
            status_j = (
                "PASS" if judge_result["correct"] is True
                else "FAIL" if judge_result["correct"] is False
                else "SKIP"
            )
            print(f"[retrieval:{status_r}] [keyword:{status_k}] [llm_judge:{status_j}] {case['question']}")
            if not r_hit or not k_hit or judge_result["correct"] is False:
                print(f"    -> answer: {answer[:200]}")
            if judge_result["correct"] is False:
                print(f"    -> judge reasoning: {judge_result['reasoning']}")

    n = len(TEST_SET)
    summary = {
        "retrieval_accuracy": retrieval_hits / n,
        "keyword_answer_accuracy": keyword_hits / n,
        "llm_judge_accuracy": (llm_judge_hits / n) if llm_judge_available else None,
        "n_cases": n,
    }
    if verbose:
        print(f"\nRetrieval accuracy:      {summary['retrieval_accuracy']:.0%} ({retrieval_hits}/{n})")
        print(f"Keyword answer accuracy: {summary['keyword_answer_accuracy']:.0%} ({keyword_hits}/{n})")
        if llm_judge_available:
            print(f"LLM judge accuracy:      {summary['llm_judge_accuracy']:.0%} ({llm_judge_hits}/{n})")
        else:
            print("LLM judge accuracy:      skipped (no ANTHROPIC_API_KEY set)")
    return summary


if __name__ == "__main__":
    with open("data/sales_calls.json") as f:
        docs = json.load(f)
    chunks = chunk_documents(docs)
    index = HybridIndex(chunks)
    run_eval(index)
