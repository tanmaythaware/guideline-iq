import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> Optional[int]:
    if not gold_ids:
        return None
    top_k = (retrieved_ids or [])[:k]
    return 1 if any(g in top_k for g in gold_ids) else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    results_path = Path(args.results)
    data = json.loads(results_path.read_text(encoding="utf-8"))
    cases: List[Dict[str, Any]] = data["cases"]

    total = len(cases)
    errors = [c for c in cases if c.get("error") or (c.get("status_code") and c["status_code"] >= 400)]
    ok = [c for c in cases if c not in errors]

    # Evaluate only cases that have an expected_decision
    eval_cases = [c for c in ok if c.get("expected_decision") is not None]

    # 1) Decision accuracy
    decision_correct = [c for c in eval_cases if c.get("actual_decision") == c.get("expected_decision")]
    decision_accuracy = (len(decision_correct) / len(eval_cases)) if eval_cases else 0.0

    # 2) Refusal precision (expected refuse -> actual refuse)
    expected_refuse = [c for c in eval_cases if c.get("expected_decision") == "refuse"]
    correctly_refused = [c for c in expected_refuse if c.get("actual_decision") == "refuse"]
    refusal_precision = (len(correctly_refused) / len(expected_refuse)) if expected_refuse else 0.0

    # 3) Refusal reason accuracy (only when expected_reason provided)
    reason_cases = [c for c in expected_refuse if c.get("expected_reason")]
    reason_correct = [c for c in reason_cases if c.get("actual_reason") == c.get("expected_reason")]
    refusal_reason_accuracy = (len(reason_correct) / len(reason_cases)) if reason_cases else None

    # 4) Per-reason confusion (expected_reason -> actual_reason)
    reason_confusion = defaultdict(Counter)
    for c in reason_cases:
        er = c.get("expected_reason")
        ar = c.get("actual_reason") or "None"
        reason_confusion[er][ar] += 1
    reason_confusion = {k: dict(v) for k, v in reason_confusion.items()}

    # 5) Empty retrieval rate
    empty_retrieval = [c for c in ok if not (c.get("retrieved_ids") or [])]
    empty_retrieval_rate = (len(empty_retrieval) / len(ok)) if ok else 0.0

    # 6) Answer citation compliance: if answer then must have citations
    answer_cases = [c for c in ok if c.get("actual_decision") == "answer"]
    answer_missing_citations = [c for c in answer_cases if not c.get("had_citations")]
    answer_citation_compliance = (
        1.0 - (len(answer_missing_citations) / len(answer_cases))
        if answer_cases else 1.0
    )

    # 7) Recall@K (only for cases with gold_chunk_ids)
    recall_cases = [c for c in ok if c.get("gold_chunk_ids")]
    recall_values = []
    for c in recall_cases:
        r = recall_at_k(c.get("retrieved_ids", []), c.get("gold_chunk_ids", []), args.k)
        if r is not None:
            recall_values.append(r)
    recall_k = (sum(recall_values) / len(recall_values)) if recall_values else None

    # 8) Useful breakdowns
    refusal_reason_counts = Counter([c.get("actual_reason") for c in ok if c.get("actual_decision") == "refuse"])
    classifier_label_counts = Counter([c.get("classifier_label") for c in ok])

    # 9) API usage and cost summary
    meta_usage = (data.get("meta") or {}).get("usage") or {}
    meta_tokens = meta_usage.get("total_tokens")
    meta_cost = meta_usage.get("estimated_cost_usd")
    case_tokens = sum(int(c.get("total_tokens", 0) or 0) for c in cases)
    case_cost = sum(float(c.get("estimated_cost_usd", 0.0) or 0.0) for c in cases)
    total_api_tokens = int(meta_tokens) if isinstance(meta_tokens, (int, float)) else case_tokens
    total_api_cost_usd = (
        float(meta_cost) if isinstance(meta_cost, (int, float)) else case_cost
    )

    # Examples (top failures)
    wrong_decisions = [c for c in eval_cases if c not in decision_correct]
    report = {
        "meta": data.get("meta", {}),
        "summary": {
            "total_cases": total,
            "ok_cases": len(ok),
            "api_errors": len(errors),

            "evaluated_cases": len(eval_cases),
            "decision_accuracy": round(decision_accuracy, 4),

            "expected_refuse_cases": len(expected_refuse),
            "refusal_precision": round(refusal_precision, 4),

            "refusal_reason_accuracy": None if refusal_reason_accuracy is None else round(refusal_reason_accuracy, 4),
            "empty_retrieval_rate": round(empty_retrieval_rate, 4),

            "answer_citation_compliance": round(answer_citation_compliance, 4),
            "recall_at_k": None if recall_k is None else round(recall_k, 4),
            "k": args.k,
            "total_api_tokens": total_api_tokens,
            "total_api_cost_usd": round(total_api_cost_usd, 8),
        },
        "breakdowns": {
            "reason_confusion": reason_confusion,
            "actual_refusal_reason_counts": dict(refusal_reason_counts),
            "classifier_label_counts": dict(classifier_label_counts),
        },
        "samples": {
            "api_errors": errors[:5],
            "wrong_decisions": wrong_decisions[:5],
            "empty_retrieval": empty_retrieval[:5],
            "answer_missing_citations": answer_missing_citations[:5],
        },
    }

    out_path = results_path.with_suffix(".metrics.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote metrics: {out_path}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
