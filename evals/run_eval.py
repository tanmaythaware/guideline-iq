import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_nested(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="evals/datasets/golden.jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    cases = load_jsonl(dataset_path)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else Path(f"evals/reports/{ts}_results.json")
    ensure_dir(out_path.parent)

    results: Dict[str, Any] = {
        "meta": {
            "dataset": str(dataset_path),
            "api_base_url": args.api_base_url,
            "timestamp": ts,
            "num_cases": len(cases),
        },
        "cases": [],
    }

    for case in cases:
        qid = case["id"]
        query = case["query"]

        expected_decision = case.get("expected_decision")
        expected_reason = case.get("expected_reason")  
        gold_chunk_ids = case.get("gold_chunk_ids", [])  

        start = time.time()
        resp_json: Optional[Dict[str, Any]] = None
        err: Optional[str] = None
        status_code: Optional[int] = None

        try:
            r = requests.get(
                f"{args.api_base_url}/ask",
                params={"q": query},
                timeout=args.timeout,
            )
            status_code = r.status_code
            r.raise_for_status()
            resp_json = r.json()
        except Exception as e:
            err = str(e)

        latency_ms = int((time.time() - start) * 1000)

        # Normalize fields from AskResponse schema
        actual_decision = resp_json.get("decision") if resp_json else None
        refusal_reason = resp_json.get("refusal_reason") if resp_json else None

        classifier = resp_json.get("classifier") if resp_json else None
        classifier_label = (classifier or {}).get("label")
        classifier_conf = (classifier or {}).get("confidence")

        citations = resp_json.get("citations", []) if resp_json else []
        had_citations = bool(citations)

        retrieval_results = get_nested(resp_json, ["retrieval", "results"], default=[]) if resp_json else []
        retrieved_ids = [x.get("id") for x in retrieval_results if x.get("id")]
        retrieved_scores = [x.get("score") for x in retrieval_results if x.get("score") is not None]
        top_score = retrieved_scores[0] if retrieved_scores else None

        retrieval_ms = get_nested(resp_json, ["retrieval", "duration_ms"], default=None) if resp_json else None

        results["cases"].append(
            {
                "id": qid,
                "query": query,

                "expected_decision": expected_decision,
                "expected_reason": expected_reason,
                "gold_chunk_ids": gold_chunk_ids,

                "status_code": status_code,
                "error": err,
                "latency_ms": latency_ms,

                "actual_decision": actual_decision,
                "actual_reason": refusal_reason,  
                "had_citations": had_citations,

                "classifier_label": classifier_label,
                "classifier_confidence": classifier_conf,

                "retrieved_ids": retrieved_ids,
                "retrieved_scores": retrieved_scores,
                "top_score": top_score,
                "retrieval_duration_ms": retrieval_ms,
            }
        )

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote eval results: {out_path}")


if __name__ == "__main__":
    main()
