"""
Evaluation suite for the movie recommender.

Runs all 20 test queries across three pipeline modes and reports:
  - precision@5 per query per mode (genre-based graded relevance)
  - aggregate average precision@5 per mode
  - delta columns showing improvement vs fast baseline
  - confidence calibration (Spearman correlation) for full mode

Requires the server to be running. Default target: http://localhost:8080
Override with --url flag.

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --url http://localhost:8080
    python eval/evaluate.py --mode full          # single mode
    python eval/evaluate.py --timeout 60         # seconds per request
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests
from scipy.stats import spearmanr

QUERIES_PATH = Path(__file__).parent / "test_queries.json"
MODES = ["fast", "balanced", "full"]
COL_WIDTH = 10


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_single(rec: dict, graded_relevance: dict) -> int:
    """
    0 = miss, 1 = weak match, 2 = strong match.

    Matching is case-insensitive substring so "TV Thrillers" matches "Thriller".
    """
    rec_genres = " ".join(rec.get("genres", [])).lower()

    for genre in graded_relevance.get("strong", []):
        if genre.lower() in rec_genres:
            return 2

    for genre in graded_relevance.get("weak", []):
        if genre.lower() in rec_genres:
            return 1

    return 0


def precision_at_5(recs: list[dict], query_data: dict) -> float:
    """Fraction of top-5 results that score >= 1 (weak match or better)."""
    if not recs:
        return 0.0
    scores = [score_single(r, query_data["graded_relevance"]) for r in recs[:5]]
    relevant = sum(1 for s in scores if s >= 1)
    return relevant / min(5, len(recs))


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def call_recommend(
    base_url: str,
    query: str,
    mode: str,
    timeout: int,
) -> dict | None:
    """
    POST /recommend and return the parsed response dict.
    Returns None on any error — evaluation continues with 0 precision for that cell.
    """
    try:
        resp = requests.post(
            f"{base_url}/recommend",
            json={"query": query, "pipeline_mode": mode},
            timeout=timeout,
        )
        if resp.status_code == 422:
            # Pydantic validation error — expected for the whitespace edge case
            return {"_status": 422}
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        print(f"    TIMEOUT ({mode})", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"    ERROR ({mode}): {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------


def compute_calibration(results_full: dict[str, dict]) -> float | None:
    """
    Spearman correlation between confidence scores and graded relevance scores.
    Only uses results from full mode where the reranker assigns confidence values.

    A positive correlation means the model assigns higher confidence to titles
    that actually match the query — a sanity check on whether confidence is
    meaningful or just noise.
    """
    confidence_values = []
    relevance_scores = []

    for qid, data in results_full.items():
        query_data = data["query_data"]
        recs = data.get("recs", [])
        for rec in recs[:5]:
            conf = rec.get("confidence")
            if conf is not None:
                score = score_single(rec, query_data["graded_relevance"])
                confidence_values.append(float(conf))
                relevance_scores.append(float(score))

    if len(confidence_values) < 3:
        return None

    corr, _ = spearmanr(confidence_values, relevance_scores)
    return float(corr)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def fmt(value: float | None, width: int = COL_WIDTH) -> str:
    if value is None:
        return "  N/A".ljust(width)
    return f"{value:.3f}".rjust(width)


def fmt_delta(v: float | None, baseline: float | None, width: int = COL_WIDTH) -> str:
    if v is None or baseline is None:
        return "  N/A".ljust(width)
    delta = v - baseline
    sign = "+" if delta >= 0 else ""
    return f"({sign}{delta:+.3f})".rjust(width)


def print_table(
    queries: list[dict],
    scores: dict[str, dict[str, float | None]],
) -> None:
    """Print a comparison table: one row per query, one column group per mode."""
    # Header
    col_id = 6
    col_type = 16
    header = (
        f"{'ID':<{col_id}}"
        f"{'Type':<{col_type}}"
        + "".join(f"{'  P@5 ' + m:>{COL_WIDTH}}" for m in MODES)
        + f"{'vs fast':>{COL_WIDTH}}"
        f"{'vs balanced':>{COL_WIDTH}}"
    )
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for q in queries:
        qid = q["id"]
        row = f"{qid:<{col_id}}{q['type']:<{col_type}}"
        mode_scores = [scores.get(qid, {}).get(m) for m in MODES]

        for ms in mode_scores:
            row += fmt(ms)

        # delta columns
        fast_score = scores.get(qid, {}).get("fast")
        balanced_score = scores.get(qid, {}).get("balanced")
        full_score = scores.get(qid, {}).get("full")

        row += fmt_delta(full_score, fast_score)
        row += fmt_delta(full_score, balanced_score)
        print(row)

    print(sep)

    # Aggregate averages (skip edge cases — they don't have meaningful graded relevance)
    eval_queries = [q for q in queries if q["type"] != "edge_case"]
    avgs: dict[str, float | None] = {}
    for mode in MODES:
        mode_vals = [
            scores.get(q["id"], {}).get(mode)
            for q in eval_queries
            if scores.get(q["id"], {}).get(mode) is not None
        ]
        avgs[mode] = sum(mode_vals) / len(mode_vals) if mode_vals else None

    avg_row = f"{'AVG':<{col_id}}{'(excl. edge cases)':<{col_type}}"
    for mode in MODES:
        avg_row += fmt(avgs.get(mode))
    avg_row += fmt_delta(avgs.get("full"), avgs.get("fast"))
    avg_row += fmt_delta(avgs.get("full"), avgs.get("balanced"))
    print(avg_row)
    print(sep)
    print()

    # Per-type averages
    for query_type in ["vague_mood", "semi_structured", "explicit_reference"]:
        type_queries = [q for q in queries if q["type"] == query_type]
        type_row = f"{'':>{col_id}}{query_type:<{col_type}}"
        for mode in MODES:
            vals = [
                scores.get(q["id"], {}).get(mode)
                for q in type_queries
                if scores.get(q["id"], {}).get(mode) is not None
            ]
            type_row += fmt(sum(vals) / len(vals) if vals else None)
        print(type_row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate movie recommender pipeline")
    parser.add_argument("--url", default="http://localhost:8080", help="Server base URL")
    parser.add_argument(
        "--mode", choices=MODES, default=None, help="Run a single mode only"
    )
    parser.add_argument(
        "--timeout", type=int, default=45, help="Request timeout in seconds"
    )
    args = parser.parse_args()

    modes_to_run = [args.mode] if args.mode else MODES

    with open(QUERIES_PATH) as f:
        queries = json.load(f)

    print(f"Evaluating {len(queries)} queries across modes: {modes_to_run}")
    print(f"Target: {args.url}\n")

    # Check server is up
    try:
        health = requests.get(f"{args.url}/health", timeout=5).json()
        print(f"Server healthy — {health['titles_loaded']:,} titles loaded\n")
    except Exception as exc:
        print(f"ERROR: Cannot reach {args.url}/health — is the server running?")
        print(f"  {exc}")
        sys.exit(1)

    # scores[query_id][mode] = precision@5
    scores: dict[str, dict[str, float | None]] = {q["id"]: {} for q in queries}
    # Store full-mode results for calibration analysis
    full_results: dict[str, dict] = {}

    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        print(f"[{qid}] {query_text[:60]!r}")

        for mode in modes_to_run:
            if q["type"] == "edge_case" and qid == "E2":
                # Whitespace query — expect 422, don't score
                result = call_recommend(args.url, query_text, mode, args.timeout)
                status = (result or {}).get("_status")
                score_val = None
                label = f"HTTP {status}" if status else "ERROR"
                print(f"  {mode:>9}: {label}")
            else:
                result = call_recommend(args.url, query_text, mode, args.timeout)
                if result and "_status" not in result:
                    recs = result.get("recommendations", [])
                    p5 = precision_at_5(recs, q)
                    scores[qid][mode] = p5
                    print(f"  {mode:>9}: P@5={p5:.3f}  ({len(recs)} results)")

                    if mode == "full":
                        full_results[qid] = {"query_data": q, "recs": recs}
                else:
                    scores[qid][mode] = None
                    print(f"  {mode:>9}: FAILED")

            # Brief pause between modes to avoid rate-limiting Groq
            time.sleep(0.5)

        print()

    # Results table
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print_table(queries, scores)

    # Confidence calibration (full mode only)
    if "full" in modes_to_run and full_results:
        corr = compute_calibration(full_results)
        if corr is not None:
            print(f"Confidence calibration (Spearman ρ, full mode): {corr:+.3f}")
            if corr > 0.3:
                print("  → Confidence scores correlate positively with relevance (well-calibrated)")
            elif corr > 0:
                print("  → Weak positive correlation — confidence scores are partially meaningful")
            else:
                print("  → Low/negative correlation — confidence scores are not well-calibrated")
        else:
            print("Confidence calibration: insufficient data (need full mode with reranker results)")
        print()


if __name__ == "__main__":
    main()
