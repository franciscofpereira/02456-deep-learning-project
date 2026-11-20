"""
Retrieval evaluation metrics.

All metrics take `relevancies`: for one query, a list of booleans in ranked order,
True when the document at that rank is relevant (see retrieveQueryAndGetRelevancies).

    Precision@k = (# relevant in top k) / k
    Recall@k    = (# relevant in top k) / (# relevant documents for the query)
    MRR@k       = mean over queries of 1 / rank of the first relevant document in the top k (0 if none)
    nDCG@k      = DCG@k / ideal DCG@k, with binary relevance and log2 rank discount

Compare saved runs from the command line:
    python -m src.evaluation.metrics outputs/results
"""
import json
import math
import os
import sys

from src.data_utils.query import Query

DEFAULT_KS = (1, 5, 10)


# ----------------------------------------------------------------------------
# Per-query metrics
# ----------------------------------------------------------------------------

def precision_at_k(relevancies: list[bool], k: int) -> float:
    """Fraction of the top-k retrieved documents that are relevant."""
    if k <= 0:
        return 0.0
    return sum(relevancies[:k]) / k


def recall_at_k(relevancies: list[bool], n_relevant: int, k: int) -> float:
    """Fraction of all relevant documents for the query that appear in the top k."""
    if n_relevant <= 0:
        return 0.0
    return sum(relevancies[:k]) / n_relevant


def reciprocal_rank_at_k(relevancies: list[bool], k: int) -> float:
    """1 / rank of the first relevant document within the top k, or 0 if there is none."""
    for rank, is_rel in enumerate(relevancies[:k], start=1):
        if is_rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevancies: list[bool], n_relevant: int, k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, is_rel in enumerate(relevancies[:k], start=1) if is_rel)
    ideal_hits = min(k, n_relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ----------------------------------------------------------------------------
# Aggregation over queries
# ----------------------------------------------------------------------------

def Evaluate(relevancies: list[list[bool]], queries: list[Query], ks=DEFAULT_KS) -> dict[str, float]:
    """Average Precision@k, Recall@k, MRR@k and nDCG@k over all queries, for every k in ks.

    The relevancy lists must contain at least max(ks) ranked results (retrieve with k = max(ks)).
    Returns a dict such as {"precision@1": ..., "recall@5": ..., "mrr@10": ..., "ndcg@10": ...}.
    """
    assert len(relevancies) == len(queries), (
        "The number of relevancy lists must match the number of queries"
    )
    ks = sorted(set(int(k) for k in ks))
    n_queries = len(queries)
    if n_queries == 0:
        return {}

    totals = {f"{name}@{k}": 0.0 for name in ("precision", "recall", "mrr", "ndcg") for k in ks}
    for rel, query in zip(relevancies, queries):
        n_relevant = query.getNumberOfRelevantDocuments()
        for k in ks:
            totals[f"precision@{k}"] += precision_at_k(rel, k)
            totals[f"recall@{k}"] += recall_at_k(rel, n_relevant, k)
            totals[f"mrr@{k}"] += reciprocal_rank_at_k(rel, k)
            totals[f"ndcg@{k}"] += ndcg_at_k(rel, n_relevant, k)

    return {name: total / n_queries for name, total in totals.items()}


# ----------------------------------------------------------------------------
# Comparing retrieval methods
# ----------------------------------------------------------------------------

def save_results(results: dict, path: str) -> None:
    """Save {split: {metric: value}} for one method/run as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(results_dir: str) -> dict[str, dict]:
    """Load every <method>.json in a folder -> {method: {split: {metric: value}}}."""
    runs = {}
    for name in sorted(os.listdir(results_dir)):
        if name.endswith(".json"):
            with open(os.path.join(results_dir, name)) as f:
                runs[name[:-len(".json")]] = json.load(f)
    return runs


def compare_methods(runs: dict[str, dict], split: str = "test", metrics: list[str] | None = None) -> str:
    """Markdown table with one row per method and one column per metric for the given split.

    The best value in every column is marked with *.
    """
    rows = {method: res[split] for method, res in runs.items() if split in res}
    if not rows:
        return f"(no results for split '{split}')"

    if metrics is None:
        all_metrics = {m for scores in rows.values() for m in scores}
        order = {"precision": 0, "recall": 1, "mrr": 2, "ndcg": 3}
        metrics = sorted(all_metrics, key=lambda m: (order.get(m.split("@")[0], 9), int(m.split("@")[1]) if "@" in m else 0))

    best = {m: max((scores.get(m, float("-inf")) for scores in rows.values())) for m in metrics}
    width = max(len("method"), *(len(name) for name in rows))

    lines = [f"Split: {split}",
             "| " + "method".ljust(width) + " | " + " | ".join(m.rjust(12) for m in metrics) + " |",
             "|" + "-" * (width + 2) + "|" + "|".join("-" * 14 for _ in metrics) + "|"]
    for method, scores in rows.items():
        cells = []
        for m in metrics:
            v = scores.get(m)
            cell = "-" if v is None else f"{v:.4f}" + ("*" if v == best[m] else " ")
            cells.append(cell.rjust(12))
        lines.append("| " + method.ljust(width) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/results"
    splits = sys.argv[2:] or ["val", "test"]
    runs = load_results(results_dir)
    for s in splits:
        print(compare_methods(runs, split=s))
        print()
