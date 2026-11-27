import math
from src.data_utils.query import Query


def calculateRecall(relevancies: list[bool], query: Query) -> float:
    return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())


def calculatePrecision(relevancies: list[bool]) -> float:
    if not relevancies:
        return 0.0
    return sum(relevancies) / len(relevancies)


def calculateMRR(relevancies: list[bool]) -> float:
    for rank, is_rel in enumerate(relevancies, start=1):
        if is_rel:
            return 1.0 / rank
    return 0.0


def calculateNDCG(relevancies: list[bool], query: Query) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, is_rel in enumerate(relevancies, start=1) if is_rel)
    ideal_hits = min(len(relevancies), query.getNumberOfRelevantDocuments())
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def Evaluate(relevancies: list[list[bool]], queries: list[Query]) -> dict[str, float]:
    assert len(relevancies) == len(queries), "The number of relevant passages must match the number of queries"
    n_queries = len(queries)
    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0

    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i])
        total_precision += calculatePrecision(relevancies[i])
        total_mrr += calculateMRR(relevancies[i])
        total_ndcg += calculateNDCG(relevancies[i], queries[i])

    return {
        "recall": total_recall / n_queries,
        "precision": total_precision / n_queries,
        "mrr": total_mrr / n_queries,
        "ndcg": total_ndcg / n_queries,
    }
