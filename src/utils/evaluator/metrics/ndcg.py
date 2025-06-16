import numpy as np

def dcg_at_k(relevance_scores, k):
   dcg = relevance_scores[0] if relevance_scores else 0.0
   for i in range(1, min(k, len(relevance_scores))):
      dcg += relevance_scores[i] / np.log2(i + 2)
   return dcg

def ndcg_at_k(true_relevance: dict, retrieved_ids: list, k: int):
   rel_scores = [true_relevance.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
   ideal_scores = sorted(true_relevance.values(), reverse=True)[:k]
   idcg = dcg_at_k(ideal_scores, k)
   return dcg_at_k(rel_scores, k) / idcg if idcg > 0 else 0.0

