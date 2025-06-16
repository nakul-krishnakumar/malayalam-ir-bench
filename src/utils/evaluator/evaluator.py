import torch
import numpy as np
from typing import List, Dict
from sentence_transformers import util
from .metrics import recall_at_k, ndcg_at_k, average_precision_at_k, mrr_at_k

class IRBenchmarkEvaluator():
    def __init__(self, doc_ids: List[str], query_ids: List[str], qrels: Dict):
        self.doc_ids = doc_ids
        self.query_ids = query_ids
        self.qrels = qrels

        # print("doc_ids: ", doc_ids[: 10])
        # print("query_ids: ", query_ids[: 10])
        # print("qrels: ", list(qrels.items())[:10])

    def compute_similarity_scores(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between queries and documents"""
        return util.dot_score(query_embeddings, doc_embeddings)

    def evaluate_model(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, k_values: List[int] = [1, 5, 10]) -> Dict:
        """Evaluate model performance across multiple metrics"""

        scores = self.compute_similarity_scores(query_embeddings, doc_embeddings)
        results = {}

        for k in k_values:
            recalls = []
            ndcgs = []
            aps = []
            mrrs = []

            for i, qid in enumerate(self.query_ids):
                relevant_docs = self.qrels.get(qid, {})
                if not relevant_docs:
                    continue

                true_ids = list(relevant_docs.keys())
                true_relevance = relevant_docs  # dict: doc_id -> relevance

                # Get top-k ranked documents for this query
                ranked_indices = torch.topk(scores[i], k=min(k, len(self.doc_ids))).indices.cpu().tolist()
                ranked_doc_ids = [self.doc_ids[idx] for idx in ranked_indices]

                # Compute metrics
                recall = recall_at_k(true_ids, ranked_doc_ids, k)
                ndcg = ndcg_at_k(true_relevance, ranked_doc_ids, k)
                ap = average_precision_at_k(true_ids, ranked_doc_ids, k)
                mrr = mrr_at_k(true_ids, ranked_doc_ids, k)

                recalls.append(recall)
                ndcgs.append(ndcg)
                aps.append(ap)
                mrrs.append(mrr)

            results[f'Recall_at_{k}'] = float(np.mean(recalls)) if recalls else 0.0
            results[f'NDCG_at_{k}'] = float(np.mean(ndcgs)) if ndcgs else 0.0
            results[f'MAP_at_{k}'] = float(np.mean(aps)) if aps else 0.0
            results[f'MRR_at_{k}'] = float(np.mean(mrrs)) if mrrs else 0.0

        return results