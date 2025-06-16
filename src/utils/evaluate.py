import torch
from sentence_transformers import util
import numpy as np
from typing import Dict, List, Tuple
from ..metrics.map import map_at_k, average_precision_at_k
from ..metrics.ndcg import ndcg_at_k
from ..metrics.recall import recall_at_k

class IRBenchmarkEvaluator:
   def __init__(self, doc_ids: List[str], query_ids: List[str], qrels: Dict):
      self.doc_ids = doc_ids
      self.query_ids = query_ids
      self.qrels = qrels
   
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
         
         for i, qid in enumerate(self.query_ids):
               relevant_docs = self.qrels.get(qid, {})
               if not relevant_docs:
                  continue
               
               # Get top-k ranked documents
               ranked_indices = torch.topk(scores[i], k=min(k+10, len(self.doc_ids))).indices.cpu().tolist()
               ranked_doc_ids = [self.doc_ids[idx] for idx in ranked_indices]
               
               # Calculate metrics
               recall = recall_at_k(list(relevant_docs.keys()), self.doc_ids, k)
               ndcg = ndcg_at_k(relevant_docs, self.doc_ids, k)
               ap = average_precision_at_k(list(relevant_docs.keys()), ranked_doc_ids, k)
               
               recalls.append(recall)
               ndcgs.append(ndcg)
               aps.append(ap)
         
         results[f'Recall_at_{k}'] = np.mean(recalls) if recalls else 0.0
         results[f'NDCG_at_{k}'] = np.mean(ndcgs) if ndcgs else 0.0
         results[f'MAP_at_{k}'] = np.mean(aps) if aps else 0.0
      
      return results