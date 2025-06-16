import numpy as np
from typing import List, Dict, Union

def dcg_at_k(relevance_scores: List[float], k: int) -> float:
   """
   Calculate Discounted Cumulative Gain (DCG) at rank k.
   
   DCG@k = rel_1 + Σ(rel_i / log2(i+1)) for i=2 to k
   
   Args:
      relevance_scores: List of relevance scores for retrieved documents
      k: Number of top documents to consider
      
   Returns:
      DCG@K score
   """
   if not relevance_scores or k <= 0:
      return 0.0
      
   # Take only top k scores
   scores = relevance_scores[:k]
   
   if len(scores) == 0:
      return 0.0
      
   # First document doesn't get discounted
   dcg = scores[0]
   
   # Apply logarithmic discount to remaining documents
   for i, score in enumerate(scores[1:], start=2):
      dcg += score / np.log2(i)
      
   return dcg


def ideal_dcg_at_k(relevance_scores: List[float], k: int) -> float:
   """
   Calculate Ideal DCG (IDCG) at rank k.
   
   IDCG is DCG calculated on perfectly ranked documents (sorted by relevance).
   
   Args:
      relevance_scores: List of all available relevance scores
      k: Number of top documents to consider
      
   Returns:
      IDCG@K score
   """
   if not relevance_scores:
      return 0.0
      
   # Sort relevance scores in descending order for ideal ranking
   ideal_scores = sorted(relevance_scores, reverse=True)
   return dcg_at_k(ideal_scores, k)


def ndcg_at_k(relevant_docs: Dict[str, float], retrieved_docs: List[str], k: int) -> float:
   """
   Calculate Normalized Discounted Cumulative Gain (NDCG) at rank k.
   
   NDCG@k = DCG@k / IDCG@k
   
   Args:
      relevant_docs: Dictionary mapping doc_id to relevance score
      retrieved_docs: List of retrieved document IDs (ranked by system)
      k: Number of top documents to consider
      
   Returns:
      NDCG@K score (0.0 to 1.0)
   """
   if not relevant_docs or not retrieved_docs:
      return 0.0
   
   # Get relevance scores for retrieved documents
   retrieved_relevance = [relevant_docs.get(doc_id, 0.0) for doc_id in retrieved_docs[:k]]
   
   # Calculate DCG for retrieved ranking
   dcg = dcg_at_k(retrieved_relevance, k)
   
   # Calculate IDCG (ideal ranking)
   all_relevance_scores = list(relevant_docs.values())
   idcg = ideal_dcg_at_k(all_relevance_scores, k)
   
   # Avoid division by zero
   if idcg == 0.0:
      return 0.0
      
   return dcg / idcg


def ndcg_at_multiple_k(relevant_docs: Dict[str, float], retrieved_docs: List[str], 
                     k_values: List[int]) -> dict:
   """
   Calculate NDCG@K for multiple K values efficiently.
   
   Args:
      relevant_docs: Dictionary mapping doc_id to relevance score
      retrieved_docs: List of retrieved document IDs (ranked)
      k_values: List of K values to compute NDCG for
      
   Returns:
      Dictionary with K values as keys and NDCG@K as values
   """
   if not relevant_docs or not retrieved_docs:
      return {k: 0.0 for k in k_values}
   
   results = {}
   all_relevance_scores = list(relevant_docs.values())
   
   for k in k_values:
      retrieved_relevance = [relevant_docs.get(doc_id, 0.0) for doc_id in retrieved_docs[:k]]
      dcg = dcg_at_k(retrieved_relevance, k)
      idcg = ideal_dcg_at_k(all_relevance_scores, k)
      
      results[k] = dcg / idcg if idcg > 0 else 0.0
   
   return results