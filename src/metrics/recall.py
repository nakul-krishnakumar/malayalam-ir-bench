import numpy as np
from typing import List, Set, Union

def recall_at_k(relevant_docs: Union[List, Set], retrieved_docs: List, k: int) -> float:
   """
   Calculate Recall@K metric.
   
   Recall@K = |relevant ∩ retrieved@k| / |relevant|
   
   Args:
      relevant_docs: List or set of relevant document IDs for a query
      retrieved_docs: List of retrieved document IDs (ranked by relevance)
      k: Number of top documents to consider
      
   Returns:
      Recall@K score (0.0 to 1.0)
   """
   if not relevant_docs:
      return 0.0
      
   relevant_set = set(relevant_docs)
   retrieved_at_k = set(retrieved_docs[:k])
   
   intersection = len(relevant_set & retrieved_at_k)
   return intersection / len(relevant_set)

def recall_at_multiple_k(relevant_docs: Union[List, Set], retrieved_docs: List, 
                        k_values: List[int]) -> dict:
   """
   Calculate Recall@K for multiple K values efficiently.
   
   Args:
      relevant_docs: List or set of relevant document IDs
      retrieved_docs: List of retrieved document IDs (ranked)
      k_values: List of K values to compute recall for
      
   Returns:
      Dictionary with K values as keys and Recall@K as values
   """
   if not relevant_docs:
      return {k: 0.0 for k in k_values}
      
   relevant_set = set(relevant_docs)
   results = {}
   
   for k in sorted(k_values):
      retrieved_at_k = set(retrieved_docs[:k])
      intersection = len(relevant_set & retrieved_at_k)
      results[k] = intersection / len(relevant_set)
   
   return results