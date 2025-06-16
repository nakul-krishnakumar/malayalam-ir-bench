from typing import List, Set, Union

def precision_at_k(relevant_docs: Union[List, Set], retrieved_docs: List, k: int) -> float:
   """
   Calculate Precision@K metric.
   
   Precision@K = |relevant ∩ retrieved@k| / k
   
   Args:
      relevant_docs: List or set of relevant document IDs
      retrieved_docs: List of retrieved document IDs (ranked by relevance)
      k: Number of top documents to consider
      
   Returns:
      Precision@K score (0.0 to 1.0)
   """
   if not retrieved_docs or k <= 0:
      return 0.0
      
   relevant_set = set(relevant_docs) if relevant_docs else set()
   retrieved_at_k = retrieved_docs[:k]
   
   # Count how many of the top k are relevant
   relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
   
   return relevant_retrieved / min(k, len(retrieved_at_k))


def precision_at_multiple_k(relevant_docs: Union[List, Set], retrieved_docs: List, 
                        k_values: List[int]) -> dict:
   """
   Calculate Precision@K for multiple K values efficiently.
   
   Args:
      relevant_docs: List or set of relevant document IDs
      retrieved_docs: List of retrieved document IDs (ranked)
      k_values: List of K values to compute precision for
      
   Returns:
      Dictionary with K values as keys and Precision@K as values
   """
   if not retrieved_docs:
      return {k: 0.0 for k in k_values}
      
   relevant_set = set(relevant_docs) if relevant_docs else set()
   results = {}
   
   for k in sorted(k_values):
      retrieved_at_k = retrieved_docs[:k]
      relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
      results[k] = relevant_retrieved / min(k, len(retrieved_at_k)) if retrieved_at_k else 0.0
   
   return results