import numpy as np
from typing import List, Set, Union

def mrr_at_k(relevant_docs: Union[List, Set], retrieved_docs: List, k: int) -> float:
   """
   Calculate Mean Reciprocal Rank (MRR) at rank k.
   
   MRR@k = 1/rank of first relevant document (if within top k, else 0)
   
   Args:
      relevant_docs: List or set of relevant document IDs
      retrieved_docs: List of retrieved document IDs (ranked by relevance)
      k: Number of top documents to consider
      
   Returns:
      MRR@K score (0.0 to 1.0)
   """
   if not relevant_docs or not retrieved_docs:
      return 0.0
      
   relevant_set = set(relevant_docs)
   
   # Find rank of first relevant document
   for rank, doc_id in enumerate(retrieved_docs[:k], start=1):
      if doc_id in relevant_set:
         return 1.0 / rank
         
   return 0.0  # No relevant document found in top k


def mrr_multiple_queries(queries_relevance: List[Union[List, Set]], 
                     queries_retrieved: List[List], k: int) -> float:
   """
   Calculate Mean Reciprocal Rank across multiple queries.
   
   Args:
      queries_relevance: List of relevant docs for each query
      queries_retrieved: List of retrieved docs for each query
      k: Number of top documents to consider
      
   Returns:
      Average MRR@K across all queries
   """
   if not queries_relevance or not queries_retrieved:
      return 0.0
      
   mrr_scores = []
   for relevant, retrieved in zip(queries_relevance, queries_retrieved):
      mrr_scores.append(mrr_at_k(relevant, retrieved, k))
      
   return np.mean(mrr_scores)