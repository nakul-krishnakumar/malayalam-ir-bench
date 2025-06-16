import numpy as np
from typing import List, Set, Union

def average_precision_at_k(relevant_docs: Union[List, Set], retrieved_docs: List, k: int) -> float:
   """
   Calculate Average Precision (AP) at rank k.
   
   AP@k = (1/|relevant|) * Σ(Precision@i * rel(i)) for i=1 to k
   where rel(i) = 1 if doc at rank i is relevant, else 0
   
   Args:
      relevant_docs: List or set of relevant document IDs
      retrieved_docs: List of retrieved document IDs (ranked by relevance)
      k: Number of top documents to consider
      
   Returns:
      AP@K score (0.0 to 1.0)
   """
   if not relevant_docs or not retrieved_docs:
      return 0.0
      
   relevant_set = set(relevant_docs)
   
   precision_sum = 0.0
   relevant_found = 0
   
   for i, doc_id in enumerate(retrieved_docs[:k], start=1):
      if doc_id in relevant_set:
         relevant_found += 1
         precision_at_i = relevant_found / i
         precision_sum += precision_at_i
   
   # Average over number of relevant documents (not retrieved documents)
   return precision_sum / len(relevant_set) if relevant_set else 0.0


def map_at_k(queries_relevance: List[Union[List, Set]], 
            queries_retrieved: List[List], k: int) -> float:
   """
   Calculate Mean Average Precision (MAP) at rank k across multiple queries.
   
   MAP@k = (1/|queries|) * Σ(AP@k for each query)
   
   Args:
      queries_relevance: List of relevant docs for each query
      queries_retrieved: List of retrieved docs for each query  
      k: Number of top documents to consider
      
   Returns:
      MAP@K score (0.0 to 1.0)
   """
   if not queries_relevance or not queries_retrieved:
      return 0.0
      
   ap_scores = []
   for relevant, retrieved in zip(queries_relevance, queries_retrieved):
      ap_scores.append(average_precision_at_k(relevant, retrieved, k))
      
   return np.mean(ap_scores)
