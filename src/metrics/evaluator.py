import numpy as np
from typing import Dict, List, Union, Set
from .recall import recall_at_k, recall_at_multiple_k
from .ndcg import ndcg_at_k, ndcg_at_multiple_k
from .mrr import mrr_at_k
from .map import map_at_k
from .precision import precision_at_k

class IRMetricsEvaluator:
   """
   Complete Information Retrieval metrics evaluator.
   Supports batch evaluation across multiple queries.
   """
   
   def __init__(self, k_values: List[int] = [5, 10, 20]):
      """
      Initialize evaluator with specific K values.
      
      Args:
         k_values: List of K values to evaluate metrics at
      """
      self.k_values = k_values
   
   def evaluate_single_query(self, relevant_docs: Union[List, Set, Dict], 
                           retrieved_docs: List[str]) -> Dict[str, float]:
      """
      Evaluate all metrics for a single query.
      
      Args:
         relevant_docs: Relevant documents (list/set of IDs or dict with relevance scores)
         retrieved_docs: Retrieved document IDs ranked by relevance
         
      Returns:
         Dictionary with metric names as keys and scores as values
      """
      results = {}
      
      # Handle different input formats for relevant docs
      if isinstance(relevant_docs, dict):
         # For NDCG (needs relevance scores)
         relevant_ids = set(relevant_docs.keys())
         relevance_dict = relevant_docs
      else:
         # For other metrics (just need IDs)
         relevant_ids = set(relevant_docs) if relevant_docs else set()
         relevance_dict = {doc_id: 1.0 for doc_id in relevant_ids}
      
      # Calculate metrics for each K value
      for k in self.k_values:
         # Recall@K
         results[f'recall@{k}'] = recall_at_k(relevant_ids, retrieved_docs, k)
         
         # Precision@K  
         results[f'precision@{k}'] = precision_at_k(relevant_ids, retrieved_docs, k)
         
         # NDCG@K
         results[f'ndcg@{k}'] = ndcg_at_k(relevance_dict, retrieved_docs, k)
         
         # MRR@K
         results[f'mrr@{k}'] = mrr_at_k(relevant_ids, retrieved_docs, k)
      
      return results
   
   def evaluate_multiple_queries(self, queries_data: List[Dict]) -> Dict[str, float]:
      """
      Evaluate metrics across multiple queries and return averages.
      
      Args:
         queries_data: List of dicts, each containing:
               - 'relevant': relevant docs (list/set/dict)
               - 'retrieved': retrieved docs (list)
               
      Returns:
         Dictionary with average metric scores
      """
      all_results = []
      
      for query_data in queries_data:
         query_results = self.evaluate_single_query(
               query_data['relevant'], 
               query_data['retrieved']
         )
         all_results.append(query_results)
      
      # Calculate averages
      averaged_results = {}
      if all_results:
         for metric in all_results[0].keys():
               scores = [result[metric] for result in all_results]
               averaged_results[metric] = np.mean(scores)
               averaged_results[f'{metric}_std'] = np.std(scores)
      
      return averaged_results
   
   def get_metric_summary(self, results: Dict[str, float]) -> str:
      """
      Generate a formatted summary of results.
      
      Args:
         results: Dictionary of metric results
         
      Returns:
         Formatted string summary
      """
      summary = "🎯 IR Metrics Summary\n"
      summary += "=" * 50 + "\n"
      
      for k in self.k_values:
         summary += f"\n📊 Top-{k} Results:\n"
         summary += f"   Recall@{k}:    {results.get(f'recall@{k}', 0):.3f}\n"
         summary += f"   Precision@{k}: {results.get(f'precision@{k}', 0):.3f}\n" 
         summary += f"   NDCG@{k}:      {results.get(f'ndcg@{k}', 0):.3f}\n"
         summary += f"   MRR@{k}:       {results.get(f'mrr@{k}', 0):.3f}\n"
      
      return summary