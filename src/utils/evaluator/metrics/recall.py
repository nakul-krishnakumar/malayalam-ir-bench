def recall_at_k(true_ids, retrieved_ids, k):
   """
   Computes Recall@k for a single query.
   """
   return len(set(true_ids) & set(retrieved_ids[:k])) / len(true_ids) if true_ids else 0.0
