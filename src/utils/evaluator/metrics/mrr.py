def mrr_at_k(true_ids, retrieved_ids, k):
   for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
      if doc_id in true_ids:
         return 1.0 / rank
   return 0.0