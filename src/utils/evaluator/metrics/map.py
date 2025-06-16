def average_precision_at_k(true_ids, retrieved_ids, k):
   hits = 0
   sum_precisions = 0.0
   for i in range(min(k, len(retrieved_ids))):
      if retrieved_ids[i] in true_ids:
         hits += 1
         sum_precisions += hits / (i + 1)
   return sum_precisions / len(true_ids) if true_ids else 0.0
