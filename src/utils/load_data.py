import os
import json
import pandas as pd
from typing import List, Tuple, Dict

class DataLoader:
   def __init__(self, dataset_path: str):
      self.dataset_path = dataset_path
    
   def download_and_save_data(self, corpus_path: str, qrels_path: str, queries_path: str, engine: str = "pyarrow"):
      """Download data from HuggingFace and save locally"""
      # Load the data
      df_corpus = pd.read_parquet(corpus_path, engine=engine)
      df_queries = pd.read_parquet(queries_path, engine=engine)
      df_qrels = pd.read_parquet(qrels_path, engine=engine)

      # Ensure directory exists
      os.makedirs(self.dataset_path, exist_ok=True)
      
      # Save as JSONL
      df_corpus.to_json(f"{self.dataset_path}/corpus.jsonl", orient="records", lines=True)
      df_queries.to_json(f"{self.dataset_path}/queries.jsonl", orient="records", lines=True)
      df_qrels.to_csv(f"{self.dataset_path}/qrels.tsv", sep='\t', index=False)

   
   def load_corpus(self) -> Tuple[List[str], List[str]]:
      """Load corpus documents"""
      with open(f"{self.dataset_path}/corpus.jsonl", "r", encoding="utf-8") as f:
         docs = [json.loads(line) for line in f]
         doc_texts = [doc["text"] for doc in docs]
         doc_ids = [str(doc["_id"]) for doc in docs]
      return doc_texts, doc_ids
   
   def load_queries(self) -> Tuple[List[str], List[str]]:
      """Load queries"""
      with open(f"{self.dataset_path}/queries.jsonl", "r", encoding="utf-8") as f:
         queries = [json.loads(line) for line in f]
         query_texts = [q["text"] for q in queries]
         query_ids = [str(q["_id"]) for q in queries]
      return query_texts, query_ids
   
   def load_qrels(self) -> Dict[str, Dict[str, int]]:
      """Load relevance judgments"""
      df_qrels = pd.read_csv(f"{self.dataset_path}/qrels.tsv", sep='\t')
      qrels = {}
      for _, row in df_qrels.iterrows():
         qid = str(row['query-id'])
         did = str(row['corpus-id'])
         qrels.setdefault(qid, {})[did] = 1
      
      print(f"Loaded qrels for {len(qrels)} queries")  # Debug info
      return qrels
   
   def load_data(self):
      return self.load_corpus(), self.load_queries(), self.load_qrels()