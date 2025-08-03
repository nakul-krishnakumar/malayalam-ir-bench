import os
import json
import pandas as pd
from typing import List, Tuple, Dict, Optional

class DataLoader():
   def __init__(self, dataset_path: str):
      self.path = dataset_path
      self.corpus_path = os.path.join(self.path, "corpus.jsonl")
      self.queries_path = os.path.join(self.path, "queries.jsonl")
      self.qrels_path = os.path.join(self.path, "qrels.tsv")

   def fetch_and_save_data(self, corpus_url: str, queries_url: str, qrels_url: str, engine: str = "pyarrow"):
      try:
         os.makedirs(self.path, exist_ok=True)

         df_corpus = pd.read_parquet(corpus_url, engine=engine)
         df_qrels = pd.read_parquet(qrels_url, engine=engine)
         df_queries = pd.read_parquet(queries_url,engine=engine)

         df_corpus.to_json(self.corpus_path, orient="records", lines=True)
         df_queries.to_json(self.queries_path, orient="records", lines=True)
         df_qrels.to_csv(self.qrels_path, sep='\t', index=False)

         print(f"[INFO] Download and saved dataset at: {self.path}")

      except Exception as e:
         print(f"[ERROR] Could not download data")

   def load_corpus(self) -> Optional[Tuple[List[str], List[str]]]:
      try:
         with open(self.corpus_path, "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
            doc_ids: List = [str(doc["_id"]) for doc in docs]
            doc_texts: List = [str(doc["text"]) for doc in docs]
      
         return doc_ids, doc_texts
   
      except Exception as e:
         print(f"[ERROR] Could not load corpus! {e}")
         return None
   
   def load_queries(self) -> Optional[Tuple[List[str], List[str]]]: 
      try:
         with open(self.queries_path, "r", encoding="utf-8") as f:
            queries = [json.loads(line) for line in f]
            query_ids: List = [str(q["_id"]) for q in queries]
            query_texts: List = [str(q["text"]) for q in queries]
         
         return query_ids, query_texts

      except Exception as e:
         print(f"[ERROR] Could not load queries! {e}")
         return None

   def load_qrels(self) -> Optional[Dict]:
      try:
         df_qrels = pd.read_csv(self.qrels_path, sep='\t')
         qrels: Dict = {}
         for _, row in df_qrels.iterrows():
            qid = str(row['query-id'])
            did = str(row['corpus-id'])
            score = int(row['score']) if 'score' in df_qrels.columns else 1 # relevance score defaults to 1 if not present
            # Initialize qrels dict
            qrels.setdefault(qid, {})[did] = score 
         # Convert to dict of dicts
         return qrels

      except Exception as e:
         print(f"[ERROR] Could not load qrels! {e}")
         return None

   def load_data(self):
      return self.load_corpus(), self.load_queries(), self.load_qrels()