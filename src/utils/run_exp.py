import torch
import yaml
from typing import Dict, List
from src.utils import DataLoader, EmbeddingModel, IRBenchmarkEvaluator

class ExperimentRunner:
   def __init__(self, config_path: str = "./src/configs/config.yaml"):
      self.config_path = config_path
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
      
      # Enable CUDA optimization
      if torch.cuda.is_available():
         torch.backends.cudnn.benchmark = True
      
      print(f"Using device: {self.device}")
      if self.device == "cuda":
         print(f"GPU: {torch.cuda.get_device_name(0)}")
   
   def run_single_model(self, model_name: str, dataset_path: str, k_values: List[int] = [1, 5, 10]) -> Dict:
      """Run evaluation for a single model"""
      print(f"\nEvaluating model: {model_name}")
      
      # Load data
      data_loader = DataLoader(dataset_path)
      doc_texts, doc_ids = data_loader.load_corpus()
      query_texts, query_ids = data_loader.load_queries()
      qrels = data_loader.load_qrels()
      
      print(f"Loaded {len(doc_texts)} documents, {len(query_texts)} queries")
      
      # Load model
      model = EmbeddingModel(model_name, self.config_path)
      
      # Encode texts
      print("Encoding documents...")
      doc_embeddings = model.encode(doc_texts)
      
      print("Encoding queries...")
      query_embeddings = model.encode(query_texts)
      
      # Evaluate
      print("Computing metrics...")
      evaluator = IRBenchmarkEvaluator(doc_ids, query_ids, qrels)
      results = evaluator.evaluate_model(query_embeddings, doc_embeddings, k_values)
      
      return results
   
   def run_all_models(self, dataset_path: str, k_values: List[int] = [1, 5, 10]) -> Dict:
      """Run evaluation for all models in config"""
      with open(self.config_path, 'r') as f:
         configs = yaml.safe_load(f)
      
      all_results = {}
      for model_name in configs.keys():
         try:
               results = self.run_single_model(model_name, dataset_path, k_values)
               all_results[model_name] = results
               
               print(f"\nResults for {model_name}:")
               for metric, score in results.items():
                  print(f"  {metric}: {score:.3f}")
                  
         except Exception as e:
               print(f"Error evaluating {model_name}: {e}")
               all_results[model_name] = {"error": str(e)}
      
      return all_results