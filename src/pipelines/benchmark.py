from src.utils import DataLoader, ExperimentRunner
from typing import List
import mlflow
import mlflow.pytorch
import json
import os
from datetime import datetime

def run_benchmark_pipeline(
   model_name: str, 
   dataset_path: str, 
   config_path: str, 
   k_values: List[int] = [1, 5, 10], 
   download_dataset: bool = False, 
   corpus_path: str = None, 
   qrels_path: str = None, 
   queries_path: str = None, 
   engine: str = "pyarrow"
):
   # Set experiment name
   mlflow.set_experiment("Malayalam-IR-Benchmark")
   
   with mlflow.start_run(run_name=f"{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
      
      # Log parameters
      mlflow.log_param("model_name", model_name)
      mlflow.log_param("dataset_path", dataset_path)
      mlflow.log_param("config_path", config_path)
      mlflow.log_param("k_values", k_values)
      mlflow.log_param("download_dataset", download_dataset)
      mlflow.log_param("engine", engine)
      mlflow.log_param("timestamp", datetime.now().isoformat())
      
      # Initialize experiment runner
      runner = ExperimentRunner(config_path=config_path)
      
      # Log model configuration
      try:
         import yaml
         with open(config_path, 'r') as f:
               configs = yaml.safe_load(f)
               model_config = configs.get(model_name, {})
               
         mlflow.log_param("batch_size", model_config.get('batch_size', 'default'))
         mlflow.log_param("max_seq_length", model_config.get('max_seq_length', 'default'))
         
         # Save model config as artifact
         config_file = "model_config.json"
         with open(config_file, 'w') as f:
               json.dump(model_config, f, indent=2)
         mlflow.log_artifact(config_file)
         os.remove(config_file)  # Clean up temp file
         
      except Exception as e:
         mlflow.log_param("config_error", str(e))

      # Setup data
      data_loader = DataLoader(dataset_path=dataset_path)
      
      if download_dataset:
         mlflow.log_param("data_download", "performed")
         data_loader.download_and_save_data(
               corpus_path=corpus_path, 
               qrels_path=qrels_path, 
               queries_path=queries_path, 
               engine=engine
         )
      else:
         mlflow.log_param("data_download", "skipped")
      
      # Log dataset statistics
      try:
         doc_texts, doc_ids = data_loader.load_corpus()
         query_texts, query_ids = data_loader.load_queries()
         qrels = data_loader.load_qrels()
         
         mlflow.log_param("num_documents", len(doc_texts))
         mlflow.log_param("num_queries", len(query_texts))
         mlflow.log_param("num_query_doc_pairs", sum(len(docs) for docs in qrels.values()))
         mlflow.log_param("avg_doc_length", sum(len(doc.split()) for doc in doc_texts) / len(doc_texts))
         mlflow.log_param("avg_query_length", sum(len(q.split()) for q in query_texts) / len(query_texts))
         
      except Exception as e:
         mlflow.log_param("dataset_stats_error", str(e))

      # Run experiment
      print(f"Running benchmark for {model_name}")
      results = runner.run_single_model(
         model_name=model_name,
         dataset_path=dataset_path,
         k_values=k_values
      )

      # Log all metrics
      print("Final Results:")
      for metric, score in results.items():
         print(f"{metric}: {score:.3f}")
         mlflow.log_metric(metric, score)
      
      # Log additional computed metrics
      for k in k_values:
         if f'Recall@{k}' in results and f'MAP@{k}' in results:
               f1_score = 2 * (results[f'Recall@{k}'] * results[f'MAP@{k}']) / (results[f'Recall@{k}'] + results[f'MAP@{k}']) if (results[f'Recall@{k}'] + results[f'MAP@{k}']) > 0 else 0
               mlflow.log_metric(f"F1@{k}", f1_score)
      
      # Save detailed results as artifact
      results_file = "detailed_results.json"
      detailed_results = {
         "model_name": model_name,
         "dataset_path": dataset_path,
         "metrics": results,
         "k_values": k_values,
         "timestamp": datetime.now().isoformat(),
         "model_config": model_config if 'model_config' in locals() else {}
      }
      
      with open(results_file, 'w') as f:
         json.dump(detailed_results, f, indent=2)
      mlflow.log_artifact(results_file)
      os.remove(results_file)
      
      # Save model information as artifact
      model_info_file = "model_info.txt"
      with open(model_info_file, 'w') as f:
         f.write(f"Model: {model_name}\n")
         f.write(f"Dataset: {dataset_path}\n")
         f.write(f"Configuration: {config_path}\n")
         f.write(f"K Values: {k_values}\n")
         f.write(f"Evaluation Date: {datetime.now().isoformat()}\n\n")
         f.write("Results:\n")
         for metric, score in results.items():
               f.write(f"  {metric}: {score:.3f}\n")
      
      mlflow.log_artifact(model_info_file)
      os.remove(model_info_file)
      
      # Log the actual model (if possible)
      try:
         # This will depend on your ExperimentRunner implementation
         # You might need to modify it to return the model object
         model_obj = runner.get_model(model_name)  # You'll need to implement this
         mlflow.pytorch.log_model(model_obj, "model")
      except Exception as e:
         mlflow.log_param("model_logging_error", str(e))
      
      # Set tags for better organization
      mlflow.set_tag("model_family", model_name.split('/')[0] if '/' in model_name else "unknown")
      mlflow.set_tag("language", "malayalam")
      mlflow.set_tag("task", "information_retrieval")
      mlflow.set_tag("dataset", os.path.basename(dataset_path))
      
      print(f"✅ Experiment logged to MLflow with run ID: {mlflow.active_run().info.run_id}")
      
      return results
