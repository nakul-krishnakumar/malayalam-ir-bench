import yaml
import torch
from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingModel():
   def __init__(self, model_name: str, config_path: str = None):
      self.model_name = model_name
      self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model config if provided
      if config_path:
         with open(config_path, 'r') as f:
               configs = yaml.safe_load(f)
               self.config = configs.get(model_name, {})
      else:
         self.config = {}
      
      self.batch_size = self.config.get('batch_size', 32)
      self.max_seq_length = self.config.get('max_seq_length', 512)
      self.cache_folder = self.config.get('cache_folder')
      
      # Load model
      self.model = SentenceTransformer(
         model_name, 
         device=self.device,
         cache_folder=self.cache_folder,
         local_files_only=False
      )
      
      # Set max sequence length if specified
      if hasattr(self.model, 'max_seq_length'):
         self.model.max_seq_length = self.max_seq_length
    
   def encode(
      self, 
      texts: List[str], 
      convert_to_tensor: bool = True, 
      normalize_embeddings: bool = True
   ):
      return self.model.encode(
         texts,
         convert_to_tensor=convert_to_tensor,
         device=self.device,
         normalize_embeddings=normalize_embeddings,
         batch_size=self.batch_size,
      )