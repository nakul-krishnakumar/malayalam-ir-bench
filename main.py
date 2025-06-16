import os
from dotenv import load_dotenv
from huggingface_hub import login
from src.pipelines import run_benchmark_pipeline

load_dotenv()
login(os.getenv("HF_TOKEN"))

if __name__ == "__main__":
    model_name = "intfloat/multilingual-e5-large"
    dataset_path = "./datasets/Bharat_NanoMSMARCO"
    config_path = "./src/configs"

    results = run_benchmark_pipeline(
        model_name=model_name,
        dataset_path=dataset_path,
        config_path=config_path,
    )
