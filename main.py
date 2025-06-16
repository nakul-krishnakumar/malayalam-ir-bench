from src.pipelines import run_benchmark_pipeline
import yaml

if __name__ == "__main__":
    model_name = "intfloat/multilingual-e5-base"
    dataset_path = "./datasets/Bharat_NanoMSMARCO"
    config_path = "./src/configs/config.yaml"

    results = run_benchmark_pipeline(
        model_name=model_name,
        dataset_path=dataset_path,
        config_path=config_path,
    )
