# Malayalam IR Benchmark

## Directory Structure

```
malayalam-ir-bench/
├── .venv/                # Python virtual environment
├── datasets/             # Datasets for IR tasks
├── mlruns/               # MLflow runs and experiment tracking
├── models/               # Saved and baseline models
├── notebooks/            # Jupyter notebooks for experiments
│   └── experiment.ipynb
├── src/                  # Source code
│   ├── configs/          # Configuration files
│   ├── pipelines/        # Benchmarking and pipeline scripts
│   │   ├── __init__.py
│   │   └── benchmark.py
│   ├── utils/            # Utility functions
│   ├── data_loader/      # Data loading modules
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── encoder/          # Model encoder modules
│   │   ├── __init__.py
│   │   └── model.py
│   ├── evaluator/        # Evaluation metrics and logic
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── metrics.py
│   └── exp_runner/       # Experiment runner scripts
│       ├── __init__.py
│       └── run.py
├── .env                  # Environment variables
├── .gitignore
├── .python-version
├── main.py               # Main entry point
├── pyproject.toml
├── readme.md             # Project documentation
├── README.md             # (Duplicate, see readme.md)
├── uv.lock
```

## Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/malayalam-ir-bench.git
    cd malayalam-ir-bench
    ```

2. **Install dependencies:**

    ```bash
    uv sync
    ```

3. **Download datasets:**

    - Place datasets in the `datasets/` directory.

4. **Run experiments:**
    - Use `main.py` to start experiments.

## Usage

-   **Notebooks:**  
    Explore and prototype in `notebooks/experiment.ipynb`.
-   **Pipelines:**  
    Use `src/pipelines/benchmark.py` for benchmarking.
-   **Data Loading:**  
    Handled by `src/data_loader/data_loader.py`.
-   **Model Encoders:**  
    Implemented in `src/encoder/model.py`.
-   **Evaluation:**  
    Use `src/evaluator/evaluator.py` and `src/evaluator/metrics.py`.


## Citations of artifacts used
```
@misc{bharat-nanobeir,
  title={Bharat-NanoBEIR: Indian Language Information Retrieval Datasets},
  year={2024},
  url={https://huggingface.co/datasets/carlfeynman/Bharat_NanoMSMARCO_ml}
}
```