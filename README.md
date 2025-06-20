# Data Pipeline with DVC and MLflow for Machine Learning

This project demonstrates how to build an end-to-end machine learning pipeline using DVC (Data Version Control) for data and model versioning, and MLflow for experiment tracking. The pipeline focuses on training a Random Forest Classifier on the Pima Indians Diabetes Dataset, with clear stages for data preprocessing, model training, and evaluation.

## Key Features of the Project

### Data Version Control (DVC)
- Tracks and versions datasets, models, and pipeline stages to ensure reproducibility.
- Pipeline structured into stages (preprocessing, training, evaluation) that auto re-run if dependencies change.
- Supports remote storage (e.g., DagsHub, S3) for large files.

### Experiment Tracking with MLflow
- Logs hyperparameters (e.g., `n_estimators`, `max_depth`) and performance metrics like accuracy.
- Enables easy comparison of different runs and models for optimization.

## Pipeline Stages

### Preprocessing
The `preprocess.py` script reads the raw dataset (`data/raw/data.csv`), performs preprocessing (e.g., renaming columns), and outputs the processed data to `data/processed/data.csv`.

### Training
The `train.py` script trains a Random Forest Classifier on the processed data. The trained model is saved as `models/random_forest.pkl`. Hyperparameters and model details are logged to MLflow.

### Evaluation
The `evaluate.py` script loads the trained model and evaluates its accuracy on the dataset. Evaluation metrics are logged to MLflow.

## Goals
- **Reproducibility:** Ensures consistent results by versioning data, parameters, and code with DVC.
- **Experimentation:** Tracks and compares experiments with MLflow.
- **Collaboration:** Supports teamwork by managing changes via DVC and MLflow.

## Use Cases
- Data Science teams managing reproducible workflows.
- ML researchers iterating experiments efficiently.

## Technology Stack
- Python (data processing, training, evaluation)
- DVC (data, model, pipeline version control)
- MLflow (experiment tracking)
- Scikit-learn (Random Forest Classifier)

---

## Adding Pipeline Stages with DVC

```bash
# Add preprocessing stage
dvc stage add -n preprocess \
  -p preprocess.input,preprocess.output \
  -d src/preprocess.py -d data/raw/data.csv \
  -o data/processed/data.csv \
  python src/preprocess.py

# Add training stage
dvc stage add -n train \
  -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
  -d src/train.py -d data/raw/data.csv \
  -o models/model.pkl \
  python src/train.py

# Add evaluation stage
dvc stage add -n evaluate \
  -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
  python src/evaluate.py
