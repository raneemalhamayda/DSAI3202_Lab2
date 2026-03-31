# DSAI3202 Assignment 2 – Model Training & Automation with Azure

## Overview
This assignment implements a full MLOps workflow for the Amazon Electronics review dataset:
code push → Azure DevOps CI → Azure ML training job → MLflow tracking → versioned model → deployed endpoint

## Dataset Splits
| Split | Size | Purpose |
|-------|------|---------|
| Train | 60% | Model training |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final offline evaluation |
| Deployment | 10% | Simulates production data (most recent reviews) |

## Model Choice
**Logistic Regression** (scikit-learn) was chosen for:
- Fast training on CPU compute
- Interpretable outputs
- Works well with high-dimensional sparse features (TF-IDF)

## Features Used
| Feature Group | Dimensions | Description |
|---------------|-----------|-------------|
| SBERT embeddings | 384 | Dense semantic text embeddings |
| TF-IDF vectors | 500 | Sparse word frequency features |
| Sentiment scores | 3 | VADER neg/neu/pos scores |
| Length features | 2 | review_length_chars, review_length_words |
| **Total** | **889** | |

## Hyperparameter Tuning (Sweep Job)
- Search space: C ~ uniform(0.01, 10.0), max_iter ~ choice(500, 1000, 2000)
- Sampling: Random, 6 trials, 2 concurrent
- Best run: C=1.183, max_iter=1000
- Objective: maximize val_accuracy

## Final Model Performance
| Split | Accuracy | F1 Score |
|-------|----------|----------|
| Train | 1.0000 | 1.0000 |
| Validation | 1.0000 | 1.0000 |
| Test | 1.0000 | 1.0000 |
| **Deployment** | **0.8589** | **0.9217** |

## screenshot in the screenshots folder (assingment2_screenshot)

The drop in deployment accuracy (85.9%) compared to test (100%) demonstrates **data drift** — the deployment split contains the most recent reviews, which exhibit different language patterns than the training data.

## Repository Structure
```
├── src/
│   ├── train.py           # Training script
│   ├── score.py           # Scoring script for endpoint
│   └── invoke_endpoint.py # Endpoint invocation with deploy dataset
├── jobs/
│   ├── train_job.yml      # Azure ML command job
│   ├── sweep_job.yml      # Hyperparameter sweep job
│   └── deployment.yml     # Online endpoint deployment config
├── env/
│   ├── conda.yaml         # Training environment
│   └── inference_conda.yml # Inference environment
└── azure-pipelines.yml    # Azure DevOps CI pipeline
```

## Bonus Question Answer
**What are we doing "not correctly" in this assignment?**

The TF-IDF features are computed directly from review text, which inherently encodes the sentiment and opinion expressed in the review. Since the label (rating >= 4) is derived from the same review, the TF-IDF vectors are highly correlated with the label — this is **data leakage**. A model that sees TF-IDF features built from "this product is amazing" will trivially predict a 5-star rating. In a real production system, TF-IDF should either be excluded, or the feature engineering pipeline should ensure no label-correlated information leaks into the feature space. This explains the perfect train/val/test scores and the more realistic deployment score.
